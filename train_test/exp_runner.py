from __future__ import annotations

import gc
import itertools
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_processing.data_processing import (
    MultimodalCollator,
    PKBatchSampler,
    RVLCDIPCachedDataset,
    get_smooth_labels,
    get_transforms,
)
from model.DIC_without_groupwise import DICWithoutGroupwise
from model.modules.ee_modules import ExitConfig, compute_ece
from model.modules.layoutlmv3 import LayoutLMv3EEForSequenceClassification
from utils.config import build_adamw_param_groups
from utils.gradient_conflict import (
    GradientConflictProber,
    compute_2d_loss_landscape,
    enable_layoutlmv3_checkpointing,
    extract_shared_layer_param_groups,
)
from utils.lars_wrapper import LARSWrapper
from utils.losses import MultiExitCELoss, MultiExitSupConLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _count_subgraph_params(model: nn.Module, exit_layer: int) -> int:
    if hasattr(model, "layoutlmv3"):
        base = model.layoutlmv3
    elif hasattr(model, "backbone") and hasattr(model.backbone, "model"):
        base = model.backbone.model
    else:
        base = model

    total = 0
    for name, p in base.named_parameters():
        if "early_exits" in name or "exit_embeddings" in name or "lte_" in name:
            continue
        if name.startswith("encoder.layer."):
            parts = name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_num = int(parts[2]) + 1
                if layer_num <= exit_layer:
                    total += p.numel()
        elif name.startswith("encoder."):
            if exit_layer >= 1:
                total += p.numel()
        else:
            total += p.numel()
    return max(1, total)


# Source: Subgraph weighting Eq. (2) in Hamed et al. (ICDAR 2024, arXiv:2405.12705) & params_per_exit/EETrainer in https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/models/EE_modules.py
def compute_exit_weights(
    model: nn.Module,
    exit_layers: Sequence[int],
    weighting_mode: str = "subgraph_paper",
    gamma: float = 0.5,
) -> Dict[int, float]:
    sorted_layers = sorted(int(l) for l in exit_layers)
    k = len(sorted_layers)
    if k == 0:
        return {}
    if k == 1 or weighting_mode == "uniform":
        return {l: 1.0 / k for l in sorted_layers}

    inv_counts = {l: 1.0 / float(_count_subgraph_params(model, l)) for l in sorted_layers}

    if weighting_mode == "subgraph_paper":
        early_layers = sorted_layers[:-1]
        final_layer = sorted_layers[-1]
        early_denom = sum(inv_counts[l] for l in early_layers) or 1.0
        weights: Dict[int, float] = {}
        for l in early_layers:
            w_sub = inv_counts[l] / early_denom
            weights[l] = gamma * w_sub if gamma != 0.0 else w_sub
        weights[final_layer] = (1.0 - gamma) if gamma != 0.0 else 1.0
        return weights

    if weighting_mode == "subgraph_repo":
        early_layers = sorted_layers[:-1]
        final_layer = sorted_layers[-1]
        early_denom = sum(inv_counts[l] for l in early_layers) or 1.0
        weights = {}
        num_early = len(early_layers)
        for l in early_layers:
            w_sub = inv_counts[l] / early_denom
            weights[l] = (gamma / float(num_early)) * w_sub if gamma != 0.0 else w_sub
        weights[final_layer] = (1.0 - gamma) if gamma != 0.0 else 1.0
        return weights

    raise ValueError(f"Unsupported weighting_mode: {weighting_mode}")


def _build_linear_warmup_scheduler(
    optimizer: Any,
    total_steps: int,
    warmup_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio > 0.0 and total_steps > 1 else 0

    def _lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        denom = float(max(1, total_steps - warmup_steps))
        return max(0.0, float(total_steps - current_step) / denom)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _build_cosine_warmup_scheduler(
    optimizer: Any,
    total_steps: int,
    warmup_ratio: float = 0.1,
    warmup_start_lr_ratio: float = 0.1,
    eta_min_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio > 0.0 and total_steps > 1 else 0

    def _lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            frac = float(current_step + 1) / float(warmup_steps)
            return warmup_start_lr_ratio + (1.0 - warmup_start_lr_ratio) * frac
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _create_synthetic_batches(
    num_samples: int,
    batch_size: int,
    num_classes: int,
    is_contrastive: bool,
    seed: int = 42,
    seq_len: int = 32,
) -> List[Dict[str, Any]]:
    gen = torch.Generator().manual_seed(seed)
    batches: List[Dict[str, Any]] = []
    num_batches = max(1, (num_samples + batch_size - 1) // batch_size)
    for b_idx in range(num_batches):
        cur_bs = min(batch_size, num_samples - b_idx * batch_size)
        if cur_bs < 2:
            cur_bs = batch_size
        labels = torch.tensor([(b_idx * batch_size + i) % num_classes for i in range(cur_bs)], dtype=torch.long)
        input_ids = torch.randint(0, 1000, (cur_bs, seq_len), generator=gen, dtype=torch.long)
        attention_mask = torch.ones((cur_bs, seq_len), dtype=torch.long)
        bbox = torch.zeros((cur_bs, seq_len, 4), dtype=torch.long)
        if is_contrastive:
            p1 = torch.randn((cur_bs, 3, 224, 224), generator=gen)
            p2 = p1 + 0.1 * torch.randn((cur_bs, 3, 224, 224), generator=gen)
            batches.append(
                {
                    "view1": {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "bbox": bbox,
                        "pixel_values": p1,
                    },
                    "view2": {
                        "input_ids": input_ids.clone(),
                        "attention_mask": attention_mask.clone(),
                        "bbox": bbox.clone(),
                        "pixel_values": p2,
                    },
                    "labels": labels,
                }
            )
        else:
            p1 = torch.randn((cur_bs, 3, 224, 224), generator=gen)
            batches.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "bbox": bbox,
                    "pixel_values": p1,
                    "labels": labels,
                }
            )
    return batches


def _build_loaders(
    data_dir: Union[str, Path],
    batch_size: int,
    num_workers: int,
    is_contrastive: bool,
    max_samples: Optional[int],
    dry_run: bool,
    num_classes: int,
    seed: int,
    eval_batch_size: Optional[int] = None,
    sampler_type: str = "random",
    pk_p: int = 4,
    pk_k: int = 2,
    transform: Optional[Any] = None,
) -> Tuple[Any, Any]:
    val_bs = eval_batch_size if eval_batch_size is not None else batch_size
    if dry_run:
        n_train = max(batch_size * 2, min(max_samples or 4, 8))
        n_val = max(val_bs, min(max_samples or 4, 4))
        train_loader = _create_synthetic_batches(
            num_samples=n_train,
            batch_size=batch_size,
            num_classes=num_classes,
            is_contrastive=is_contrastive,
            seed=seed,
        )
        val_loader = _create_synthetic_batches(
            num_samples=n_val,
            batch_size=val_bs,
            num_classes=num_classes,
            is_contrastive=is_contrastive,
            seed=seed + 100,
        )
        return train_loader, val_loader

    collator = MultimodalCollator(max_length=512)
    train_tf = transform if transform is not None else (get_transforms() if is_contrastive else None)
    val_tf = train_tf if is_contrastive else None
    train_ds = RVLCDIPCachedDataset(
        data_dir=data_dir,
        split="train",
        transform=train_tf,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )
    val_ds = RVLCDIPCachedDataset(
        data_dir=data_dir,
        split="val",
        transform=val_tf,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )
    pin = torch.cuda.is_available()
    if sampler_type == "pk" and len(train_ds) > 0:
        train_labels = [int(r["label"]) for r in train_ds.records]
        batch_sampler = PKBatchSampler(train_labels, p=pk_p, k=pk_k, seed=seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin,
            drop_last=is_contrastive and len(train_ds) >= batch_size,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin,
        drop_last=False,
    )
    return train_loader, val_loader


def _move_view(view: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {
        "input_ids": view["input_ids"].to(device, non_blocking=True),
        "bbox": view["bbox"].to(device, non_blocking=True),
        "pixel_values": view["pixel_values"].to(device, non_blocking=True),
    }
    if view.get("attention_mask") is not None:
        out["attention_mask"] = view["attention_mask"].to(device, non_blocking=True)
    return out


def _evaluate_ee_model(
    model: LayoutLMv3EEForSequenceClassification,
    val_loader: Any,
    exit_layers: Sequence[int],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    model.eval()
    logits_buf: Dict[int, List[torch.Tensor]] = {l: [] for l in exit_layers}
    labels_buf: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = _move_view(batch, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                outputs = model(**inputs)
            for idx, l in enumerate(exit_layers[:-1]):
                st = outputs.exit_states[idx]
                lg = st[0] if isinstance(st, tuple) else st
                logits_buf[l].append(lg.detach().float().cpu())
            logits_buf[exit_layers[-1]].append(outputs.logits.detach().float().cpu())
            labels_buf.append(labels.detach().cpu())

    all_labels = torch.cat(labels_buf, dim=0).numpy()
    per_exit_acc: Dict[str, float] = {}
    per_exit_ece: Dict[str, float] = {}
    for l in exit_layers:
        all_logits = torch.cat(logits_buf[l], dim=0).numpy()
        preds = np.argmax(all_logits, axis=-1)
        per_exit_acc[str(l)] = float(np.mean(preds == all_labels) * 100.0)
        per_exit_ece[str(l)] = float(compute_ece(all_labels, all_logits))

    model.train()
    return {"accuracy": per_exit_acc, "ece": per_exit_ece}


def _evaluate_dic_model(
    model: DICWithoutGroupwise,
    val_loader: Any,
    exit_layers: Sequence[int],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    model.eval()
    logits_buf: Dict[int, List[torch.Tensor]] = {l: [] for l in exit_layers}
    labels_buf: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = _move_view(batch, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                out = model(**inputs, compute_projections=False, compute_logits=True)
            for l in exit_layers:
                logits_buf[l].append(out.logits_dict[l].detach().float().cpu())
            labels_buf.append(labels.detach().cpu())

    all_labels = torch.cat(labels_buf, dim=0).numpy()
    per_exit_acc: Dict[str, float] = {}
    per_exit_ece: Dict[str, float] = {}
    for l in exit_layers:
        all_logits = torch.cat(logits_buf[l], dim=0).numpy()
        preds = np.argmax(all_logits, axis=-1)
        per_exit_acc[str(l)] = float(np.mean(preds == all_labels) * 100.0)
        per_exit_ece[str(l)] = float(compute_ece(all_labels, all_logits))

    model.train()
    if model.is_encoder_frozen:
        model.backbone.eval()
    return {"accuracy": per_exit_acc, "ece": per_exit_ece}


# Source: Baseline EE training (one_stage_subgraphs_weighted, AdamW, max_confidence) from https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/configs.py & https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/IC_only.py
def run_exp1_hamed_ce(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
    num_classes: int = 16,
    weighting_mode: str = "subgraph_repo",
    gamma: float = 0.5,
    batch_size: int = 2,
    eval_batch_size: int = 1,
    grad_accum_steps: int = 1,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.0,
    scheduler_type: str = "linear",
    max_grad_norm: float = 1.0,
    probe_every_n_steps: int = 10,
    landscape_grid_size: int = 11,
    landscape_span: float = 0.5,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "cuda",
    use_amp: bool = True,
    use_checkpointing: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    dev = _resolve_device(device)
    amp_dt = _amp_dtype(dev)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_exits = sorted({int(l) for l in exit_layers})
    if not sorted_exits or any(l < 0 or l > 12 for l in sorted_exits):
        raise ValueError(f"exit_layers must be integers in [0, 12], got {exit_layers}")
    if sorted_exits[-1] != 12:
        raise ValueError(
            f"LayoutLMv3EEForSequenceClassification requires final exit layer 12, got {sorted_exits}"
        )
    early_ee_spec: List[Union[str, int]] = [
        "text_visual_concat" if l == 0 else l for l in sorted_exits[:-1]
    ]
    ee_cfg = ExitConfig(
        training_strategy="one_stage_subgraphs_weighted",
        inference_strategy="max_confidence",
        exits=early_ee_spec,
        encoder_layer_strategy="ramp",
        exit_head_num_layers=2,
        gamma=gamma,
    )
    model = LayoutLMv3EEForSequenceClassification.from_local_checkpoint(
        num_labels=num_classes,
        ee_config=ee_cfg,
    ).to(dev)

    if use_checkpointing:
        enable_layoutlmv3_checkpointing(model)

    exit_weights = compute_exit_weights(
        model=model,
        exit_layers=sorted_exits,
        weighting_mode=weighting_mode,
        gamma=gamma,
    )

    probe_layers = tuple(sorted_exits[:-1])
    probe_groups = extract_shared_layer_param_groups(model, probe_layers=probe_layers)
    prober = GradientConflictProber(probe_param_groups=probe_groups)

    train_loader, val_loader = _build_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        is_contrastive=False,
        max_samples=max_samples,
        dry_run=dry_run,
        num_classes=num_classes,
        seed=seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda" and amp_dt == torch.float16)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)))
    total_steps = steps_per_epoch * epochs
    if scheduler_type == "linear":
        scheduler = _build_linear_warmup_scheduler(optimizer, total_steps=total_steps, warmup_ratio=warmup_ratio)
    elif scheduler_type in ("cosine", "warmup_cosine"):
        scheduler = _build_cosine_warmup_scheduler(optimizer, total_steps=total_steps, warmup_ratio=warmup_ratio)
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t0 = time.perf_counter()
    epoch_logs: List[Dict[str, Any]] = []
    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        running_total = 0.0
        running_per_exit = {l: 0.0 for l in sorted_exits}
        n_batches = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            global_step += 1
            n_batches += 1
            labels = batch["labels"].to(dev, non_blocking=True)
            inputs = _move_view(batch, dev)

            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                outputs = model(**inputs, labels=labels)
                layer_losses: Dict[int, torch.Tensor] = {}
                for i, l in enumerate(sorted_exits[:-1]):
                    layer_losses[l] = outputs.exit_losses[i]
                layer_losses[sorted_exits[-1]] = outputs.loss

                total_loss = torch.zeros((), device=dev, dtype=outputs.loss.dtype)
                for l, lv in layer_losses.items():
                    total_loss = total_loss + exit_weights[l] * lv

            if global_step == 1 or (probe_every_n_steps > 0 and global_step % probe_every_n_steps == 0):
                prober.probe(step=global_step, exit_losses=layer_losses, weights=exit_weights)

            scaler.scale(total_loss / float(max(1, grad_accum_steps))).backward()
            running_total += float(total_loss.detach().float().item())
            for l, lv in layer_losses.items():
                running_per_exit[l] += float(lv.detach().float().item())

            if step_idx % max(1, grad_accum_steps) == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_metrics = _evaluate_ee_model(model, val_loader, sorted_exits, dev, use_amp, amp_dt)
        epoch_rec = {
            "epoch": epoch,
            "train_total_loss": running_total / max(1, n_batches),
            "train_exit_losses": {str(l): running_per_exit[l] / max(1, n_batches) for l in sorted_exits},
            "val_accuracy": val_metrics["accuracy"],
            "val_ece": val_metrics["ece"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        epoch_logs.append(epoch_rec)
        print(
            f"[Exp1 Hamed CE | {weighting_mode}] Epoch {epoch}/{epochs} "
            f"loss={epoch_rec['train_total_loss']:.4f} "
            f"L{sorted_exits[0]}={val_metrics['accuracy'][str(sorted_exits[0])]:.2f}% "
            f"L{sorted_exits[-1]}={val_metrics['accuracy'][str(sorted_exits[-1])]:.2f}%"
        )

    del optimizer, scheduler, scaler
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    probe_batches = list(itertools.islice(val_loader, 2))

    def _eval_landscape_losses() -> Dict[int, torch.Tensor]:
        acc_losses: Dict[int, List[torch.Tensor]] = {l: [] for l in sorted_exits}
        for b in probe_batches:
            labels = b["labels"].to(dev, non_blocking=True)
            inputs = _move_view(b, dev)
            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                outputs = model(**inputs, labels=labels)
            for i, l in enumerate(sorted_exits[:-1]):
                acc_losses[l].append(outputs.exit_losses[i])
            acc_losses[sorted_exits[-1]].append(outputs.loss)
        return {l: torch.stack(acc_losses[l]).mean() for l in sorted_exits}

    landscape = compute_2d_loss_landscape(
        model=model,
        eval_losses_fn=_eval_landscape_losses,
        shallow_key=sorted_exits[0],
        deep_key=sorted_exits[-1],
        weights=exit_weights,
        grid_size=landscape_grid_size,
        span=landscape_span,
        seed=seed,
    )

    wall_time = float(time.perf_counter() - t0)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2))
        if dev.type == "cuda"
        else 0.0
    )

    payload: Dict[str, Any] = {
        "experiment": "exp1_hamed_ce",
        "config": {
            "exit_layers": sorted_exits,
            "weighting_mode": weighting_mode,
            "exit_weights": {str(k): v for k, v in exit_weights.items()},
            "gamma": gamma,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "scheduler_type": scheduler_type,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "dry_run": dry_run,
        },
        "epoch_logs": epoch_logs,
        "step_logs": prober.history,
        "summary": prober.summarize(),
        "loss_landscape": landscape,
        "runtime_sec": wall_time,
        "peak_vram_mb": peak_vram_mb,
    }

    log_file = out_path / "gradient_conflict_log.json"
    log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_path / "model_final.pt")
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return payload


# Source: Stage 1 (SimCLR arXiv:2002.05709 + ConOSR AAAI 2023 SupCon-ST without Mixup) & Stage 2 (ConOSR Step 2 Classifier + SimCLR Linear Eval)
def run_exp2_supcon(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
    num_classes: int = 16,
    weighting_mode: str = "uniform",
    gamma: float = 0.5,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    sampler: str = "pk",
    pk_p: int = 2,
    pk_k: int = 2,
    epochs: int = 30,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    exclude_bias_norm: bool = True,
    warmup_ratio: float = 0.1,
    warmup_start_lr_ratio: float = 0.1,
    eta_min_ratio: float = 0.0,
    temperature: float = 0.1,
    base_temperature: Optional[float] = None,
    smoothing_coeff: float = 0.1,
    use_lars: bool = False,
    lars_eta: float = 0.02,
    max_grad_norm: float = 1.0,
    max_shear: float = 4.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    p_blur: float = 0.5,
    blur_sigma: Tuple[float, float] = (0.1, 2.0),
    noise_std: float = 8.0,
    p_erase: float = 0.3,
    epochs_stage2: int = 20,
    batch_size_stage2: Optional[int] = None,
    eval_batch_size: int = 1,
    grad_accum_steps_stage2: int = 1,
    lr_stage2: float = 3e-4,
    weight_decay_stage2: float = 0.0,
    label_smoothing_stage2: float = 0.1,
    use_aug_stage2: bool = True,
    probe_every_n_steps: int = 10,
    landscape_grid_size: int = 11,
    landscape_span: float = 0.5,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "cuda",
    use_amp: bool = True,
    use_checkpointing: bool = True,
    dry_run: bool = False,
    run_stage2_probe: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    dev = _resolve_device(device)
    amp_dt = _amp_dtype(dev)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_exits = sorted({int(l) for l in exit_layers})
    if not sorted_exits or any(l < 0 or l > 12 for l in sorted_exits):
        raise ValueError(f"exit_layers must be integers in [0, 12], got {exit_layers}")
    model = DICWithoutGroupwise(
        num_classes=num_classes,
        exit_layers=sorted_exits,
        projection_dim=128,
    ).to(dev)

    if use_checkpointing:
        enable_layoutlmv3_checkpointing(model)

    exit_weights = compute_exit_weights(
        model=model,
        exit_layers=sorted_exits,
        weighting_mode=weighting_mode,
        gamma=gamma,
    )

    probe_layers = tuple(sorted_exits[:-1])
    probe_groups = extract_shared_layer_param_groups(model, probe_layers=probe_layers)
    prober = GradientConflictProber(probe_param_groups=probe_groups)

    resolved_base_temp = temperature if base_temperature is None else base_temperature
    supcon_criterion = MultiExitSupConLoss(
        exit_layers=sorted_exits,
        temperature=temperature,
        base_temperature=resolved_base_temp,
    ).to(dev)
    ce_criterion = MultiExitCELoss(
        exit_layers=sorted_exits,
        label_smoothing=label_smoothing_stage2,
    ).to(dev)

    aug_tf = get_transforms(
        max_shear=max_shear,
        brightness_range=brightness_range,
        p_erase=p_erase,
        noise_std=noise_std,
        p_blur=p_blur,
        blur_sigma=blur_sigma,
    )

    train_loader_s1, val_loader_s1 = _build_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        is_contrastive=True,
        max_samples=max_samples,
        dry_run=dry_run,
        num_classes=num_classes,
        seed=seed,
        sampler_type=sampler,
        pk_p=pk_p,
        pk_k=pk_k,
        transform=aug_tf,
    )

    stage1_named_params = [
        (f"backbone.{n}", p) for n, p in model.backbone.named_parameters()
    ] + [
        (f"projection_heads.{n}", p) for n, p in model.projection_heads.named_parameters()
    ]
    stage1_params = [p for _, p in stage1_named_params]
    stage1_param_groups = build_adamw_param_groups(
        stage1_named_params,
        weight_decay=weight_decay,
        exclude_bias_norm=exclude_bias_norm,
    )
    base_opt = torch.optim.AdamW(stage1_param_groups, lr=lr)
    optimizer = LARSWrapper(base_opt, eta=lars_eta, clip=True) if use_lars else base_opt
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda" and amp_dt == torch.float16)
    steps_per_epoch = max(1, math.ceil(len(train_loader_s1) / max(1, grad_accum_steps)))
    scheduler = _build_cosine_warmup_scheduler(
        base_opt,
        total_steps=steps_per_epoch * epochs,
        warmup_ratio=warmup_ratio,
        warmup_start_lr_ratio=warmup_start_lr_ratio,
        eta_min_ratio=eta_min_ratio,
    )

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t0 = time.perf_counter()
    epoch_logs: List[Dict[str, Any]] = []
    global_step = 0
    model.unfreeze_encoder()
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        running_total = 0.0
        running_per_exit = {l: 0.0 for l in sorted_exits}
        n_batches = 0

        for step_idx, batch in enumerate(train_loader_s1, start=1):
            global_step += 1
            n_batches += 1
            labels = batch["labels"].to(dev, non_blocking=True)
            soft_targets = get_smooth_labels(
                labels,
                classid_list=num_classes,
                smoothing_coeff=smoothing_coeff,
            ).to(dev)
            v1 = _move_view(batch["view1"], dev)
            v2 = _move_view(batch["view2"], dev)

            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                out1 = model(**v1, compute_projections=True, compute_logits=False)
                out2 = model(**v2, compute_projections=True, compute_logits=False)
            proj1_f32 = {l: t.float() for l, t in out1.projections_dict.items()}
            proj2_f32 = {l: t.float() for l, t in out2.projections_dict.items()}
            layer_losses = supcon_criterion(
                (proj1_f32, proj2_f32),
                soft_targets,
            )
            first_loss = next(iter(layer_losses.values()))
            total_loss = torch.zeros((), device=dev, dtype=first_loss.dtype)
            for l, lv in layer_losses.items():
                total_loss = total_loss + exit_weights[l] * lv

            if global_step == 1 or (probe_every_n_steps > 0 and global_step % probe_every_n_steps == 0):
                prober.probe(step=global_step, exit_losses=layer_losses, weights=exit_weights)

            scaler.scale(total_loss / float(max(1, grad_accum_steps))).backward()
            running_total += float(total_loss.detach().float().item())
            for l, lv in layer_losses.items():
                running_per_exit[l] += float(lv.detach().float().item())

            if step_idx % max(1, grad_accum_steps) == 0 or step_idx == len(train_loader_s1):
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(stage1_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        epoch_rec: Dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": running_total / max(1, n_batches),
            "train_exit_losses": {str(l): running_per_exit[l] / max(1, n_batches) for l in sorted_exits},
            "lr": float(scheduler.get_last_lr()[0]),
        }
        epoch_logs.append(epoch_rec)
        print(
            f"[Exp2 SupCon Stage1 | {weighting_mode}] Epoch {epoch}/{epochs} "
            f"loss={epoch_rec['train_total_loss']:.4f}"
        )

    del optimizer, base_opt, scheduler, scaler
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    probe_batches = list(itertools.islice(val_loader_s1, 2))

    def _eval_landscape_losses() -> Dict[int, torch.Tensor]:
        acc_losses: Dict[int, List[torch.Tensor]] = {l: [] for l in sorted_exits}
        for b in probe_batches:
            labels = b["labels"].to(dev, non_blocking=True)
            soft_targets = get_smooth_labels(
                labels,
                classid_list=num_classes,
                smoothing_coeff=smoothing_coeff,
            ).to(dev)
            v1 = _move_view(b["view1"], dev)
            v2 = _move_view(b["view2"], dev)
            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                o1 = model(**v1, compute_projections=True, compute_logits=False)
                o2 = model(**v2, compute_projections=True, compute_logits=False)
            p1_f32 = {l: t.float() for l, t in o1.projections_dict.items()}
            p2_f32 = {l: t.float() for l, t in o2.projections_dict.items()}
            l_dict = supcon_criterion((p1_f32, p2_f32), soft_targets)
            for l in sorted_exits:
                acc_losses[l].append(l_dict[l])
        return {l: torch.stack(acc_losses[l]).mean() for l in sorted_exits}

    landscape = compute_2d_loss_landscape(
        model=model,
        eval_losses_fn=_eval_landscape_losses,
        shallow_key=sorted_exits[0],
        deep_key=sorted_exits[-1],
        weights=exit_weights,
        grid_size=landscape_grid_size,
        span=landscape_span,
        seed=seed,
    )

    stage2_logs: List[Dict[str, Any]] = []
    bs2 = batch_size_stage2 if batch_size_stage2 is not None else batch_size
    if run_stage2_probe and epochs_stage2 > 0:
        train_loader_s2, val_loader_s2 = _build_loaders(
            data_dir=data_dir,
            batch_size=bs2,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            is_contrastive=False,
            max_samples=max_samples,
            dry_run=dry_run,
            num_classes=num_classes,
            seed=seed,
            transform=aug_tf if use_aug_stage2 else None,
        )
        model.freeze_encoder()
        model.train()
        model.backbone.eval()

        head_params = list(model.exit_heads.parameters())
        opt2 = torch.optim.AdamW(head_params, lr=lr_stage2, weight_decay=weight_decay_stage2)
        scaler2 = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda" and amp_dt == torch.float16)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, epochs_stage2), eta_min=0.0)

        for ep2 in range(1, epochs_stage2 + 1):
            opt2.zero_grad(set_to_none=True)
            tot_ce = 0.0
            nb2 = 0
            for st2, batch in enumerate(train_loader_s2, start=1):
                nb2 += 1
                labels = batch["labels"].to(dev, non_blocking=True)
                inputs = _move_view(batch, dev)
                with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                    out = model(**inputs, compute_projections=False, compute_logits=True)
                    ce_dict = ce_criterion(out.logits_dict, labels)
                    ce_loss = ce_criterion.compute_total_loss(ce_dict)
                scaler2.scale(ce_loss / float(max(1, grad_accum_steps_stage2))).backward()
                tot_ce += float(ce_loss.detach().float().item())
                if st2 % max(1, grad_accum_steps_stage2) == 0 or st2 == len(train_loader_s2):
                    scaler2.step(opt2)
                    scaler2.update()
                    opt2.zero_grad(set_to_none=True)
            sched2.step()
            val_metrics = _evaluate_dic_model(model, val_loader_s2, sorted_exits, dev, use_amp, amp_dt)
            stage2_logs.append(
                {
                    "epoch": ep2,
                    "train_ce_loss": tot_ce / max(1, nb2),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_ece": val_metrics["ece"],
                }
            )
            print(
                f"[Exp2 SupCon Stage2 Probe] Epoch {ep2}/{epochs_stage2} "
                f"ce={tot_ce / max(1, nb2):.4f} "
                f"L{sorted_exits[0]}={val_metrics['accuracy'][str(sorted_exits[0])]:.2f}% "
                f"L{sorted_exits[-1]}={val_metrics['accuracy'][str(sorted_exits[-1])]:.2f}%"
            )

    wall_time = float(time.perf_counter() - t0)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2))
        if dev.type == "cuda"
        else 0.0
    )

    payload: Dict[str, Any] = {
        "experiment": "exp2_supcon",
        "config": {
            "exit_layers": sorted_exits,
            "weighting_mode": weighting_mode,
            "exit_weights": {str(k): v for k, v in exit_weights.items()},
            "gamma": gamma,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "sampler": sampler,
            "pk_p": pk_p,
            "pk_k": pk_k,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "exclude_bias_norm": exclude_bias_norm,
            "warmup_ratio": warmup_ratio,
            "temperature": temperature,
            "base_temperature": resolved_base_temp,
            "smoothing_coeff": smoothing_coeff,
            "use_lars": use_lars,
            "lars_eta": lars_eta,
            "epochs_stage2": epochs_stage2,
            "batch_size_stage2": bs2,
            "eval_batch_size": eval_batch_size,
            "grad_accum_steps_stage2": grad_accum_steps_stage2,
            "lr_stage2": lr_stage2,
            "weight_decay_stage2": weight_decay_stage2,
            "label_smoothing_stage2": label_smoothing_stage2,
            "use_aug_stage2": use_aug_stage2,
            "seed": seed,
            "dry_run": dry_run,
        },
        "epoch_logs": epoch_logs,
        "stage2_logs": stage2_logs,
        "step_logs": prober.history,
        "summary": prober.summarize(),
        "loss_landscape": landscape,
        "runtime_sec": wall_time,
        "peak_vram_mb": peak_vram_mb,
    }

    log_file = out_path / "gradient_conflict_log.json"
    log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_path / "model_final.pt")
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return payload
