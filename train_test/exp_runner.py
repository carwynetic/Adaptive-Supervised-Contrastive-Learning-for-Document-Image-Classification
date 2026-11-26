from __future__ import annotations

import gc
import itertools
import json
import math
import os
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
from model.DIC import DIC
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        cpu_st = state["torch_cpu"]
        if isinstance(cpu_st, torch.Tensor):
            torch.set_rng_state(cpu_st.cpu().to(torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if isinstance(cuda_states, list) and len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in cuda_states])


class _SeededWorkerInit:
    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        w_seed = (self.base_seed + worker_id) % (2**32)
        random.seed(w_seed)
        np.random.seed(w_seed)
        torch.manual_seed(w_seed)


def _save_checkpoint_atomic(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _split_sequential_group_named_params(
    model: DIC,
    group_specs: Sequence[Tuple[str, int, int]],
) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
    grouped: Dict[str, List[Tuple[str, nn.Parameter]]] = {g_name: [] for g_name, _, _ in group_specs}
    first_group = group_specs[0][0]
    first_transformer_group = next(
        (g_name for g_name, _, target_exit in group_specs if target_exit >= 1),
        first_group,
    )
    exit_to_group = {target_exit: g_name for g_name, _, target_exit in group_specs}

    for name, param in model.named_parameters():
        if name.startswith("exit_heads."):
            continue
        if name.startswith("projection_heads."):
            parts = name.split(".")
            layer_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else group_specs[-1][2]
            target_g = exit_to_group.get(layer_id, group_specs[-1][0])
            grouped[target_g].append((name, param))
            continue
        if "encoder.layer." in name:
            after = name.split("encoder.layer.", 1)[1]
            idx_str = after.split(".", 1)[0]
            layer_num = int(idx_str) + 1 if idx_str.isdigit() else 12
            assigned = group_specs[-1][0]
            for g_name, prev_exit, target_exit in group_specs:
                if prev_exit < layer_num <= target_exit:
                    assigned = g_name
                    break
            grouped[assigned].append((name, param))
            continue
        if "encoder.rel_pos_" in name:
            grouped[first_transformer_group].append((name, param))
            continue
        grouped[first_group].append((name, param))
    return grouped


def _split_groupwise_named_params(
    model: DIC,
    split_layer: int = 4,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    g1_named: List[Tuple[str, nn.Parameter]] = []
    g2_named: List[Tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if name.startswith("exit_heads."):
            continue
        if name.startswith("projection_heads."):
            parts = name.split(".")
            layer_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 12
            if layer_id <= split_layer:
                g1_named.append((name, param))
            else:
                g2_named.append((name, param))
            continue
        if "encoder.layer." in name:
            after = name.split("encoder.layer.", 1)[1]
            idx_str = after.split(".", 1)[0]
            layer_num = int(idx_str) + 1 if idx_str.isdigit() else 12
            if layer_num <= split_layer:
                g1_named.append((name, param))
            else:
                g2_named.append((name, param))
        else:
            g1_named.append((name, param))
    return g1_named, g2_named


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
    worker_init = _SeededWorkerInit(seed) if num_workers > 0 else None
    val_worker_init = _SeededWorkerInit(seed + 10000) if num_workers > 0 else None
    train_gen = torch.Generator().manual_seed(seed)
    if sampler_type == "pk" and len(train_ds) > 0:
        train_labels = [int(r["label"]) for r in train_ds.records]
        batch_sampler = PKBatchSampler(train_labels, p=pk_p, k=pk_k, seed=seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin,
            worker_init_fn=worker_init,
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
            worker_init_fn=worker_init,
            generator=train_gen,
        )
    if is_contrastive and sampler_type == "pk" and len(val_ds) > 0:
        val_labels = [int(r["label"]) for r in val_ds.records]
        val_pk = PKBatchSampler(val_labels, p=pk_p, k=pk_k, seed=seed + 10000)
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_pk,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin,
            worker_init_fn=val_worker_init,
        )
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=val_bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin,
            drop_last=False,
            worker_init_fn=val_worker_init,
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


def _format_eta(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _set_pk_sampler_epoch(loader: Any, epoch_zero_based: int) -> None:
    bs = getattr(loader, "batch_sampler", None)
    if bs is not None and hasattr(bs, "set_epoch"):
        bs.set_epoch(epoch_zero_based)


def _evaluate_dic_supcon_loss(
    model: nn.Module,
    val_loader: Any,
    supcon_criterion: MultiExitSupConLoss,
    exit_weights: Dict[int, float],
    exit_layers: Sequence[int],
    num_classes: int,
    smoothing_coeff: float,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    eval_seed: int,
) -> Dict[str, Any]:
    saved_rng = _capture_rng_state()
    was_training = model.training
    model.eval()
    set_seed(eval_seed)
    _set_pk_sampler_epoch(val_loader, 0)

    total_loss = 0.0
    per_exit = {l: 0.0 for l in exit_layers}
    n_batches = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                n_batches += 1
                labels = batch["labels"].to(device, non_blocking=True)
                soft_targets = get_smooth_labels(
                    labels,
                    classid_list=num_classes,
                    smoothing_coeff=smoothing_coeff,
                ).to(device)
                v1 = _move_view(batch["view1"], device)
                v2 = _move_view(batch["view2"], device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                    o1 = model(**v1, compute_projections=True, compute_logits=False)
                    o2 = model(**v2, compute_projections=True, compute_logits=False)
                p1_f32 = {l: t.float() for l, t in o1.projections_dict.items()}
                p2_f32 = {l: t.float() for l, t in o2.projections_dict.items()}
                l_dict = supcon_criterion((p1_f32, p2_f32), soft_targets)
                step_total = 0.0
                for l in exit_layers:
                    lv = float(l_dict[l].detach().item())
                    per_exit[l] += lv
                    step_total += float(exit_weights[l]) * lv
                total_loss += step_total
                del o1, o2, l_dict
    finally:
        model.train(was_training)
        _restore_rng_state(saved_rng)

    return {
        "total_loss": total_loss / max(1, n_batches),
        "exit_losses": {str(l): per_exit[l] / max(1, n_batches) for l in exit_layers},
    }


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
    resume: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    dev = _resolve_device(device)
    amp_dt = _amp_dtype(dev)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "gradient_conflict_log.json"
    final_ckpt = out_path / "model_final.pt"
    last_ckpt = out_path / "checkpoint_last.pt"
    if resume and log_file.exists() and final_ckpt.exists():
        print(f"[Exp1 Hamed CE] Already completed at {out_path}, loading saved log.")
        return json.loads(log_file.read_text(encoding="utf-8"))

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
    start_epoch = 1
    best_val_acc = -1.0
    elapsed_prior = 0.0
    if resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        if "scaler_state_dict" in ck and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        global_step = int(ck.get("global_step", 0))
        epoch_logs = list(ck.get("epoch_logs", []))
        prober.history = list(ck.get("step_logs", []))
        best_val_acc = float(ck.get("best_val_acc", -1.0))
        elapsed_prior = float(ck.get("elapsed_sec", 0.0))
        _restore_rng_state(ck.get("rng_state"))
        print(f"[Exp1 Hamed CE] Resumed from {last_ckpt} at epoch {start_epoch}/{epochs}", flush=True)

    model.train()

    for epoch in range(start_epoch, epochs + 1):
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
        val_l12 = float(val_metrics["accuracy"][str(sorted_exits[-1])])
        if val_l12 >= best_val_acc:
            best_val_acc = val_l12
            _save_checkpoint_atomic(model.state_dict(), out_path / "model_best.pt")
        _save_checkpoint_atomic(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch_logs": epoch_logs,
                "step_logs": prober.history,
                "best_val_acc": best_val_acc,
                "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                "rng_state": _capture_rng_state(),
            },
            last_ckpt,
        )
        print(
            f"[Exp1 Hamed CE | {weighting_mode}] Epoch {epoch}/{epochs} "
            f"loss={epoch_rec['train_total_loss']:.4f} "
            f"L{sorted_exits[0]}={val_metrics['accuracy'][str(sorted_exits[0])]:.2f}% "
            f"L{sorted_exits[-1]}={val_l12:.2f}%",
            flush=True,
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

    wall_time = elapsed_prior + float(time.perf_counter() - t0)
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

    log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_checkpoint_atomic(model.state_dict(), final_ckpt)
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
    resume: bool = True,
    stage1_ckpt_for_stage2: str = "best",
) -> Dict[str, Any]:
    set_seed(seed)
    dev = _resolve_device(device)
    amp_dt = _amp_dtype(dev)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "gradient_conflict_log.json"
    final_ckpt = out_path / "model_final.pt"
    s1_ckpt = out_path / "checkpoint_stage1_last.pt"
    s1_best_ckpt = out_path / "stage1_best.pt"
    s1_final_ckpt = out_path / "stage1_final.pt"
    s2_ckpt = out_path / "checkpoint_stage2_last.pt"
    want_stage2 = bool(run_stage2_probe and epochs_stage2 > 0)
    if resume and log_file.exists() and final_ckpt.exists():
        saved_log = json.loads(log_file.read_text(encoding="utf-8"))
        done_s1 = len(saved_log.get("epoch_logs", [])) >= epochs
        done_s2 = (not want_stage2) or (len(saved_log.get("stage2_logs", [])) >= epochs_stage2)
        if done_s1 and done_s2:
            print(f"[Exp2 SupCon] Already completed at {out_path}, loading saved log.", flush=True)
            return saved_log

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
    start_epoch = 1
    stage1_done = False
    landscape: Optional[Dict[str, Any]] = None
    elapsed_prior = 0.0
    best_s1_val_loss = float("inf")
    if resume and s1_ckpt.exists():
        ck1 = torch.load(s1_ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(ck1["model_state_dict"])
        if ck1.get("optimizer_state_dict"):
            base_opt.load_state_dict(ck1["optimizer_state_dict"])
        if ck1.get("scheduler_state_dict"):
            scheduler.load_state_dict(ck1["scheduler_state_dict"])
        if ck1.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(ck1["scaler_state_dict"])
        start_epoch = int(ck1.get("epoch", 0)) + 1
        global_step = int(ck1.get("global_step", 0))
        best_s1_val_loss = float(ck1.get("best_s1_val_loss", float("inf")))
        epoch_logs = list(ck1.get("epoch_logs", []))
        prober.history = list(ck1.get("step_logs", []))
        stage1_done = bool(ck1.get("stage1_done", False))
        landscape = ck1.get("landscape")
        elapsed_prior = float(ck1.get("elapsed_sec", 0.0))
        _restore_rng_state(ck1.get("rng_state"))
        print(
            f"[Exp2 SupCon Stage1] Resumed from {s1_ckpt} (start_epoch={start_epoch}/{epochs}, stage1_done={stage1_done})",
            flush=True,
        )

    model.unfreeze_encoder()
    model.train()

    for epoch in range(start_epoch, epochs + 1):
        _set_pk_sampler_epoch(train_loader_s1, epoch - 1)
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

        val_s1 = _evaluate_dic_supcon_loss(
            model=model,
            val_loader=val_loader_s1,
            supcon_criterion=supcon_criterion,
            exit_weights=exit_weights,
            exit_layers=sorted_exits,
            num_classes=num_classes,
            smoothing_coeff=smoothing_coeff,
            device=dev,
            use_amp=use_amp,
            amp_dtype=amp_dt,
            eval_seed=seed + 10000,
        )
        epoch_rec: Dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": running_total / max(1, n_batches),
            "train_exit_losses": {str(l): running_per_exit[l] / max(1, n_batches) for l in sorted_exits},
            "val_total_loss": val_s1["total_loss"],
            "val_exit_losses": val_s1["exit_losses"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        epoch_logs.append(epoch_rec)
        if val_s1["total_loss"] <= best_s1_val_loss:
            best_s1_val_loss = val_s1["total_loss"]
            _save_checkpoint_atomic(model.state_dict(), s1_best_ckpt)
        _save_checkpoint_atomic(
            {
                "epoch": epoch,
                "stage1_done": epoch >= epochs,
                "global_step": global_step,
                "best_s1_val_loss": best_s1_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": base_opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch_logs": epoch_logs,
                "step_logs": prober.history,
                "landscape": None,
                "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                "rng_state": _capture_rng_state(),
            },
            s1_ckpt,
        )
        print(
            f"[Exp2 SupCon Stage1 | {weighting_mode}] Epoch {epoch}/{epochs} "
            f"train={epoch_rec['train_total_loss']:.4f} val={val_s1['total_loss']:.4f} "
            f"best_val={best_s1_val_loss:.4f}",
            flush=True,
        )

    if not s1_final_ckpt.exists() or start_epoch <= epochs:
        _save_checkpoint_atomic(model.state_dict(), s1_final_ckpt)
    del optimizer, base_opt, scheduler, scaler
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if landscape is None and landscape_grid_size > 0 and probe_every_n_steps > 0:
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
    if not s1_ckpt.exists():
        _save_checkpoint_atomic(
            {
                "epoch": epochs,
                "stage1_done": True,
                "global_step": global_step,
                "best_s1_val_loss": best_s1_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "scaler_state_dict": {},
                "epoch_logs": epoch_logs,
                "step_logs": prober.history,
                "landscape": landscape,
                "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                "rng_state": _capture_rng_state(),
            },
            s1_ckpt,
        )
    elif landscape is not None:
        ck_keep = torch.load(s1_ckpt, map_location="cpu", weights_only=False)
        ck_keep["landscape"] = landscape
        ck_keep["stage1_done"] = True
        _save_checkpoint_atomic(ck_keep, s1_ckpt)

    stage2_logs: List[Dict[str, Any]] = []
    bs2 = batch_size_stage2 if batch_size_stage2 is not None else batch_size
    if want_stage2:
        if not (resume and s2_ckpt.exists()):
            src_s1 = s1_best_ckpt if (stage1_ckpt_for_stage2 == "best" and s1_best_ckpt.exists()) else s1_final_ckpt
            if src_s1.exists():
                model.load_state_dict(torch.load(src_s1, map_location=dev, weights_only=False))
                print(f"[Exp2 SupCon Stage2] Loaded Stage 1 weights from {src_s1}", flush=True)
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
        start_ep2 = 1
        best_val_acc2 = -1.0
        if resume and s2_ckpt.exists():
            ck2 = torch.load(s2_ckpt, map_location=dev, weights_only=False)
            model.load_state_dict(ck2["model_state_dict"])
            opt2.load_state_dict(ck2["optimizer_state_dict"])
            sched2.load_state_dict(ck2["scheduler_state_dict"])
            if "scaler_state_dict" in ck2 and scaler2.is_enabled():
                scaler2.load_state_dict(ck2["scaler_state_dict"])
            start_ep2 = int(ck2.get("epoch", 0)) + 1
            stage2_logs = list(ck2.get("stage2_logs", []))
            best_val_acc2 = float(ck2.get("best_val_acc", -1.0))
            elapsed_prior = max(elapsed_prior, float(ck2.get("elapsed_sec", 0.0)))
            model.freeze_encoder()
            model.train()
            model.backbone.eval()
            print(f"[Exp2 SupCon Stage2] Resumed from {s2_ckpt} at epoch {start_ep2}/{epochs_stage2}")

        for ep2 in range(start_ep2, epochs_stage2 + 1):
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
            val_l12 = float(val_metrics["accuracy"][str(sorted_exits[-1])])
            if val_l12 >= best_val_acc2:
                best_val_acc2 = val_l12
                _save_checkpoint_atomic(model.state_dict(), out_path / "model_best.pt")
            _save_checkpoint_atomic(
                {
                    "epoch": ep2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt2.state_dict(),
                    "scheduler_state_dict": sched2.state_dict(),
                    "scaler_state_dict": scaler2.state_dict(),
                    "stage2_logs": stage2_logs,
                    "best_val_acc": best_val_acc2,
                    "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                },
                s2_ckpt,
            )
            print(
                f"[Exp2 SupCon Stage2 Probe] Epoch {ep2}/{epochs_stage2} "
                f"ce={tot_ce / max(1, nb2):.4f} "
                f"L{sorted_exits[0]}={val_metrics['accuracy'][str(sorted_exits[0])]:.2f}% "
                f"L{sorted_exits[-1]}={val_l12:.2f}%"
            )

    wall_time = elapsed_prior + float(time.perf_counter() - t0)
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

    log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_checkpoint_atomic(model.state_dict(), final_ckpt)
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _forward_incremental_group_view(
    model: DIC,
    view: Dict[str, torch.Tensor],
    cache: Optional[Dict[str, Any]],
    prev_exit: int,
    target_exit: int,
) -> torch.Tensor:
    bb = model.backbone
    if target_exit == 0:
        emb_out, _, _, _ = bb.forward_embeddings_state(
            input_ids=view["input_ids"],
            bbox=view["bbox"],
            pixel_values=view["pixel_values"],
            attention_mask=view["attention_mask"],
            eval_mode=False,
        )
        feat = emb_out.mean(dim=1)
    elif prev_exit < 0:
        emb_out, ext_mask, pos_ids, bbox = bb.forward_embeddings_state(
            input_ids=view["input_ids"],
            bbox=view["bbox"],
            pixel_values=view["pixel_values"],
            attention_mask=view["attention_mask"],
            eval_mode=False,
        )
        rel_pos, rel_2d_pos = bb._compute_rel_pos_biases(pos_ids, bbox)
        h = bb.forward_layer_slice(emb_out, ext_mask, rel_pos, rel_2d_pos, 0, target_exit, eval_mode=False)
        feat = h[:, 0, :]
    elif prev_exit == 0:
        assert cache is not None
        rel_pos, rel_2d_pos = bb._compute_rel_pos_biases(cache["pos_ids"], cache["bbox"])
        h = bb.forward_layer_slice(
            cache["h"], cache["ext_mask"], rel_pos, rel_2d_pos, 0, target_exit, eval_mode=False
        )
        feat = h[:, 0, :]
    else:
        assert cache is not None
        h = bb.forward_layer_slice(
            cache["h"],
            cache["ext_mask"],
            cache["rel_pos"],
            cache["rel_2d_pos"],
            prev_exit,
            target_exit,
            eval_mode=False,
        )
        feat = h[:, 0, :]
    return model.projection_heads[str(target_exit)](feat)


def _recompute_incremental_group_view(
    model: DIC,
    view: Dict[str, torch.Tensor],
    cache: Optional[Dict[str, Any]],
    prev_exit: int,
    target_exit: int,
) -> Dict[str, Any]:
    bb = model.backbone
    if target_exit == 0:
        h0, ext_mask, pos_ids, bbox = bb.forward_embeddings_state(
            input_ids=view["input_ids"],
            bbox=view["bbox"],
            pixel_values=view["pixel_values"],
            attention_mask=view["attention_mask"],
            eval_mode=True,
        )
        return {
            "h": h0,
            "ext_mask": ext_mask,
            "pos_ids": pos_ids,
            "bbox": bbox,
            "rel_pos": None,
            "rel_2d_pos": None,
        }
    if prev_exit < 0:
        h0, ext_mask, pos_ids, bbox = bb.forward_embeddings_state(
            input_ids=view["input_ids"],
            bbox=view["bbox"],
            pixel_values=view["pixel_values"],
            attention_mask=view["attention_mask"],
            eval_mode=True,
        )
        with torch.no_grad():
            rel_pos_new, rel_2d_pos_new = bb._compute_rel_pos_biases(pos_ids, bbox)
        rel_pos_d = rel_pos_new.detach() if isinstance(rel_pos_new, torch.Tensor) else rel_pos_new
        rel_2d_d = rel_2d_pos_new.detach() if isinstance(rel_2d_pos_new, torch.Tensor) else rel_2d_pos_new
        h_new = bb.forward_layer_slice(h0, ext_mask, rel_pos_d, rel_2d_d, 0, target_exit, eval_mode=True)
        return {
            "h": h_new,
            "ext_mask": ext_mask,
            "pos_ids": pos_ids,
            "bbox": bbox,
            "rel_pos": rel_pos_d,
            "rel_2d_pos": rel_2d_d,
        }
    assert cache is not None
    if prev_exit == 0:
        with torch.no_grad():
            rel_pos_new, rel_2d_pos_new = bb._compute_rel_pos_biases(cache["pos_ids"], cache["bbox"])
        rel_pos_d = rel_pos_new.detach() if isinstance(rel_pos_new, torch.Tensor) else rel_pos_new
        rel_2d_d = rel_2d_pos_new.detach() if isinstance(rel_2d_pos_new, torch.Tensor) else rel_2d_pos_new
        h_new = bb.forward_layer_slice(
            cache["h"], cache["ext_mask"], rel_pos_d, rel_2d_d, 0, target_exit, eval_mode=True
        )
        return {
            "h": h_new,
            "ext_mask": cache["ext_mask"],
            "pos_ids": cache["pos_ids"],
            "bbox": cache["bbox"],
            "rel_pos": rel_pos_d,
            "rel_2d_pos": rel_2d_d,
        }
    h_new = bb.forward_layer_slice(
        cache["h"],
        cache["ext_mask"],
        cache["rel_pos"],
        cache["rel_2d_pos"],
        prev_exit,
        target_exit,
        eval_mode=True,
    )
    return {
        "h": h_new,
        "ext_mask": cache["ext_mask"],
        "pos_ids": cache["pos_ids"],
        "bbox": cache["bbox"],
        "rel_pos": cache["rel_pos"],
        "rel_2d_pos": cache["rel_2d_pos"],
    }


# Source: Exp 3 6-Group Incremental Sequential Recompute SupCon-ST ([0],[1],[4],[8],[10],[12], AIC-Lab Layerwise + SimCLR + ConOSR)
def run_exp3_groupwise(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
    num_classes: int = 16,
    weighting_mode: str = "uniform",
    gamma: float = 0.5,
    batch_size: int = 16,
    grad_accum_steps: int = 1,
    sampler: str = "pk",
    pk_p: int = 8,
    pk_k: int = 2,
    epochs: int = 60,
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
    resume: bool = True,
    stage1_ckpt_for_stage2: str = "best",
) -> Dict[str, Any]:
    set_seed(seed)
    dev = _resolve_device(device)
    amp_dt = _amp_dtype(dev)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "gradient_conflict_log.json"
    final_ckpt = out_path / "model_final.pt"
    s1_ckpt = out_path / "checkpoint_stage1_last.pt"
    s1_best_ckpt = out_path / "stage1_best.pt"
    s1_final_ckpt = out_path / "stage1_final.pt"
    s2_ckpt = out_path / "checkpoint_stage2_last.pt"
    want_stage2 = bool(run_stage2_probe and epochs_stage2 > 0)
    if resume and log_file.exists() and final_ckpt.exists():
        saved_log = json.loads(log_file.read_text(encoding="utf-8"))
        done_s1 = len(saved_log.get("epoch_logs", [])) >= epochs
        done_s2 = (not want_stage2) or (len(saved_log.get("stage2_logs", [])) >= epochs_stage2)
        if done_s1 and done_s2:
            print(f"[Exp3 Groupwise] Already completed at {out_path}, loading saved log.", flush=True)
            return saved_log

    sorted_exits = sorted({int(l) for l in exit_layers})
    if not sorted_exits or any(l < 0 or l > 12 for l in sorted_exits):
        raise ValueError(f"exit_layers must be integers in [0, 12], got {exit_layers}")

    group_specs: List[Tuple[str, int, int]] = []
    prev_boundary = -1
    for idx, exit_l in enumerate(sorted_exits):
        group_specs.append((f"group{idx + 1}", prev_boundary, exit_l))
        prev_boundary = exit_l
    groups = {g_name: (target_exit,) for g_name, _, target_exit in group_specs}
    model = DIC(
        num_classes=num_classes,
        exit_layers=sorted_exits,
        groups=groups,
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

    grouped_named = _split_sequential_group_named_params(model, group_specs)
    group_params: Dict[str, List[nn.Parameter]] = {
        g_name: [p for _, p in named_list] for g_name, named_list in grouped_named.items()
    }
    all_stage1_params = [p for plist in group_params.values() for p in plist]
    base_opts: Dict[str, torch.optim.Optimizer] = {}
    opts: Dict[str, Any] = {}
    scalers: Dict[str, torch.amp.GradScaler] = {}
    scheds: Dict[str, Any] = {}
    steps_per_epoch = max(1, math.ceil(len(train_loader_s1) / max(1, grad_accum_steps)))
    use_fp16_scaler = use_amp and dev.type == "cuda" and amp_dt == torch.float16
    for g_name, _, _ in group_specs:
        g_param_groups = build_adamw_param_groups(
            grouped_named[g_name],
            weight_decay=weight_decay,
            exclude_bias_norm=exclude_bias_norm,
        )
        if not g_param_groups:
            continue
        b_opt = torch.optim.AdamW(g_param_groups, lr=lr)
        base_opts[g_name] = b_opt
        opts[g_name] = LARSWrapper(b_opt, eta=lars_eta, clip=True) if use_lars else b_opt
        scalers[g_name] = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
        scheds[g_name] = _build_cosine_warmup_scheduler(
            b_opt,
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
    start_epoch = 1
    best_s1_val_loss = float("inf")
    landscape: Optional[Dict[str, Any]] = None
    elapsed_prior = 0.0
    if resume and s1_ckpt.exists():
        ck1 = torch.load(s1_ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(ck1["model_state_dict"])
        saved_opts = ck1.get("opt_state_dicts", {})
        saved_scheds = ck1.get("sched_state_dicts", {})
        saved_scalers = ck1.get("scaler_state_dicts", {})
        for g_name, b_opt in base_opts.items():
            if g_name in saved_opts and saved_opts[g_name]:
                b_opt.load_state_dict(saved_opts[g_name])
            if g_name in saved_scheds and saved_scheds[g_name]:
                scheds[g_name].load_state_dict(saved_scheds[g_name])
            if g_name in saved_scalers and saved_scalers[g_name] and scalers[g_name].is_enabled():
                scalers[g_name].load_state_dict(saved_scalers[g_name])
        start_epoch = int(ck1.get("epoch", 0)) + 1
        global_step = int(ck1.get("global_step", 0))
        best_s1_val_loss = float(ck1.get("best_s1_val_loss", ck1.get("best_s1_loss", float("inf"))))
        epoch_logs = list(ck1.get("epoch_logs", []))
        prober.history = list(ck1.get("step_logs", []))
        landscape = ck1.get("landscape")
        elapsed_prior = float(ck1.get("elapsed_sec", 0.0))
        _restore_rng_state(ck1.get("rng_state"))
        print(
            f"[Exp3 6-Group Recompute Stage1] Resumed from {s1_ckpt} at epoch {start_epoch}/{epochs} "
            f"(best_val={best_s1_val_loss:.4f})",
            flush=True,
        )

    model.unfreeze_encoder()
    model.train()
    total_s1_batches = len(train_loader_s1)
    log_interval = max(1, min(5, total_s1_batches))

    for epoch in range(start_epoch, epochs + 1):
        _set_pk_sampler_epoch(train_loader_s1, epoch - 1)
        ep_t0 = time.perf_counter()
        for g_opt in opts.values():
            g_opt.zero_grad(set_to_none=True)
        running_total = 0.0
        running_per_exit = {l: 0.0 for l in sorted_exits}
        n_batches = 0

        for step_idx, batch in enumerate(train_loader_s1, start=1):
            global_step += 1
            n_batches += 1
            labels = batch["labels"].to(dev, non_blocking=True)
            soft_targets = get_smooth_labels(labels, classid_list=num_classes, smoothing_coeff=smoothing_coeff).to(dev)
            v1 = _move_view(batch["view1"], dev)
            v2 = _move_view(batch["view2"], dev)

            if probe_every_n_steps > 0 and (global_step == 1 or global_step % probe_every_n_steps == 0):
                for p in all_stage1_params:
                    p.requires_grad = True
                with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                    po1 = model(**v1, compute_projections=True, compute_logits=False)
                    po2 = model(**v2, compute_projections=True, compute_logits=False)
                p1_f32 = {l: t.float() for l, t in po1.projections_dict.items()}
                p2_f32 = {l: t.float() for l, t in po2.projections_dict.items()}
                probe_losses = supcon_criterion((p1_f32, p2_f32), soft_targets)
                prober.probe(step=global_step, exit_losses=probe_losses, weights=exit_weights)
                del po1, po2, probe_losses

            do_step = (step_idx % max(1, grad_accum_steps) == 0) or (step_idx == total_s1_batches)
            step_loss_sum = 0.0
            cache_v1: Optional[Dict[str, Any]] = None
            cache_v2: Optional[Dict[str, Any]] = None
            last_group_idx = len(group_specs) - 1

            for g_idx, (g_name, prev_exit, target_exit) in enumerate(group_specs):
                g_opt = opts.get(g_name)
                if g_opt is None:
                    continue
                for other_name, plist in group_params.items():
                    req = other_name == g_name
                    for p in plist:
                        p.requires_grad = req
                with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                    p1 = _forward_incremental_group_view(model, v1, cache_v1, prev_exit, target_exit)
                    p2 = _forward_incremental_group_view(model, v2, cache_v2, prev_exit, target_exit)
                lv = supcon_criterion.supcon(torch.stack([p1.float(), p2.float()], dim=1), soft_targets)
                lv_val = float(lv.detach().item())
                running_per_exit[target_exit] += lv_val
                step_loss_sum += float(exit_weights[target_exit]) * lv_val
                scalers[g_name].scale(lv / float(max(1, grad_accum_steps))).backward()
                del p1, p2, lv
                if do_step:
                    scalers[g_name].unscale_(g_opt)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(group_params[g_name], max_grad_norm)
                    scalers[g_name].step(g_opt)
                    scalers[g_name].update()
                    scheds[g_name].step()
                    g_opt.zero_grad(set_to_none=True)
                if g_idx < last_group_idx:
                    with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp and dev.type == "cuda"):
                        cache_v1 = _recompute_incremental_group_view(model, v1, cache_v1, prev_exit, target_exit)
                        cache_v2 = _recompute_incremental_group_view(model, v2, cache_v2, prev_exit, target_exit)

            del cache_v1, cache_v2
            running_total += step_loss_sum

            if step_idx == 1 or step_idx % log_interval == 0 or step_idx == total_s1_batches:
                ep_elapsed = time.perf_counter() - ep_t0
                sec_per_step = ep_elapsed / float(step_idx)
                eta_ep = sec_per_step * float(total_s1_batches - step_idx)
                vram_now = (
                    float(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2))
                    if dev.type == "cuda"
                    else 0.0
                )
                print(
                    f"[Exp3 S1 Ep {epoch}/{epochs} | Step {step_idx}/{total_s1_batches} | "
                    f"loss={running_total / step_idx:.4f} | {sec_per_step:.2f}s/step | "
                    f"ETA {_format_eta(eta_ep)} | VRAM {vram_now:.0f}MB]",
                    flush=True,
                )

        first_g = group_specs[0][0]
        epoch_loss = running_total / max(1, n_batches)
        val_s1 = _evaluate_dic_supcon_loss(
            model=model,
            val_loader=val_loader_s1,
            supcon_criterion=supcon_criterion,
            exit_weights=exit_weights,
            exit_layers=sorted_exits,
            num_classes=num_classes,
            smoothing_coeff=smoothing_coeff,
            device=dev,
            use_amp=use_amp,
            amp_dtype=amp_dt,
            eval_seed=seed + 10000,
        )
        epoch_rec = {
            "epoch": epoch,
            "train_total_loss": epoch_loss,
            "train_exit_losses": {str(l): running_per_exit[l] / max(1, n_batches) for l in sorted_exits},
            "val_total_loss": val_s1["total_loss"],
            "val_exit_losses": val_s1["exit_losses"],
            "lr": float(scheds[first_g].get_last_lr()[0]) if first_g in scheds else lr,
            "epoch_sec": float(time.perf_counter() - ep_t0),
        }
        epoch_logs.append(epoch_rec)
        if val_s1["total_loss"] <= best_s1_val_loss:
            best_s1_val_loss = val_s1["total_loss"]
            _save_checkpoint_atomic(model.state_dict(), s1_best_ckpt)
        _save_checkpoint_atomic(
            {
                "epoch": epoch,
                "stage1_done": epoch >= epochs,
                "global_step": global_step,
                "best_s1_val_loss": best_s1_val_loss,
                "model_state_dict": model.state_dict(),
                "opt_state_dicts": {gn: bo.state_dict() for gn, bo in base_opts.items()},
                "sched_state_dicts": {gn: sc.state_dict() for gn, sc in scheds.items()},
                "scaler_state_dicts": {gn: sl.state_dict() for gn, sl in scalers.items()},
                "epoch_logs": epoch_logs,
                "step_logs": prober.history,
                "landscape": None,
                "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                "rng_state": _capture_rng_state(),
            },
            s1_ckpt,
        )
        print(
            f"[Exp3 6-Group Recompute Stage1] Epoch {epoch}/{epochs} "
            f"train={epoch_loss:.4f} val={val_s1['total_loss']:.4f} "
            f"best_val={best_s1_val_loss:.4f} time={_format_eta(epoch_rec['epoch_sec'])}",
            flush=True,
        )

    for p in all_stage1_params:
        p.requires_grad = True
    if not s1_final_ckpt.exists() or start_epoch <= epochs:
        _save_checkpoint_atomic(model.state_dict(), s1_final_ckpt)
    del opts, base_opts, scheds, scalers
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if landscape is None and landscape_grid_size > 0 and probe_every_n_steps > 0:
        probe_batches = list(itertools.islice(val_loader_s1, 2))

        def _eval_landscape_losses() -> Dict[int, torch.Tensor]:
            acc_losses: Dict[int, List[torch.Tensor]] = {l: [] for l in sorted_exits}
            for b in probe_batches:
                labels = b["labels"].to(dev, non_blocking=True)
                soft_targets = get_smooth_labels(labels, classid_list=num_classes, smoothing_coeff=smoothing_coeff).to(dev)
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
    if not s1_ckpt.exists():
        _save_checkpoint_atomic(
            {
                "epoch": epochs,
                "stage1_done": True,
                "global_step": global_step,
                "best_s1_val_loss": best_s1_val_loss,
                "model_state_dict": model.state_dict(),
                "epoch_logs": epoch_logs,
                "step_logs": prober.history,
                "landscape": landscape,
                "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                "rng_state": _capture_rng_state(),
            },
            s1_ckpt,
        )
    elif landscape is not None:
        ck_keep = torch.load(s1_ckpt, map_location="cpu", weights_only=False)
        ck_keep["landscape"] = landscape
        ck_keep["stage1_done"] = True
        _save_checkpoint_atomic(ck_keep, s1_ckpt)

    stage2_logs: List[Dict[str, Any]] = []
    bs2 = batch_size_stage2 if batch_size_stage2 is not None else batch_size
    if want_stage2:
        if not (resume and s2_ckpt.exists()):
            src_s1 = s1_best_ckpt if (stage1_ckpt_for_stage2 == "best" and s1_best_ckpt.exists()) else s1_final_ckpt
            if src_s1.exists():
                model.load_state_dict(torch.load(src_s1, map_location=dev, weights_only=False))
                print(f"[Exp3 Groupwise Stage2] Loaded Stage 1 weights from {src_s1}", flush=True)
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
        start_ep2 = 1
        best_val_acc2 = -1.0
        if resume and s2_ckpt.exists():
            ck2 = torch.load(s2_ckpt, map_location=dev, weights_only=False)
            model.load_state_dict(ck2["model_state_dict"])
            opt2.load_state_dict(ck2["optimizer_state_dict"])
            sched2.load_state_dict(ck2["scheduler_state_dict"])
            if "scaler_state_dict" in ck2 and scaler2.is_enabled():
                scaler2.load_state_dict(ck2["scaler_state_dict"])
            start_ep2 = int(ck2.get("epoch", 0)) + 1
            stage2_logs = list(ck2.get("stage2_logs", []))
            best_val_acc2 = float(ck2.get("best_val_acc", -1.0))
            elapsed_prior = max(elapsed_prior, float(ck2.get("elapsed_sec", 0.0)))
            _restore_rng_state(ck2.get("rng_state"))
            model.freeze_encoder()
            model.train()
            model.backbone.eval()
            print(f"[Exp3 Groupwise Stage2] Resumed from {s2_ckpt} at epoch {start_ep2}/{epochs_stage2}", flush=True)

        for ep2 in range(start_ep2, epochs_stage2 + 1):
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
            val_l12 = float(val_metrics["accuracy"][str(sorted_exits[-1])])
            if val_l12 >= best_val_acc2:
                best_val_acc2 = val_l12
                _save_checkpoint_atomic(model.state_dict(), out_path / "model_best.pt")
            _save_checkpoint_atomic(
                {
                    "epoch": ep2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt2.state_dict(),
                    "scheduler_state_dict": sched2.state_dict(),
                    "scaler_state_dict": scaler2.state_dict(),
                    "stage2_logs": stage2_logs,
                    "best_val_acc": best_val_acc2,
                    "elapsed_sec": elapsed_prior + float(time.perf_counter() - t0),
                    "rng_state": _capture_rng_state(),
                },
                s2_ckpt,
            )
            print(
                f"[Exp3 Groupwise Stage2 Probe] Epoch {ep2}/{epochs_stage2} "
                f"ce={tot_ce / max(1, nb2):.4f} "
                f"L{sorted_exits[0]}={val_metrics['accuracy'][str(sorted_exits[0])]:.2f}% "
                f"L{sorted_exits[-1]}={val_l12:.2f}%",
                flush=True,
            )

    wall_time = elapsed_prior + float(time.perf_counter() - t0)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2))
        if dev.type == "cuda"
        else 0.0
    )

    payload: Dict[str, Any] = {
        "experiment": "exp3_groupwise",
        "config": {
            "exit_layers": sorted_exits,
            "groups": {k: list(v) for k, v in groups.items()},
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

    log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_checkpoint_atomic(model.state_dict(), final_ckpt)
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return payload
