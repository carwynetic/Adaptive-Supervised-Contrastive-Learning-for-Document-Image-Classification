"""
Anytime Early-Exiting Evaluation Pipeline for Document Image Classification (DIC).

Source:
    - Hamed et al. (ICDAR 2024, arXiv:2405.12705), Section 3.3 (Eq. 6-12) & Section 4
    - https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/eval.py
    - https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/policy.py
    - https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/analysis.py

Capabilities:
    1. Model-agnostic evaluation for `LayoutLMv3EEForSequenceClassification` (`exp1_hamed_ce`),
       `DICWithoutGroupwise` (`exp2_supcon`), and `DIC` (`groupwise`).
    2. Real multimodal cached-OCR evaluation (`RVLCDIPCachedDataset` + `MultimodalCollator`).
    3. Leakage-free Post-hoc Temperature Scaling (Eq. 7) fitted strictly on validation set.
    4. Exit policies: Global Thresholding (Eq. 8) & Accuracy-Calibration Heuristic (Eq. 10 & 12).
    5. Full metrics: Accuracy, Macro-F1, ECE, NLL, Brier, AURC, Params/Latency Reduction,
       Pareto frontier extraction, optional true wall-clock latency benchmark, JSON + plot export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_processing.data_processing import MultimodalCollator, RVLCDIPCachedDataset
from model.DIC import DIC
from model.DIC_without_groupwise import DICWithoutGroupwise
from model.modules.ee_modules import (
    EarlyExitPolicy,
    ExitConfig,
    TemperatureScaler,
    entropy,
    max_confidence,
    top12_margin,
)
from model.modules.layoutlmv3 import LayoutLMv3EEForSequenceClassification
from utils.metrics import (
    compute_classification_metrics,
    compute_efficiency_metrics,
    compute_subgraph_param_ratios,
    extract_pareto_frontier,
)


def build_cached_eval_loader(
    data_dir: Union[str, Path],
    split: str = "test",
    batch_size: int = 1,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    max_length: int = 512,
) -> DataLoader:
    """
    Build a multimodal DataLoader reading real cached OCR words & bounding boxes
    (`ocr_cache/{split}.jsonl`). Fails explicitly if OCR cache is missing rather than
    silently substituting zero tokens.
    """
    ds = RVLCDIPCachedDataset(
        data_dir=data_dir,
        split=split,
        transform=None,
        is_contrastive=False,
        max_samples=max_samples,
    )
    collator = MultimodalCollator(max_length=max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def load_model_for_evaluation(
    model_type: str = "auto",
    checkpoint_path: Optional[Union[str, Path]] = None,
    num_classes: int = 16,
    exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, str]:
    """
    Instantiate and optionally load checkpoint weights for `ee_hamed`, `dic`, or `dic_groupwise`.
    When `model_type='auto'`, detects architecture automatically from checkpoint keys.
    """
    sorted_exits = tuple(sorted(int(l) for l in exit_layers))
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    ckpt_file = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_file is not None and ckpt_file.exists():
        raw = torch.load(ckpt_file, map_location=dev, weights_only=False)
        state_dict = _extract_state_dict(raw)

    resolved_type = model_type.lower()
    if resolved_type == "auto":
        if state_dict is not None and any(k.startswith("layoutlmv3.") for k in state_dict.keys()):
            resolved_type = "ee_hamed"
        else:
            resolved_type = "dic"

    if resolved_type == "ee_hamed":
        if sorted_exits[-1] != 12:
            raise ValueError(f"ee_hamed requires final exit layer 12, got {sorted_exits}")
        early_spec: List[Union[str, int]] = ["text_visual_concat" if l == 0 else l for l in sorted_exits[:-1]]
        ee_cfg = ExitConfig(
            training_strategy="one_stage_subgraphs_weighted",
            inference_strategy="max_confidence",
            exits=early_spec,
            encoder_layer_strategy="ramp",
            exit_head_num_layers=2,
            enable_early_exit=False,
        )
        model: nn.Module = LayoutLMv3EEForSequenceClassification.from_local_checkpoint(
            num_labels=num_classes,
            ee_config=ee_cfg,
        )
    elif resolved_type == "dic_groupwise":
        model = DIC(num_classes=num_classes, exit_layers=sorted_exits)
    elif resolved_type == "dic":
        model = DICWithoutGroupwise(num_classes=num_classes, exit_layers=sorted_exits)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model = model.to(dev)
    model.eval()
    return model, resolved_type


class EarlyExitEvaluator:
    def __init__(
        self,
        model: nn.Module,
        exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True,
    ) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()
        self.exit_layers = tuple(sorted(int(l) for l in exit_layers))
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if (self.use_amp and torch.cuda.is_bf16_supported())
            else (torch.float16 if self.use_amp else torch.float32)
        )
        self.is_ee_hamed = isinstance(self.model, LayoutLMv3EEForSequenceClassification) or (
            hasattr(self.model, "layoutlmv3") and not hasattr(self.model, "exit_heads")
        )
        self.param_ratios = compute_subgraph_param_ratios(self.model, self.exit_layers)

    # ==========================================================================
    # 1. COLLECT LOGITS ACROSS ALL EXIT LAYERS
    # ==========================================================================

    @torch.no_grad()
    def collect_all_logits(self, dataloader: Any) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Collect logits for all configured exit layers on CPU (float32).
        Supports both `DICWithoutGroupwise`/`DIC` and `LayoutLMv3EEForSequenceClassification`.
        """
        self.model.eval()
        prev_ee_enabled = None
        if self.is_ee_hamed and hasattr(self.model, "set_early_exit_mode"):
            prev_ee_enabled = getattr(self.model.layoutlmv3, "enable_early_exit", False)
            self.model.set_early_exit_mode(enabled=False)

        logits_collector: Dict[int, List[torch.Tensor]] = {l: [] for l in self.exit_layers}
        labels_collector: List[torch.Tensor] = []

        try:
            for batch in dataloader:
                labels = batch["labels"]
                labels_collector.append(labels.detach().cpu().long())

                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                b = pixel_values.shape[0]
                if "input_ids" not in batch or "bbox" not in batch:
                    raise KeyError(
                        "Evaluation batch is missing OCR `input_ids`/`bbox`. "
                        "Use `build_cached_eval_loader` (RVLCDIPCachedDataset + MultimodalCollator)."
                    )
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                bbox = batch["bbox"].to(self.device, non_blocking=True)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                else:
                    attention_mask = torch.ones((b, input_ids.shape[1]), dtype=torch.long, device=self.device)

                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    if self.is_ee_hamed:
                        out = self.model(
                            input_ids=input_ids,
                            bbox=bbox,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                        )
                        for idx, l in enumerate(self.exit_layers[:-1]):
                            st = out.exit_states[idx]
                            lg = st[0] if isinstance(st, tuple) else st
                            logits_collector[l].append(lg.detach().float().cpu())
                        logits_collector[self.exit_layers[-1]].append(out.logits.detach().float().cpu())
                    else:
                        out = self.model(
                            input_ids=input_ids,
                            bbox=bbox,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            compute_projections=False,
                            compute_logits=True,
                        )
                        for l in self.exit_layers:
                            logits_collector[l].append(out.logits_dict[l].detach().float().cpu())
        finally:
            if prev_ee_enabled is not None and hasattr(self.model, "set_early_exit_mode"):
                self.model.set_early_exit_mode(enabled=prev_ee_enabled)

        all_logits = {l: torch.cat(logits_collector[l], dim=0) for l in self.exit_layers}
        all_labels = torch.cat(labels_collector, dim=0)
        return all_logits, all_labels

    # ==========================================================================
    # 2. LEAKAGE-FREE POST-HOC CALIBRATION (EQ. 7) & HEURISTIC THRESHOLDS (EQ. 10, 12)
    # ==========================================================================

    def fit_temperature_scaling(
        self,
        val_logits: Dict[int, torch.Tensor],
        val_labels: torch.Tensor,
    ) -> Dict[int, float]:
        """Fit per-exit TemperatureScaler strictly on the validation set (Eq. 7)."""
        temperatures: Dict[int, float] = {}
        for l in self.exit_layers:
            scaler = TemperatureScaler()
            temperatures[l] = float(scaler.fit(val_labels, val_logits[l]))
        return temperatures

    def apply_temperature_scaling(
        self,
        logits_dict: Dict[int, torch.Tensor],
        temperatures: Dict[int, float],
    ) -> Dict[int, torch.Tensor]:
        """Scale logits per exit layer using pre-fitted validation temperatures."""
        return {l: logits_dict[l] / float(temperatures.get(l, 1.0)) for l in self.exit_layers}

    def compute_heuristic_thresholds_from_val(
        self,
        val_logits_calibrated: Dict[int, torch.Tensor],
        val_labels: torch.Tensor,
        epsilon: float = 0.1,
    ) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
        """
        Compute per-exit Accuracy-Calibration Heuristic thresholds (Eq. 10 & Eq. 12)
        strictly from calibrated validation logits (no test-set leakage).
        """
        val_static = self.evaluate_static_exits(val_logits_calibrated, val_labels)
        accs = [val_static[l]["accuracy_01"] for l in self.exit_layers]
        eces = [val_static[l]["ece"] for l in self.exit_layers]
        tau_arr = EarlyExitPolicy.compute_heuristic_thresholds(
            accuracies=accs,
            eces=eces,
            epsilon=epsilon,
            exclude_final=True,
        )
        thresholds = {l: float(tau_arr[idx]) for idx, l in enumerate(self.exit_layers)}
        return thresholds, val_static

    # ==========================================================================
    # 3. STATIC PER-EXIT EVALUATION
    # ==========================================================================

    def evaluate_static_exits(
        self,
        all_logits: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[int, Dict[str, float]]:
        results: Dict[int, Dict[str, float]] = {}
        for l in self.exit_layers:
            metrics = compute_classification_metrics(labels, all_logits[l])
            metrics["param_ratio"] = float(self.param_ratios.get(l, 1.0))
            results[l] = metrics
        return results

    # ==========================================================================
    # 4. ADAPTIVE EARLY-EXITING INFERENCE (GLOBAL & PER-EXIT/HEURISTIC)
    # ==========================================================================

    def evaluate_adaptive_inference(
        self,
        all_logits: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        criterion: str = "max_confidence",
        threshold: float = 0.8,
        exit_thresholds: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        num_samples = labels.size(0)
        num_classes = all_logits[self.exit_layers[0]].size(-1)
        dev = all_logits[self.exit_layers[0]].device

        selected_logits = torch.zeros((num_samples, num_classes), dtype=torch.float32, device=dev)
        exit_at_layer = torch.zeros(num_samples, dtype=torch.long, device=dev)
        is_exited = torch.zeros(num_samples, dtype=torch.bool, device=dev)
        used_thresholds: Dict[int, float] = {}

        for l in self.exit_layers:
            active_mask = ~is_exited
            if not active_mask.any():
                break

            logits = all_logits[l][active_mask]
            tau = float(exit_thresholds[l]) if (exit_thresholds is not None and l in exit_thresholds) else float(threshold)
            used_thresholds[l] = tau

            if l == self.exit_layers[-1]:
                should_exit = torch.ones(logits.size(0), dtype=torch.bool, device=dev)
            else:
                if criterion == "max_confidence":
                    should_exit = max_confidence(logits) >= tau
                elif criterion == "entropy":
                    should_exit = entropy(logits) <= tau
                elif criterion == "top12_margin":
                    should_exit = top12_margin(logits) >= tau
                else:
                    raise ValueError(f"Không hỗ trợ criterion: {criterion}")

            active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            exited_indices = active_indices[should_exit]
            if exited_indices.numel() > 0:
                selected_logits[exited_indices] = logits[should_exit].float()
                exit_at_layer[exited_indices] = l
                is_exited[exited_indices] = True

        cls_metrics = compute_classification_metrics(labels, selected_logits)
        exit_distribution = {
            l: float((exit_at_layer == l).sum().item() / max(1, num_samples) * 100.0)
            for l in self.exit_layers
        }
        eff_metrics = compute_efficiency_metrics(
            exit_distribution_pct=exit_distribution,
            exit_layers=self.exit_layers,
            param_ratios=self.param_ratios,
        )

        return {
            "adaptive_accuracy": cls_metrics["accuracy"],
            "accuracy_01": cls_metrics["accuracy_01"],
            "f1_macro": cls_metrics["f1_macro"],
            "f1_micro": cls_metrics["f1_micro"],
            "ece": cls_metrics["ece"],
            "nll": cls_metrics["nll"],
            "brier": cls_metrics["brier"],
            "aurc": cls_metrics["aurc"],
            "avg_confidence": cls_metrics["avg_confidence"],
            "avg_entropy": cls_metrics["avg_entropy"],
            "avg_exit_layer": eff_metrics["avg_exit_layer"],
            "avg_exit_index": eff_metrics["avg_exit_index"],
            "relative_latency_reduction": eff_metrics["relative_latency_reduction"],
            "params_reduction": eff_metrics["params_reduction"],
            "compute_saved_percent": eff_metrics["compute_saved_percent"],
            "exit_distribution": exit_distribution,
            "exit_thresholds": used_thresholds,
        }

    # ==========================================================================
    # 5. THRESHOLD SWEEP (0.00 -> 1.00 STEP 0.05 PER HAMED ET AL. FIG. 3-4)
    # ==========================================================================

    def sweep_thresholds(
        self,
        all_logits: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        criterion: str = "max_confidence",
        thresholds: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        if thresholds is None:
            if criterion == "entropy":
                thresholds = [round(float(x), 2) for x in np.linspace(0.05, 2.5, 20)]
            else:
                thresholds = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)]

        sweep_results: List[Dict[str, Any]] = []
        for t in thresholds:
            res = self.evaluate_adaptive_inference(all_logits, labels, criterion=criterion, threshold=t)
            res["threshold"] = float(t)
            sweep_results.append(res)
        return sweep_results

    # ==========================================================================
    # 6. TRUE WALL-CLOCK EARLY-EXIT LATENCY BENCHMARK (BATCH SIZE = 1)
    # ==========================================================================

    @torch.no_grad()
    def benchmark_true_early_exit_latency(
        self,
        dataloader: Any,
        criterion: str = "max_confidence",
        threshold: float = 0.8,
        exit_thresholds: Optional[Dict[int, float]] = None,
        temperatures: Optional[Dict[int, float]] = None,
        max_samples: int = 32,
        warmup_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Benchmark real wall-clock inference latency (ms/sample) comparing full 12-layer forward
        vs true Anytime Early Exit stopping intermediate Transformer layers.
        """
        self.model.eval()
        samples: List[Dict[str, torch.Tensor]] = []
        for batch in dataloader:
            bs = batch["pixel_values"].shape[0]
            for i in range(bs):
                item = {
                    "input_ids": batch["input_ids"][i : i + 1].to(self.device),
                    "bbox": batch["bbox"][i : i + 1].to(self.device),
                    "pixel_values": batch["pixel_values"][i : i + 1].to(self.device),
                }
                if batch.get("attention_mask") is not None:
                    item["attention_mask"] = batch["attention_mask"][i : i + 1].to(self.device)
                samples.append(item)
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break

        if not samples:
            return {"num_samples": 0, "full_ms_per_sample": 0.0, "ee_ms_per_sample": 0.0, "wall_clock_speedup_percent": 0.0}

        def _sync() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        # Warmup
        for s in samples[: min(warmup_samples, len(samples))]:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                if self.is_ee_hamed:
                    self.model.set_early_exit_mode(enabled=False)
                    _ = self.model(**s)
                else:
                    _ = self.model(**s, compute_projections=False, compute_logits=True)

        # Full-depth timing
        _sync()
        t0 = time.perf_counter()
        for s in samples:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                if self.is_ee_hamed:
                    self.model.set_early_exit_mode(enabled=False)
                    _ = self.model(**s)
                else:
                    _ = self.model(**s, compute_projections=False, compute_logits=True)
        _sync()
        full_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Early-exit timing
        exit_counts = {l: 0 for l in self.exit_layers}
        if self.is_ee_hamed:
            ee_thresh_map: Optional[Dict[Union[int, str], float]] = None
            if exit_thresholds is not None:
                ee_thresh_map = {("text_visual_concat" if k == 0 else k): v for k, v in exit_thresholds.items()}
            self.model.set_early_exit_mode(enabled=True, global_threshold=threshold, exit_thresholds=ee_thresh_map)

        _sync()
        t1 = time.perf_counter()
        for s in samples:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                if self.is_ee_hamed:
                    out = self.model(**s)
                    if out.exited_at is None:
                        exited_layer = self.exit_layers[-1]
                    elif "concat" in out.exited_at:
                        exited_layer = 0
                    else:
                        digits = "".join(ch for ch in out.exited_at if ch.isdigit())
                        exited_layer = int(digits) if digits else self.exit_layers[-1]
                else:
                    _, exited_layer = self.model.forward_early_exit(
                        **s,
                        criterion=criterion,
                        threshold=threshold,
                        exit_thresholds=exit_thresholds,
                        temperatures=temperatures,
                    )
            if exited_layer in exit_counts:
                exit_counts[exited_layer] += 1
        _sync()
        ee_elapsed_ms = (time.perf_counter() - t1) * 1000.0

        if self.is_ee_hamed:
            self.model.set_early_exit_mode(enabled=False)

        n = len(samples)
        full_ms = full_elapsed_ms / n
        ee_ms = ee_elapsed_ms / n
        speedup_pct = (1.0 - ee_ms / max(full_ms, 1e-9)) * 100.0
        return {
            "num_samples": n,
            "full_ms_per_sample": float(full_ms),
            "ee_ms_per_sample": float(ee_ms),
            "wall_clock_speedup_percent": float(speedup_pct),
            "exit_distribution": {l: float(exit_counts[l] / n * 100.0) for l in self.exit_layers},
        }

    # ==========================================================================
    # 7. FULL EVALUATION PIPELINE + PARETO CURVE VISUALIZATION
    # ==========================================================================

    @staticmethod
    def plot_pareto_tradeoff(report: Dict[str, Any], output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
        x_specs = [
            ("params_reduction", "Params Reduction (%)"),
            ("relative_latency_reduction", "Relative Latency Reduction (%)"),
        ]

        uncal_sweep = report.get("uncalibrated", {}).get("global_sweep", [])
        cal_sweep = report.get("calibrated", {}).get("global_sweep", [])
        heur = report.get("calibrated", {}).get("heuristic")
        heur_sweep = report.get("calibrated", {}).get("heuristic_epsilon_sweep", [])

        for ax, (x_key, x_label) in zip(axes, x_specs):
            if uncal_sweep:
                ux = [p[x_key] for p in uncal_sweep]
                uy = [p["adaptive_accuracy"] for p in uncal_sweep]
                ax.plot(ux, uy, "--o", markersize=3.5, color="#D55E00", label="Global (Uncalibrated)")
            if cal_sweep:
                cx = [p[x_key] for p in cal_sweep]
                cy = [p["adaptive_accuracy"] for p in cal_sweep]
                ax.plot(cx, cy, "-s", markersize=3.5, color="#0072B2", label="Global (Calibrated)")
                frontier = extract_pareto_frontier(cal_sweep, x_key=x_key, y_key="adaptive_accuracy")
                if frontier:
                    ax.plot(
                        [p[x_key] for p in frontier],
                        [p["adaptive_accuracy"] for p in frontier],
                        "-",
                        linewidth=2.2,
                        color="#009E73",
                        alpha=0.85,
                        label="Pareto Frontier (Calibrated)",
                    )
            if heur_sweep:
                hx = [p[x_key] for p in heur_sweep]
                hy = [p["adaptive_accuracy"] for p in heur_sweep]
                ax.scatter(hx, hy, marker="^", s=38, color="#CC79A7", alpha=0.8, label="Heuristic Sweep (eps)")
            if heur is not None:
                ax.scatter(
                    [heur[x_key]],
                    [heur["adaptive_accuracy"]],
                    marker="*",
                    s=150,
                    color="#000000",
                    zorder=6,
                    label=f"Heuristic (eps={heur.get('epsilon', 0.1):.2f})",
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Adaptive Accuracy (%)")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=8, loc="lower left")

        fig.suptitle("Early-Exit Pareto Efficiency: Accuracy vs. Compute Reduction", fontsize=12)
        fig.tight_layout()

        png_path = out_path / "fig_pareto_tradeoff.png"
        pdf_path = out_path / "fig_pareto_tradeoff.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        return png_path, pdf_path

    def run_full_evaluation(
        self,
        test_loader: Any,
        val_loader: Optional[Any] = None,
        criterion: str = "max_confidence",
        threshold: float = 0.8,
        epsilon: float = 0.1,
        calibrate: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        benchmark_latency: bool = False,
        latency_max_samples: int = 32,
    ) -> Dict[str, Any]:
        """
        Execute the complete Hamed et al. evaluation protocol:
            - Static per-exit metrics (uncalibrated & calibrated)
            - Global threshold sweep (0.0 -> 1.0 step 0.05) + Pareto frontier
            - Accuracy-Calibration Heuristic policy (fitted on val, evaluated on test)
            - Optional true wall-clock latency benchmark
            - JSON report + Pareto figure export
        """
        test_logits, test_labels = self.collect_all_logits(test_loader)
        uncal_static = self.evaluate_static_exits(test_logits, test_labels)
        uncal_single = self.evaluate_adaptive_inference(test_logits, test_labels, criterion=criterion, threshold=threshold)
        uncal_sweep = self.sweep_thresholds(test_logits, test_labels, criterion=criterion)
        uncal_pareto = extract_pareto_frontier(uncal_sweep, x_key="params_reduction", y_key="adaptive_accuracy")

        report: Dict[str, Any] = {
            "exit_layers": list(self.exit_layers),
            "criterion": criterion,
            "num_test_samples": int(test_labels.numel()),
            "param_ratios": {str(k): v for k, v in self.param_ratios.items()},
            "uncalibrated": {
                "static": {str(k): v for k, v in uncal_static.items()},
                "single_threshold": {"threshold": threshold, **uncal_single},
                "global_sweep": uncal_sweep,
                "pareto_frontier": uncal_pareto,
            },
        }

        temperatures: Optional[Dict[int, float]] = None
        heur_thresholds: Optional[Dict[int, float]] = None

        if calibrate and val_loader is not None:
            val_logits, val_labels = self.collect_all_logits(val_loader)
            temperatures = self.fit_temperature_scaling(val_logits, val_labels)
            val_logits_cal = self.apply_temperature_scaling(val_logits, temperatures)
            test_logits_cal = self.apply_temperature_scaling(test_logits, temperatures)

            cal_static = self.evaluate_static_exits(test_logits_cal, test_labels)
            cal_single = self.evaluate_adaptive_inference(
                test_logits_cal, test_labels, criterion=criterion, threshold=threshold
            )
            cal_sweep = self.sweep_thresholds(test_logits_cal, test_labels, criterion=criterion)
            cal_pareto = extract_pareto_frontier(cal_sweep, x_key="params_reduction", y_key="adaptive_accuracy")

            heur_thresholds, val_cal_static = self.compute_heuristic_thresholds_from_val(
                val_logits_cal, val_labels, epsilon=epsilon
            )
            heur_res = self.evaluate_adaptive_inference(
                test_logits_cal,
                test_labels,
                criterion="max_confidence",
                exit_thresholds=heur_thresholds,
            )
            heur_res["epsilon"] = float(epsilon)

            eps_sweep: List[Dict[str, Any]] = []
            for eps_val in (0.05, 0.1, 0.2, 0.3, 0.5):
                tau_eps, _ = self.compute_heuristic_thresholds_from_val(val_logits_cal, val_labels, epsilon=eps_val)
                r_eps = self.evaluate_adaptive_inference(
                    test_logits_cal, test_labels, criterion="max_confidence", exit_thresholds=tau_eps
                )
                r_eps["epsilon"] = float(eps_val)
                eps_sweep.append(r_eps)

            report["num_val_samples"] = int(val_labels.numel())
            report["temperatures"] = {str(k): v for k, v in temperatures.items()}
            report["val_calibrated_static"] = {str(k): v for k, v in val_cal_static.items()}
            report["calibrated"] = {
                "static": {str(k): v for k, v in cal_static.items()},
                "single_threshold": {"threshold": threshold, **cal_single},
                "global_sweep": cal_sweep,
                "pareto_frontier": cal_pareto,
                "heuristic": heur_res,
                "heuristic_epsilon_sweep": eps_sweep,
            }

        if benchmark_latency:
            report["latency_benchmark"] = self.benchmark_true_early_exit_latency(
                dataloader=test_loader,
                criterion=criterion,
                threshold=threshold,
                exit_thresholds=heur_thresholds if heur_thresholds is not None else None,
                temperatures=temperatures,
                max_samples=latency_max_samples,
            )

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "evaluation_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            png_path, pdf_path = self.plot_pareto_tradeoff(report, out_path)
            report["artifacts"] = {
                "report_json": str(out_path / "evaluation_report.json"),
                "pareto_png": str(png_path),
                "pareto_pdf": str(pdf_path),
            }

        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá Anytime Early-Exiting Inference cho DIC")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "dic", "dic_groupwise", "ee_hamed"])
    parser.add_argument("--checkpoint_path", type=str, default="outputs/exp2_supcon/model_final.pt")
    parser.add_argument("--data_dir", type=str, default="data/rvl_cdip_100")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--exit_layers", type=int, nargs="+", default=[0, 1, 4, 8, 10, 12])
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--criterion", type=str, default="max_confidence", choices=["max_confidence", "entropy", "top12_margin"])
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark_latency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, resolved_type = load_model_for_evaluation(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        exit_layers=args.exit_layers,
        device=args.device,
    )
    evaluator = EarlyExitEvaluator(model=model, exit_layers=args.exit_layers, device=args.device)

    if args.dry_run:
        dummy_loader = [
            {
                "pixel_values": torch.randn(4, 3, 224, 224),
                "input_ids": torch.randint(0, 1000, (4, 16), dtype=torch.long),
                "bbox": torch.zeros((4, 16, 4), dtype=torch.long),
                "attention_mask": torch.ones((4, 16), dtype=torch.long),
                "labels": torch.tensor([0, 5, 10, 15]),
            }
            for _ in range(3)
        ]
        val_loader = dummy_loader
        test_loader = dummy_loader
    else:
        val_loader = (
            build_cached_eval_loader(args.data_dir, split="val", batch_size=args.batch_size, max_samples=args.max_samples)
            if args.calibrate
            else None
        )
        test_loader = build_cached_eval_loader(
            args.data_dir, split="test", batch_size=args.batch_size, max_samples=args.max_samples
        )

    report = evaluator.run_full_evaluation(
        test_loader=test_loader,
        val_loader=val_loader,
        criterion=args.criterion,
        threshold=args.threshold,
        epsilon=args.epsilon,
        calibrate=args.calibrate,
        output_dir=args.output_dir,
        benchmark_latency=args.benchmark_latency,
    )
    print(f"[Model={resolved_type}] Evaluated {report['num_test_samples']} test samples.")
    for l_str, m in report["uncalibrated"]["static"].items():
        print(
            f"  [Uncal] L{int(l_str):2d}: Acc={m['accuracy']:6.2f}% | F1={m['f1_macro']:.4f} "
            f"| ECE={m['ece']:.4f} | NLL={m['nll']:.4f} | AURC={m['aurc']:.4f}"
        )
    if "calibrated" in report:
        h = report["calibrated"]["heuristic"]
        print(
            f"  [Cal Heuristic eps={h['epsilon']:.2f}] Acc={h['adaptive_accuracy']:.2f}% | "
            f"ECE={h['ece']:.4f} | ParamsSaved={h['params_reduction']:.2f}% | "
            f"RelLatSaved={h['relative_latency_reduction']:.2f}%"
        )
    if "latency_benchmark" in report:
        lb = report["latency_benchmark"]
        print(
            f"  [True Latency] Full={lb['full_ms_per_sample']:.2f}ms -> EE={lb['ee_ms_per_sample']:.2f}ms "
            f"(Speedup={lb['wall_clock_speedup_percent']:.2f}%)"
        )
    if "artifacts" in report:
        print(f"  Saved report & Pareto plot to: {report['artifacts']['report_json']}")


if __name__ == "__main__":
    main()
