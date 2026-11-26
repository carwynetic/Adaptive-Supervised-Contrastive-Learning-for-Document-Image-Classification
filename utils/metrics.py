"""
Offline Evaluation Metrics & Efficiency Analysis for Multi-Exit Document Image Classification.

Source:
    - Hamed et al. (ICDAR 2024, arXiv:2405.12705), Section 3.3 & Section 4
    - https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/eval.py
    - https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/analysis.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from model.modules.ee_modules import compute_ece


def _to_numpy_1d(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr).reshape(-1)


def _to_numpy_2d(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def logits_to_probs(logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    arr = _to_numpy_2d(logits)
    return softmax(arr, axis=-1)


def compute_accuracy(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
) -> float:
    y = _to_numpy_1d(labels)
    if y.size == 0:
        return 0.0
    preds = np.argmax(_to_numpy_2d(logits), axis=-1)
    return float(np.mean(preds == y))


def compute_f1_macro(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
) -> float:
    y = _to_numpy_1d(labels)
    if y.size == 0:
        return 0.0
    preds = np.argmax(_to_numpy_2d(logits), axis=-1)
    return float(f1_score(y, preds, average="macro", zero_division=0))


def compute_f1_micro(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
) -> float:
    y = _to_numpy_1d(labels)
    if y.size == 0:
        return 0.0
    preds = np.argmax(_to_numpy_2d(logits), axis=-1)
    return float(f1_score(y, preds, average="micro", zero_division=0))


def compute_nll(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
) -> float:
    y = _to_numpy_1d(labels).astype(np.int64)
    if y.size == 0:
        return 0.0
    log_p = log_softmax(_to_numpy_2d(logits), axis=-1)
    return float(-np.mean(log_p[np.arange(y.size), y]))


def compute_brier_score(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
) -> float:
    y = _to_numpy_1d(labels).astype(np.int64)
    if y.size == 0:
        return 0.0
    probs = logits_to_probs(logits)
    num_classes = probs.shape[-1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(y.size), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def compute_aurc(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Area Under the Risk-Coverage Curve (AURC) sorted by descending MSP confidence.
    Lower is better (0.0 for a perfect classifier).
    """
    y = _to_numpy_1d(labels)
    n = y.size
    if n == 0:
        return 0.0
    probs = logits_to_probs(logits)
    confidences = np.max(probs, axis=-1)
    preds = np.argmax(probs, axis=-1)
    residuals = (preds != y).astype(np.float64)

    order = np.argsort(-confidences, kind="mergesort")
    sorted_residuals = residuals[order]
    cum_errors = np.cumsum(sorted_residuals)
    coverages = np.arange(1, n + 1, dtype=np.float64)
    risks = cum_errors / coverages
    return float(np.mean(risks))


def compute_classification_metrics(
    labels: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
) -> Dict[str, float]:
    y = _to_numpy_1d(labels)
    arr = _to_numpy_2d(logits)
    probs = softmax(arr, axis=-1)
    log_p = log_softmax(arr, axis=-1)
    avg_conf = float(np.mean(np.max(probs, axis=-1))) if y.size > 0 else 0.0
    avg_ent = float(np.mean(-np.sum(probs * log_p, axis=-1))) if y.size > 0 else 0.0
    acc_01 = compute_accuracy(y, arr)
    return {
        "accuracy": acc_01 * 100.0,
        "accuracy_01": acc_01,
        "f1_macro": compute_f1_macro(y, arr),
        "f1_micro": compute_f1_micro(y, arr),
        "ece": float(compute_ece(y, arr, n_bins=n_bins)),
        "nll": compute_nll(y, arr),
        "brier": compute_brier_score(y, arr),
        "aurc": compute_aurc(y, arr),
        "avg_confidence": avg_conf,
        "avg_entropy": avg_ent,
    }


def _count_subgraph_params_for_exit(model: nn.Module, exit_layer: int) -> int:
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

    if hasattr(model, "exit_heads") and str(exit_layer) in model.exit_heads:
        total += sum(p.numel() for p in model.exit_heads[str(exit_layer)].parameters())
    elif hasattr(model, "classifier") and exit_layer == 12:
        total += sum(p.numel() for p in model.classifier.parameters())
    return max(1, total)


def compute_subgraph_param_ratios(
    model: nn.Module,
    exit_layers: Sequence[int],
) -> Dict[int, float]:
    sorted_layers = sorted(int(l) for l in exit_layers)
    if not sorted_layers:
        return {}
    counts = {l: _count_subgraph_params_for_exit(model, l) for l in sorted_layers}
    full_count = float(counts[sorted_layers[-1]])
    return {l: float(counts[l]) / full_count for l in sorted_layers}


def compute_efficiency_metrics(
    exit_distribution_pct: Dict[int, float],
    exit_layers: Sequence[int],
    param_ratios: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    sorted_layers = sorted(int(l) for l in exit_layers)
    num_exits = max(1, len(sorted_layers))
    fracs = {l: float(exit_distribution_pct.get(l, 0.0)) / 100.0 for l in sorted_layers}

    avg_exit_layer = sum(l * fracs[l] for l in sorted_layers)
    avg_exit_index = sum((idx + 1) * fracs[l] for idx, l in enumerate(sorted_layers))

    # Relative latency proxy per Hamed et al. (EE/analysis.py): (exit_index + 1) / num_exits
    rel_latency_used = sum(fracs[l] * ((idx + 1) / float(num_exits)) for idx, l in enumerate(sorted_layers))
    relative_latency_reduction = (1.0 - rel_latency_used) * 100.0

    if param_ratios:
        params_used_ratio = sum(fracs[l] * float(param_ratios.get(l, 1.0)) for l in sorted_layers)
        params_reduction = (1.0 - params_used_ratio) * 100.0
    else:
        params_reduction = relative_latency_reduction

    return {
        "avg_exit_layer": float(avg_exit_layer),
        "avg_exit_index": float(avg_exit_index),
        "relative_latency_reduction": float(relative_latency_reduction),
        "params_reduction": float(params_reduction),
        "compute_saved_percent": float(params_reduction),
    }


def extract_pareto_frontier(
    points: Sequence[Dict[str, Any]],
    x_key: str = "params_reduction",
    y_key: str = "adaptive_accuracy",
) -> List[Dict[str, Any]]:
    """
    Extract non-dominated Pareto-optimal points maximizing both `x_key` (efficiency) and `y_key` (accuracy).
    """
    sorted_pts = sorted(points, key=lambda d: (-float(d[x_key]), -float(d[y_key])))
    frontier: List[Dict[str, Any]] = []
    best_y = -float("inf")
    for pt in sorted_pts:
        y_val = float(pt[y_key])
        if y_val >= best_y - 1e-12:
            if frontier and abs(float(frontier[-1][x_key]) - float(pt[x_key])) < 1e-12:
                continue
            frontier.append(pt)
            best_y = max(best_y, y_val)
    frontier.sort(key=lambda d: float(d[x_key]))
    return frontier
