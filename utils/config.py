"""
Structured experiment configurations separating:
    1. EEBaselineConfig      : Hamed et al. (ICDAR 2024, arXiv:2405.12705)
                               https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/configs.py
    2. Stage1SupConConfig    : SimCLR (ICML 2020, arXiv:2002.05709) + ConOSR SupCon-ST (AAAI 2023)
                               https://github.com/google-research/simclr
                               https://github.com/nguhcv/cl-tad
                               https://github.com/NJU-RINC/ConOSR
    3. Stage2ClassifierConfig: ConOSR Step 2 Classifier + SimCLR Linear Eval + EE Calibration
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch.nn as nn


@dataclass
class EEBaselineConfig:
    """Hyperparameters for the 1-stage Early-Exit CE baseline (Hamed et al., ICDAR 2024)."""

    epochs: int = 20
    batch_size: int = 2
    eval_batch_size: int = 1
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "linear"
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    weighting_mode: str = "subgraph_repo"
    gamma: float = 0.5


@dataclass
class Stage1SupConConfig:
    """Hyperparameters for Stage 1 multi-exit contrastive representation learning (SimCLR + ConOSR SupCon-ST, no Mixup)."""

    epochs: int = 30
    batch_size: int = 4
    grad_accum_steps: int = 4
    sampler: str = "pk"
    pk_p: int = 2
    pk_k: int = 2

    temperature: float = 0.1
    base_temperature: float = 0.1
    smoothing_coeff: float = 0.1
    projection_hidden_dim: int = 768
    projection_dim: int = 128

    lr: float = 5e-5
    weight_decay: float = 1e-4
    exclude_bias_norm: bool = True
    use_lars: bool = False
    lars_eta: float = 0.02
    lars_clip: bool = True

    scheduler: str = "warmup_cosine"
    warmup_ratio: float = 0.1
    warmup_start_lr_ratio: float = 0.1
    eta_min: float = 0.0
    max_grad_norm: float = 1.0

    max_shear: float = 4.0
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    p_blur: float = 0.5
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    noise_std: float = 8.0
    p_erase: float = 0.3

    exit_weighting: str = "uniform"
    gamma: float = 0.5
    train_type: str = "without_groupwise"


@dataclass
class Stage2ClassifierConfig:
    """Hyperparameters for Stage 2 frozen-backbone multi-exit classifier training (ConOSR Step 2 + SimCLR Linear Eval + EE Calibration)."""

    epochs: int = 20
    batch_size: int = 8
    eval_batch_size: int = 1
    grad_accum_steps: int = 1

    label_smoothing: float = 0.1
    use_augmentation: bool = True
    exit_weighting: str = "uniform"

    lr: float = 3e-4
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.0
    eta_min: float = 0.0

    calibrate_temperature: bool = True
    exit_criterion: str = "max_confidence"


@dataclass
class ExperimentConfig:
    """Top-level configuration grouping shared settings with EE, Stage 1, and Stage 2 sub-configs."""

    exit_layers: Tuple[int, ...] = (0, 1, 4, 8, 10, 12)
    num_classes: int = 16
    seed: int = 42
    data_dir: str = "data/rvl_cdip_100"
    output_dir: str = "outputs"
    num_workers: int = 0
    use_amp: bool = True
    use_checkpointing: bool = True
    dry_run: bool = False

    ee_baseline: EEBaselineConfig = field(default_factory=EEBaselineConfig)
    stage1: Stage1SupConConfig = field(default_factory=Stage1SupConConfig)
    stage2: Stage2ClassifierConfig = field(default_factory=Stage2ClassifierConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_adamw_param_groups(
    named_parameters: Iterable[Tuple[str, nn.Parameter]],
    weight_decay: float,
    exclude_bias_norm: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build AdamW parameter groups separating decay vs no-decay (bias & LayerNorm/BatchNorm).
    Sets `exclude_from_lars=True` on the no-decay group per SimCLR (`google-research/simclr/lars_optimizer.py`).
    """
    if not exclude_bias_norm:
        params = [p for _, p in named_parameters if p.requires_grad]
        return [{"params": params, "weight_decay": weight_decay, "exclude_from_lars": False}]

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    no_decay_keywords = ("bias", "layernorm", "layer_norm", "batchnorm", "batch_norm", ".norm.")

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim <= 1 or any(kw in lname for kw in no_decay_keywords) or lname.endswith(".norm.weight"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay, "exclude_from_lars": False})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "exclude_from_lars": True})
    return groups
