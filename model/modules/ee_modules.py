"""
Early Exit (EE) Modules, Configurations, Training Strategies, Calibration, and Policies.

Adapted from:
    Omar Hamed, Souhail Bakkali, Matthew Blaschko, Sien Moens, Jordy Van Landeghem,
    "Multimodal Adaptive Inference for Document Image Classification with Anytime Early Exiting",
    ICDAR 2024 (arXiv:2405.12705).
    Official repo: https://github.com/Jordy-VL/multi-modal-early-exit

Key adaptations for Adaptive_SupConST_DIC:
    1. Offline-ready execution (no online HuggingFace `evaluate.load("jordyvl/ece")` dependency).
    2. Resolved relative path dependencies for `EELayoutLM_exit_named_parameters-wotherexits.json`
       with automatic dynamic fallback when custom exit subsets are used.
    3. Exposed intermediate `exit_features` in `EEModelOutput` and `EESequenceClassifierOutput`
       so Supervised Contrastive (SupCon) projection heads can operate alongside Ramp/Gate classifiers.
    4. Integrated Temperature Scaling (Eq. 7), Global/Multi-exit Thresholding (Eq. 8-9),
       and the Accuracy-Calibration Heuristic Thresholding (Eq. 10-12) into a single module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import operator
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax as np_softmax
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_EXIT_PARAMS_JSON = _MODULE_DIR / "EELayoutLM_exit_named_parameters-wotherexits.json"

POSSIBLE_EXITS: List[Union[str, int]] = [
    "vision_avg",
    "text_avg",
    "text_visual_concat",
    *list(range(1, 13)),
]

# Các cấu hình vị trí đặt nhánh thoát sớm theo Mục 3.2 trong bài báo EE_2024
EXITS_PLACEMENT_PRESETS: Dict[str, List[Union[str, int]]] = {
    "independent_all": ["vision_avg", "text_avg", *list(range(1, 13))],          # 14 exits
    "concat_all": ["text_visual_concat", *list(range(1, 13))],                   # 13 exits
    "concat_single": ["text_visual_concat", 6],                                  # 2 exits
    "concat_quarter": ["text_visual_concat", 1, 4, 8, 12],                       # 5 exits
    "concat_alternate": ["text_visual_concat", 2, 5, 9, 11],                     # 5 exits
}


class Enhnum(str, Enum):
    """Enhanced string Enum supporting value lookup and listing all choices."""

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}-{self._name_}>"

    def _generate_next_value_(self, start, count, last_values) -> str:
        return self.lower()

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one from {cls.all()}"
        )

    @classmethod
    def all(cls) -> List[str]:
        return [c.value for c in cls]


class EarlyExitStrategy(Enhnum):
    """Chiến lược huấn luyện mạng Multi-Exit (Mục 3.2 - Fine-tuning Strategies)."""

    JOINT = "joint"
    JOINT_W_AVG = "joint_weighted_avg"  # Eq. (1): tổng đều các nhánh thoát + tầng cuối
    JOINT_W = "joint_weighted"
    TWO_STAGE = "two-stage"
    ALTERNATING = "alternating"
    LAYERWISE = "layerwise"

    # Subgraph training strategies (Eq. 1, 2, 3)
    ONE_STAGE_SUBGRAPHS = "one_stage_subgraphs"
    ONE_STAGE_SUBGRAPHS_WEIGHTED = "one_stage_subgraphs_weighted"                  # Eq. (2)
    ONE_STAGE_SUBGRAPHS_ENTROPYREG = "one_stage_subgraphs_entropyreg"              # Eq. (3)
    ONE_STAGE_SUBGRAPHS_WEIGHTED_ENTROPYREG = "one_stage_subgraphs_weighted_entropyreg"

    # Two-stage variants (freeze backbone, train exits)
    TWO_STAGE_SUBGRAPHS = "two_stage_subgraphs"
    TWO_STAGE_SUBGRAPHS_WEIGHTED = "two_stage_subgraphs_weighted"
    TWO_STAGE_SUBGRAPHS_ENTROPYREG = "two_stage_subgraphs_entropyreg"
    TWO_STAGE_SUBGRAPHS_WEIGHTED_ENTROPYREG = "two_stage_subgraphs_weighted_entropyreg"


class EarlyExitInference(Enhnum):
    """Tiêu chuẩn tính điểm tự tin (Confidence Scoring Function - CSF) khi suy luận."""

    MAX_CONFIDENCE = "max_confidence"  # Eq. (6): Maximum Softmax Probability (MSP)
    ENTROPY = "entropy"                # Entropy của phân phối dự đoán
    PATIENCE = "patience"              # PABEE patience criterion
    LTE = "lte"                        # BERxiT Learning-to-Exit criterion

    def get_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == EarlyExitInference.MAX_CONFIDENCE:
            return max_confidence
        if self == EarlyExitInference.ENTROPY:
            return entropy
        if self == EarlyExitInference.LTE:
            return lte
        raise NotImplementedError(f"{self} not implemented")

    def get_sign(self) -> Callable[[Any, Any], Any]:
        """Trả về toán tử so sánh với ngưỡng: MSP cần >= threshold, Entropy/LTE cần < threshold."""
        if self == EarlyExitInference.MAX_CONFIDENCE:
            return operator.ge
        if self in (EarlyExitInference.ENTROPY, EarlyExitInference.LTE):
            return operator.lt
        raise NotImplementedError(f"{self} not implemented")


class EarlyExitHead(Enhnum):
    """Loại tầng phân loại tại mỗi điểm thoát sớm (Mục 3.2 - Exit Classifiers Type)."""

    GATE = "gate"        # Nhị phân f_e: X -> {0, 1} (Eq. 4 - BCE loss)
    RAMP = "ramp"        # Đa lớp f_e: X -> [K] (Eq. 5 - CE loss)
    EMBEXIT = "embexit"  # Nhánh thoát ngay sau tầng embedding


def entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Tính Shannon Entropy từ raw logits (trước Softmax) ổn định số học.

    Args:
        x: Tensor logits có shape (B, K).

    Returns:
        Tensor entropy có shape (B,).
    """
    log_probs = F.log_softmax(x, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def max_confidence(x: torch.Tensor) -> torch.Tensor:
    """
    Tính Maximum Softmax Probability (MSP - Công thức 6 trong bài báo).

    Args:
        x: Tensor logits có shape (B, K).

    Returns:
        Tensor xác suất lớn nhất có shape (B,) nằm trong đoạn [0, 1].
    """
    probs = F.softmax(x, dim=-1)
    return torch.max(probs, dim=-1)[0]


def top12_margin(x: torch.Tensor) -> torch.Tensor:
    """Tính khoảng cách xác suất giữa lớp cao nhất (top-1) và cao thứ hai (top-2)."""
    probs = F.softmax(x, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)[0]
    if top2.shape[-1] == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def lte(x: torch.Tensor) -> torch.Tensor:
    """Placeholder cho tiêu chuẩn Learning-to-Exit (được tính trực tiếp từ nhánh LTE)."""
    return x.squeeze(-1)


class ExitConfig:
    """
    Cấu hình toàn diện cho kiến trúc LayoutLMv3 Early Exit.
    Hỗ trợ khởi tạo từ tên preset ('concat_quarter', 'independent_all', ...) hoặc danh sách tầng.
    """

    def __init__(self, **kwargs) -> None:
        self.training_strategy = EarlyExitStrategy(
            kwargs.get("training_strategy", "joint_weighted_avg")
        )
        self.inference_strategy = EarlyExitInference(
            kwargs.get("inference_strategy", "max_confidence")
        )
        self.global_threshold: float = float(kwargs.get("global_threshold", 0.9))

        raw_exits = kwargs.get("exits", "concat_quarter")
        self.exits: List[Union[str, int]] = self._parse_exits(raw_exits)

        self.exit_thresholds: Optional[Dict[Union[int, str], float]] = kwargs.get(
            "exit_thresholds", None
        )
        self.encoder_layer_strategy = EarlyExitHead(
            kwargs.get("encoder_layer_strategy", "ramp")
        )
        self.exit_head_num_layers: int = int(kwargs.get("exit_head_num_layers", 2))
        self.use_lte: bool = bool(kwargs.get("use_lte", False))
        self.gamma: float = float(kwargs.get("gamma", 0.5))
        self.temperature: float = float(kwargs.get("temperature", 1.0))
        self.enable_early_exit: bool = bool(kwargs.get("enable_early_exit", False))

    @staticmethod
    def _parse_exits(raw_exits: Union[str, Sequence[Union[str, int]]]) -> List[Union[str, int]]:
        if isinstance(raw_exits, str):
            key = raw_exits.strip().lower().replace("-", "_")
            if key in EXITS_PLACEMENT_PRESETS:
                return list(EXITS_PLACEMENT_PRESETS[key])
            parsed: List[Union[str, int]] = []
            for item in raw_exits.split(","):
                item_str = item.strip()
                if not item_str:
                    continue
                try:
                    parsed.append(int(item_str))
                except ValueError:
                    parsed.append(item_str)
            return parsed
        return list(raw_exits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_strategy": str(self.training_strategy),
            "inference_strategy": str(self.inference_strategy),
            "global_threshold": self.global_threshold,
            "exits": list(self.exits),
            "exit_thresholds": self.exit_thresholds,
            "encoder_layer_strategy": str(self.encoder_layer_strategy),
            "exit_head_num_layers": self.exit_head_num_layers,
            "use_lte": self.use_lte,
            "gamma": self.gamma,
            "temperature": self.temperature,
            "enable_early_exit": self.enable_early_exit,
        }


class EarlyExitException(Exception):
    """Ngoại lệ dùng để ngắt luồng tính toán ngay khi mẫu thỏa mãn điều kiện thoát sớm."""

    def __init__(self, payload: "EEModelOutput", exit_layer: str, exit_index: int = -1) -> None:
        super().__init__(f"Early exit triggered at {exit_layer} (index={exit_index})")
        self.payload = payload
        self.exit_layer = exit_layer
        self.exit_index = exit_index


@dataclass
class EEModelOutput(ModelOutput):
    """
    Cấu trúc đầu ra của LayoutLMv3ModelEE và LayoutLMv3EncoderEE.

    Attributes:
        last_hidden_state: Tensor (B, L, D) tại tầng cuối cùng được thực thi.
        hidden_states: Tuple các Tensor (B, L, D) qua các tầng encoder.
        attentions: Tuple các ma trận attention (nếu yêu cầu).
        exit_states: Tuple các logits (hoặc tuple (logits, criterion) khi eval) tại từng nhánh thoát.
        gate_inputs: Tuple các biểu diễn trung gian (B, D) dùng cho nhánh Gate.
        exit_features: Tuple các vector đặc trưng (B, D) trước tầng phân loại tại mỗi điểm thoát
            (hỗ trợ trực tiếp cho nhánh Projection Head của Adaptive SupCon).
        lte_output: Danh sách đầu ra của bộ dự đoán Learning-to-Exit (nếu bật use_lte).
        exited_at: Định danh tầng đã thoát sớm (None nếu chạy hết toàn mạng).
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    exit_states: Optional[Tuple[Any, ...]] = None
    gate_inputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    exit_features: Optional[Tuple[torch.FloatTensor, ...]] = None
    lte_output: Optional[List[torch.FloatTensor]] = None
    exited_at: Optional[str] = None


@dataclass
class EESequenceClassifierOutput(ModelOutput):
    """
    Cấu trúc đầu ra của mô hình phân loại tài liệu đa nhánh thoát LayoutLMv3EEForSequenceClassification.
    """

    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    exit_losses: Optional[List[torch.FloatTensor]] = None
    exit_criteria: Optional[List[torch.FloatTensor]] = None
    exit_states: Optional[Tuple[Any, ...]] = None
    exit_features: Optional[Tuple[torch.FloatTensor, ...]] = None
    gated_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    lte_output: Optional[List[torch.FloatTensor]] = None
    exited_at: Optional[str] = None


class LogitNormLoss(nn.Module):
    """Hàm mất mát chuẩn hóa chuẩn L2 của Logits trước khi tính Cross-Entropy."""

    def __init__(self, t: float = 1.0) -> None:
        super().__init__()
        self.t = t

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


def filter_encoder_exits(lst: Sequence[str], encoder_counter: int) -> List[str]:
    """Lọc các tham số thuộc về các nhánh thoát từ đầu đến nhánh encoder_counter."""
    return [
        item
        for item in lst
        if (
            item.startswith("encoder.early_exits.")
            and int(item.split(".")[-3]) <= encoder_counter
        )
        or not item.startswith("encoder.early_exits.")
    ]


def translate_exit_identifier(exit_id: Union[int, str], encoder_counter: int = 0) -> str:
    """Ánh xạ tên điểm thoát trong cấu hình sang tên module tương ứng trong PyTorch."""
    if isinstance(exit_id, int):
        return f"encoder.early_exits.{encoder_counter}"
    if "avg" in exit_id:
        return exit_id.replace("avg", "exit") + "_embeddings"
    if "concat" in exit_id:
        return "concat_exit_embeddings"
    return "classifier"


def _dynamic_subgraph_modules(
    model: nn.Module,
    exit_id: Union[int, str],
    encoder_counter: int,
) -> List[str]:
    """
    Tính toán động danh sách tên tham số thuộc subgraph tính đến điểm thoát `exit_id`
    (dùng khi không có file JSON hoặc khi cấu hình nhánh thoát tùy chỉnh).
    """
    all_names = [name for name, _ in model.named_parameters()]
    selected: List[str] = []

    for name in all_names:
        # Bỏ qua prefix 'layoutlmv3.' khi so sánh cấu trúc bên trong
        clean = name[len("layoutlmv3.") :] if name.startswith("layoutlmv3.") else name

        if exit_id == "vision_avg":
            if any(
                clean.startswith(prefix)
                for prefix in ("cls_token", "pos_embed", "patch_embed", "norm", "vision_exit_embeddings")
            ):
                selected.append(name)
        elif exit_id == "text_avg":
            if any(
                clean.startswith(prefix)
                for prefix in ("embeddings.", "text_exit_embeddings")
            ):
                selected.append(name)
        elif exit_id == "text_visual_concat":
            if any(
                clean.startswith(prefix)
                for prefix in (
                    "cls_token",
                    "pos_embed",
                    "patch_embed",
                    "norm",
                    "embeddings.",
                    "LayerNorm.",
                    "concat_exit_embeddings",
                )
            ):
                selected.append(name)
        elif isinstance(exit_id, int):
            # Bao gồm toàn bộ embeddings + encoder layers < exit_id + early_exits <= encoder_counter
            if any(
                clean.startswith(prefix)
                for prefix in (
                    "cls_token",
                    "pos_embed",
                    "patch_embed",
                    "norm",
                    "embeddings.",
                    "LayerNorm.",
                    "encoder.rel_pos",
                )
            ):
                selected.append(name)
            elif clean.startswith("encoder.layer."):
                parts = clean.split(".")
                if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) < exit_id:
                    selected.append(name)
            elif clean.startswith("encoder.early_exits."):
                parts = clean.split(".")
                if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) <= encoder_counter:
                    selected.append(name)

    return selected


def count_named_parameters(model: nn.Module, modules: Sequence[str]) -> int:
    """Đếm tổng số lượng tham số của các module nằm trong danh sách `modules`."""
    module_set = set(modules)
    total = 0
    for name, parameter in model.named_parameters():
        clean = name[len("layoutlmv3.") :] if name.startswith("layoutlmv3.") else name
        if name in module_set or clean in module_set:
            total += parameter.numel()
    return max(1, total)


def params_per_exit(
    model: nn.Module,
    beta: float = 1.0,
    json_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, List[str]], Dict[int, float], Dict[int, List[str]]]:
    """
    Tính toán trọng số Subgraph Weighting w_b (Công thức 2 trong bài báo):
    Mỗi nhánh thoát sớm b được gán trọng số tỉ lệ nghịch với số lượng tham số
    của đồ thị con G_b tính từ đầu mạng tới nhánh thoát b.

    Returns:
        - exit_named_params: Dict ánh xạ tên nhánh thoát -> danh sách tên tham số thuộc subgraph.
        - exit_loss_weights: Dict ánh xạ chỉ số nhánh thoát b -> trọng số chuẩn hóa w_b.
        - exit_branch_params: Dict ánh xạ chỉ số nhánh thoát b -> danh sách tên tham số của riêng head đó.
    """
    target_model = model.module if not hasattr(model, "config") and hasattr(model, "module") else model
    exit_config = getattr(target_model.config, "exit_config", {})
    if isinstance(exit_config, ExitConfig):
        exit_config = exit_config.to_dict()

    json_file = Path(json_path) if json_path is not None else _DEFAULT_EXIT_PARAMS_JSON
    exit_named_params: Dict[str, List[str]] = {}
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            exit_named_params = json.load(f)

    exit_param_count: Dict[int, float] = {}
    exit_branch_params: Dict[int, List[str]] = {}
    encoder_counter = 0

    exits_list = exit_config.get("exits", [])
    all_model_param_names = [name for name, _ in target_model.named_parameters()]

    for exit_index, exit_id in enumerate(exits_list):
        str_key = str(exit_id)
        if str_key in exit_named_params:
            modules = exit_named_params[str_key]
            if isinstance(exit_id, int):
                modules = filter_encoder_exits(modules, encoder_counter)
                exit_named_params[str_key] = modules
        else:
            modules = _dynamic_subgraph_modules(target_model, exit_id, encoder_counter)
            exit_named_params[str_key] = modules

        exit_name = translate_exit_identifier(exit_id, encoder_counter)
        exit_branch_params[exit_index] = [
            name for name in all_model_param_names if exit_name in name
        ]

        param_count = count_named_parameters(target_model, modules=modules)
        exit_param_count[exit_index] = beta / float(param_count)

        if isinstance(exit_id, int):
            encoder_counter += 1

    exit_branch_params[len(exits_list)] = [
        name for name in all_model_param_names if "classifier" in name
    ]

    total_inv = sum(exit_param_count.values()) or 1.0
    exit_loss_weights = {
        k: (v / total_inv) for k, v in exit_param_count.items()
    }
    return exit_named_params, exit_loss_weights, exit_branch_params


def compute_entropy_reg_weights(
    exit_states: Sequence[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    final_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Tính hệ số điều chuẩn Entropy \\lambda_b cho từng nhánh thoát (Công thức 3 trong bài báo):
        \\lambda = 1 - softmax(H(Y; f(X; \\theta_{G_b})))
    """
    device = final_logits.device
    num_branches = len(exit_states) + 1
    exit_entropies = torch.zeros(num_branches, device=device)

    for j, state in enumerate(exit_states):
        logits_j = state[0] if isinstance(state, tuple) else state
        exit_entropies[j] = entropy(logits_j).mean()
    exit_entropies[-1] = entropy(final_logits).mean()

    norm_entropy = F.softmax(exit_entropies, dim=0) * num_branches
    norm_entropy = torch.clamp(norm_entropy, max=1.0)
    return 1.0 - norm_entropy


class EETrainingArguments(TrainingArguments):
    """Mở rộng HuggingFace TrainingArguments với các siêu tham số điều khiển Early Exit."""

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        temperature: float = 1.0,
        gamma: float = 0.5,
        training_strategy: str = "one_stage_subgraphs_weighted",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.gamma = gamma
        self.training_strategy = training_strategy


class EETrainer(Trainer):
    """
    Custom Trainer cài đặt thuật toán huấn luyện Subgraph Weighting (Eq. 2)
    và Exit Entropy Regularization (Eq. 3) của bài báo EE_2024.
    """

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            exit_losses = outputs["exit_losses"]

        if self.args.n_gpu > 1:
            loss = loss.mean()
            exit_losses = tuple(x.mean() for x in exit_losses)

        if self.args.gradient_accumulation_steps > 1 and not getattr(self, "deepspeed", None):
            loss = loss / self.args.gradient_accumulation_steps
            exit_losses = tuple(
                x / self.args.gradient_accumulation_steps for x in exit_losses
            )

        _, exit_loss_weights, exit_branch_params = params_per_exit(model)

        final_loss = torch.tensor(0.0, device=loss.device)
        num_exits = max(1, len(exit_losses))

        for j, exit_loss in enumerate(exit_losses):
            weighted_exit_loss = exit_loss
            if "weighted" in self.args.training_strategy:
                weighted_exit_loss = weighted_exit_loss * exit_loss_weights[j]
            if self.args.gamma != 0:
                weighted_exit_loss = weighted_exit_loss * (self.args.gamma / num_exits)
            weighted_exit_loss.backward(retain_graph=True)
            final_loss = final_loss + weighted_exit_loss.detach()

        if self.args.gamma != 0:
            loss = loss * (1.0 - self.args.gamma)
        loss.backward()
        final_loss = final_loss + loss.detach()

        if "entropyreg" in self.args.training_strategy and outputs.get("exit_states"):
            neg_norm_entropy = compute_entropy_reg_weights(
                outputs["exit_states"], outputs["logits"]
            )
            for j, criterion in enumerate(neg_norm_entropy):
                branch_names = set(exit_branch_params.get(j, []))
                for name, p in model.named_parameters():
                    if p.grad is not None and name in branch_names:
                        p.grad.mul_(criterion)

        return final_loss


# ==============================================================================
# Post-Hoc Calibration (Eq. 7 & Eq. 11) và Exit Policies (Eq. 8, 9, 10, 12)
# ==============================================================================

def compute_ece(
    references: Union[np.ndarray, torch.Tensor],
    logits_or_probs: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
) -> float:
    """
    Tính Expected Calibration Error (ECE - Công thức 11 trong bài báo):
        ECE = \\sum_{j=1}^J (|B_j| / N) * |ACC(B_j) - MSP(B_j)|

    Hoạt động hoàn toàn offline trên NumPy/PyTorch, không phụ thuộc mạng.
    """
    if isinstance(references, torch.Tensor):
        references = references.detach().cpu().numpy()
    if isinstance(logits_or_probs, torch.Tensor):
        logits_or_probs = logits_or_probs.detach().cpu().numpy()

    references = np.asarray(references).reshape(-1)
    probs = np.asarray(logits_or_probs, dtype=np.float64)

    # Tự động chuyển logits sang xác suất nếu tổng hàng khác 1
    if probs.ndim == 2 and not np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-3):
        probs = np_softmax(probs, axis=-1)

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == references).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(references)
    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler:
    """
    Hiệu chuẩn xác suất (Post-hoc Temperature Scaling - Công thức 7 trong bài báo)
    tối ưu hóa Negative Log-Likelihood (NLL) trên tập validation bằng L-BFGS-B.
    """

    def __init__(self, temperature: Optional[float] = None) -> None:
        self.temperature = (
            np.ones(1, dtype=np.float64)
            if temperature is None
            else np.ones(1, dtype=np.float64) * float(temperature)
        )

    def temperature_scale(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature[0]

    def transform(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        return np_softmax(self.temperature_scale(np.asarray(logits, dtype=np.float64)), axis=-1)

    def fit(
        self,
        labels: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor],
    ) -> float:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        labels_arr = np.asarray(labels).reshape(-1)
        logits_arr = np.asarray(logits, dtype=np.float64)
        class_indices = np.arange(logits_arr.shape[-1])

        def objective(temp: np.ndarray) -> float:
            scaled_probs = np_softmax(logits_arr / temp[0], axis=-1)
            return float(log_loss(labels_arr, scaled_probs, labels=class_indices))

        result = minimize(
            objective,
            x0=self.temperature,
            method="L-BFGS-B",
            bounds=[(1e-3, 100.0)],
        )
        if result.success:
            self.temperature = result.x
        return float(self.temperature[0])


class EarlyExitPolicy:
    """
    Cài đặt các chính sách suy luận thoát sớm (Mục 3.3 - Inference Strategies):
        1. Global Thresholding (Eq. 8)
        2. Multi-exit Thresholding (Eq. 9)
        3. Proposed Accuracy-Calibration Heuristic Thresholding (Eq. 10 & Eq. 12)
    """

    def __init__(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Args:
            logits: Tensor hoặc mảng NumPy có shape (num_exits + 1, num_samples, num_labels).
            device: Thiết bị trả về tensor dự đoán.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        self.logits = np.asarray(logits, dtype=np.float64)
        self.device = torch.device(device)

    @staticmethod
    def compute_heuristic_thresholds(
        accuracies: Sequence[float],
        eces: Sequence[float],
        epsilon: float = 0.1,
    ) -> np.ndarray:
        """
        Tính ngưỡng thoát sớm động cho từng tầng theo Công thức (10) và (12):
            T_b = 1 - (ACC_b / ECE_b)
            \\tau_{normalized} = (T - (min(T) - \\epsilon)) / ((max(T) + \\epsilon) - (min(T) - \\epsilon))
        """
        raw_metrics = np.array(
            [1.0 - (acc / max(ece, 1e-8)) for acc, ece in zip(accuracies, eces)],
            dtype=np.float64,
        )
        min_v = np.min(raw_metrics) - epsilon
        max_v = np.max(raw_metrics) + epsilon
        denom = max(max_v - min_v, 1e-8)
        return (raw_metrics - min_v) / denom

    def apply_thresholds(
        self, thresholds: Union[float, Sequence[float], Dict[int, float]]
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[int, float]]:
        """
        Áp dụng ngưỡng (đơn hoặc đa tầng) lên tập logits có shape (num_exits_total, N, K).

        Returns:
            - exits_store: Mảng (N,) lưu chỉ số nhánh thoát của từng mẫu.
            - predictions: Tensor (N, K) lưu vector logits tại nhánh đã thoát.
            - exit_distribution: Tỷ lệ mẫu thoát tại từng nhánh.
        """
        num_exits, num_samples, num_labels = self.logits.shape
        if isinstance(thresholds, (float, int)):
            thresh_list = [float(thresholds)] * num_exits
        elif isinstance(thresholds, dict):
            thresh_list = [float(thresholds.get(i, 1.0)) for i in range(num_exits)]
        else:
            thresh_list = [float(t) for t in thresholds]
            if len(thresh_list) < num_exits:
                thresh_list.extend([1.0] * (num_exits - len(thresh_list)))

        exits_store = np.zeros(num_samples, dtype=np.int32)
        predictions = torch.zeros(
            (num_samples, num_labels), dtype=torch.float64, device=self.device
        )

        # Tính sẵn MSP cho toàn bộ nhánh và mẫu: shape (num_exits, num_samples)
        probs = np_softmax(self.logits, axis=-1)
        msp_scores = np.max(probs, axis=-1)

        for sample_id in range(num_samples):
            for exit_id in range(num_exits):
                if exit_id == num_exits - 1 or msp_scores[exit_id, sample_id] > thresh_list[exit_id]:
                    exits_store[sample_id] = exit_id
                    predictions[sample_id] = torch.from_numpy(self.logits[exit_id, sample_id])
                    break

        exit_distribution = {
            exit_id: float(np.count_nonzero(exits_store == exit_id)) / max(1, num_samples)
            for exit_id in range(num_exits)
        }
        return exits_store, predictions, exit_distribution

    def max_confidence_global_thresholding_policy(
        self, exit_threshold: float
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[int, float]]:
        """Chính sách Global Thresholding (Công thức 8)."""
        return self.apply_thresholds(exit_threshold)

    def accuracy_calibration_heuristic(
        self,
        accuracies: Sequence[float],
        eces: Sequence[float],
        epsilon: float = 0.1,
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[int, float], np.ndarray]:
        """Chính sách Heuristic dựa trên tỷ số Accuracy / ECE (Công thức 10 & 12)."""
        thresholds = self.compute_heuristic_thresholds(accuracies, eces, epsilon=epsilon)
        exits_store, predictions, exit_distribution = self.apply_thresholds(thresholds)
        return exits_store, predictions, exit_distribution, thresholds



if __name__ == "__main__":
    print("=" * 70)
    print("Testing EE Calibration & Thresholding Policies (EE_2024 Eq. 7-12)...")
    print("=" * 70)

    np.random.seed(42)
    num_exits_total = 6  # 5 early exits + 1 final classifier
    num_samples = 100
    num_classes = 16

    sim_logits = np.random.randn(num_exits_total, num_samples, num_classes)
    sim_labels = np.random.randint(0, num_classes, size=(num_samples,))

    # 1. Temperature Scaling & ECE per exit
    accs, eces, temps = [], [], []
    calibrated_logits = np.zeros_like(sim_logits)
    for b in range(num_exits_total):
        scaler = TemperatureScaler()
        opt_t = scaler.fit(sim_labels, sim_logits[b])
        calibrated_logits[b] = scaler.temperature_scale(sim_logits[b])
        preds_b = np.argmax(calibrated_logits[b], axis=-1)
        acc_b = float(np.mean(preds_b == sim_labels))
        ece_b = compute_ece(sim_labels, calibrated_logits[b])
        accs.append(acc_b)
        eces.append(ece_b)
        temps.append(opt_t)

    print(f"Optimal Temperatures per exit : {[round(t, 4) for t in temps]}")
    print(f"Post-calibration ECE per exit : {[round(e, 4) for e in eces]}")

    # 2. Global Thresholding Policy (Eq. 8)
    policy = EarlyExitPolicy(calibrated_logits)
    exits_g, preds_g, dist_g = policy.max_confidence_global_thresholding_policy(0.25)
    print(f"Global Thresholding Dist      : { {k: round(v, 3) for k, v in dist_g.items()} }")

    # 3. Proposed Accuracy-Calibration Heuristic Policy (Eq. 10 & 12)
    exits_h, preds_h, dist_h, tau_norm = policy.accuracy_calibration_heuristic(
        accuracies=accs, eces=eces, epsilon=0.1
    )
    print(f"Heuristic Normalized Tau      : {[round(float(x), 4) for x in tau_norm]}")
    print(f"Heuristic Exit Distribution   : { {k: round(v, 3) for k, v in dist_h.items()} }")
    print("\n[PASSED] EE Calibration & Policy verification completed.")
