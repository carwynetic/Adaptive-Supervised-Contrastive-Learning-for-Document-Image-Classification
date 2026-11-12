"""
Loss Functions for Document Image Classification (DIC).
Tái hiện nguyên bản từ:
    - ConOSR (AAAI 2023): https://github.com/NJU-RINC/ConOSR (models/ContrasiveLoss_SoftLabel.py)
    - Supervised Contrastive Learning with Soft Targets (SupCon-ST)
Kiến trúc tinh gọn phục vụ cả 2 chiến lược huấn luyện (điều phối qua train_type trong Trainer):
    - MultiExitSupConLoss: Tính toán SupCon-ST loss tại từng tầng exit [1, 4, 8, 10, 12] (Stage 1).
    - MultiExitCELoss    : Tính toán Cross-Entropy loss tại từng tầng exit [1, 4, 8, 10, 12] (Stage 2).
    
    Cả 2 module đều trả về Dict[layer -> Tensor (có grad_fn)].
    - Khi train_type == "without_groupwise": Dùng hàm `compute_total_loss()` để lấy 1 scalar tổng.
    - Khi train_type == "groupwise"        : Lấy trực tiếp loss theo từng cụm (Group) để backward riêng.
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss với Soft Targets (SupCon-ST).
    từ `ContrasiveLoss_SoftLabel.py` của ConOSR (AAAI 2023).
     toán học:
        - Độ tương đồng nhãn mềm:
            s(y_i, y_j) = cos(y_i, y_j) = (y_i . y_j) / (||y_i||_2 * ||y_j||_2)
        - Phân phối xác suất mục tiêu:
            S_{ij} = s(y_i, y_j) / sum_{k != i} s(y_i, y_k)
        - Phân phối xác suất biểu diễn:
            P_{ij} = exp(z_i . z_j / tau) / sum_{k != i} exp(z_i . z_k / tau)
        - Hàm mất mát:
            L = - sum_{i} sum_{j != i} S_{ij} * log(P_{ij})
    """
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: Tensor vector chiếu [batch_size, n_views, projection_dim].
            labels: Tensor nhãn mềm [batch_size, num_classes] (từ get_smooth_labels)
                    hoặc nhãn cứng 1D [batch_size].
            mask: Ma trận tương phản tùy chọn [batch_size, batch_size].

        Returns:
            Scalar tensor biểu diễn giá trị mất mát SupCon-ST.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                f"`features` cần có kích thước tối thiểu 3 chiều [batch_size, n_views, projection_dim], "
                f"nhưng nhận được shape {features.shape}"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Xây dựng ma trận tương phản mask theo chuẩn ConOSR
        if labels is not None and mask is not None:
            raise ValueError("Không thể truyền đồng thời cả `labels` và `mask`.")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            if labels.shape[0] != batch_size:
                raise ValueError(f"Số lượng labels ({labels.shape[0]}) không khớp với batch size ({batch_size}).")
            
            # Hỗ trợ tự động chuyển nhãn 1D sang one-hot nếu người dùng truyền nhãn nguyên
            if labels.dim() == 1:
                labels = F.one_hot(labels).float().to(device)
            else:
                labels = labels.float().to(device)

            # Chuẩn hóa L2 vector nhãn và tính Cosine Similarity giữa các vector nhãn: s(y_i, y_j)
            labels = F.normalize(labels, p=2, dim=1)
            mask = torch.mm(labels, torch.transpose(labels, 0, 1))
        else:
            mask = mask.float().to(device)

        mask.requires_grad = False
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Tích vô hướng độ tương đồng biểu diễn chia cho nhiệt độ tau
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Nhân bản mask và triệt tiêu self-contrast
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # Tính toán log xác suất
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Trọng số mục tiêu S_ij * log P_ij chuẩn ConOSR
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Loss trung bình trên toàn bộ batch
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Helper: Gom 2 views từ tuple (view1_dict, view2_dict) thành [B, 2, D]
def _format_projections(
    projections_input: Union[Dict[int, torch.Tensor], Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]],
    exit_layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    if isinstance(projections_input, tuple) and len(projections_input) == 2:
        dict_v1, dict_v2 = projections_input
        return {
            layer: torch.stack([dict_v1[layer], dict_v2[layer]], dim=1)
            for layer in exit_layers
        }
    return projections_input


# ==============================================================================
# 2. MultiExitSupConLoss (Stage 1 - Phục vụ cả 2 kiểu train_type)
# ==============================================================================

class MultiExitSupConLoss(nn.Module):
    """
    Hàm mất mát SupCon-ST đa tầng thoát sớm cho Giai đoạn 1 (Stage 1).
    Tính riêng lẻ loss tại từng tầng exit và trả về Dict[layer -> Tensor (có grad_fn)].

    Args:
        exit_layers: Danh sách các tầng thoát sớm (mặc định (1, 4, 8, 10, 12)).
        temperature: Nhiệt độ của hàm mất mát SupCon.
        default_weights: Trọng số mặc định cho từng tầng khi tính tổng (nếu None thì chia đều).
    """

    def __init__(
        self,
        exit_layers: Sequence[int] = (1, 4, 8, 10, 12),
        temperature: float = 0.07,
        default_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__()
        self.exit_layers = tuple(sorted(exit_layers))
        self.supcon = SupConLoss(temperature=temperature)

        if default_weights is not None:
            total_w = sum(default_weights.values())
            self.default_weights = {k: v / total_w for k, v in default_weights.items()}
        else:
            uniform_w = 1.0 / len(self.exit_layers)
            self.default_weights = {l: uniform_w for l in self.exit_layers}

    def forward(
        self,
        projections_input: Union[Dict[int, torch.Tensor], Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]],
        labels: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Tính toán loss tương phản riêng lẻ tại từng tầng.

        Args:
            projections_input: Dict[layer -> Tensor [B, 2, D]] hoặc Tuple(dict_v1, dict_v2).
            labels: Tensor nhãn mềm [B, K] từ get_smooth_labels hoặc nhãn cứng [B].

        Returns:
            Dict[layer -> Tensor loss (giữ nguyên grad_fn)].
        """
        proj_dict = _format_projections(projections_input, self.exit_layers)
        layer_losses: Dict[int, torch.Tensor] = {}

        for layer in self.exit_layers:
            layer_losses[layer] = self.supcon(proj_dict[layer], labels)

        return layer_losses

    def compute_total_loss(
        self,
        layer_losses: Dict[int, torch.Tensor],
        weights: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Hàm tiện ích gom tổng các loss tầng thành 1 scalar duy nhất (phục vụ train_type == 'without_groupwise').
        """
        w_dict = weights or self.default_weights
        first_loss = next(iter(layer_losses.values()))
        total_loss = torch.tensor(0.0, device=first_loss.device)

        for layer, loss_val in layer_losses.items():
            w = w_dict.get(layer, 1.0 / len(layer_losses))
            total_loss = total_loss + w * loss_val

        return total_loss


# ==============================================================================
# 3. MultiExitCELoss (Stage 2 - Phục vụ cả 2 kiểu train_type)
# ==============================================================================

class MultiExitCELoss(nn.Module):
    """
    Hàm mất mát Cross-Entropy đa tầng thoát sớm cho Giai đoạn 2 (Stage 2).
    Tính riêng lẻ CE loss tại từng tầng exit và trả về Dict[layer -> Tensor (có grad_fn)].

    Args:
        exit_layers: Danh sách các tầng thoát sớm (mặc định (1, 4, 8, 10, 12)).
        label_smoothing: Hệ số làm mịn nhãn cho Cross-Entropy (mặc định 0.0).
        default_weights: Trọng số mặc định cho từng tầng khi tính tổng (nếu None thì chia đều).
    """

    def __init__(
        self,
        exit_layers: Sequence[int] = (1, 4, 8, 10, 12),
        label_smoothing: float = 0.0,
        default_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__()
        self.exit_layers = tuple(sorted(exit_layers))
        self.ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if default_weights is not None:
            total_w = sum(default_weights.values())
            self.default_weights = {k: v / total_w for k, v in default_weights.items()}
        else:
            uniform_w = 1.0 / len(self.exit_layers)
            self.default_weights = {l: uniform_w for l in self.exit_layers}

    def forward(
        self,
        logits_dict: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Tính toán CE loss riêng lẻ tại từng tầng exit.

        Args:
            logits_dict: Dict[layer -> Tensor Logits [B, num_classes]].
            labels: Tensor nhãn lớp ground truth [B].

        Returns:
            Dict[layer -> Tensor CE loss (giữ nguyên grad_fn)].
        """
        layer_losses: Dict[int, torch.Tensor] = {}

        for layer in self.exit_layers:
            layer_losses[layer] = self.ce_fn(logits_dict[layer], labels)

        return layer_losses

    def compute_total_loss(
        self,
        layer_losses: Dict[int, torch.Tensor],
        weights: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Hàm tiện ích gom tổng các CE loss thành 1 scalar duy nhất (phục vụ train_type == 'without_groupwise').
        """
        w_dict = weights or self.default_weights
        first_loss = next(iter(layer_losses.values()))
        total_loss = torch.tensor(0.0, device=first_loss.device)

        for layer, loss_val in layer_losses.items():
            w = w_dict.get(layer, 1.0 / len(layer_losses))
            total_loss = total_loss + w * loss_val

        return total_loss


# ==============================================================================
# Unit Test tự kiểm chứng
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Testing Unified Loss Modules with ConOSR (utils/losses.py)...")
    print("=" * 65)

    from data_processing.data_preprocessing import get_smooth_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    batch_size = 4
    num_views = 2
    dim = 128
    num_classes = 16
    exit_layers = (1, 4, 8, 10, 12)

    hard_labels = torch.tensor([0, 0, 1, 15], dtype=torch.long, device=device)
    soft_targets = get_smooth_labels(hard_labels, classid_list=num_classes, smoothing_coeff=0.1).to(device)

    # --------------------------------------------------------------------------
    # 1. Test MultiExitSupConLoss (Stage 1)
    # --------------------------------------------------------------------------
    print("\n--- 1. Testing MultiExitSupConLoss (ConOSR Soft Targets) ---")
    supcon_criterion = MultiExitSupConLoss(exit_layers=exit_layers, temperature=0.1).to(device)
    dict_v1 = {l: F.normalize(torch.randn(batch_size, dim, device=device, requires_grad=True), p=2, dim=-1) for l in exit_layers}
    dict_v2 = {l: F.normalize(torch.randn(batch_size, dim, device=device, requires_grad=True), p=2, dim=-1) for l in exit_layers}

    # Hàm forward trả về Dict loss từng tầng
    supcon_losses = supcon_criterion((dict_v1, dict_v2), soft_targets)
    print(f"Computed layer losses: {list(supcon_losses.keys())}")
    for l, loss_val in supcon_losses.items():
        print(f"  - Exit Layer {l:2d} SupCon Loss: {loss_val.item():.4f}")

    # A. Mô phỏng train_type == "without_groupwise" (dùng compute_total_loss)
    total_supcon = supcon_criterion.compute_total_loss(supcon_losses)
    print(f"\n[without_groupwise] Total Scalar Loss: {total_supcon.item():.4f}")
    total_supcon.backward(retain_graph=True)
    print("  -> Backward successfully in without_groupwise mode.")

    # B. Mô phỏng train_type == "groupwise" (tự gom nhóm và backward riêng)
    loss_group1 = supcon_losses[1] + supcon_losses[4]
    loss_group2 = supcon_losses[8] + supcon_losses[10] + supcon_losses[12]
    loss_group1.backward(retain_graph=True)
    loss_group2.backward()
    print("  -> Backward successfully in groupwise mode.")
    print("[PASSED] MultiExitSupConLoss verified.")

    # --------------------------------------------------------------------------
    # 2. Test MultiExitCELoss (Stage 2)
    # --------------------------------------------------------------------------
    print("\n--- 2. Testing MultiExitCELoss (Stage 2) ---")
    ce_criterion = MultiExitCELoss(exit_layers=exit_layers).to(device)
    dummy_logits = {l: torch.randn(batch_size, num_classes, device=device, requires_grad=True) for l in exit_layers}

    ce_losses = ce_criterion(dummy_logits, hard_labels)
    for l, loss_val in ce_losses.items():
        print(f"  - Exit Layer {l:2d} CE Loss: {loss_val.item():.4f}")

    total_ce = ce_criterion.compute_total_loss(ce_losses)
    print(f"\n[without_groupwise] Total CE Scalar Loss: {total_ce.item():.4f}")
    total_ce.backward(retain_graph=True)

    ce_group1 = ce_losses[1] + ce_losses[4]
    ce_group1.backward()
    print("  -> Backward successfully in groupwise mode.")
    print("[PASSED] MultiExitCELoss verified.")

    print("\n" + "=" * 65)
    print("[ALL PASSED] Unified loss architecture verified successfully!")
    print("=" * 65)