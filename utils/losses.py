"""
loss function for DIC

dược tách thành 2 options rõ phục vụ cho nhiều mục đích :
    1. without groupwise : joing backprop - hammed et al.:  
        - joint multiple exit supcon-st loss : tính trực tiếp 1 loss total từ các projection heads rồi backprop 1 luồng (stage 1 trainng)
        - joint multple exit cross-entropy loss:  tính 1 loss total từ các exit classifier heads (stage 2 trainign)
    2. group-wise trainign:
        - LayerWiseSupCon-ST loss: trả về một Dict[int, torch.Tensor] chứa các loss riêng lẻ từng tầng để gom nhóm
        - LayerWiseCR loss : trả một Dict[int, torch.Tensor] chứa các loss của các CE loss ở các layer exits
"""


from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss với Soft Targets (SupCon-ST).
    Tái hiện nguyên bản 100% từ 'ContrasiveLoss_SoftLabel.py' của bài báo ConOSR (AAAI 2023).

    Bản chất toán học:
        - Độ tương đồng nhãn mềm giữa 2 mẫu i và j:
            s(y_i, y_j) = cos(y_i, y_j) = (y_i . y_j) / (||y_i||_2 * ||y_j||_2)
        - Phân phối mục tiêu:
            S_{ij} = s(y_i, y_j) / sum_{k != i} s(y_i, y_k)
        - Phân phối embedding:
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