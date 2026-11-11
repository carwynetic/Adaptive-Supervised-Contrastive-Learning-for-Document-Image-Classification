"""
Submodule Projection Head , chuẩn hóa cho Supervised COntrastive Learning (supcon-st)
ánh xạ đặc trưng feature [cls] từ không gian của Transformer sang không gian siêu cầu đơn vị L2-normalization
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn 
import torch.nn.functional as F

class SupConProjector(nn.Module):
    """
    Projection head chuẩn hóa :
        Linear (in_dim, hidden_dim)-> LayerNorm -> GELU -> Linear(hidden_dim, out_dim) -> L2-Normalization
    Args:
        in_dim: Kích thước vector đặc trưng đầu vào (mặc định 768 cho LayoutLMv3 Base).
        hidden_dim: Kích thước tầng ẩn trung gian của MLP (mặc định bằng in_dim = 768).
        out_dim: Kích thước vector chiếu đầu ra (mặc định 128).
        num_layers: Số tầng Linear (mặc định 2).
        dropout_prob: Xác suất dropout (mặc định 0.0).
        l2_normalize: Tự động chuẩn hóa L2 về siêu cầu đơn vị ở cuối forward pass.
    """
    def __init__(
        self,
        in_dim : int=768,
        hidden_dim : Optional[int] = None, 
        out_dim : int = 128,
        num_layers : int = 2,
        dropout_prob : float = 0.0,
        l2_normalize : bool= True,
    )-> None:
        super().__init__()
        self.in_dim =in_dim
        self.hidden_dim = hidden_dim or in_dim 
        self.out_dim = out_dim
        self.l2_normalize = l2_normalize
        
        layers : list[nn.Module] = []
        current_dim = self.in_dim

        for i in range(num_layers -1 ) :
            layers.append(nn.Linear(current_dim, self.hidden_dim, bias=True))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.GELU())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = self.hidden_dim
        
        #không áp dụng phi tuyến trước khi chuẩn hóa , tránh các điểm sample nằm trên góc 1/4 hình cầu
        #-> mất tính phong phú biểu diễn tương phản
        layers.append(nn.Linear(current_dim, self.out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        forward của projector.
        args: 
            x : Tensor biểu diễn đặc trưng 
        Returns :
            Tensor vector dã chuẩn hóa L2 (B, out_dim) nằm trên siêu cầu đơn vị
        """

        z = self.mlp(x)
        if self.l2_normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z

if __name__ == "__main__":
    projector = SupConProjector(in_dim=768, hidden_dim=768, out_dim=128, l2_normalize=True)
    dummy_feat = torch.randn(4, 768)
    z = projector(dummy_feat)

    print(f"Input shape : {dummy_feat.shape}")
    print(f"Output shape: {z.shape}")

    # Kiểm tra tính chất siêu cầu L2-norm: ||z||_2 phải bằng 1.0
    norms = torch.norm(z, p=2, dim=-1)
    print(f"L2 Norms    : {norms}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Error: Output is not L2-normalized!"
    print("[PASSED] SupConProjector verified successfully.")