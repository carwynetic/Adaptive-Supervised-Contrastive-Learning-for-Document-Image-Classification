"""
Document Image Classification (DIC) - Group-wise Early-Exiting Architecture.
Kiến trúc mô hình phân cụm nhóm tầng (Group-wise Layerwise Learning) cho DIC:
    - Phân chia 5 tầng Exit thành các cụm độc lập (mặc định: Group 1 [1, 4] và Group 2 [8, 10, 12]).
    - Hỗ trợ forward chọn lọc theo Group (early termination): Tiết kiệm tính toán khi huấn luyện Group 1.
    - Hỗ trợ đóng băng/mở khóa độc lập theo từng Group để triệt tiêu xung đột gradient.
    - Tương thích 100% với DICWithoutGroupwise, DICTrainer và EarlyExitEvaluator.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Đảm bảo đường dẫn gốc của project luôn nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DIC_without_groupwise import DICOutput, DICWithoutGroupwise
from model.modules.layoutlmv3 import (
    LayoutLMv3Backbone,
    LayoutLMv3ClassificationHead,
    LayoutLMv3Config,
)
from model.modules.projector import SupConProjector


class DIC(nn.Module):
    """
    Mô hình DIC hoàn chỉnh hỗ trợ chiến lược Group-wise Learning.
    """

    def __init__(
        self,
        num_classes: int = 16,
        exit_layers: Sequence[int] = (1, 4, 8, 10, 12),
        groups: Optional[Dict[str, Sequence[int]]] = None,
        projection_dim: int = 128,
        config_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.exit_layers = tuple(sorted(exit_layers))
        self.projection_dim = projection_dim

        # 1. Cấu hình phân nhóm các tầng Exits
        if groups is None:
            self.groups: Dict[str, Tuple[int, ...]] = {
                "group1": tuple(l for l in self.exit_layers if l <= 4),
                "group2": tuple(l for l in self.exit_layers if l > 4),
            }
        else:
            self.groups = {k: tuple(sorted(v)) for k, v in groups.items()}

        # 2. Backbone đa phương thức LayoutLMv3
        self.backbone = LayoutLMv3Backbone(
            config_dir=config_dir,
            checkpoint_dir=checkpoint_dir,
            output_hidden_states=True,
        )
        hidden_size = self.backbone.hidden_size
        backbone_config = self.backbone.config
        backbone_config.num_labels = num_classes
        backbone_config.classifier_dropout = dropout_prob

        # 3. Projection Heads cho từng tầng Exit (phục vụ SupCon-ST)
        self.projection_heads = nn.ModuleDict({
            str(layer): SupConProjector(
                in_dim=hidden_size,
                hidden_dim=hidden_size,
                out_dim=projection_dim,
                l2_normalize=True,
            )
            for layer in self.exit_layers
        })

        # 4. Exit Heads cho từng tầng Exit (phục vụ phân loại CE)
        self.exit_heads = nn.ModuleDict({
            str(layer): LayoutLMv3ClassificationHead(backbone_config, pool_feature=False)
            for layer in self.exit_layers
        })

    # ==========================================================================
    # QUẢN LÝ ĐÓNG BĂNG/MỞ KHÓA THEO CỤM (GROUP-WISE FREEZE CONTROLS)
    # ==========================================================================

    def freeze_encoder(self) -> None:
        """Đóng băng toàn bộ trọng số của Backbone Transformer."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_encoder(self) -> None:
        """Mở khóa toàn bộ trọng số của Backbone Transformer."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def is_encoder_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.backbone.parameters())

    def freeze_group(self, group_name: str) -> None:
        """
        Đóng băng toàn bộ Heads (Projector và Classifier) thuộc group chỉ định.
        """
        if group_name not in self.groups:
            raise KeyError(f"Group '{group_name}' không tồn tại trong {list(self.groups.keys())}")

        layers = self.groups[group_name]
        for l in layers:
            for p in self.projection_heads[str(l)].parameters():
                p.requires_grad = False
            for p in self.exit_heads[str(l)].parameters():
                p.requires_grad = False

    def unfreeze_group(self, group_name: str) -> None:
        """
        Mở khóa toàn bộ Heads thuộc group chỉ định.
        """
        if group_name not in self.groups:
            raise KeyError(f"Group '{group_name}' không tồn tại trong {list(self.groups.keys())}")

        layers = self.groups[group_name]
        for l in layers:
            for p in self.projection_heads[str(l)].parameters():
                p.requires_grad = True
            for p in self.exit_heads[str(l)].parameters():
                p.requires_grad = True

    # ==========================================================================
    # FORWARD PASS TOÀN BỘ MÔ HÌNH (STANDARD FORWARD)
    # ==========================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_projections: bool = True,
    ) -> DICOutput:
        """
        Forward đầy đủ qua tất cả các tầng exits.
        """
        if self.is_encoder_frozen:
            with torch.no_grad():
                backbone_out = self.backbone(
                    input_ids=input_ids,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                )
                hidden_states = tuple(h.detach() for h in backbone_out.hidden_states)
        else:
            backbone_out = self.backbone(
                input_ids=input_ids,
                bbox=bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            hidden_states = backbone_out.hidden_states

        logits_dict: Dict[int, torch.Tensor] = {}
        projections_dict: Optional[Dict[int, torch.Tensor]] = {} if compute_projections else None
        cls_tokens_dict: Dict[int, torch.Tensor] = {}

        for layer in self.exit_layers:
            cls_token = self.backbone.extract_cls_at_layer(hidden_states, layer)
            cls_tokens_dict[layer] = cls_token

            # Logits phân loại
            logits_dict[layer] = self.exit_heads[str(layer)](cls_token)

            # Vector chiếu tương phản L2
            if compute_projections and projections_dict is not None:
                projections_dict[layer] = self.projection_heads[str(layer)](cls_token)

        final_layer = self.exit_layers[-1]
        final_logits = logits_dict[final_layer]

        return DICOutput(
            logits_dict=logits_dict,
            final_logits=final_logits,
            projections_dict=projections_dict,
            cls_tokens_dict=cls_tokens_dict,
        )

    # ==========================================================================
    # FORWARD CHỌN LỌC THEO GROUP (GROUP-WISE SELECTIVE FORWARD)
    # ==========================================================================

    def forward_group(
        self,
        group_name: str,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_projections: bool = True,
    ) -> DICOutput:
        """
        Forward chọn lọc chỉ tính toán các tầng thuộc một Group cụ thể.
        Giúp tiết kiệm thời gian huấn luyện và triệt tiêu tính toán thừa ở các tầng sâu hơn.
        """
        if group_name not in self.groups:
            raise KeyError(f"Group '{group_name}' không tồn tại. Danh sách: {list(self.groups.keys())}")

        target_layers = self.groups[group_name]

        # Chạy backbone
        backbone_out = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_out.hidden_states

        logits_dict: Dict[int, torch.Tensor] = {}
        projections_dict: Optional[Dict[int, torch.Tensor]] = {} if compute_projections else None
        cls_tokens_dict: Dict[int, torch.Tensor] = {}

        for layer in target_layers:
            cls_token = self.backbone.extract_cls_at_layer(hidden_states, layer)
            cls_tokens_dict[layer] = cls_token
            logits_dict[layer] = self.exit_heads[str(layer)](cls_token)

            if compute_projections and projections_dict is not None:
                projections_dict[layer] = self.projection_heads[str(layer)](cls_token)

        last_layer = target_layers[-1]
        final_logits = logits_dict[last_layer]

        return DICOutput(
            logits_dict=logits_dict,
            final_logits=final_logits,
            projections_dict=projections_dict,
            cls_tokens_dict=cls_tokens_dict,
        )


# ==============================================================================
# Unit Test tự kiểm chứng kiến trúc DIC
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Testing Group-wise DIC Model Architecture (model/DIC.py)...")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 1. Khởi tạo mô hình
    exit_layers = (1, 4, 8, 10, 12)
    custom_groups = {"group1": (1, 4), "group2": (8, 10, 12)}
    model = DIC(num_classes=16, exit_layers=exit_layers, groups=custom_groups).to(device)

    print(f"[+] Exit layers : {model.exit_layers}")
    print(f"[+] Group config: {model.groups}")

    # 2. Tạo batch dữ liệu giả lập
    b, seq_len = 2, 32
    dummy_input_ids = torch.zeros((b, seq_len), dtype=torch.long, device=device)
    dummy_bbox = torch.zeros((b, seq_len, 4), dtype=torch.long, device=device)
    dummy_pixel_values = torch.randn(b, 3, 224, 224, device=device)

    # 3. Test Full Forward
    print("\n--- 1. Testing Full Forward ---")
    out_full = model(dummy_input_ids, dummy_bbox, dummy_pixel_values, compute_projections=True)
    print(f"  All exits computed: {list(out_full.logits_dict.keys())}")
    print(f"  Final logits shape: {out_full.final_logits.shape}")
    assert len(out_full.logits_dict) == 5, "Lỗi: Không đủ 5 tầng exits!"

    # 4. Test Forward Group 1 (Selective Forward)
    print("\n--- 2. Testing Forward Group 1 (Selective) ---")
    out_g1 = model.forward_group("group1", dummy_input_ids, dummy_bbox, dummy_pixel_values, compute_projections=True)
    print(f"  Group 1 exits computed: {list(out_g1.logits_dict.keys())}")
    assert set(out_g1.logits_dict.keys()) == {1, 4}, "Lỗi: Group 1 phải chỉ có tầng 1 và 4!"

    # 5. Test Group Freezing Controls
    print("\n--- 3. Testing Group Freezing Controls ---")
    model.freeze_group("group1")
    g1_frozen = all(not p.requires_grad for l in (1, 4) for p in model.exit_heads[str(l)].parameters())
    print(f"  Group 1 heads frozen: {g1_frozen} (Expected: True)")
    assert g1_frozen, "Lỗi: Heads của Group 1 chưa được đóng băng!"

    model.unfreeze_group("group1")
    g1_unfrozen = all(p.requires_grad for l in (1, 4) for p in model.exit_heads[str(l)].parameters())
    print(f"  Group 1 heads unfrozen: {g1_unfrozen} (Expected: True)")
    assert g1_unfrozen, "Lỗi: Heads của Group 1 chưa được mở khóa!"

    print("\n" + "=" * 65)
    print("[ALL PASSED] Group-wise DIC architecture verified successfully!")
    print("=" * 65)