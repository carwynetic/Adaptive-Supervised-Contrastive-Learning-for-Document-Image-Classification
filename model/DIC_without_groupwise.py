"""
Document Image Classification (DIC) Model Architecture - Without Group-wise Training.
Pure PyTorch nn.Module architecture unifying:
    - Multi-modal Transformer Backbone: LayoutLMv3 with intermediate exits at [1, 4, 8, 10, 12].
    - Contrastive Projection Heads: Multi-depth MLP (768 -> 768 -> 128, L2-normalized) for SupCon-ST.
    - Early Exit Classification Heads: LayoutLMv3ClassificationHead for classification at each depth.
    - Encoder freezing/unfreezing controls for two-stage training.
Note:
    Training loops, loss calculations, data loaders, and evaluation routines belong
    strictly to `train_test/` and `utils/`.
"""

from __future__     import annotations
from dataclasses    import dataclass
from pathlib        import Path
from typing         import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn 
import torch.nn.functional as F

from model.modules.layoutlmv3 import (
    LayoutLMv3Backbone, 
    LayoutLMv3ClassificationHead,
    LayoutLMv3Config,
)
from model.modules.projector import SupConProjector

@dataclass 
class DICOutput:
    logits_dict     : Dict[int, torch.Tensor]
    final_logits    : Optional[torch.Tensor] = None
    projections_dict: Optional[Dict[int , torch.Tensor]] = None
    cls_tokens_dict : Optional[Dict[int, torch.Tensor]] = None

class DICWithoutGroupwise(nn.Module):
    """
    mô hình kiến trúc DIC không dùng group-wise training
    chỉ mô tả kiến trúc mạng (layers, forward pass,  freeze/unfreeze)
    """
    def __init__(
        self,
        num_classes: int = 16,
        exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
        projection_dim: int = 128,
        config_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        dropout_prob: float = 0.1,
    ) -> None:
    
        super().__init__()
        self.num_classes = num_classes
        self.exit_layers = tuple(sorted(exit_layers))
        self.projection_dim = projection_dim

        # 1. Khung mạng Transformer LayoutLMv3Backbone
        self.backbone = LayoutLMv3Backbone(
            config_dir=config_dir,
            checkpoint_dir=checkpoint_dir,
            output_hidden_states=True,
        )
        hidden_size = self.backbone.hidden_size
        backbone_config = self.backbone.config
        backbone_config.num_labels = num_classes
        backbone_config.classifier_dropout = dropout_prob

        # 2. Các Projection Heads cho SupCon-ST
        self.projection_heads = nn.ModuleDict({
            str(layer): SupConProjector(
                in_dim=hidden_size,
                hidden_dim=hidden_size,
                out_dim=projection_dim,
                l2_normalize=True,
            )
            for layer in self.exit_layers
        })

        # 3. Các nhánh phân loại thoát sớm (Exit Heads cho Cross-Entropy)
        self.exit_heads = nn.ModuleDict({
            str(layer): LayoutLMv3ClassificationHead(backbone_config, pool_feature=False)
            for layer in self.exit_layers
        })


    def freeze_encoder(self) -> None:
        """Đóng băng toàn bộ trọng số của Backbone Encoder."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()


    def unfreeze_encoder(self) -> None:
        """Mở khóa toàn bộ trọng số của Backbone Encoder."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def is_encoder_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.backbone.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_projections: bool = True,
        compute_logits: bool = True,
    ) -> DICOutput:
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
            if layer == 0:
                cls_token = hidden_states[0].mean(dim=1)
            else:
                cls_token = self.backbone.extract_cls_at_layer(hidden_states, layer)
            cls_tokens_dict[layer] = cls_token

            if compute_logits:
                logits_dict[layer] = self.exit_heads[str(layer)](cls_token)

            if compute_projections and projections_dict is not None:
                projections_dict[layer] = self.projection_heads[str(layer)](cls_token)

        final_layer = self.exit_layers[-1]
        final_logits = logits_dict.get(final_layer) if compute_logits else None

        return DICOutput(
            logits_dict=logits_dict,
            final_logits=final_logits,
            projections_dict=projections_dict,
            cls_tokens_dict=cls_tokens_dict,
        )

    @torch.no_grad()
    def forward_early_exit(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        criterion: str = "max_confidence",
        threshold: float = 0.8,
        exit_thresholds: Optional[Dict[int, float]] = None,
        temperatures: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        True Anytime Early-Exit inference (stops Transformer execution as soon as an exit satisfies threshold).
        Intended for batch_size=1 wall-clock latency benchmarking and real-time deployment.
        """
        from model.modules.ee_modules import entropy, max_confidence, top12_margin

        final_layer = self.exit_layers[-1]

        def _exit_fn(layer: int, feat: torch.Tensor) -> Tuple[torch.Tensor, bool]:
            logits = self.exit_heads[str(layer)](feat)
            if temperatures is not None and layer in temperatures:
                logits = logits / float(temperatures[layer])
            if layer == final_layer:
                return logits, True
            tau = float(exit_thresholds[layer]) if (exit_thresholds is not None and layer in exit_thresholds) else float(threshold)
            if criterion == "max_confidence":
                should_stop = bool((max_confidence(logits) >= tau).all())
            elif criterion == "entropy":
                should_stop = bool((entropy(logits) <= tau).all())
            elif criterion == "top12_margin":
                should_stop = bool((top12_margin(logits) >= tau).all())
            else:
                raise ValueError(f"Unsupported criterion: {criterion}")
            return logits, should_stop

        return self.backbone.forward_early_exit(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            exit_layers=self.exit_layers,
            exit_fn=_exit_fn,
        )


if __name__ == "__main__":
    print("=" * 65)
    print("Testing DICWithoutGroupwise (Pure Architecture Verification)...")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    model = DICWithoutGroupwise(num_classes=16, exit_layers=(0, 1, 4, 8, 10, 12)).to(device)
    
    # Dữ liệu giả lập kiểm tra luồng tính toán
    batch_size = 2
    seq_len = 64
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_bbox = torch.randint(0, 1000, (batch_size, seq_len, 4), dtype=torch.long, device=device)
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)

    # 1. Forward khi mở Encoder (phục vụ Stage 1 - tính cả Projections)
    out = model(dummy_input_ids, dummy_bbox, dummy_pixel_values, compute_projections=True)
    print(f"Logits Exits               : {list(out.logits_dict.keys())}")
    print(f"Final Logits Shape (L12)   : {out.final_logits.shape}")
    print(f"Projections Shape (L1)     : {out.projections_dict[1].shape}")

    # 2. Kiểm thử đóng băng Encoder (phục vụ Stage 2)
    model.freeze_encoder()
    print(f"Is Encoder Frozen          : {model.is_encoder_frozen} (Expected: True)")
    assert model.is_encoder_frozen, "Error: Encoder was not frozen!"

    out_frozen = model(dummy_input_ids, dummy_bbox, dummy_pixel_values, compute_projections=False)
    print(f"Forward while frozen OK    : {out_frozen.final_logits.shape}")

    print("\n[PASSED] DICWithoutGroupwise pure architecture verified successfully.")