"""
LayoutLMv3 Backbone Module for Multi-Modal Document Representation.

This module wraps the LayoutLMv3 transformer backbone, supporting offline checkpoint
loading, intermediate hidden states extraction across all 12 encoder layers, and token
pooling for early-exit multi-modal document classification.

Reference:
    - Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking",
      ACM MM 2022. https://arxiv.org/abs/2204.08387
    - Hamed et al., "Multi-Modal Early Exit for Document Classification", ICDAR 2024.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from third_party.layoutlmv3 import LayoutLMv3Config, LayoutLMv3Model
except ImportError:
    from transformers import LayoutLMv3Config, LayoutLMv3Model

# Xác định đường dẫn thư mục gốc độc lập với vị trí thực thi
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "configs" / "layoutlmv3-base"
_DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints" / "layoutlmv3-base"


@dataclass
class LayoutLMv3BackboneOutput:
    """Dataclass chứa các đặc trưng đầu ra từ LayoutLMv3 backbone."""
    last_hidden_state: torch.Tensor
    cls_token: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]


class LayoutLMv3Backbone(nn.Module):
    """
    LayoutLMv3 Backbone wrapper phục vụ trích xuất đặc trưng đa phương thái (Text + Layout + Vision)
    và hỗ trợ lấy hidden states tại tất cả các tầng cho mô hình Early Exit.
    """

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        output_hidden_states: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.config_dir = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else _DEFAULT_CHECKPOINT_DIR

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        # Khởi tạo Config với output_hidden_states=True cho Early Exit
        self.config = LayoutLMv3Config.from_pretrained(str(self.config_dir))
        self.config.output_hidden_states = output_hidden_states

        # Nạp weights từ local safetensors
        self.model, loading_info = LayoutLMv3Model.from_pretrained(
            str(self.checkpoint_dir),
            config=self.config,
            local_files_only=True,
            output_loading_info=True,
        )

        missing = loading_info.get("missing_keys", [])
        unexpected = loading_info.get("unexpected_keys", [])
        if missing or unexpected:
            raise RuntimeError(
                f"Weight mismatch loading LayoutLMv3: missing={missing}, unexpected={unexpected}"
            )

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.hidden_size: int = self.config.hidden_size
        self.num_layers: int = self.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> LayoutLMv3BackboneOutput:
        """
        Forward pass qua LayoutLMv3 encoder.

        Args:
            input_ids: (B, L) LongTensor chứa token IDs.
            bbox: (B, L, 4) LongTensor chứa bounding boxes chuẩn hóa [0, 1000].
            pixel_values: (B, 3, 224, 224) FloatTensor chứa ảnh tài liệu đã chuẩn hóa.
            attention_mask: (B, L) Float/LongTensor mask padding cho text tokens.

        Returns:
            LayoutLMv3BackboneOutput với:
                - last_hidden_state: (B, total_seq_len, hidden_size)
                - cls_token: (B, hidden_size) đặc trưng đại diện toàn bộ văn bản tại tầng cuối
                - hidden_states: Tuple 13 tensors (tầng 0 là embedding, 1..12 là các encoder layers)
        """
        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        hidden_states = outputs.hidden_states

        return LayoutLMv3BackboneOutput(
            last_hidden_state=last_hidden_state,
            cls_token=cls_token,
            hidden_states=hidden_states,
        )

    @staticmethod
    def extract_cls_at_layer(hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        """
        Trích xuất vector [CLS] tại tầng chỉ định.
        
        Args:
            hidden_states: Tuple các tensor từ self.forward().hidden_states.
            layer_idx: Vị trí tầng (0: embedding, 1..12: encoder layers).
            
        Returns:
            (B, hidden_size) Tensor biểu diễn văn bản tại tầng đó.
        """
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(hidden_states) - 1}]")
        return hidden_states[layer_idx][:, 0, :]

    def extract_exit_cls_features(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        exit_layers: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """
        Trích xuất đồng thời các vector [CLS] tại danh sách các tầng Early Exit.

        Args:
            hidden_states: Tuple các tensor ẩn.
            exit_layers: Danh sách chỉ số tầng (ví dụ [1, 4, 8, 10, 12]).

        Returns:
            Dictionary mapping layer_idx -> Tensor [B, hidden_size].
        """
        return {layer: self.extract_cls_at_layer(hidden_states, layer) for layer in exit_layers}


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LayoutLMv3Backbone Module...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # Khởi tạo backbone
    backbone = LayoutLMv3Backbone().to(device)
    backbone.eval()
    print(f"Loaded successfully: hidden_size={backbone.hidden_size}, num_layers={backbone.num_layers}")

    # Tạo batch dữ liệu giả lập (batch_size=2, sequence_length=128)
    batch_size = 2
    seq_len = 128
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_bbox = torch.randint(0, 1000, (batch_size, seq_len, 4), dtype=torch.long, device=device)
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    print("\nRunning forward pass with dummy inputs...")
    with torch.no_grad():
        out = backbone(
            input_ids=dummy_input_ids,
            bbox=dummy_bbox,
            pixel_values=dummy_pixel_values,
            attention_mask=dummy_mask,
        )

    print(f"Last hidden state shape : {out.last_hidden_state.shape}")
    print(f"Final CLS token shape   : {out.cls_token.shape}")
    print(f"Total hidden states     : {len(out.hidden_states)} layers (embedding + 12 layers)")

    # Kiểm tra trích xuất đặc trưng tại các tầng Early Exit: 1, 4, 8, 10, 12
    exit_layers = [1, 4, 8, 10, 12]
    exit_features = backbone.extract_exit_cls_features(out.hidden_states, exit_layers)

    print("\nEarly Exit CLS Tensors extracted:")
    for layer, feat in exit_features.items():
        print(f"  - Exit Layer {layer:2d}: shape = {feat.shape}, norm = {feat.norm(dim=-1).mean().item():.4f}")

    print("\n[PASSED] LayoutLMv3Backbone Module verified successfully.")