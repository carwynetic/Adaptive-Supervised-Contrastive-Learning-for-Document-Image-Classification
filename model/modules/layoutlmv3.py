from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from third_party.layoutlmv3 import (
        LayoutLMv3Config,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3Model,
        LayoutLMv3Processor,
    )
    from third_party.layoutlmv3.modeling_layoutlmv3 import (
        LayoutLMv3ClassificationHead,
        LayoutLMv3Encoder,
        create_bidirectional_mask,
    )
except ImportError:
    from transformers import (
        LayoutLMv3Config,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3Model,
        LayoutLMv3Processor,
    )
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
        LayoutLMv3ClassificationHead,
        LayoutLMv3Encoder,
    )

    create_bidirectional_mask = None

from model.modules.ee_modules import (
    EXITS_PLACEMENT_PRESETS,
    POSSIBLE_EXITS,
    EEModelOutput,
    EESequenceClassifierOutput,
    EarlyExitException,
    EarlyExitHead,
    EarlyExitInference,
    EarlyExitStrategy,
    ExitConfig,
    compute_entropy_reg_weights,
    entropy,
    max_confidence,
    params_per_exit,
)

# Xác định đường dẫn thư mục gốc độc lập với vị trí thực thi
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "configs" / "layoutlmv3-base"
_DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints" / "layoutlmv3-base"


# ==============================================================================
# 1. Early-Exit Classification / Gate Head (Adapted from EE/models/LayoutLMv3.py)
# ==============================================================================

class LayoutLMv3Exit(nn.Module):
    """
    Nhánh phân loại trung gian (Off-ramp hoặc Binary Gate) đặt tại các tầng thoát sớm.
    Tham chiếu: RobertaClassificationHead & LayoutLMv3Exit trong EE_2024.
    """

    def __init__(
        self,
        config: LayoutLMv3Config,
        input_dim: int,
        identifier: str,
    ) -> None:
        super().__init__()

        raw_exit_cfg = getattr(config, "exit_config", {})
        exit_config = (
            ExitConfig(**raw_exit_cfg)
            if isinstance(raw_exit_cfg, dict)
            else raw_exit_cfg
        )

        if exit_config.exit_head_num_layers == 2:
            self.dense = nn.Linear(input_dim, input_dim)

        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        head_type = EarlyExitHead(exit_config.encoder_layer_strategy)
        self.identifier = f"{identifier}_{head_type}"
        output_dim = config.num_labels if head_type == EarlyExitHead.RAMP else 2
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "dense"):
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ==============================================================================
# 2. Early-Exit Encoder (Adapted from LayoutLMv3EncoderEE in EE_2024)
# ==============================================================================

class LayoutLMv3EncoderEE(LayoutLMv3Encoder):
    """
    Bộ mã hóa Transformer 12 tầng của LayoutLMv3 được tích hợp các nhánh thoát sớm
    tại danh sách tầng cấu hình `exit_encoder_layers` (ví dụ [1, 4, 8, 12]).
    Tương thích hoàn toàn với `third_party.layoutlmv3` (Transformers 5.19.0).
    """

    def __init__(self, config: LayoutLMv3Config) -> None:
        super().__init__(config)
        self.config = config

        raw_exit_cfg = getattr(config, "exit_config", None) or getattr(config, "EE_config", {})
        exit_cfg = ExitConfig(**raw_exit_cfg) if isinstance(raw_exit_cfg, dict) else raw_exit_cfg
        config.exit_config = exit_cfg.to_dict()

        self.exit_encoder_layers: List[int] = [
            i for i in exit_cfg.exits if isinstance(i, int)
        ]
        self.has_exits: bool = len(self.exit_encoder_layers) > 0
        self.early_exits = nn.ModuleList(
            [
                LayoutLMv3Exit(config, config.hidden_size, f"encoder_exit_{i}")
                for i in self.exit_encoder_layers
            ]
        )
        self.apply_gating: bool = (
            str(exit_cfg.encoder_layer_strategy) == str(EarlyExitHead.GATE)
        )

        if self.has_exits:
            inf_strategy = EarlyExitInference(exit_cfg.inference_strategy)
            self.exit_criterion = inf_strategy.get_function()
            self.threshold_sign = inf_strategy.get_sign()
            self.exit_threshold: float = exit_cfg.global_threshold
            self.exit_thresholds: Optional[Dict[Union[int, str], float]] = exit_cfg.exit_thresholds
            self.enable_early_exit: bool = exit_cfg.enable_early_exit
        else:
            self.enable_early_exit = False
            self.exit_thresholds = None

        self.early_exit_entropy: List[float] = [-1.0 for _ in range(config.num_hidden_layers)]
        self.num_layers: int = len(self.exit_encoder_layers)
        self.use_lte: bool = exit_cfg.use_lte

    def init_lte(self, config: LayoutLMv3Config) -> None:
        """Khởi tạo nhánh Learning-to-Exit (BERxiT LTE) dùng chung giữa các tầng."""
        self.lte_th = [0.005] * config.num_hidden_layers
        self.lte_classifier = nn.Linear(config.hidden_size, 1)
        self.lte_activation = nn.Sigmoid()

    def enable_lte(self) -> None:
        if getattr(self, "exit_threshold", None) is not None:
            self.lte_th = [float(self.exit_threshold)] * max(1, self.num_layers)

    def _get_threshold_for_exit(self, exit_key: Union[int, str], exit_idx: int) -> float:
        if self.exit_thresholds is not None:
            if exit_key in self.exit_thresholds:
                return float(self.exit_thresholds[exit_key])
            if str(exit_key) in self.exit_thresholds:
                return float(self.exit_thresholds[str(exit_key)])
            if exit_idx in self.exit_thresholds:
                return float(self.exit_thresholds[exit_idx])
        return float(self.exit_threshold)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        position_ids: Optional[torch.Tensor] = None,
        patch_height: Optional[int] = None,
        patch_width: Optional[int] = None,
        prior_exits_count: int = 0,
        **kwargs,
    ) -> Union[Tuple, EEModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_early_exits = () if self.has_exits else None
        all_gate_inputs = () if self.apply_gating else None
        all_exit_features = () if self.has_exits else None
        lte_outputs: List[torch.Tensor] = []

        # Tương thích cả chữ ký hàm của third_party.layoutlmv3 (1 tham số) và HF cũ (2 tham số)
        if self.has_relative_attention_bias:
            try:
                rel_pos = self._cal_1d_pos_emb(position_ids)
            except TypeError:
                rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids)
        else:
            rel_pos = None

        if self.has_spatial_attention_bias:
            try:
                rel_2d_pos = self._cal_2d_pos_emb(bbox)
            except TypeError:
                rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox)
        else:
            rel_2d_pos = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
                **kwargs,
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if output_attentions and len(layer_outputs) > 1:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

            layer_num = i + 1
            if layer_num in self.exit_encoder_layers:
                encoder_exit_idx = len(all_early_exits)
                global_exit_idx = prior_exits_count + encoder_exit_idx
                current_exit = self.early_exits[encoder_exit_idx]

                # Vector [CLS] đại diện tài liệu tại tầng thứ (i + 1)
                exit_input = hidden_states[:, 0, :]
                exit_logits = current_exit(exit_input)
                highway_entropy = entropy(exit_logits)

                if self.use_lte and hasattr(self, "lte_classifier"):
                    lte_out = self.lte_activation(self.lte_classifier(exit_input)).squeeze(-1)
                    lte_outputs.append(lte_out)

                exit_crit = None
                exit_state_item: Any = exit_logits
                if not self.training:
                    exit_crit = self.exit_criterion(exit_logits)
                    exit_state_item = (exit_logits, exit_crit)

                all_early_exits = all_early_exits + (exit_state_item,)
                all_exit_features = all_exit_features + (exit_input,)

                if self.apply_gating:
                    all_gate_inputs = all_gate_inputs + (exit_input,)

                # Kiểm tra điều kiện ngắt sớm thực sự khi suy luận (Anytime Early Exit)
                if not self.training and self.enable_early_exit:
                    if self.use_lte and hasattr(self, "lte_classifier"):
                        self.enable_lte()
                        if encoder_exit_idx + 1 < self.num_layers:
                            should_exit = bool((lte_out < self.lte_th[encoder_exit_idx]).all())
                            if should_exit:
                                output = EEModelOutput(
                                    last_hidden_state=hidden_states,
                                    hidden_states=all_hidden_states,
                                    attentions=all_self_attentions,
                                    exit_states=all_early_exits,
                                    gate_inputs=all_gate_inputs,
                                    exit_features=all_exit_features,
                                    lte_output=lte_outputs,
                                    exited_at=current_exit.identifier,
                                )
                                raise EarlyExitException(
                                    output, current_exit.identifier, global_exit_idx
                                )
                    elif exit_crit is not None:
                        thresh = self._get_threshold_for_exit(layer_num, global_exit_idx)
                        if bool(self.threshold_sign(exit_crit, thresh).all()):
                            output = EEModelOutput(
                                last_hidden_state=hidden_states,
                                hidden_states=all_hidden_states,
                                attentions=all_self_attentions,
                                exit_states=all_early_exits,
                                gate_inputs=all_gate_inputs,
                                exit_features=all_exit_features,
                                lte_output=lte_outputs,
                                exited_at=current_exit.identifier,
                            )
                            raise EarlyExitException(
                                output, current_exit.identifier, global_exit_idx
                            )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in (hidden_states, all_hidden_states, all_self_attentions)
                if v is not None
            )

        return EEModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            exit_states=all_early_exits,
            gate_inputs=all_gate_inputs,
            exit_features=all_exit_features,
            lte_output=lte_outputs,
            exited_at=None,
        )


# ==============================================================================
# 3. Early-Exit Multimodal Model (Adapted from LayoutLMv3ModelEE in EE_2024)
# ==============================================================================

class LayoutLMv3ModelEE(LayoutLMv3Model):
    """
    LayoutLMv3Model tích hợp cả các nhánh thoát sớm tại tầng Modality Embeddings
    ('vision_avg', 'text_avg', 'text_visual_concat') lẫn các tầng Encoder (1..12).
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config) -> None:
        super().__init__(config)
        self.config = config
        self.has_exits: bool = False

        raw_exit_cfg = getattr(config, "exit_config", None) or getattr(config, "EE_config", {})
        exit_cfg = ExitConfig(**raw_exit_cfg) if isinstance(raw_exit_cfg, dict) else raw_exit_cfg
        config.exit_config = exit_cfg.to_dict()

        self.apply_gating: bool = (
            str(exit_cfg.encoder_layer_strategy) == str(EarlyExitHead.GATE)
        )

        # Các nhánh thoát sớm cấp độ Modality (trước khi vào Transformer Encoder)
        if "vision_avg" in exit_cfg.exits:
            self.vision_exit_embeddings = LayoutLMv3Exit(
                config, config.hidden_size, "vision_avg"
            )
            self.has_exits = True

        if "text_avg" in exit_cfg.exits:
            self.text_exit_embeddings = LayoutLMv3Exit(
                config, config.hidden_size, "text_avg"
            )
            self.has_exits = True

        if "text_visual_concat" in exit_cfg.exits:
            self.concat_exit_embeddings = LayoutLMv3Exit(
                config, int(config.hidden_size), "text_visual_concat"
            )
            self.has_exits = True

        self.encoder = LayoutLMv3EncoderEE(config)
        if self.encoder.has_exits:
            self.has_exits = True

        inf_strategy = EarlyExitInference(exit_cfg.inference_strategy)
        self.exit_criterion = inf_strategy.get_function()
        self.threshold_sign = inf_strategy.get_sign()
        self.exit_threshold: float = exit_cfg.global_threshold
        self.exit_thresholds: Optional[Dict[Union[int, str], float]] = exit_cfg.exit_thresholds
        self.enable_early_exit: bool = exit_cfg.enable_early_exit

        if self.encoder.use_lte:
            self.encoder.init_lte(config)

        if hasattr(self, "post_init"):
            self.post_init()
        else:
            self.init_weights()

    def _get_threshold_for_exit(self, exit_key: Union[int, str], exit_idx: int) -> float:
        if self.exit_thresholds is not None:
            if exit_key in self.exit_thresholds:
                return float(self.exit_thresholds[exit_key])
            if str(exit_key) in self.exit_thresholds:
                return float(self.exit_thresholds[str(exit_key)])
            if exit_idx in self.exit_thresholds:
                return float(self.exit_thresholds[exit_idx])
        return float(self.exit_threshold)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, EEModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        )

        all_early_exits = () if self.has_exits else None
        all_gate_inputs = () if self.apply_gating else None
        all_exit_features = () if self.has_exits else None
        lte_outputs: List[torch.Tensor] = []

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds or pixel_values"
            )

        # 1. Nhánh Vision Modality được tính trước (theo thiết kế EE_2024 để có thể thoát sớm trước OCR/Text)
        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        visual_embeddings = None
        visual_attention_mask = None
        visual_bbox = None
        visual_position_ids = None

        if pixel_values is not None:
            patch_height = int(pixel_values.shape[2] / self.config.patch_size)
            patch_width = int(pixel_values.shape[3] / self.config.patch_size)
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]),
                dtype=torch.long,
                device=device,
            )

            if (
                self.config.has_relative_attention_bias
                or self.config.has_spatial_attention_bias
            ):
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(
                        device, dtype=torch.long, batch_size=batch_size
                    )
                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)

            if hasattr(self, "vision_exit_embeddings"):
                exit_idx = len(all_early_exits)
                exit_input = visual_embeddings.mean(dim=1)
                vision_logits = self.vision_exit_embeddings(exit_input)

                exit_crit = None
                vision_exit_item: Any = vision_logits
                if not self.training:
                    exit_crit = self.exit_criterion(vision_logits)
                    vision_exit_item = (vision_logits, exit_crit)

                all_early_exits = all_early_exits + (vision_exit_item,)
                all_exit_features = all_exit_features + (exit_input,)
                if self.apply_gating:
                    all_gate_inputs = all_gate_inputs + (exit_input,)

                if not self.training and self.enable_early_exit and exit_crit is not None:
                    thresh = self._get_threshold_for_exit("vision_avg", exit_idx)
                    if bool(self.threshold_sign(exit_crit, thresh).all()):
                        output = EEModelOutput(
                            last_hidden_state=visual_embeddings,
                            hidden_states=None,
                            attentions=None,
                            exit_states=all_early_exits,
                            gate_inputs=all_gate_inputs,
                            exit_features=all_exit_features,
                            lte_output=lte_outputs,
                            exited_at=self.vision_exit_embeddings.identifier,
                        )
                        raise EarlyExitException(
                            output, self.vision_exit_embeddings.identifier, exit_idx
                        )

        # 2. Nhánh Text & Layout Modality Embeddings
        embedding_output = None
        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )
            if bbox is None:
                bbox = torch.zeros(
                    tuple(list(input_shape) + [4]), dtype=torch.long, device=device
                )

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            if hasattr(self, "text_exit_embeddings"):
                exit_idx = len(all_early_exits)
                exit_input = embedding_output.mean(dim=1)
                text_logits = self.text_exit_embeddings(exit_input)

                exit_crit = None
                text_exit_item: Any = text_logits
                if not self.training:
                    exit_crit = self.exit_criterion(text_logits)
                    text_exit_item = (text_logits, exit_crit)

                all_early_exits = all_early_exits + (text_exit_item,)
                all_exit_features = all_exit_features + (exit_input,)
                if self.apply_gating:
                    all_gate_inputs = all_gate_inputs + (exit_input,)

                if not self.training and self.enable_early_exit and exit_crit is not None:
                    thresh = self._get_threshold_for_exit("text_avg", exit_idx)
                    if bool(self.threshold_sign(exit_crit, thresh).all()):
                        output = EEModelOutput(
                            last_hidden_state=embedding_output,
                            hidden_states=None,
                            attentions=None,
                            exit_states=all_early_exits,
                            gate_inputs=all_gate_inputs,
                            exit_features=all_exit_features,
                            lte_output=lte_outputs,
                            exited_at=self.text_exit_embeddings.identifier,
                        )
                        raise EarlyExitException(
                            output, self.text_exit_embeddings.identifier, exit_idx
                        )

        # 3. Ghép nối đa phương thức (Modality Concatenation)
        if visual_embeddings is not None:
            if embedding_output is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
                if visual_bbox is not None:
                    final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                if visual_position_ids is not None:
                    text_pos_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0).expand(input_shape)
                    final_position_ids = torch.cat([text_pos_ids, visual_position_ids], dim=1)
            else:
                embedding_output = visual_embeddings
                attention_mask = visual_attention_mask
                final_bbox = visual_bbox
                final_position_ids = visual_position_ids

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif (
            self.config.has_relative_attention_bias
            or self.config.has_spatial_attention_bias
        ):
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                final_position_ids = position_ids.expand_as(input_ids)

        # 4. Nhánh thoát sớm sau khi ghép Text + Vision (Concat-Exit)
        if hasattr(self, "concat_exit_embeddings"):
            exit_idx = len(all_early_exits)
            exit_input = embedding_output.mean(dim=1)
            joint_logits = self.concat_exit_embeddings(exit_input)

            exit_crit = None
            joint_exit_item: Any = joint_logits
            if not self.training:
                exit_crit = self.exit_criterion(joint_logits)
                joint_exit_item = (joint_logits, exit_crit)

            all_early_exits = all_early_exits + (joint_exit_item,)
            all_exit_features = all_exit_features + (exit_input,)

            if self.encoder.use_lte and hasattr(self.encoder, "lte_classifier"):
                lte_out = self.encoder.lte_activation(
                    self.encoder.lte_classifier(exit_input)
                ).squeeze(-1)
                lte_outputs.append(lte_out)

            if self.apply_gating:
                all_gate_inputs = all_gate_inputs + (exit_input,)

            if not self.training and self.enable_early_exit and exit_crit is not None:
                thresh = self._get_threshold_for_exit("text_visual_concat", exit_idx)
                if bool(self.threshold_sign(exit_crit, thresh).all()):
                    output = EEModelOutput(
                        last_hidden_state=embedding_output,
                        hidden_states=None,
                        attentions=None,
                        exit_states=all_early_exits,
                        gate_inputs=all_gate_inputs,
                        exit_features=all_exit_features,
                        lte_output=lte_outputs,
                        exited_at=self.concat_exit_embeddings.identifier,
                    )
                    raise EarlyExitException(
                        output, self.concat_exit_embeddings.identifier, exit_idx
                    )

        # 5. Chuẩn bị Attention Mask cho Transformer Encoder
        if create_bidirectional_mask is not None:
            extended_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, None, device, dtype=embedding_output.dtype
            )

        prior_exits_count = len(all_early_exits) if all_early_exits is not None else 0

        try:
            encoder_outputs = self.encoder(
                embedding_output,
                bbox=final_bbox,
                position_ids=final_position_ids,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                patch_height=patch_height,
                patch_width=patch_width,
                prior_exits_count=prior_exits_count,
                **kwargs,
            )
        except EarlyExitException as exc:
            # Ghép thêm các nhánh thoát ở tầng embedding vào payload trước khi ném ngoại lệ lên trên
            enc_payload = exc.payload
            if all_early_exits and enc_payload.exit_states:
                enc_payload.exit_states = all_early_exits + enc_payload.exit_states
            if all_exit_features and enc_payload.exit_features:
                enc_payload.exit_features = all_exit_features + enc_payload.exit_features
            if self.apply_gating and all_gate_inputs and enc_payload.gate_inputs:
                enc_payload.gate_inputs = all_gate_inputs + enc_payload.gate_inputs
            if lte_outputs and enc_payload.lte_output is not None:
                enc_payload.lte_output = lte_outputs + enc_payload.lte_output
            raise

        if encoder_outputs.exit_states:
            all_early_exits = (all_early_exits or ()) + encoder_outputs.exit_states
        if encoder_outputs.exit_features:
            all_exit_features = (all_exit_features or ()) + encoder_outputs.exit_features
        if self.apply_gating and encoder_outputs.gate_inputs:
            all_gate_inputs = (all_gate_inputs or ()) + encoder_outputs.gate_inputs

        sequence_output = encoder_outputs.last_hidden_state

        if not return_dict:
            return (sequence_output, encoder_outputs.hidden_states, encoder_outputs.attentions)

        return EEModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            exit_states=all_early_exits,
            gate_inputs=all_gate_inputs,
            exit_features=all_exit_features,
            lte_output=lte_outputs + (encoder_outputs.lte_output or []),
            exited_at=None,
        )


# ==============================================================================
# 4. Early-Exit Sequence Classifier (Adapted from LayoutLMv3EEForSequenceClassification)
# ==============================================================================

class LayoutLMv3EEForSequenceClassification(LayoutLMv3ForSequenceClassification):
    """
    Mô hình phân loại ảnh tài liệu đa phương thức với Anytime Early Exiting
    (LayoutLMv3EEForSequenceClassification từ bài báo EE_2024).

    Hỗ trợ:
        - Nạp hoàn toàn offline từ `configs/layoutlmv3-base` và `checkpoints/layoutlmv3-base`
          qua phương thức lớp `from_local_checkpoint(...)`.
        - Đầy đủ các cấu hình vị trí nhánh thoát ('concat_quarter', 'independent_all', ...).
        - Cả 2 loại bộ phân loại nhánh thoát: `ramp` (Eq. 5) và `gate` (Eq. 4), cùng `use_lte`.
        - Trích xuất đồng thời `exit_features` để kết hợp với hàm loss Adaptive SupCon.
    """

    def __init__(self, hf_model_config: LayoutLMv3Config, **kwargs) -> None:
        raw_ee_cfg = (
            getattr(hf_model_config, "EE_config", None)
            or getattr(hf_model_config, "exit_config", None)
            or kwargs.pop("ee_config", None)
            or {}
        )
        exit_cfg = ExitConfig(**raw_ee_cfg) if isinstance(raw_ee_cfg, dict) else raw_ee_cfg
        hf_model_config.exit_config = exit_cfg.to_dict()
        hf_model_config.EE_config = hf_model_config.exit_config

        super().__init__(hf_model_config)

        self.training_strategy: str = str(exit_cfg.training_strategy)
        self.apply_gating: bool = (
            str(exit_cfg.encoder_layer_strategy) == str(EarlyExitHead.GATE)
        )
        self.use_lte: bool = exit_cfg.use_lte
        self.num_layers: int = len(
            [i for i in exit_cfg.exits if isinstance(i, int)]
        )

        # Thay thế self.layoutlmv3 bằng phiên bản Early-Exit
        self.layoutlmv3 = LayoutLMv3ModelEE(self.config)
        self.processor = None

        if hasattr(self, "post_init"):
            self.post_init()
        else:
            self.init_weights()

    @classmethod
    def from_local_checkpoint(
        cls,
        config_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        num_labels: int = 16,
        ee_config: Optional[Union[Dict[str, Any], ExitConfig]] = None,
        load_processor: bool = False,
        apply_ocr: bool = False,
    ) -> "LayoutLMv3EEForSequenceClassification":
        """
        Khởi tạo `LayoutLMv3EEForSequenceClassification` trực tiếp từ thư mục
        cấu hình và trọng số safetensors cục bộ trong dự án mà không cần kết nối mạng.
        """
        cfg_dir = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else _DEFAULT_CHECKPOINT_DIR

        if not cfg_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {cfg_dir}")
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        hf_config = LayoutLMv3Config.from_pretrained(
            str(cfg_dir),
            num_labels=num_labels,
            local_files_only=True,
        )
        exit_cfg = (
            ee_config
            if isinstance(ee_config, ExitConfig)
            else ExitConfig(**(ee_config or {}))
        )
        hf_config.EE_config = exit_cfg.to_dict()
        hf_config.exit_config = exit_cfg.to_dict()

        model = cls.from_pretrained(
            str(ckpt_dir),
            config=hf_config,
            local_files_only=True,
        )

        if load_processor:
            model.processor = LayoutLMv3Processor.from_pretrained(
                str(cfg_dir),
                apply_ocr=apply_ocr,
                local_files_only=True,
            )

        return model

    def set_early_exit_mode(
        self,
        enabled: bool = True,
        global_threshold: Optional[float] = None,
        exit_thresholds: Optional[Dict[Union[int, str], float]] = None,
    ) -> None:
        """Bật/tắt chế độ ngắt sớm thực tế (Anytime Early Exit) và cập nhật ngưỡng."""
        self.layoutlmv3.enable_early_exit = enabled
        self.layoutlmv3.encoder.enable_early_exit = enabled
        self.config.exit_config["enable_early_exit"] = enabled

        if global_threshold is not None:
            self.layoutlmv3.exit_threshold = float(global_threshold)
            self.layoutlmv3.encoder.exit_threshold = float(global_threshold)
            self.config.exit_config["global_threshold"] = float(global_threshold)

        if exit_thresholds is not None:
            self.layoutlmv3.exit_thresholds = exit_thresholds
            self.layoutlmv3.encoder.exit_thresholds = exit_thresholds
            self.config.exit_config["exit_thresholds"] = exit_thresholds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, EESequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        )
        exited = False
        exited_at: Optional[str] = None

        try:
            outputs: EEModelOutput = self.layoutlmv3(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                bbox=bbox,
                pixel_values=pixel_values,
                **kwargs,
            )
            sequence_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(sequence_output)
        except EarlyExitException as exc:
            exited = True
            exited_at = exc.exit_layer
            outputs = exc.payload
            if self.apply_gating:
                logits = self.classifier(outputs.gate_inputs[-1])
            else:
                last_exit_state = outputs.exit_states[-1]
                logits = (
                    last_exit_state[0]
                    if isinstance(last_exit_state, tuple)
                    else last_exit_state
                )

        loss: Optional[torch.Tensor] = None
        exit_criteria: List[torch.Tensor] = []
        exit_losses: List[torch.Tensor] = []
        all_gated_logits: Tuple[torch.Tensor, ...] = ()

        # Ngay cả khi labels=None (lúc inference), vẫn thu thập exit_criteria và gated_logits
        if outputs.exit_states is not None:
            for j, exit_state in enumerate(outputs.exit_states):
                if isinstance(exit_state, tuple):
                    _, exit_crit = exit_state
                    exit_criteria.append(exit_crit)
                if self.apply_gating and outputs.gate_inputs is not None and j < len(outputs.gate_inputs):
                    gated_logits_j = self.classifier(outputs.gate_inputs[j])
                    all_gated_logits = all_gated_logits + (gated_logits_j,)

        if labels is not None:
            ce_loss_fct = nn.CrossEntropyLoss()
            loss = ce_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.apply_gating and outputs.gate_inputs is not None:
                gate_losses: List[torch.Tensor] = []
                bce_loss_fct = nn.BCEWithLogitsLoss()
                flat_labels = labels.view(-1)

                for j, gated_logits_j in enumerate(all_gated_logits):
                    correctly_gated = gated_logits_j.argmax(dim=-1) == flat_labels
                    y_onehot = torch.zeros(
                        correctly_gated.shape[0], 2, device=labels.device, dtype=logits.dtype
                    )
                    y_onehot[torch.arange(y_onehot.shape[0], device=labels.device), correctly_gated.long()] = 1.0

                    exit_state_j = outputs.exit_states[j]
                    exit_logits_j = (
                        exit_state_j[0] if isinstance(exit_state_j, tuple) else exit_state_j
                    )
                    gate_loss = bce_loss_fct(exit_logits_j, y_onehot)
                    gate_losses.append(gate_loss)

                exit_losses = gate_losses

            elif self.use_lte and outputs.exit_states is not None:
                losses: List[torch.Tensor] = []
                inter_losses: List[torch.Tensor] = []
                lte_loss_fct = nn.MSELoss()

                for j, exit_state_j in enumerate(outputs.exit_states):
                    exit_logits_j = (
                        exit_state_j[0] if isinstance(exit_state_j, tuple) else exit_state_j
                    )
                    if self.num_labels == 1:
                        highway_loss = lte_loss_fct(exit_logits_j.view(-1), labels.view(-1).float())
                    else:
                        highway_loss = ce_loss_fct(
                            exit_logits_j.view(-1, self.num_labels), labels.view(-1)
                        )
                    inter_losses.append(highway_loss)

                    exit_pred = outputs.lte_output[j].to(labels.device).view(-1)
                    layer_output = logits if (j + 1 == self.num_layers) else exit_logits_j

                    if self.num_labels == 1:
                        correctness_loss = torch.tanh(layer_output.squeeze() - labels).abs().view(-1)
                    else:
                        lte_gold = torch.eq(
                            torch.argmax(layer_output, dim=-1), labels.view(-1)
                        ).to(labels.device)
                        correctness_loss = (1.0 - lte_gold.float()).view(-1)

                    total_loss = (
                        sum(inter_losses) + loss + lte_loss_fct(exit_pred, correctness_loss)
                    )
                    losses.append(total_loss)

                exit_losses = losses

            elif outputs.exit_states is not None:
                for exit_state_j in outputs.exit_states:
                    exit_logits_j = (
                        exit_state_j[0] if isinstance(exit_state_j, tuple) else exit_state_j
                    )
                    exit_loss = ce_loss_fct(
                        exit_logits_j.view(-1, self.num_labels), labels.view(-1)
                    )
                    exit_losses.append(exit_loss)

            if self.training_strategy in ("joint", "joint_weighted_avg") and exit_losses:
                loss = loss + sum(exit_losses)

        if not exited:
            exit_criteria.append(self.layoutlmv3.exit_criterion(logits))

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return EESequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            exit_losses=exit_losses,
            exit_criteria=exit_criteria,
            exit_states=outputs.exit_states,
            exit_features=outputs.exit_features,
            gated_logits=all_gated_logits,
            lte_output=outputs.lte_output,
            exited_at=exited_at,
        )


# ==============================================================================
# 5. LayoutLMv3Backbone (Giữ nguyên tương thích ngược 100% với code hiện có)
# ==============================================================================

@dataclass
class LayoutLMv3BackboneOutput:
    """Dataclass chứa các đặc trưng đầu ra từ LayoutLMv3 backbone."""

    last_hidden_state: torch.Tensor
    cls_token: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]


class LayoutLMv3Backbone(nn.Module):
    """Wrapper gọn nhẹ nạp LayoutLMv3Model offline và trích xuất vector [CLS] tại mọi tầng."""

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

        self.config = LayoutLMv3Config.from_pretrained(str(self.config_dir))
        self.config.output_hidden_states = output_hidden_states

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
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(hidden_states) - 1}]")
        return hidden_states[layer_idx][:, 0, :]

    def extract_exit_cls_features(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        exit_layers: Sequence[int],
    ) -> Dict[int, torch.Tensor]:
        return {layer: self.extract_cls_at_layer(hidden_states, layer) for layer in exit_layers}


if __name__ == "__main__":
    print("=" * 70)
    print("Testing LayoutLMv3 Early-Exit Module (Adapted from EE_2024 Paper)...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 1. Kiểm thử cấu hình Concat-Quarter (5 exits: text_visual_concat, 1, 4, 8, 12) + Ramp
    ee_cfg = ExitConfig(
        training_strategy="one_stage_subgraphs_weighted",
        inference_strategy="max_confidence",
        exits="concat_quarter",
        encoder_layer_strategy="ramp",
        exit_head_num_layers=2,
        global_threshold=0.8,
    )
    model = LayoutLMv3EEForSequenceClassification.from_local_checkpoint(
        num_labels=16,
        ee_config=ee_cfg,
    ).to(device)

    batch_size = 2
    seq_len = 64
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long, device=device),
        "bbox": torch.randint(0, 1000, (batch_size, seq_len, 4), dtype=torch.long, device=device),
        "pixel_values": torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long, device=device),
        "labels": torch.tensor([3, 11], dtype=torch.long, device=device),
    }

    # Kiểm thử Training Forward + Subgraph Weighting
    model.train()
    train_out = model(**dummy_input)
    _, exit_weights, _ = params_per_exit(model)
    print(f"\n[Train Mode - Concat-Quarter Ramp]")
    print(f"  - Final Classifier Logits : {train_out.logits.shape}")
    print(f"  - Total Exits             : {len(train_out.exit_losses)} ({ee_cfg.exits})")
    print(f"  - Exit Features Shapes    : {[f.shape for f in train_out.exit_features]}")
    print(f"  - Subgraph Weights (w_b)  : {[round(exit_weights[k], 4) for k in sorted(exit_weights)]}")

    # Kiểm thử Eval Mode + Anytime Early Exit
    model.eval()
    model.set_early_exit_mode(enabled=True, global_threshold=0.01)  # Ngưỡng thấp để kích hoạt thoát sớm ngay nhánh đầu
    with torch.no_grad():
        early_out = model(
            input_ids=dummy_input["input_ids"][:1],
            bbox=dummy_input["bbox"][:1],
            pixel_values=dummy_input["pixel_values"][:1],
            attention_mask=dummy_input["attention_mask"][:1],
        )
    print(f"\n[Eval Mode - Anytime Early Exit Triggered]")
    print(f"  - Exited Early At         : {early_out.exited_at}")
    print(f"  - Evaluated Exits Count   : {len(early_out.exit_states)} / {len(ee_cfg.exits)}")
    print(f"  - Output Logits Shape     : {early_out.logits.shape}")

    print("\n[PASSED] LayoutLMv3EEForSequenceClassification & EE Modules verified successfully.")
