"""
Two-Stage Trainer for Document Image Classification (DIC).
Hỗ trợ 2 cơ chế điều phối huấn luyện (train_type):
    1. 'without_groupwise': Tính tổng loss toàn mạng và backward 1 luồng duy nhất.
    2. 'groupwise': Phân cụm các tầng Exit (Group 1: [1, 4], Group 2: [8, 10, 12]) 
                    và backward độc lập để triệt tiêu xung đột gradient.

Chiến lược 2 giai đoạn (Two-Stage Training):
    - Stage 1: Huấn luyện biểu diễn tương phản đa tầng (MultiExitSupConLoss) 
               với LARSWrapper(AdamW), Warmup Cosine LR và Soft Targets (ConOSR).
    - Stage 2: Đóng băng Encoder (Linear Probe), chỉ huấn luyện các nhánh phân loại 
               thoát sớm (MultiExitCELoss) bằng AdamW.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_processing import get_smooth_labels
from utils.lars_wrapper import LARSWrapper
from utils.losses import MultiExitCELoss, MultiExitSupConLoss
from utils.schedulers import LinearWarmupCosineAnnealingLR


class DICTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_type: str = "without_groupwise",
        exit_layers: Sequence[int] = (1, 4, 8, 10, 12),
        groups: Optional[Dict[str, Sequence[int]]] = None,
        num_classes: int = 16,
        smoothing_coeff: float = 0.1,
        temperature: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        gradient_accumulation_steps: int = 4,
        use_amp: bool = True,
    ) -> None:
        """
        Args:
            model: Mô hình DIC (DICWithoutGroupwise hoặc DIC).
            train_type: 'without_groupwise' hoặc 'groupwise'.
            exit_layers: Danh sách tầng exit [1, 4, 8, 10, 12].
            groups: Cấu hình phân nhóm cho groupwise (mặc định: g1=[1,4], g2=[8,10,12]).
            num_classes: Số lượng lớp (16 cho RVL-CDIP).
            smoothing_coeff: Hệ số làm mịn nhãn mềm ConOSR (epsilon=0.1).
            temperature: Tham số nhiệt độ tau cho SupCon-ST.
            device: 'cuda' hoặc 'cpu'.
            gradient_accumulation_steps: Số bước tích lũy gradient (mô phỏng large batch size).
            use_amp: Bật/tắt Mixed Precision (float16/bfloat16).
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_type = train_type.lower()
        self.exit_layers = tuple(sorted(exit_layers))
        self.num_classes = num_classes
        self.smoothing_coeff = smoothing_coeff
        self.grad_accum_steps = max(1, gradient_accumulation_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Định nghĩa các nhóm tầng cho groupwise
        if groups is None:
            self.groups = {
                "group1": tuple(l for l in self.exit_layers if l <= 4),
                "group2": tuple(l for l in self.exit_layers if l > 4),
            }
        else:
            self.groups = {k: tuple(v) for k, v in groups.items()}

        # Khởi tạo 2 hàm mất mát đa tầng
        self.supcon_criterion = MultiExitSupConLoss(exit_layers=self.exit_layers, temperature=temperature).to(self.device)
        self.ce_criterion = MultiExitCELoss(exit_layers=self.exit_layers).to(self.device)

        print(f"[+] Khởi tạo DICTrainer trên thiết bị: {self.device}")
        print(f"[+] Kiểu huấn luyện (train_type): {self.train_type}")
        print(f"[+] Phân nhóm tầng (groups): {self.groups}")
        print(f"[+] Tích lũy gradient: {self.grad_accum_steps} steps | AMP: {self.use_amp}")

    # ==========================================================================
    # STAGE 1: HUẤN LUYỆN TƯƠNG PHẢN (SUPCON-ST)
    # ==========================================================================

    def train_stage1(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 2,
        use_lars: bool = True,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Giai đoạn 1: Huấn luyện biểu diễn tương phản đa tầng (SupCon-ST).
        """
        print("\n" + "=" * 65)
        print(f"BẮT ĐẦU STAGE 1: TƯƠNG PHẢN ĐA TẦNG (SUPCON-ST) - {epochs} EPOCHS")
        print("=" * 65)

        self.model.unfreeze_encoder()
        self.model.train()

        # 1. Optimizer cho Stage 1: Cập nhật Backbone + 5 Projection Heads
        stage1_params = [p for p in self.model.parameters() if p.requires_grad]
        base_optimizer = torch.optim.AdamW(stage1_params, lr=lr, weight_decay=weight_decay)

        optimizer = LARSWrapper(base_optimizer, eta=0.02, clip=True) if use_lars else base_optimizer
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)

        for epoch in range(1, epochs + 1):
            total_epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, start=1):
                # 2 views từ TwoCropTransform
                v1, v2 = batch["view1"], batch["view2"]
                labels = batch["labels"].to(self.device)

                # Sinh nhãn mềm ConOSR Soft Targets
                soft_targets = get_smooth_labels(
                    labels, classid_list=self.num_classes, smoothing_coeff=self.smoothing_coeff
                ).to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    # Forward view 1
                    out_v1 = self._forward_model_batch(v1, compute_projections=True)
                    # Forward view 2
                    out_v2 = self._forward_model_batch(v2, compute_projections=True)

                    # Tính SupCon-ST loss tại từng tầng exit
                    layer_losses = self.supcon_criterion(
                        (out_v1.projections_dict, out_v2.projections_dict), soft_targets
                    )

                # Điều phối Backward pass theo train_type
                if self.train_type == "without_groupwise":
                    total_loss = self.supcon_criterion.compute_total_loss(layer_losses)
                    loss_to_backward = total_loss / self.grad_accum_steps
                    self.scaler.scale(loss_to_backward).backward()
                    step_loss_val = total_loss.item()
                else:
                    # Group-wise backward
                    group_loss_vals = []
                    for i, (g_name, g_layers) in enumerate(self.groups.items()):
                        g_loss = sum(layer_losses[l] for l in g_layers)
                        loss_to_backward = g_loss / self.grad_accum_steps
                        is_last_group = (i == len(self.groups) - 1)
                        self.scaler.scale(loss_to_backward).backward(retain_graph=not is_last_group)
                        group_loss_vals.append(g_loss.item())
                    step_loss_val = sum(group_loss_vals)

                total_epoch_loss += step_loss_val

                # Cập nhật trọng số theo chu kỳ tích lũy gradient
                if step % self.grad_accum_steps == 0 or step == len(train_loader):
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

            avg_epoch_loss = total_epoch_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Stage 1 | Epoch [{epoch:2d}/{epochs:2d}] - SupCon Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.6f}")
            scheduler.step()

        print("[+] Hoàn tất Stage 1!")
        if save_dir:
            self._save_checkpoint(Path(save_dir) / "stage1_pretrained.pt")

    # ==========================================================================
    # STAGE 2: HUẤN LUYỆN BỘ PHÂN LOẠI (LINEAR PROBE CE)
    # ==========================================================================

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Giai đoạn 2: Đóng băng Backbone Encoder, chỉ huấn luyện 5 nhánh phân loại Exit.
        """
        print("\n" + "=" * 65)
        print(f"BẮT ĐẦU STAGE 2: HUẤN LUYỆN BỘ PHÂN LOẠI (CE LOSS) - {epochs} EPOCHS")
        print("=" * 65)

        # 1. Đóng băng Backbone Encoder
        self.model.freeze_encoder()
        self.model.train()

        # 2. Optimizer chỉ tối ưu các tham số của exit_heads
        head_params = [p for p in self.model.exit_heads.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            total_epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, start=1):
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    # Forward single view (compute_projections=False để tiết kiệm VRAM)
                    out = self._forward_model_batch(batch, compute_projections=False)
                    # Tính CE loss tại từng tầng exit
                    layer_losses = self.ce_criterion(out.logits_dict, labels)

                # Điều phối Backward pass theo train_type
                if self.train_type == "without_groupwise":
                    total_loss = self.ce_criterion.compute_total_loss(layer_losses)
                    loss_to_backward = total_loss / self.grad_accum_steps
                    self.scaler.scale(loss_to_backward).backward()
                    step_loss_val = total_loss.item()
                else:
                    # Group-wise backward
                    group_loss_vals = []
                    for i, (g_name, g_layers) in enumerate(self.groups.items()):
                        g_loss = sum(layer_losses[l] for l in g_layers)
                        loss_to_backward = g_loss / self.grad_accum_steps
                        is_last_group = (i == len(self.groups) - 1)
                        self.scaler.scale(loss_to_backward).backward(retain_graph=not is_last_group)
                        group_loss_vals.append(g_loss.item())
                    step_loss_val = sum(group_loss_vals)

                total_epoch_loss += step_loss_val

                if step % self.grad_accum_steps == 0 or step == len(train_loader):
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

            avg_epoch_loss = total_epoch_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Stage 2 | Epoch [{epoch:2d}/{epochs:2d}] - CE Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.6f}")
            scheduler.step()

            # Đánh giá validation accuracy tại từng tầng Exit
            if val_loader:
                val_accs = self.evaluate(val_loader)
                final_acc = val_accs[self.exit_layers[-1]]
                if final_acc > best_acc and save_dir:
                    best_acc = final_acc
                    self._save_checkpoint(Path(save_dir) / "best_model.pt")

        print("[+] Hoàn tất Stage 2!")

    # ==========================================================================
    # ĐÁNH GIÁ (EVALUATION)
    # ==========================================================================

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[int, float]:
        """
        Đánh giá độ chính xác (Accuracy Top-1) tại từng tầng Exit [1, 4, 8, 10, 12].
        """
        self.model.eval()
        correct_counts = {l: 0 for l in self.exit_layers}
        total_samples = 0

        for batch in val_loader:
            labels = batch["labels"].to(self.device)
            batch_size = labels.size(0)
            total_samples += batch_size

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self._forward_model_batch(batch, compute_projections=False)

            for l in self.exit_layers:
                preds = torch.argmax(out.logits_dict[l], dim=-1)
                correct_counts[l] += (preds == labels).sum().item()

        accuracies = {l: (correct_counts[l] / total_samples) * 100.0 for l in self.exit_layers}
        acc_str = " | ".join([f"L{l:2d}: {accuracies[l]:.2f}%" for l in self.exit_layers])
        print(f"  [Validation Accuracy] -> {acc_str}")

        self.model.train()
        return accuracies

    # ==========================================================================
    # HÀM TIỆN ÍCH NỘI BỘ
    # ==========================================================================

    def _forward_model_batch(self, batch_data: Dict[str, Any], compute_projections: bool = False):
        """
        Trích xuất tensor và forward qua mô hình. Tự động hỗ trợ cả dữ liệu có OCR lẫn ảnh thuần.
        """
        pixel_values = batch_data["pixel_values"].to(self.device)
        b, _, _, _ = pixel_values.shape

        # Nếu chưa chạy pipeline OCR, tạo input_ids và bbox giả lập định dạng LayoutLMv3
        if "input_ids" in batch_data:
            input_ids = batch_data["input_ids"].to(self.device)
            bbox = batch_data["bbox"].to(self.device)
            attention_mask = batch_data.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            # Fallback tensor khi chỉ có ảnh (chuẩn 1 token [CLS])
            input_ids = torch.zeros((b, 1), dtype=torch.long, device=self.device)
            bbox = torch.zeros((b, 1, 4), dtype=torch.long, device=self.device)
            attention_mask = torch.ones((b, 1), dtype=torch.long, device=self.device)

        return self.model(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            compute_projections=compute_projections,
        )

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "train_type": self.train_type,
            "exit_layers": self.exit_layers,
        }, str(path))
        print(f"  [+] Đã lưu checkpoint tại: {path.name}")


# ==============================================================================
# Unit Test tự kiểm chứng Trainer
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Testing DICTrainer Pipeline (train_test/trainer.py)...")
    print("=" * 65)

    from model.DIC_without_groupwise import DICWithoutGroupwise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 1. Khởi tạo mô hình
    exit_layers = (1, 4, 8, 10, 12)
    model = DICWithoutGroupwise(num_classes=16, exit_layers=exit_layers).to(device)

    # 2. Khởi tạo Trainer với mode 'without_groupwise'
    trainer = DICTrainer(
        model=model,
        train_type="without_groupwise",
        exit_layers=exit_layers,
        gradient_accumulation_steps=2,
    )

    # 3. Tạo dummy dataloader 2 views giả lập cho Stage 1
    dummy_contrastive_loader = [
        {
            "view1": {
                "pixel_values": torch.randn(2, 3, 224, 224),
                "input_ids": torch.zeros((2, 16), dtype=torch.long),
                "bbox": torch.zeros((2, 16, 4), dtype=torch.long),
            },
            "view2": {
                "pixel_values": torch.randn(2, 3, 224, 224),
                "input_ids": torch.zeros((2, 16), dtype=torch.long),
                "bbox": torch.zeros((2, 16, 4), dtype=torch.long),
            },
            "labels": torch.tensor([0, 15]),
        }
    ]

    print("\n--- 1. Testing Stage 1 (SupCon-ST with LARS + Warmup) ---")
    trainer.train_stage1(dummy_contrastive_loader, epochs=1, warmup_epochs=1, use_lars=True)

    # 4. Tạo dummy dataloader 1 view giả lập cho Stage 2
    dummy_single_loader = [
        {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "input_ids": torch.zeros((2, 16), dtype=torch.long),
            "bbox": torch.zeros((2, 16, 4), dtype=torch.long),
            "labels": torch.tensor([0, 15]),
        }
    ]

    print("\n--- 2. Testing Stage 2 (Linear Probe CE on Frozen Encoder) ---")
    trainer.train_stage2(dummy_single_loader, val_loader=dummy_single_loader, epochs=1)

    print("\n" + "=" * 65)
    print("[PASSED] DICTrainer verified successfully!")
    print("=" * 65)