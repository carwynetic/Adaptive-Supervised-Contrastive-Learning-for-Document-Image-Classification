"""
CLI Training Entry Point for Document Image Classification (DIC).
Chạy huấn luyện 2 giai đoạn (Two-Stage Pipeline):
    - Stage 1: Pre-training với Multi-Exit SupCon-ST + LARS + Cosine Warmup.
    - Stage 2: Fine-tuning bộ phân loại Exit Heads với CE Loss (Linear Probe).

Hỗ trợ 2 phương pháp huấn luyện điều phối qua CLI:
    --train_type without_groupwise  (Joint backprop - Hameed et al.)
    --train_type groupwise          (Layerwise/Groupwise learning - AIC Lab)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

# Đảm bảo đường dẫn gốc của project luôn nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from data_processing import get_transforms
from model import DIC, DICWithoutGroupwise
from train_test.trainer import DICTrainer
from utils.dataloader import build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện Adaptive Contrastive Early-Exiting DIC")

    # 1. Cấu hình chế độ huấn luyện
    parser.add_argument(
        "--train_type",
        type=str,
        default="without_groupwise",
        choices=["without_groupwise", "groupwise"],
        help="Chiến lược backward: 'without_groupwise' (tính tổng loss) hoặc 'groupwise' (backward từng cụm layer)",
    )
    parser.add_argument(
        "--exit_layers",
        type=int,
        nargs="+",
        default=[0, 1, 4, 8, 10, 12],
        help="Danh sách các tầng đặt exit (mặc định: 0 1 4 8 10 12)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=16,
        help="Số lượng lớp phân loại (mặc định 16 lớp RVL-CDIP)",
    )

    # 2. Dữ liệu và đường dẫn
    parser.add_argument("--data_dir", type=str, default="data/rvl-cdip", help="Đường dẫn thư mục tập dữ liệu")
    parser.add_argument("--save_dir", type=str, default="checkpoints/run_01", help="Thư mục lưu mô hình")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu train (dùng để debug nhanh)")

    # 3. Tham số siêu tham số huấn luyện (Hyperparameters)
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size trên mỗi GPU")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Số bước tích lũy gradient (mô phỏng large batch size theo GS. Hinton)",
    )
    parser.add_argument("--epochs_stage1", type=int, default=10, help="Số epochs huấn luyện tương phản Stage 1")
    parser.add_argument("--epochs_stage2", type=int, default=10, help="Số epochs huấn luyện phân loại Stage 2")
    parser.add_argument("--lr_stage1", type=float, default=1e-4, help="Learning rate Stage 1")
    parser.add_argument("--lr_stage2", type=float, default=3e-4, help="Learning rate Stage 2")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Số epochs Linear Warmup cho Stage 1")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--use_lars", action="store_true", default=True, help="Sử dụng LARS Optimizer Wrapper")
    parser.add_argument("--smoothing_coeff", type=float, default=0.1, help="Hệ số làm mịn nhãn mềm ConOSR")
    parser.add_argument("--temperature", type=float, default=0.1, help="Nhiệt độ tau trong hàm SupConLoss")

    # 4. Phần cứng và Chạy thử
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2, help="Số worker của DataLoader")
    parser.add_argument("--dry_run", action="store_true", help="Chạy thử với dữ liệu giả lập để kiểm tra luồng")

    return parser.parse_args()


def create_dry_run_loaders() -> tuple[list, list]:
    """Tạo dataloader giả lập để chạy thử nghiệm tính năng nhanh chóng."""
    dummy_stage1 = [
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
    dummy_stage2 = [
        {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "input_ids": torch.zeros((2, 16), dtype=torch.long),
            "bbox": torch.zeros((2, 16, 4), dtype=torch.long),
            "labels": torch.tensor([0, 15]),
        }
    ]
    return dummy_stage1, dummy_stage2


def main():
    args = parse_args()
    print("=" * 70)
    print("HUẤN LUYỆN ADAPTIVE CONTRASTIVE EARLY-EXITING DIC")
    print("=" * 70)
    print(f"Chiến lược huấn luyện    : {args.train_type.upper()}")
    print(f"Danh sách tầng Exits      : {args.exit_layers}")
    print(f"Thiết bị tính toán        : {args.device}")
    print(f"Micro Batch Size          : {args.batch_size} (Tích lũy gradient: {args.gradient_accumulation_steps})")
    print(f"Effective Batch Size      : {args.batch_size * args.gradient_accumulation_steps}")
    print(f"LARS Optimizer Wrapper    : {args.use_lars}")
    print(f"Thư mục lưu checkpoint   : {args.save_dir}")
    print("=" * 70)

    # 1. Khởi tạo mô hình mạng
    print("\n[1/3] Khởi tạo mô hình DIC...")
    if args.train_type == "groupwise":
        model = DIC(
            num_classes=args.num_classes,
            exit_layers=args.exit_layers,
            projection_dim=128,
        )
    else:
        model = DICWithoutGroupwise(
            num_classes=args.num_classes,
            exit_layers=args.exit_layers,
            projection_dim=128,
        )

    # 2. Khởi tạo DICTrainer
    print("\n[2/3] Khởi tạo DICTrainer...")
    trainer = DICTrainer(
        model=model,
        train_type=args.train_type,
        exit_layers=args.exit_layers,
        num_classes=args.num_classes,
        smoothing_coeff=args.smoothing_coeff,
        temperature=args.temperature,
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=True,
    )

    # 3. Chuẩn bị DataLoader
    print("\n[3/3] Chuẩn bị DataLoaders...")
    if args.dry_run:
        print("[!] Chế độ DRY-RUN: Sử dụng dữ liệu giả lập trong bộ nhớ để test luồng.")
        train_loader_stage1, train_loader_stage2 = create_dry_run_loaders()
        val_loader = train_loader_stage2
    else:
        print(f"[+] Đang tải dữ liệu từ: {args.data_dir}")
        # Stage 1: Dữ liệu 2 views đối xứng cho contrastive learning
        train_loader_stage1 = build_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_contrastive=True,
            transform=get_transforms(),
            max_samples=args.max_samples,
        )
        # Stage 2: Dữ liệu 1 view chuẩn cho huấn luyện phân loại
        train_loader_stage2 = build_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_contrastive=False,
            transform=None,
            max_samples=args.max_samples,
        )
        val_loader = build_dataloader(
            data_dir=args.data_dir,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_contrastive=False,
            transform=None,
            max_samples=args.max_samples,
        )

    # 4. Thực thi Stage 1: Tương phản đa tầng (SupCon-ST)
    trainer.train_stage1(
        train_loader=train_loader_stage1,
        epochs=args.epochs_stage1,
        lr=args.lr_stage1,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_lars=args.use_lars,
        save_dir=args.save_dir,
    )

    # 5. Thực thi Stage 2: Phân loại CE (Đóng băng Encoder)
    trainer.train_stage2(
        train_loader=train_loader_stage2,
        val_loader=val_loader,
        epochs=args.epochs_stage2,
        lr=args.lr_stage2,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
    )

    print("\n" + "=" * 70)
    print("HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN 2 GIAI ĐOẠN THÀNH CÔNG!")
    print("=" * 70)


if __name__ == "__main__":
    main()