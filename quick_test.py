"""
File kịch bản kiểm tra nhanh (Quick Test) cho Data Preprocessing và DataLoader.
Chạy trực tiếp từ terminal: python quick_test.py hoặc py -3.14 quick_test.py
"""

import io
import os
from pathlib import Path
import sys

# Đảm bảo console Windows in tiếng Việt UTF-8 không lỗi
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import AutoProcessor

from data_processing.data_preprocessing import (
    read_image_as_tensor,
    get_contrastive_transforms,
    RVLCDIPDataset,
)
from utils.dataloader import build_dataloader


def main():
    data_dir = Path("dataset/rvl-cdip")
    print("=" * 70)
    print("PHẦN 1: TEST DATA PREPROCESSING VỚI 1 ẢNH THỰC TẾ")
    print("=" * 70)

    # 1. Tìm 1 ảnh thực tế từ nhãn train.txt
    label_file = data_dir / "labels" / "train.txt"
    if not label_file.exists():
        print(f"[-] Không tìm thấy file nhãn: {label_file}")
        return

    with open(label_file, "r", encoding="utf-8") as f:
        rel_path, label_str = f.readline().strip().split()[:2]
    img_path = data_dir / "images" / rel_path
    label_id = int(label_str)

    print(f"[+] File ảnh kiểm thử : {img_path}")
    print(f"[+] Nhãn số (Label ID): {label_id} -> Tên nhãn: {RVLCDIPDataset.LABEL_NAMES[label_id]}")

    # 2. Đọc ảnh thành Tensor nguyên bản (C, H, W) bằng read_image_as_tensor
    raw_tensor = read_image_as_tensor(img_path)
    print(f"\n[+] [read_image_as_tensor] Output ảnh gốc:")
    print(f"    - Kích thước (Shape) : {list(raw_tensor.shape)} (Kênh C={raw_tensor.shape[0]}, H={raw_tensor.shape[1]}, W={raw_tensor.shape[2]})")
    print(f"    - Kiểu dữ liệu (Dtype): {raw_tensor.dtype}")
    print(f"    - Dải giá trị Pixel  : Min={raw_tensor.min().item()}, Max={raw_tensor.max().item()}, Mean={raw_tensor.float().mean().item():.2f}")

    # 3. Thử áp dụng chuỗi biến đổi Contrastive Transforms (Augmentation)
    transforms = get_contrastive_transforms(max_shear=10.0, noise_std=8.0, p_erase=0.5)
    view1 = transforms(raw_tensor)
    view2 = transforms(raw_tensor)
    print(f"\n[+] [get_contrastive_transforms] Tăng cường ảnh thành 2 Views:")
    print(f"    - View 1 Shape/Dtype : {list(view1.shape)} | {view1.dtype}")
    print(f"    - View 2 Shape/Dtype : {list(view2.shape)} | {view2.dtype}")
    diff = torch.abs(view1.float() - view2.float()).mean().item()
    print(f"    - Mức độ sai khác giữa 2 View (Mean Diff): {diff:.2f}")

    print("\n" + "=" * 70)
    print("PHẦN 2: XEM DATALOADER HOẠT ĐỘNG VÀ GOM BATCH NHƯ THẾ NÀO")
    print("=" * 70)

    # Nạp LayoutLMv3 AutoProcessor từ configs cục bộ (nếu có)
    processor_path = Path("configs/layoutlmv3-base")
    processor = None
    if processor_path.exists():
        print(f"[+] Đang nạp Processor cục bộ từ '{processor_path}'...")
        processor = AutoProcessor.from_pretrained(str(processor_path), apply_ocr=False)
    else:
        print("[!] Không có thư mục configs, DataLoader sẽ dùng chế độ resize mặc định.")

    # 4. DataLoader ở chế độ Single-view (dùng cho huấn luyện phân loại chuẩn hoặc inference)
    print("\n---> [2.1] SINGLE-VIEW DATALOADER (batch_size=2):")
    loader_single = build_dataloader(
        data_dir=data_dir,
        split="train",
        processor=processor,
        batch_size=2,
        num_workers=0,
        is_contrastive=False,
        max_samples=4,
    )

    batch_single = next(iter(loader_single))
    print(f"  * Các khóa (Keys) trong Batch: {list(batch_single.keys())}")
    print(f"  * batch['pixel_values']      : Shape = {list(batch_single['pixel_values'].shape)} | Dtype = {batch_single['pixel_values'].dtype}")
    print(f"  * batch['labels']            : Shape = {list(batch_single['labels'].shape)} | Values = {batch_single['labels'].tolist()}")
    print(f"  * batch['image_paths']       : {batch_single['image_paths']}")

    # 5. DataLoader ở chế độ Contrastive Learning (SupCon 2 views)
    print("\n---> [2.2] CONTRASTIVE SUPCON DATALOADER (batch_size=2, song song 2 views):")
    loader_contrastive = build_dataloader(
        data_dir=data_dir,
        split="train",
        processor=processor,
        batch_size=2,
        num_workers=0,
        transform=transforms,
        is_contrastive=True,
        max_samples=4,
    )

    batch_con = next(iter(loader_contrastive))
    print(f"  * Các khóa trong SupCon Batch: {list(batch_con.keys())}")
    print(f"  * batch['view1']['pixel_values']: Shape = {list(batch_con['view1']['pixel_values'].shape)} | Dtype = {batch_con['view1']['pixel_values'].dtype}")
    print(f"  * batch['view2']['pixel_values']: Shape = {list(batch_con['view2']['pixel_values'].shape)} | Dtype = {batch_con['view2']['pixel_values'].dtype}")
    print(f"  * batch['labels']              : Shape = {list(batch_con['labels'].shape)} | Values = {batch_con['labels'].tolist()}")
    print(f"  * batch['image_paths']         : {batch_con['image_paths']}")

    print("\n" + "=" * 70)
    print("[V] HOÀN TẤT KIỂM THỬ: DATA PREPROCESSING & DATALOADER CHẠY TỐT!")
    print("=" * 70)


if __name__ == "__main__":
    main()
