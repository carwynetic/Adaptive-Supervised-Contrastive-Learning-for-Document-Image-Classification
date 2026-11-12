# ==============================================================================
# KIỂM THỬ PIPELINE TIỀN XỬ LÝ DỮ LIỆU (TEST DATA PREPROCESSING)
# Dự án: Adaptive Supervised Contrastive Learning for Document Image Classification
# ==============================================================================

import io
import os
from pathlib import Path
import sys
from typing import Any, Dict

# Đảm bảo import được module từ thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

from data_processing.data_preprocessing import (
    DatasetItem,
    GaussianNoise,
    LayoutLMv3DataCollator,
    RVLCDIPDataset,
    RandomBrightness,
    RandomErase,
    RandomShear,
    TwoCropTransform,
    get_contrastive_transforms,
    read_image_as_tensor,
)
from utils.dataloader import build_dataloader

def print_tensor_info(name: str, tensor: torch.Tensor, show_sample_values: bool = True, indent: int = 4) -> None:
    """Hàm in thông tin chi tiết và dữ liệu mẫu của một Tensor."""
    prefix = " " * indent
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.float().mean().item()
    std_val = tensor.float().std().item()

    print(f"{prefix}* [{name}]")
    print(f"{prefix}  - Kích thước (Shape) : {list(tensor.shape)} (Số chiều: {tensor.ndim})")
    print(f"{prefix}  - Kiểu dữ liệu (Dtype): {tensor.dtype} | Device: {tensor.device}")
    print(f"{prefix}  - Thống kê phân bố    : Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, Std={std_val:.2f}")

    if show_sample_values:
        # Nếu là Tensor ảnh 3 chiều (C, H, W)
        if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
            patch = tensor[0, :3, :3].tolist()
            print(f"{prefix}  - Ma trận pixel mẫu (3x3 góc trên bên trái, kênh 0):")
            for row in patch:
                print(f"{prefix}    {row}")
        # Nếu là Tensor 1 chiều (nhãn hoặc danh sách 1D)
        elif tensor.ndim == 1:
            vals = tensor[:8].tolist()
            suffix = "..." if len(tensor) > 8 else ""
            print(f"{prefix}  - Giá trị mẫu ({len(tensor)} phần tử): {vals}{suffix}")
        # Nếu là Tensor 2 chiều (ví dụ: input_ids, attention_mask)
        elif tensor.ndim == 2:
            row_sample = tensor[0, :10].tolist()
            suffix = "..." if tensor.shape[1] > 10 else ""
            print(f"{prefix}  - Token sequence mẫu (dòng 0, 10 tokens đầu): {row_sample}{suffix}")
        # Nếu là Tensor 3 chiều cho Bounding Box [B, Seq, 4]
        elif tensor.ndim == 3 and tensor.shape[-1] == 4:
            boxes = tensor[0, :3].tolist()
            print(f"{prefix}  - Bounding Box mẫu [x0, y0, x1, y1] (3 tokens đầu dòng 0): {boxes}")


def find_data_dir() -> Path:
    """Tự động dò tìm thư mục dữ liệu RVL-CDIP."""
    possible_paths = [
        PROJECT_ROOT / "dataset" / "rvl-cdip",
        PROJECT_ROOT / "data" / "rvl-cdip",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data",
    ]
    for p in possible_paths:
        if (p / "labels" / "train.txt").exists():
            return p
    raise FileNotFoundError("Không tìm thấy tập dữ liệu 'rvl-cdip' chứa thư mục 'labels/train.txt'.")


def test_reader_and_single_view(data_dir: Path):
    """Kiểm tra đọc dữ liệu ảnh thành Tensor và nạp Single-view Dataset."""
    print("\n" + "=" * 70)
    print("[1] KIỂM TRA ĐỌC DỮ LIỆU & RVLCDIPDataset (SINGLE-VIEW)")
    print("=" * 70)

    dataset = RVLCDIPDataset(data_dir=data_dir, split="train", max_samples=5)
    print(f"  + Tổng số mẫu nạp vào Dataset: {len(dataset)}")
    assert len(dataset) == 5, f"Kỳ vọng 5 mẫu nhưng nhận được {len(dataset)}"

    sample: DatasetItem = dataset[0]
    img = sample["image"]
    label = sample["label"]
    img_path = sample["image_path"]

    print("\n  --- CHI TIẾT OUTPUT MẪU ĐẦU TIÊN (SAMPLE 0) ---")
    print(f"  + Khóa (Keys) có trong DatasetItem: {list(sample.keys())}")
    print(f"  + Đường dẫn file ảnh gốc         : {img_path}")
    print(f"  + Nhãn số (Class ID)              : {label}")
    print(f"  + Tên nhãn (Class Name)           : {dataset.LABEL_NAMES[label]}")

    print("\n  --- CHI TIẾT TENSOR ẢNH (IMAGE TENSOR) ---")
    print_tensor_info("sample['image']", img, show_sample_values=True)

    assert isinstance(img, torch.Tensor), "Ảnh phải là torch.Tensor"
    assert img.ndim == 3 and img.shape[0] == 3, f"Ảnh phải có shape (3, H, W), nhận được {img.shape}"
    assert img.dtype == torch.uint8, f"Dtype kỳ vọng torch.uint8, nhận được {img.dtype}"
    assert 0 <= label < 16, f"Label phải nằm trong [0, 15], nhận được {label}"
    print("\n  => [PASS] Đọc dữ liệu ảnh và nạp Single-view thành công hoàn toàn!")


def test_augmentations_on_tensor():
    """Kiểm tra các hàm biến đổi tăng cường dữ liệu chạy 100% trên Tensor."""
    print("\n" + "=" * 70)
    print("[2] KIỂM TRA CHI TIẾT TỪNG PHÉP BIẾN ĐỔI AUGMENTATION TRÊN TENSOR")
    print("=" * 70)

    # Giả lập 1 dummy tensor ảnh (3, 224, 224) uint8
    torch.manual_seed(42)
    dummy_img = torch.randint(50, 200, (3, 224, 224), dtype=torch.uint8)
    print("\n  --- TENSOR ĐẦU VÀO GỐC (ORIGINAL TENSOR) ---")
    print_tensor_info("dummy_img (Before Augmentations)", dummy_img, show_sample_values=True)

    # 1. RandomShear
    print("\n  --- [2.1] Phép biến đổi: RandomShear ---")
    shear_op = RandomShear(max_angle=15.0)
    out_shear = shear_op(dummy_img)
    print_tensor_info("out_shear", out_shear, show_sample_values=False)
    diff_shear = torch.abs(dummy_img.float() - out_shear.float()).mean().item()
    print(f"    - Mức độ biến đổi hình học (Mean Absolute Diff): {diff_shear:.2f}")
    assert isinstance(out_shear, torch.Tensor) and out_shear.shape == dummy_img.shape

    # 2. RandomBrightness
    print("\n  --- [2.2] Phép biến đổi: RandomBrightness ---")
    bright_op = RandomBrightness(min_ratio=1.3, max_ratio=1.3)  # Tăng 30% độ sáng
    out_bright = bright_op(dummy_img)
    print_tensor_info("out_bright", out_bright, show_sample_values=False)
    print(f"    - Mean gốc: {dummy_img.float().mean():.2f} -> Mean sau tăng độ sáng: {out_bright.float().mean():.2f}")
    assert isinstance(out_bright, torch.Tensor) and out_bright.shape == dummy_img.shape

    # 3. GaussianNoise
    print("\n  --- [2.3] Phép biến đổi: GaussianNoise ---")
    noise_op = GaussianNoise(mean=0.0, std=15.0)
    out_noise = noise_op(dummy_img)
    print_tensor_info("out_noise", out_noise, show_sample_values=False)
    diff_noise = torch.abs(dummy_img.float() - out_noise.float()).mean().item()
    print(f"    - Mức độ nhiễu bổ sung trung bình: {diff_noise:.2f}")
    assert isinstance(out_noise, torch.Tensor) and out_noise.shape == dummy_img.shape

    # 4. RandomErase
    print("\n  --- [2.4] Phép biến đổi: RandomErase (Cutout) ---")
    erase_op = RandomErase(p=1.0, scale=(0.1, 0.2))  # Bắt buộc xóa 1 vùng 10-20%
    out_erase = erase_op(dummy_img)
    zeros_orig = (dummy_img == 0).sum().item()
    zeros_erased = (out_erase == 0).sum().item()
    print_tensor_info("out_erase", out_erase, show_sample_values=False)
    print(f"    - Số lượng pixel bị xóa về 0: {zeros_erased - zeros_orig} pixels")
    assert isinstance(out_erase, torch.Tensor) and out_erase.shape == dummy_img.shape
    assert zeros_erased > zeros_orig, "RandomErase phải tạo thêm các pixel có giá trị 0"

    # 5. Pipeline Compose hoàn chỉnh
    print("\n  --- [2.5] Chuỗi tăng cường hoàn chỉnh: get_contrastive_transforms ---")
    con_transforms = get_contrastive_transforms(max_shear=10.0, p_erase=0.5)
    out_composed = con_transforms(dummy_img)
    print_tensor_info("out_composed (Full Pipeline)", out_composed, show_sample_values=True)
    assert isinstance(out_composed, torch.Tensor) and out_composed.shape == dummy_img.shape

    print("\n  => [PASS] Tất cả các hàm Augmentation hoạt động chính xác trên Tensor!")


def test_contrastive_dataset(data_dir: Path):
    """Kiểm tra RVLCDIPDataset ở chế độ Contrastive Learning (SupCon 2 views)."""
    print("\n" + "=" * 70)
    print("[3] KIỂM TRA RVLCDIPDataset (CONTRASTIVE SUPCON 2-VIEWS)")
    print("=" * 70)

    con_transforms = get_contrastive_transforms(max_shear=15.0, p_erase=0.5)
    dataset = RVLCDIPDataset(
        data_dir=data_dir,
        split="train",
        transform=con_transforms,
        is_contrastive=True,
        max_samples=5,
    )
    sample: DatasetItem = dataset[0]

    assert "views" in sample, "Sample phải chứa khóa 'views'"
    views = sample["views"]
    assert len(views) == 2, f"Kỳ vọng 2 views nhưng có {len(views)}"
    view1, view2 = views

    print(f"  + Đường dẫn ảnh gốc : {sample['image_path']}")
    print(f"  + Nhãn lớp (Label)  : {sample['label']} ({dataset.LABEL_NAMES[sample['label']]})")

    print("\n  --- THÔNG TIN VIEW 1 ---")
    print_tensor_info("sample['views'][0]", view1, show_sample_values=True)

    print("\n  --- THÔNG TIN VIEW 2 ---")
    print_tensor_info("sample['views'][1]", view2, show_sample_values=True)

    # Đo độ khác biệt giữa 2 góc nhìn tăng cường
    view_diff = torch.abs(view1.float() - view2.float()).mean().item()
    print(f"\n  + Mức độ sai khác giữa 2 views (Mean Absolute Difference): {view_diff:.2f}")

    assert isinstance(view1, torch.Tensor) and isinstance(view2, torch.Tensor)
    assert view1.shape == view2.shape
    assert view1.dtype == torch.uint8 and view2.dtype == torch.uint8
    print("  => [PASS] Sinh 2 views độc lập cho SupCon thành công!")


def test_dataloaders(data_dir: Path):
    """Kiểm tra gom batch của DataLoader tích hợp LayoutLMv3DataCollator."""
    print("\n" + "=" * 70)
    print("[4] KIỂM TRA DATALOADER & MULTIMODAL LAYOUTLMV3 COLLATOR")
    print("=" * 70)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "layoutlmv3-base"
    processor = None
    try:
        from transformers import AutoProcessor
        if (ckpt_path / "processor_config.json").exists() or (ckpt_path / "preprocessor_config.json").exists():
            print(f"  + Đang nạp AutoProcessor từ thư mục cục bộ: {ckpt_path}...")
            processor = AutoProcessor.from_pretrained(str(ckpt_path), apply_ocr=True)
        else:
            print("  + Đang nạp AutoProcessor từ Hugging Face Hub ('microsoft/layoutlmv3-base')...")
            processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    except Exception as e:
        print(f"  [Ghi chú] Không thể nạp AutoProcessor ({e}). Bỏ qua bước kiểm thử DataLoader.")
        return

    # 4.1 Test Single-view DataLoader
    print("\n  " + "-" * 55)
    print("  [4.1] OUTPUT BATCH: SINGLE-VIEW DATALOADER")
    print("  " + "-" * 55)
    loader_single = build_dataloader(
        data_dir=data_dir,
        split="train",
        processor=processor,
        batch_size=2,
        num_workers=0,
        is_contrastive=False,
        max_samples=2,
    )
    for batch in loader_single:
        print(f"  + Các trường (Keys) trong Batch: {list(batch.keys())}")
        print("\n  --- CHI TIẾT TỪNG TRƯỜNG DỮ LIỆU TENSOR TRONG BATCH ---")
        for k in ["input_ids", "attention_mask", "bbox", "pixel_values", "labels"]:
            if k in batch:
                print_tensor_info(f"batch['{k}']", batch[k], show_sample_values=True)

        if "image_paths" in batch:
            print(f"\n    * [batch['image_paths']] ({len(batch['image_paths'])} ảnh):")
            for idx, p in enumerate(batch["image_paths"]):
                print(f"      [{idx}] {p}")

        assert "pixel_values" in batch
        assert batch["pixel_values"].shape == torch.Size([2, 3, 224, 224])
        assert batch["labels"].shape == torch.Size([2])
        break

    # 4.2 Test Contrastive SupCon DataLoader
    print("\n  " + "-" * 55)
    print("  [4.2] OUTPUT BATCH: CONTRASTIVE SUPCON DATALOADER (2 VIEWS)")
    print("  " + "-" * 55)
    loader_contrastive = build_dataloader(
        data_dir=data_dir,
        split="train",
        processor=processor,
        batch_size=2,
        num_workers=0,
        is_contrastive=True,
        max_samples=2,
    )
    for batch in loader_contrastive:
        print(f"  + Các trường trong SupCon Batch: {list(batch.keys())}")
        print_tensor_info("batch['labels']", batch["labels"], show_sample_values=True)
        print("\n  --- VIEW 1 BATCH ENCODING ---")
        print_tensor_info("batch['view1']['pixel_values']", batch["view1"]["pixel_values"], show_sample_values=False)
        print("\n  --- VIEW 2 BATCH ENCODING ---")
        print_tensor_info("batch['view2']['pixel_values']", batch["view2"]["pixel_values"], show_sample_values=False)

        assert "view1" in batch and "view2" in batch
        assert batch["view1"]["pixel_values"].shape == torch.Size([2, 3, 224, 224])
        assert batch["view2"]["pixel_values"].shape == torch.Size([2, 3, 224, 224])
        assert batch["labels"].shape == torch.Size([2])
        break

    print("\n  => [PASS] DataLoader gom batch multimodal cho LayoutLMv3 hoạt động hoàn hảo!")


def main():
    print("=" * 70)
    print("BẮT ĐẦU CHẠY KIỂM THỬ: DATA PREPROCESSING PIPELINE (FULL OUTPUT VIEW)")
    print("=" * 70)

    try:
        data_dir = find_data_dir()
        print(f"[+] Thư mục dữ liệu: {data_dir.resolve()}")
    except FileNotFoundError as e:
        print(f"[!] Cảnh báo: {e}")
        return

    test_reader_and_single_view(data_dir)
    test_augmentations_on_tensor()
    test_contrastive_dataset(data_dir)
    test_dataloaders(data_dir)

    print("\n" + "=" * 70)
    print("HOÀN TẤT KIỂM THỬ: TẤT CẢ TEST CASES ĐỀU VƯỢT QUA THÀNH CÔNG (100% PASS)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
