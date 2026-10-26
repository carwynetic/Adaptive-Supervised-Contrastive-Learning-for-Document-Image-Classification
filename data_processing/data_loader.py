# ==============================================================================
# DATA LOADER: DataLoader Builder cho LayoutLMv3 & Supervised Contrastive Learning (SupCon)
# ==============================================================================
# DANH SÁCH CHI TIẾT CÁC NGUỒN MÃ NGUỒN THAM KHẢO (EVIDENCE & CITATIONS):
#
# 1. Cơ chế đóng gói PyTorch DataLoader với Custom Collator:
#    - Repository: NielsRogge/Transformers-Tutorials
#    - File nguồn: LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
#    - Link trực tiếp: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
#    - Cell tham chiếu: Cell 18
#    - Đoạn code gốc:
#          dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=4)
#
# 2. Cấu hình DataLoader cho Supervised Contrastive Learning (SupCon):
#    - Repository: HobbitLong/SupContrast (Tác giả: Yonglong Tian)
#    - File nguồn: main_supcon.py
#    - Link trực tiếp: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
#    - Hàm tham chiếu: `set_loader(opt)`
#    - Dòng tham chiếu: Cấu hình `pin_memory=True`, `drop_last=True` để cố định kích thước batch
#      (tránh trường hợp batch cuối cùng bị lẻ mẫu làm sai lệch ma trận tương đồng SupCon):
#          train_loader = torch.utils.data.DataLoader(
#              train_dataset, batch_size=opt.batch_size, shuffle=True,
#              num_workers=opt.num_workers, pin_memory=True, drop_last=True
#          )
#
# 3. Quản lý phân chia tập Train / Val / Test cho RVL-CDIP:
#    - Repository: huggingface/datasets
#    - File nguồn: datasets/rvl_cdip/rvl_cdip.py
#    - Link trực tiếp: https://github.com/huggingface/datasets/blob/main/datasets/rvl_cdip/rvl_cdip.py
#    - Hàm tham chiếu: `_split_generators(...)`
#    - Logic: Chia thành 3 split tương ứng với 3 file: train.txt, val.txt, test.txt.
# ==============================================================================

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from data_processing.collator import LayoutLMv3DataCollator
from data_processing.dataset import RVLCDIPDataset


def build_dataloader(
    data_dir: Union[str, Path],
    split: str = "train",
    processor: Optional[Any] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    transform: Optional[Callable] = None,
    is_contrastive: bool = False,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """
    Xây dựng PyTorch DataLoader cho tập RVL-CDIP với collator LayoutLMv3.

    Args:
        data_dir: Đường dẫn thư mục gốc dữ liệu (chứa 'images' và 'labels').
        split: 'train', 'val', hoặc 'test'.
        processor: Đối tượng LayoutLMv3Processor (nạp từ checkpoints/layoutlmv3-base).
        batch_size: Kích thước batch (số ảnh mỗi lượt nạp).
        num_workers: Số tiến trình phụ nạp dữ liệu (trên Windows nên để 0 hoặc 2).
        shuffle: Xáo trộn dữ liệu (mặc định True cho tập train, False cho val/test).
        transform: Hàm biến đổi / tăng cường ảnh tài liệu.
        is_contrastive: True nếu nạp dữ liệu cho Supervised Contrastive Learning (SupCon).
        max_samples: Giới hạn số mẫu nạp (dùng để kiểm thử nhanh).
        pin_memory: Tự động sao chép tensor vào CUDA pinned memory để tăng tốc nạp lên GPU.
                    Tham khảo: HobbitLong/SupContrast/main_supcon.py.
        drop_last: Bỏ batch cuối nếu không đủ batch_size (bắt buộc cho SupCon để ổn định kích thước batch).

    Returns:
        Đối tượng DataLoader sẵn sàng cho vòng lặp huấn luyện.
    """
    # --------------------------------------------------------------------------
    # Mặc định xáo trộn khi train, giữ nguyên thứ tự khi evaluate
    # --------------------------------------------------------------------------
    if shuffle is None:
        shuffle = (split == "train")

    # --------------------------------------------------------------------------
    # Tham khảo: HobbitLong/SupContrast/main_supcon.py (hàm set_loader)
    # SupCon yêu cầu kích thước batch cố định trong suốt quá trình train
    # --------------------------------------------------------------------------
    if drop_last is None:
        drop_last = (split == "train" and is_contrastive)

    # Khởi tạo RVLCDIPDataset
    dataset = RVLCDIPDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )

    # Khởi tạo LayoutLMv3DataCollator (dynamic padding và on-the-fly OCR)
    collator = LayoutLMv3DataCollator(processor=processor) if processor is not None else None

    # --------------------------------------------------------------------------
    # Khởi tạo DataLoader chuẩn của PyTorch
    # Tham khảo: NielsRogge/Transformers-Tutorials (Cell 18)
    # --------------------------------------------------------------------------
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


def get_dataloaders(
    data_dir: Union[str, Path],
    processor: Optional[Any] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    transform: Optional[Callable] = None,
    is_contrastive: bool = False,
    max_samples_per_split: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Hàm tiện ích tạo nhanh cả 3 DataLoader: (train_loader, val_loader, test_loader).
    
    Tham khảo cấu trúc 3 split chuẩn của RVL-CDIP:
    huggingface/datasets/datasets/rvl_cdip/rvl_cdip.py (_split_generators)
    """
    # 1. Tập Train (có thể bật is_contrastive=True cho SupCon)
    train_loader = build_dataloader(
        data_dir=data_dir,
        split="train",
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples_per_split,
    )

    # 2. Tập Validation (luôn chạy single-view để đánh giá độ chính xác phân loại)
    val_loader = build_dataloader(
        data_dir=data_dir,
        split="val",
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        transform=None,
        is_contrastive=False,
        max_samples=max_samples_per_split,
    )

    # 3. Tập Test
    test_loader = build_dataloader(
        data_dir=data_dir,
        split="test",
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        transform=None,
        is_contrastive=False,
        max_samples=max_samples_per_split,
    )

    return train_loader, val_loader, test_loader
