from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from data_processing.data_processing import (
    LayoutLMv3DataCollator,
    RVLCDIPDataset,
)


def build_dataset(
    root: Path | str,
    split: str = "train",
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    is_contrastive: bool = False,
    max_samples: Optional[int] = None,
) -> RVLCDIPDataset:
    """Hàm tiện ích (utility) khởi tạo RVLCDIPDataset."""
    return RVLCDIPDataset(
        data_dir=Path(root),
        split=split,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )


def build_dataloader(
    data_dir: Path | str,
    split: str = "train",
    processor: Optional[Any] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    is_contrastive: bool = False,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """Hàm tiện ích (utility) tạo 1 DataLoader đơn lẻ cho 1 split (train/val/test)."""
    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split == "train" and is_contrastive)

    dataset = RVLCDIPDataset(
        data_dir=Path(data_dir),
        split=split,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )
    collator = LayoutLMv3DataCollator(processor=processor)

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
    data_dir: Path | str,
    processor: Optional[Any] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    is_contrastive: bool = False,
    max_samples_per_split: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Hàm tiện ích (utility) tạo đồng thời cả 3 DataLoader: train, val, test."""
    dir_path = Path(data_dir)
    train_loader = build_dataloader(
        data_dir=dir_path,
        split="train",
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples_per_split,
    )
    val_loader = build_dataloader(
        data_dir=dir_path,
        split="val",
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        transform=None,
        is_contrastive=False,
        max_samples=max_samples_per_split,
    )
    test_loader = build_dataloader(
        data_dir=dir_path,
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

