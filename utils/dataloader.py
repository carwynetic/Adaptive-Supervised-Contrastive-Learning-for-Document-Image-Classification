from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_processing.data_processing import (
    LayoutLMv3DataCollator,
    RVLCDIPDataset,
)


def build_dataset(
    root: Path,
    split: str = "train",
    transform = None,
    is_contrastive: bool = False,
    max_samples = None,
):
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
    processor = None,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle = None,
    transform = None,
    is_contrastive: bool = False,
    max_samples = None,
    pin_memory: bool = True,
    drop_last = None,
):
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
    processor = None,
    batch_size: int = 16,
    num_workers: int = 2,
    transform = None,
    is_contrastive: bool = False,
    max_samples_per_split = None,
):

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

