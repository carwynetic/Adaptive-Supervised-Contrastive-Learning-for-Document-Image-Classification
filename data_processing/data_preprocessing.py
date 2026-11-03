import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import Compose
# ==============================================================================
# Đọc ảnh và Augmentation
# ==============================================================================

def read_image(img_path: Path | str) -> torch.Tensor:
    path_obj = Path(img_path)
    with Image.open(path_obj) as img:
        img_rgb = img.convert("RGB")
        return TF.pil_to_tensor(img_rgb)


read_image_as_tensor = read_image


class RandomShear:
    def __init__(self, max_angle: float = 10.0):
        self.max_angle = max_angle

    def __call__(self, img: torch.Tensor):
        angle = random.uniform(-self.max_angle, self.max_angle)
        return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[angle, 0.0])


class RandomBrightness:
    def __init__(self, min_ratio: float = 0.7, max_ratio: float = 1.3):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: torch.Tensor):
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        return TF.adjust_brightness(img, ratio)

class RandomErase:
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.2), ratio: Tuple[float, float] = (0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: torch.Tensor):
        if random.random() > self.p:
            return img

        _, h, w = img.shape
        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h_erase = int(np.sqrt(target_area / aspect_ratio))
        w_erase = int(target_area / h_erase)
        h_erase = max(1, min(h_erase, h))
        w_erase = max(1, min(w_erase, w))

        top = random.randint(0, h - h_erase)
        left = random.randint(0, w - w_erase)

        img_erased = img.clone()
        img_erased[:, top:top + h_erase, left:left + w_erase] = 0
        return img_erased


class GaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 10.0):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor):
        noise = torch.randn_like(img.float()) * self.std + self.mean
        if img.dtype == torch.uint8:
            return (img.float() + noise).clamp(0, 255).to(torch.uint8)
        return (img.float() + noise).clamp(0.0, 1.0).to(img.dtype)


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x: torch.Tensor):
        return self.transform(x), self.transform(x)


def get_transforms(
    max_shear: float = 10.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    p_erase: float = 0.5,
    noise_std: float = 8.0,
):
    return Compose([
        RandomShear(max_angle=max_shear),
        RandomBrightness(min_ratio=brightness_range[0], max_ratio=brightness_range[1]),
        GaussianNoise(mean=0.0, std=noise_std),
        RandomErase(p=p_erase),
    ])


get_contrastive_transforms = get_transforms


# ==============================================================================
# PYTORCH DATASET CHO RVL-CDIP
# ==============================================================================

class DatasetItem(TypedDict, total=False):
    image: torch.Tensor
    views: Tuple[torch.Tensor, torch.Tensor]
    label: int
    image_path: str


class RVLCDIPDataset(Dataset):
    LABEL_NAMES: List[str] = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific report", "scientific publication", "specification",
        "file folder", "news article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo",
    ]
    ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}
    LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_contrastive: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.is_contrastive = is_contrastive
        self.images_dir = self.data_dir / "images"

        label_file = self.data_dir / "labels" / f"{split}.txt"
        self.samples: List[Tuple[str, int]] = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], int(parts[1])))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetItem:
        rel_img_path, label = self.samples[idx]
        full_img_path = self.images_dir / rel_img_path.replace("/", os.sep).replace("\\", os.sep)

        if not full_img_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {full_img_path}")

        tensor_img = read_image(full_img_path)

        if self.is_contrastive:
            v1 = self.transform(tensor_img) if self.transform is not None else tensor_img.clone()
            v2 = self.transform(tensor_img) if self.transform is not None else tensor_img.clone()
            return {"views": (v1, v2), "label": label, "image_path": str(full_img_path)}

        if self.transform is not None:
            tensor_img = self.transform(tensor_img)

        return {"image": tensor_img, "label": label, "image_path": str(full_img_path)}


def build_dataset(
    root: Path | str,
    split: str = "train",
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    is_contrastive: bool = False,
    max_samples: Optional[int] = None,
) -> RVLCDIPDataset:
    return RVLCDIPDataset(
        data_dir=Path(root),
        split=split,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )


# ==============================================================================
# DATA COLLATOR
# ==============================================================================

class LayoutLMv3DataCollator:
    def __init__(
        self,
        processor: Optional[Any] = None,
        max_length: int = 512,
        return_tensors: str = "pt",
    ):
        self.processor = processor
        self.max_length = max_length
        self.return_tensors = return_tensors

        # Tắt Tesseract OCR mặc định của HF processor nếu có
        if hasattr(self.processor, "apply_ocr"):
            self.processor.apply_ocr = False
        if hasattr(getattr(self.processor, "image_processor", None), "apply_ocr"):
            self.processor.image_processor.apply_ocr = False

    def _encode_images(
        self,
        images: List[torch.Tensor],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
    ):
        if self.processor is None:
            return {"pixel_values": torch.stack(images)}

        # Nếu có words và boxes (khi đã chạy OCR trước) -> tokenize đầy đủ
        if words is not None and boxes is not None:
            return self.processor(
                images=images,
                text=words,
                boxes=boxes,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors=self.return_tensors,
            )

        # Nếu chỉ có ảnh (contrastive views hoặc trích xuất visual backbone)
        # Chỉ gọi image_processor để tạo pixel_values (B, 3, 224, 224), không đụng vào text
        img_proc = getattr(self.processor, "image_processor", self.processor)
        return img_proc(images=images, return_tensors=self.return_tensors)

    def __call__(self, batch: List[DatasetItem]) -> Dict[str, Any]:
        batch = [item for item in batch if item]
        if not batch:
            return {}

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        image_paths = [item["image_path"] for item in batch if "image_path" in item]

        # 1. Chế độ Contrastive Learning (2 views)
        if "views" in batch[0]:
            views_1 = [item["views"][0] for item in batch]
            views_2 = [item["views"][1] for item in batch]
            res = {
                "view1": self._encode_images(views_1),
                "view2": self._encode_images(views_2),
                "labels": labels,
            }
            if image_paths:
                res["image_paths"] = image_paths
            return res

        # 2. Chế độ Single-view
        images = [item["image"] for item in batch]
        words = [item["words"] for item in batch] if "words" in batch[0] else None
        boxes = [item["boxes"] for item in batch] if "boxes" in batch[0] else None

        encoding = self._encode_images(images, words=words, boxes=boxes)
        encoding["labels"] = labels
        if image_paths:
            encoding["image_paths"] = image_paths
        return encoding


# ==============================================================================
# BUILD DATALOADER
# ==============================================================================

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
    collator = LayoutLMv3DataCollator(processor=processor) if processor is not None else None

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





