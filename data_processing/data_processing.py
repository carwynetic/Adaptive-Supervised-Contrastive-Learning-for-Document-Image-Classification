import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

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

def read_image(img_path: Path | str):
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
        data_dir: Path,
        split: str = "train",
        transform = None,
        is_contrastive = False,
        max_samples = None,
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_img_path, label = self.samples[idx]
        full_img_path = self.images_dir / rel_img_path

        tensor_img = read_image(full_img_path)

        if self.is_contrastive:
            v1 = self.transform(tensor_img) if self.transform is not None else tensor_img.clone()
            v2 = self.transform(tensor_img) if self.transform is not None else tensor_img.clone()
            return {"views": (v1, v2), "label": label, "image_path": str(full_img_path)}
        else:
            img = self.transform(tensor_img) if self.transform is not None else tensor_img
            return {"image": img, "label": label, "image_path": str(full_img_path)}


# ==============================================================================
# DATA COLLATOR
# ==============================================================================

class LayoutLMv3DataCollator:
    def __init__(self, processor = None):
        self.processor = processor

    def _process_images(self, images):
        if self.processor is not None:
            img_proc = getattr(self.processor, "image_processor", self.processor)
            return img_proc(images=images, apply_ocr=False, return_tensors="pt")
        return {"pixel_values": torch.stack(images)}

    def __call__(self, batch):
        batch = [item for item in batch if item]
        if not batch:
            return {}

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        image_paths = [item["image_path"] for item in batch if "image_path" in item]

        if "views" in batch[0]:
            return {
                "view1": self._process_images([item["views"][0] for item in batch]),
                "view2": self._process_images([item["views"][1] for item in batch]),
                "labels": labels,
                "image_paths": image_paths,
            }

        res = self._process_images([item["image"] for item in batch])
        res["labels"] = labels
        res["image_paths"] = image_paths
        return res


#
#   soft labels generator cho supcon-st (theo ConOSR - AAAI 2023)
#

def get_smooth_labels(
    labels : torch.Tensor,
    classid_list : Union[int, Sequence[int]] = 16,
    smoothing_coeff : float = 0.1, 
)-> torch.Tensor:
    """
    Tạo nhãn mềm (Soft Targets) cho Supervised Contrastive Learning (SupCon-ST).
    Tái hiện chính xác nguyên bản hàm `get_smooth_labels` trong ConOSR (utils.py).
    Args:
        labels: Tensor 1D nhãn lớp nguyên [batch_size].
        classid_list: Số lượng lớp (int) hoặc danh sách các ID lớp (ví dụ: range(16)).
        smoothing_coeff: Hệ số làm mịn nhãn (epsilon, mặc định 0.1).
    Returns:
        targets: Tensor 2D [batch_size, num_classes] đã được làm mịn.
    """
    if isinstance(classid_list, int):
        classes = list(range(classid_list))
    else:
        classes = list(classid_list)
    num_classes = len(classes)
    label_positive = 1.0 - smoothing_coeff
    label_negative = smoothing_coeff / (num_classes - 1)
    targets = label_negative * torch.ones(
        [labels.shape[0], num_classes],
        dtype=torch.float32,
        device=labels.device,
        requires_grad=False,
    )
    for j, label in enumerate(labels):
        val = label.item()
        if val not in classes:
            continue
        idx = classes.index(val)
        targets[j, idx] = label_positive
    return targets