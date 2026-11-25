import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
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


class RandomGaussianBlur:
    """Random Gaussian Blur adapted from SimCLR Appendix A for document images."""

    def __init__(self, p: float = 0.5, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, img: torch.Tensor):
        if self.p <= 0.0 or random.random() > self.p:
            return img
        _, h, w = img.shape
        k = max(3, int(round(0.05 * min(h, w))))
        if k % 2 == 0:
            k += 1
        k = min(k, 11)
        sigma_val = random.uniform(self.sigma[0], self.sigma[1])
        return TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma_val, sigma_val])


class PKBatchSampler(Sampler[List[int]]):
    """Class-balanced P x K batch sampler ensuring positive and negative pairs per mini-batch."""

    def __init__(
        self,
        labels: Sequence[int],
        p: int = 4,
        k: int = 2,
        seed: int = 42,
    ) -> None:
        self.labels = [int(y) for y in labels]
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(self.labels):
            self.class_to_indices.setdefault(lbl, []).append(idx)
        self.classes = sorted(self.class_to_indices.keys())
        self.p = max(1, min(int(p), len(self.classes))) if self.classes else 1
        self.k = max(1, int(k))
        self.batch_size = self.p * self.k
        self.seed = int(seed)
        self._epoch = 0
        self.num_batches = max(1, len(self.labels) // max(1, self.batch_size)) if self.labels else 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        if not self.classes:
            return

        pools: Dict[int, List[int]] = {}
        for c in self.classes:
            idxs = list(self.class_to_indices[c])
            rng.shuffle(idxs)
            pools[c] = idxs

        class_order = list(self.classes)
        for _ in range(self.num_batches):
            if len(class_order) < self.p:
                refill = list(self.classes)
                rng.shuffle(refill)
                class_order.extend(refill)
            chosen_classes = class_order[: self.p]
            class_order = class_order[self.p :]

            batch: List[int] = []
            for c in chosen_classes:
                src = self.class_to_indices[c]
                if len(pools[c]) < self.k:
                    refill = list(src)
                    rng.shuffle(refill)
                    pools[c].extend(refill)
                while len(pools[c]) < self.k:
                    pools[c].append(rng.choice(src))
                batch.extend(pools[c][: self.k])
                pools[c] = pools[c][self.k :]
            yield batch


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x: torch.Tensor):
        return self.transform(x), self.transform(x)


def get_transforms(
    max_shear: float = 4.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    p_erase: float = 0.3,
    noise_std: float = 8.0,
    p_blur: float = 0.5,
    blur_sigma: Tuple[float, float] = (0.1, 2.0),
):
    return Compose([
        RandomShear(max_angle=max_shear),
        RandomBrightness(min_ratio=brightness_range[0], max_ratio=brightness_range[1]),
        RandomGaussianBlur(p=p_blur, sigma=blur_sigma),
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


class RVLCDIPCachedDataset(Dataset):
    LABEL_NAMES: List[str] = RVLCDIPDataset.LABEL_NAMES

    def __init__(
        self,
        data_dir: Union[Path, str],
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_contrastive: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.is_contrastive = is_contrastive

        cache_path = self.data_dir / "ocr_cache" / f"{split}.jsonl"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached OCR file not found: {cache_path}")

        all_records: List[Dict[str, Any]] = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_records.append(json.loads(line))

        if max_samples is not None and len(all_records) > max_samples:
            buckets: Dict[int, List[Dict[str, Any]]] = {}
            for rec in all_records:
                lbl = int(rec.get("label", 0))
                buckets.setdefault(lbl, []).append(rec)
            ordered_labels = sorted(buckets.keys())
            selected: List[Dict[str, Any]] = []
            max_len = max((len(buckets[l]) for l in ordered_labels), default=0)
            for pos in range(max_len):
                for lbl in ordered_labels:
                    if pos < len(buckets[lbl]):
                        selected.append(buckets[lbl][pos])
                        if len(selected) >= max_samples:
                            break
                if len(selected) >= max_samples:
                    break
            self.records = selected
        else:
            self.records = all_records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _sanitize_words_and_boxes(
        words: Sequence[str],
        boxes: Sequence[Sequence[int]],
    ) -> Tuple[List[str], List[List[int]]]:
        clean_words: List[str] = []
        clean_boxes: List[List[int]] = []
        for w, b in zip(words, boxes):
            w_str = str(w).strip()
            if not w_str or len(b) < 4:
                continue
            x0 = max(0, min(1000, int(b[0])))
            y0 = max(0, min(1000, int(b[1])))
            x1 = max(x0, min(1000, int(b[2])))
            y1 = max(y0, min(1000, int(b[3])))
            clean_words.append(w_str)
            clean_boxes.append([x0, y0, x1, y1])
        if not clean_words:
            clean_words = ["empty"]
            clean_boxes = [[0, 0, 1000, 1000]]
        return clean_words, clean_boxes

    def _resolve_image_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute() and p.exists():
            return p
        return self.data_dir / p

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = self._resolve_image_path(rec["image_path"])
        tensor_img = read_image(img_path)
        words, boxes = self._sanitize_words_and_boxes(rec.get("words", []), rec.get("boxes", []))
        label = int(rec["label"])

        if self.is_contrastive:
            v1_t = self.transform(tensor_img) if self.transform is not None else tensor_img.clone()
            v2_t = self.transform(tensor_img) if self.transform is not None else tensor_img.clone()
            return {
                "views": (TF.to_pil_image(v1_t).convert("RGB"), TF.to_pil_image(v2_t).convert("RGB")),
                "words": words,
                "boxes": boxes,
                "label": label,
                "image_path": str(img_path),
            }

        img_t = self.transform(tensor_img) if self.transform is not None else tensor_img
        return {
            "image": TF.to_pil_image(img_t).convert("RGB"),
            "words": words,
            "boxes": boxes,
            "label": label,
            "image_path": str(img_path),
        }


class MultimodalCollator:
    def __init__(
        self,
        processor: Optional[Any] = None,
        config_dir: Optional[Union[str, Path]] = None,
        max_length: int = 512,
    ) -> None:
        self.max_length = max_length
        if processor is not None:
            self.processor = processor
        else:
            default_cfg = (
                Path(config_dir)
                if config_dir is not None
                else Path(__file__).resolve().parents[1] / "configs" / "layoutlmv3-base"
            )
            try:
                from third_party.layoutlmv3 import LayoutLMv3Processor
            except ImportError:
                from transformers import LayoutLMv3Processor
            self.processor = LayoutLMv3Processor.from_pretrained(
                str(default_cfg),
                apply_ocr=False,
                local_files_only=True,
            )

    def _encode_view(
        self,
        images: List[Image.Image],
        words_list: List[List[str]],
        boxes_list: List[List[List[int]]],
    ) -> Dict[str, torch.Tensor]:
        enc = self.processor(
            images=images,
            text=words_list,
            boxes=boxes_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "bbox": enc["bbox"],
            "pixel_values": enc["pixel_values"],
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [item for item in batch if item]
        if not batch:
            return {}

        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        words_list = [item["words"] for item in batch]
        boxes_list = [item["boxes"] for item in batch]
        image_paths = [item.get("image_path", "") for item in batch]

        if "views" in batch[0]:
            v1_imgs = [item["views"][0] for item in batch]
            v2_imgs = [item["views"][1] for item in batch]
            return {
                "view1": self._encode_view(v1_imgs, words_list, boxes_list),
                "view2": self._encode_view(v2_imgs, words_list, boxes_list),
                "labels": labels,
                "image_paths": image_paths,
            }

        imgs = [item["image"] for item in batch]
        out = self._encode_view(imgs, words_list, boxes_list)
        out["labels"] = labels
        out["image_paths"] = image_paths
        return out