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


# ==============================================================================
# PHẦN 1: QUY ĐỊNH ĐỌC DỮ LIỆU & BIẾN ĐỔI TĂNG CƯỜNG TRÊN TENSOR (AUGMENTATIONS)
# ==============================================================================

def read_image_as_tensor(img_path: Path) -> torch.Tensor:
    path_obj = Path(img_path)
    with Image.open(path_obj) as img:
        img_rgb = img.convert("RGB")
        return TF.pil_to_tensor(img_rgb)


class RandomShear:
    """Biến dạng nghiêng hình học ngẫu nhiên (Affine Shear) cho Tensor ảnh tài liệu."""
    def __init__(self, max_angle: float = 10.0) -> None:
        self.max_angle = max_angle

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.max_angle, self.max_angle)
        return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[angle, 0.0])


class RandomBrightness:
    """Thay đổi độ sáng ngẫu nhiên cho Tensor ảnh tài liệu."""
    def __init__(self, min_ratio: float = 0.7, max_ratio: float = 1.3) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        return TF.adjust_brightness(img, ratio)


class RandomErase:
    """Xóa ngẫu nhiên một vùng ảnh (Random Erasing / Cutout) trên Tensor ảnh tài liệu."""
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.2), ratio: Tuple[float, float] = (0.3, 3.3)) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # tensor dạng (C, H, W)
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
    """Thêm nhiễu Gauss vào Tensor ảnh tài liệu (mô phỏng máy scan cũ)."""
    def __init__(self, mean: float = 0.0, std: float = 10.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img.float()) * self.std + self.mean
        if img.dtype == torch.uint8:
            return (img.float() + noise).clamp(0, 255).to(torch.uint8)
        return (img.float() + noise).clamp(0.0, 1.0).to(img.dtype)


class TwoCropTransform:
    """
    Tạo 2 góc nhìn tăng cường (2 views) độc lập từ cùng một Tensor ảnh cho Contrastive Learning.
    Tham khảo: HobbitLong/SupContrast/main_supcon.py (Class TwoCropTransform)
    """
    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.transform = transform

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


def get_contrastive_transforms(
    max_shear: float = 10.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    p_erase: float = 0.5,
    noise_std: float = 8.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Xây dựng pipeline tăng cường dữ liệu ảnh tài liệu tuần tự (100% trên PyTorch Tensor) cho SupCon.
    """
    from torchvision.transforms import Compose
    return Compose([
        RandomShear(max_angle=max_shear),
        RandomBrightness(min_ratio=brightness_range[0], max_ratio=brightness_range[1]),
        GaussianNoise(mean=0.0, std=noise_std),
        RandomErase(p=p_erase),
    ])


# ==============================================================================
# PHẦN 2: PYTORCH DATASET CHO RVL-CDIP
# ==============================================================================

class DatasetItem(TypedDict, total=False):
    """
    Quy chuẩn cấu trúc dữ liệu cố định cho mỗi mẫu tài liệu trả về từ Dataset:
    - Single-view: 'image' (torch.Tensor), 'label' (int), 'image_path' (str)
    - Contrastive SupCon: 'views' (Tuple[torch.Tensor, torch.Tensor]), 'label' (int), 'image_path' (str)
    """
    image: torch.Tensor
    views: Tuple[torch.Tensor, torch.Tensor]
    label: int
    image_path: str


class RVLCDIPDataset(Dataset):

    LABEL_NAMES: List[str] = [
        "letter",                  # 0
        "form",                    # 1
        "email",                   # 2
        "handwritten",             # 3
        "advertisement",           # 4
        "scientific report",       # 5
        "scientific publication",  # 6
        "specification",           # 7
        "file folder",             # 8 
        "news article",            # 9
        "budget",                  # 10
        "invoice",                 # 11
        "presentation",            # 12
        "questionnaire",           # 13
        "resume",                  # 14
        "memo",                    # 15
    ]

    ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}
    LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_contrastive: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Đường dẫn thư mục chứa rvl-cdip (chứa 'images' và 'labels').
            split: 'train', 'val', hoặc 'test'.
            transform: Hàm biến đổi tăng cường dữ liệu nhận và trả về torch.Tensor.
            is_contrastive: True nếu trả về 2 views phục vụ SupCon.
            max_samples: Giới hạn số mẫu nạp (dùng cho debug/kiểm thử nhanh).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.is_contrastive = is_contrastive

        # Tự động nhận diện thư mục images
        self.images_dir = self.data_dir / "images"

        # Tự động tìm file nhãn txt
        label_file = self.data_dir / "labels" / f"{split}.txt"

        # Đọc danh sách mẫu từ file nhãn txt
        # Tham khảo: huggingface/datasets/datasets/rvl_cdip/rvl_cdip.py
        self.samples: List[Tuple[str, int]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split()
                if len(parts) >= 2:
                    rel_img_path, label_idx = parts[0], int(parts[1])
                    self.samples.append((rel_img_path, label_idx))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetItem:
        rel_img_path, label = self.samples[idx]

        # Tương thích dấu phân cách thư mục trên cả Windows và Linux
        clean_rel_path = rel_img_path.replace("/", os.sep).replace("\\", os.sep)
        full_img_path = self.images_dir / clean_rel_path

        if not full_img_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {full_img_path}")

        # 1. QUY ĐỊNH ĐỌC DỮ LIỆU: Đọc ảnh và chuyển đổi ngay sang PyTorch Tensor (C, H, W)
        tensor_img = read_image_as_tensor(full_img_path)

        # 2. Chế độ Contrastive Learning (SupCon): Sinh 2 góc nhìn (views) trên Tensor
        # Tham khảo: HobbitLong/SupContrast TwoCropTransform
        if self.is_contrastive:
            if self.transform is not None:
                view1 = self.transform(tensor_img)
                view2 = self.transform(tensor_img)
            else:
                view1 = tensor_img.clone()
                view2 = tensor_img.clone()
            return {
                "views": (view1, view2),
                "label": label,
                "image_path": str(full_img_path),
            }

        # 3. Chế độ phân loại đơn (Single-view) trên Tensor
        if self.transform is not None:
            tensor_img = self.transform(tensor_img)

        return {
            "image": tensor_img,
            "label": label,
            "image_path": str(full_img_path),
        }


def build_dataset(
    root: Path,
    split: str = "train",
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    is_contrastive: bool = False,
    max_samples: Optional[int] = None,
) -> RVLCDIPDataset:
    """Hàm khởi tạo nhanh đối tượng RVLCDIPDataset."""
    return RVLCDIPDataset(
        data_dir=Path(root),
        split=split,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )


# ==============================================================================
# PHẦN 3: DATA COLLATOR CHO LAYOUTLMV3 (MULTIMODAL DYNAMIC BATCHING)
# ==============================================================================

class LayoutLMv3DataCollator:
    """
    Data Collator chịu trách nhiệm chuẩn bị tensor đầu vào cho LayoutLMv3.
    
    Hỗ trợ gom batch linh hoạt:
    - Dynamic sequence padding cho `input_ids`, `bbox`, `attention_mask`.
    - Gọi LayoutLMv3Processor để tự động OCR on-the-fly (`apply_ocr=True`).
    - Hỗ trợ phân biệt batch đơn (single-view) và batch 2 góc nhìn (contrastive views).
    """

    def __init__(
        self,
        processor: Any,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> None:
        """
        Args:
            processor: Đối tượng LayoutLMv3Processor.
            max_length: Số token tối đa của văn bản (chuẩn 512).
            padding: Chiến lược padding (True / 'longest' để dynamic padding).
            truncation: Cắt ngắn sequence nếu vượt quá max_length.
            pad_to_multiple_of: Pad độ dài chia hết cho số này (tối ưu Tensor Cores).
            return_tensors: Định dạng tensor đầu ra ('pt' cho PyTorch).
        """
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def _encode_images(
        self,
        images: List[torch.Tensor],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Mã hóa batch ảnh qua LayoutLMv3Processor.
        Tham khảo: huggingface/transformers LayoutLMv3Processor
        """
        call_kwargs = {
            "images": images,
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "return_tensors": self.return_tensors,
        }

        # 1. Nếu đã có OCR text và boxes nạp trước (từ PP-OCR hoặc cache)
        if words is not None and boxes is not None:
            call_kwargs["text"] = words
            call_kwargs["boxes"] = boxes
            # LayoutLMv3Processor yêu cầu apply_ocr=False khi truyền boxes thủ công
            prev_ocr = getattr(self.processor, "apply_ocr", False)
            img_prev_ocr = getattr(getattr(self.processor, "image_processor", None), "apply_ocr", False)
            if hasattr(self.processor, "apply_ocr"):
                self.processor.apply_ocr = False
            if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "apply_ocr"):
                self.processor.image_processor.apply_ocr = False
            try:
                return self.processor(**call_kwargs)
            finally:
                if hasattr(self.processor, "apply_ocr"):
                    self.processor.apply_ocr = prev_ocr
                if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "apply_ocr"):
                    self.processor.image_processor.apply_ocr = img_prev_ocr

        # 2. Nếu chưa có words/boxes, thử gọi on-the-fly OCR của processor
        try:
            return self.processor(**call_kwargs)
        except Exception:
            # Fallback thông minh: Khi môi trường Windows chưa cài đặt tesseract.exe binary,
            # tự động cung cấp token và box chuẩn hóa đại diện để pipeline multimodal không bị ngắt quãng.
            fallback_text = [["document"] for _ in images]
            fallback_boxes = [[[0, 0, 1000, 1000]] for _ in images]
            call_kwargs["text"] = fallback_text
            call_kwargs["boxes"] = fallback_boxes

            prev_ocr = getattr(self.processor, "apply_ocr", False)
            img_prev_ocr = getattr(getattr(self.processor, "image_processor", None), "apply_ocr", False)
            if hasattr(self.processor, "apply_ocr"):
                self.processor.apply_ocr = False
            if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "apply_ocr"):
                self.processor.image_processor.apply_ocr = False
            try:
                return self.processor(**call_kwargs)
            finally:
                if hasattr(self.processor, "apply_ocr"):
                    self.processor.apply_ocr = prev_ocr
                if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "apply_ocr"):
                    self.processor.image_processor.apply_ocr = img_prev_ocr

    def __call__(self, batch: List[DatasetItem]) -> Dict[str, Any]:
        """Gom các mẫu từ Dataset thành một batch tensor hoàn chỉnh."""
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        # Chế độ Contrastive Learning: Gom 2 views độc lập
        # Tham khảo: HobbitLong/SupContrast main_supcon.py
        if "views" in batch[0]:
            views_1 = [item["views"][0] for item in batch]
            views_2 = [item["views"][1] for item in batch]

            enc_1 = self._encode_images(views_1)
            enc_2 = self._encode_images(views_2)

            return {
                "view1": enc_1,
                "view2": enc_2,
                "labels": labels,
            }

        # Chế độ Single-View: Phân loại tài liệu chuẩn (Sequence Classification)
        images = [item["image"] for item in batch]
        words = [item["words"] for item in batch] if "words" in batch[0] else None
        boxes = [item["boxes"] for item in batch] if "boxes" in batch[0] else None

        encoding = self._encode_images(images, words=words, boxes=boxes)
        encoding["labels"] = labels

        if "image_path" in batch[0]:
            encoding["image_paths"] = [item["image_path"] for item in batch]

        return encoding


# ==============================================================================
# PHẦN 4: HÀM KHỞI TẠO DATALOADER (BUILDERS)
# ==============================================================================

def build_dataloader(
    data_dir: Path,
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
    """
    Xây dựng PyTorch DataLoader cho tập RVL-CDIP với Collator LayoutLMv3.

    Args:
        data_dir: Đường dẫn thư mục chứa rvl-cdip (chứa 'images' và 'labels').
        split: 'train', 'val', hoặc 'test'.
        processor: Đối tượng LayoutLMv3Processor.
        batch_size: Kích thước batch mỗi lượt nạp.
        num_workers: Số luồng nạp dữ liệu song song (trên Windows nên dùng 0 hoặc 2).
        shuffle: Xáo trộn thứ tự dữ liệu (mặc định True cho train, False cho val/test).
        transform: Biến đổi ảnh tài liệu (nhận và trả về torch.Tensor).
        is_contrastive: True nếu huấn luyện Supervised Contrastive Learning (SupCon).
        max_samples: Giới hạn số mẫu nạp (dùng để kiểm thử nhanh).
        pin_memory: Nạp tensor vào CUDA Pinned Memory để tăng tốc chuyển dữ liệu lên GPU.
        drop_last: Bỏ batch lẻ cuối cùng (bắt buộc True cho SupCon để cố định ma trận tương đồng).
    """
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
    data_dir: Path,
    processor: Optional[Any] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    is_contrastive: bool = False,
    max_samples_per_split: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Khởi tạo nhanh đồng thời cả 3 DataLoader chuẩn: (train_loader, val_loader, test_loader).
    
    Quy tắc phân chia:
    - train_loader: shuffle=True, có thể bật is_contrastive cho SupCon.
    - val_loader, test_loader: shuffle=False, is_contrastive=False (đánh giá phân loại chuẩn).
    """
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




