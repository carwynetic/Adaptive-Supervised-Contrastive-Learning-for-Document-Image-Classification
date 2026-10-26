# ==============================================================================
# DATASET: RVL-CDIP Dataset cho LayoutLMv3 & Supervised Contrastive Learning (SupCon)
# ==============================================================================
# DANH SÁCH CHI TIẾT CÁC NGUỒN MÃ NGUỒN THAM KHẢO (EVIDENCE & CITATIONS):
#
# 1. Danh sách 16 lớp nhãn RVL-CDIP (LABEL_NAMES):
#    - Repository: huggingface/datasets
#    - File nguồn: datasets/rvl_cdip/rvl_cdip.py
#    - Link trực tiếp: https://github.com/huggingface/datasets/blob/main/datasets/rvl_cdip/rvl_cdip.py
#    - Dòng tham chiếu: Biến `_CLASSES = [...]` chứa đúng 16 class từ index 0 đến 15.
#    - Bài báo gốc: Harley et al., "Evaluation of Deep Convolutional Nets for Document Image
#      Classification and Retrieval", ICDAR 2015 (https://www.cs.cmu.edu/~aharley/rvl-cdip/)
#
# 2. Logic đọc và parse file nhãn (train.txt / val.txt / test.txt):
#    - Repository: huggingface/datasets
#    - File nguồn: datasets/rvl_cdip/rvl_cdip.py
#    - Link trực tiếp: https://github.com/huggingface/datasets/blob/main/datasets/rvl_cdip/rvl_cdip.py
#    - Hàm tham chiếu: `_generate_examples(self, filepath, images_dir)`
#    - Đoạn code gốc:
#          with open(filepath, encoding="utf-8") as f:
#              for line in f:
#                  line = line.strip()
#                  image_path, label = line.split(" ")
#
# 3. Đọc ảnh TIFF tài liệu và chuyển sang RGB cho Transformer:
#    - Repository: NielsRogge/Transformers-Tutorials
#    - File nguồn: LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
#    - Link trực tiếp: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
#    - Cell tham chiếu: Cell 6 & Cell 16 (`Image.open(path).convert("RGB")`)
#    - Lý do: Ảnh RVL-CDIP gốc là ảnh đen trắng 1 kênh (grayscale TIFF), trong khi Vision
#      Backbone của LayoutLMv3 nhận đầu vào là tensor chuẩn 3 kênh RGB.
#
# 4. Cơ chế sinh 2 view ảnh độc lập (Two-Crop Transform) cho SupCon:
#    - Repository: HobbitLong/SupContrast (Tác giả: Yonglong Tian - NeurIPS 2020)
#    - File nguồn: main_supcon.py
#    - Link trực tiếp: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
#    - Class tham chiếu: `TwoCropTransform`
#    - Đoạn code gốc:
#          class TwoCropTransform:
#              def __call__(self, x):
#                  return [self.transform(x), self.transform(x)]
# ==============================================================================

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset


class RVLCDIPDataset(Dataset):
    """
    Dataset loader cho tập RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing).

    Cấu trúc thư mục dữ liệu tại workspace:
        data/rvl-cdip/
            images/
                imagesa/..., imagesb/..., ..., imagesz/
            labels/
                train.txt, val.txt, test.txt
                Mỗi dòng có dạng: 'imagesq/q/o/c/qoc54c00/80035521.tif 15'
    """

    # --------------------------------------------------------------------------
    # Tham khảo: huggingface/datasets/datasets/rvl_cdip/rvl_cdip.py (biến _CLASSES)
    # Link: https://github.com/huggingface/datasets/blob/main/datasets/rvl_cdip/rvl_cdip.py
    # --------------------------------------------------------------------------
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

    ID2LABEL : Dict[int, str]={
        "0": "letter",
        "1": "form",
        "2": "email",
        "3": "handwritten",
        "4": "advertisement",
        "5": "scientific report",
        "6": "scientific publication",
        "7": "specification",
        "8": "file folder",
        "9": "news article",
        "10": "budget",
        "11": "invoice",
        "12": "presentation",
        "13": "questionnaire",
        "14": "resume",
        "15": "memo"
    }
            
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        is_contrastive: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_dir: Đường dẫn thư mục gốc RVL-CDIP (chứa thư mục 'images' và 'labels').
            split: Tập phân chia ('train', 'val', hoặc 'test').
            transform: Hàm biến đổi dữ liệu (từ data_processing.augmentations hoặc torchvision).
            is_contrastive: True nếu trả về 2 views cho Supervised Contrastive Learning (SupCon).
            max_samples: Giới hạn số lượng mẫu để kiểm tra nhanh / debug.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.split = split
        self.transform = transform
        self.is_contrastive = is_contrastive

        # Tìm file danh sách nhãn tương ứng với split
        label_file = self.data_dir / "labels" / f"{split}.txt"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        # ----------------------------------------------------------------------
        # Đọc danh sách mẫu từ file nhãn txt
        # Tham khảo: logic đọc _generate_examples trong huggingface/datasets/datasets/rvl_cdip/rvl_cdip.py
        # Link: https://github.com/huggingface/datasets/blob/main/datasets/rvl_cdip/rvl_cdip.py
        # ----------------------------------------------------------------------
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

    def __getitem__(self, idx: int) -> Dict[str, Union[Image.Image, int, str, Tuple]]:
        rel_img_path, label = self.samples[idx]

        # Chuẩn hóa dấu phân cách thư mục tương thích cả Windows ('\') và Linux ('/')
        clean_rel_path = rel_img_path.replace("/", os.sep).replace("\\", os.sep)
        full_img_path = self.images_dir / clean_rel_path

        if not full_img_path.exists():
            raise FileNotFoundError(f"Image not found: {full_img_path}")

        # ----------------------------------------------------------------------
        # Mở ảnh TIFF và chuyển sang RGB 3 kênh
        # Tham khảo: NielsRogge/Transformers-Tutorials (Cell 6 & Cell 16)
        # Link: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
        # ----------------------------------------------------------------------
        image = Image.open(full_img_path).convert("RGB")

        # ----------------------------------------------------------------------
        # Chế độ Contrastive Learning (SupCon): Sinh 2 góc nhìn tăng cường (TwoCropTransform)
        # Tham khảo: HobbitLong/SupContrast/main_supcon.py (Class TwoCropTransform)
        # Link: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
        # ----------------------------------------------------------------------
        if self.is_contrastive:
            if self.transform is not None:
                view1 = self.transform(image)
                view2 = self.transform(image)
            else:
                view1 = image.copy()
                view2 = image.copy()
            return {
                "views": (view1, view2),
                "label": label,
                "image_path": str(full_img_path),
            }

        # Chế độ Single-View: Huấn luyện phân loại hoặc đánh giá (Evaluation)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "image_path": str(full_img_path),
        }


def build_dataset(
    root: Union[str, Path],
    split: str = "train",
    transform: Optional[Callable] = None,
    is_contrastive: bool = False,
    max_samples: Optional[int] = None,
) -> RVLCDIPDataset:
    """Hàm khởi tạo nhanh RVLCDIPDataset."""
    return RVLCDIPDataset(
        data_dir=root,
        split=split,
        transform=transform,
        is_contrastive=is_contrastive,
        max_samples=max_samples,
    )