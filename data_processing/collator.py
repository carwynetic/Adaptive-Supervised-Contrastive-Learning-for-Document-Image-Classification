# ==============================================================================
# COLLATOR: Data Collator cho LayoutLMv3 & Supervised Contrastive Learning (SupCon)
# ==============================================================================
# DANH SÁCH CHI TIẾT CÁC NGUỒN MÃ NGUỒN THAM KHẢO (EVIDENCE & CITATIONS):
#
# 1. Cơ chế gom batch và Dynamic Padding cho mô hình LayoutLM:
#    - Repository: microsoft/unilm
#    - File nguồn: layoutlmft/layoutlmft/data/data_collator.py
#    - Link trực tiếp: https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/data/data_collator.py
#    - Class tham chiếu: `DataCollatorForKeyValueExtraction`
#    - Dòng tham chiếu: Đoạn pad dynamic cho `input_ids`, `bbox`, `attention_mask` theo độ dài
#      lớn nhất của batch hiện tại thay vì pad cứng cố định:
#          batch = self.tokenizer.pad(
#              features,
#              padding=self.padding,
#              max_length=self.max_length,
#              pad_to_multiple_of=self.pad_to_multiple_of,
#              return_tensors="pt",
#          )
#          batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
#
# 2. Xử lý Multimodal (Ảnh + Text + Bounding Box) và On-the-fly OCR của LayoutLMv3:
#    - Repository: huggingface/transformers
#    - File nguồn: src/transformers/models/layoutlmv3/processing_layoutlmv3.py
#    - Link trực tiếp: https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/processing_layoutlmv3.py
#    - Method tham chiếu: `LayoutLMv3Processor.__call__(self, images, text=None, boxes=None, ...)`
#    - Logic:
#          features = self.image_processor(images=images, return_tensors=return_tensors)
#          encoded_inputs = self.tokenizer(text=..., boxes=..., padding=..., truncation=...)
#          encoded_inputs["pixel_values"] = features.pop("pixel_values")
#
# 3. Thuật toán chạy OCR và chuẩn hóa bounding box trên từng ảnh (Khi apply_ocr=True):
#    - Repository: huggingface/transformers
#    - File nguồn: src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py
#    - Link trực tiếp: https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py
#    - Hàm tham chiếu: `apply_tesseract(...)` và `normalize_box(...)` (chuẩn hóa box theo thang đo 0 - 1000)
#
# 4. Gom batch 2 views cho Supervised Contrastive Learning (SupCon):
#    - Repository: HobbitLong/SupContrast
#    - File nguồn: main_supcon.py
#    - Link trực tiếp: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
#    - Dòng tham chiếu: Vòng lặp huấn luyện `train(...)`, nhận diện dữ liệu đầu vào gồm 2 views
#      `images = torch.cat([images[0], images[1]], dim=0)` để tính SupCon Loss giữa các cặp cùng nhãn.
# ==============================================================================

from typing import Any, Dict, List, Optional, Union
import torch


class LayoutLMv3DataCollator:
    """
    Data Collator chịu trách nhiệm chuẩn bị tensor đầu vào cho LayoutLMv3.
    
    Hỗ trợ 2 chế độ:
    1. Chế độ phân loại (Single-view):
       - Nhận ảnh thô -> Gọi LayoutLMv3Processor để OCR on-the-fly -> dynamic pad sequence.
    2. Chế độ Contrastive Learning (Two-view SupCon):
       - Nhận 2 view tăng cường (view1, view2) -> Encode độc lập thành 2 batch tensors để tính loss.
    """

    def __init__(
        self,
        processor: Any,
        max_length: Optional[int] = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> None:
        """
        Args:
            processor: Đối tượng LayoutLMv3Processor (nạp từ checkpoints/layoutlmv3-base).
            max_length: Giới hạn số token tối đa (mặc định 512).
            padding: Chiến lược padding (True / 'longest' để dynamic padding theo batch).
                     Tham khảo: microsoft/unilm DataCollatorForKeyValueExtraction.
            truncation: Cắt ngắn sequence nếu vượt quá max_length.
            pad_to_multiple_of: Pad độ dài chia hết cho số này (tối ưu hóa Tensor Core GPU).
            return_tensors: Định dạng tensor trả về ('pt' cho PyTorch).
        """
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def _encode_images(
        self,
        images: List[Any],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Mã hóa batch ảnh qua LayoutLMv3Processor.
        
        Tham khảo: huggingface/transformers/src/transformers/models/layoutlmv3/processing_layoutlmv3.py
        Link: https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/processing_layoutlmv3.py
        
        - Nếu 'words' và 'boxes' là None: processor (với apply_ocr=True) sẽ tự động chạy
          apply_tesseract trên từng ảnh để lấy text và tọa độ hộp box.
        - Trả về dictionary gồm: pixel_values (B, 3, 224, 224), input_ids (B, L),
          attention_mask (B, L), bbox (B, L, 4).
        """
        call_kwargs = {
            "images": images,
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "return_tensors": self.return_tensors,
        }

        # Nếu đã có OCR text và boxes được cung cấp trước (ví dụ từ PP-OCR)
        if words is not None and boxes is not None:
            call_kwargs["text"] = words
            call_kwargs["boxes"] = boxes

        encoded = self.processor(**call_kwargs)
        return encoded

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gom các mẫu từ Dataset thành một batch hoàn chỉnh.
        
        Tham khảo cấu trúc batching của:
        - microsoft/unilm/layoutlmft/layoutlmft/data/data_collator.py
        - HobbitLong/SupContrast/main_supcon.py
        """
        # Lọc bỏ các mẫu None nếu có lỗi khi đọc file
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        # ----------------------------------------------------------------------
        # Chế độ Contrastive Learning: Hai góc nhìn tăng cường (view1, view2)
        # Tham khảo: HobbitLong/SupContrast/main_supcon.py (phần input của SupConLoss)
        # Link: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # Chế độ Single-View: Phân loại tài liệu chuẩn (Sequence Classification)
        # Tham khảo: NielsRogge/Transformers-Tutorials (Cell 18)
        # Link: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
        # ----------------------------------------------------------------------
        images = [item["image"] for item in batch]
        words = [item["words"] for item in batch] if "words" in batch[0] else None
        boxes = [item["boxes"] for item in batch] if "boxes" in batch[0] else None

        encoding = self._encode_images(images, words=words, boxes=boxes)
        encoding["labels"] = labels

        if "image_path" in batch[0]:
            encoding["image_paths"] = [item["image_path"] for item in batch]

        return encoding
