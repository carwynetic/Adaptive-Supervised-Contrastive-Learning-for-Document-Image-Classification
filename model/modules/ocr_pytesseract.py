"""
Baseline OCR Module using Tesseract
input : PIL.Image, torch.Tensor, str or Path
output :
    - words : List[str] - list of regconized words
    - boxes : List[List[int]] - normlized bounding boxes [x0, y0, x1, y1] in [0,1000]
"""

import os 
from pathlib import Path
from typing import List , Tuple, Union, Optional

import numpy as np 
from PIL import Image
import torch
from torchvision.transforms import functional as TF

import pytesseract
from pytesseract import Output

class TesseractOCR:
    def __init__ (
        self, 
        tesseract_cmd   : Optional[str] = None,
        lang            : str = "eng",
        config          : str = "",
        device          : Optional[Union[str, torch.device]] = None,
    )-> None:
        cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")

        if not cmd and os.name == "nt":
            win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(win_default):
               cmd = win_default
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        
        self.lang = lang
        self.config = config
        self.device = torch.device(device) if device else torch.device("cpu")

    
    @staticmethod
    def _to_pil_image(image : Union[Image.Image, torch.Tensor, str, Path]) -> Image.Image:
        if isinstance(image, (str,Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            if image.ndim==3:
                return TF.to_pil_image(image).convert("RGB")
            if image.ndim ==4 and image.shape[0]==1:
                return TF.to_pil_image(image.squeeze(0)).convert("RGB")
            raise ValueError(f"Unsupported Tensor dimensions: {image.shape}")
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise TypeError(f"Unsupported image input type : {type(image)}")

    @staticmethod
    def normalize_box (box : List[int] , width : int, height:int) -> List[int]:
        # normalize box to [0,1000]
        return [
            min(max(0, int(1000 * (box[0] / width))), 1000),
            min(max(0, int(1000 * (box[1] / height))), 1000),
            min(max(0, int(1000 * (box[2] / width))), 1000),
            min(max(0, int(1000 * (box[3] / height))), 1000),
        ]
    
    def extract_words_and_boxes(
        self, image: Union[Image.Image, torch.Tensor, str, Path]
    )-> Tuple[List[str], List[List[int]]]:
        """
        extract words and spatial bounding boxes (following the baseline of layoutlmv3)
        """
        pil_img = self._to_pil_image(image)
        image_width, image_height = pil_img.size

        #1. run tesseract to extract word-level tokens and bounding coordinates
        data = pytesseract.image_to_data(
            pil_img, 
            lang=self.lang,
            output_type = Output.DICT,
            config = self.config
        )
        words_raw   = data['text']    # even white space
        left        = data['left']    # x0
        top         = data['top']     # y0
        width       = data['width']
        height      = data['height']
        # x1​=left+width
        # y1​=top+height​

        #2. filter out whitespace-only strings, retain synchronized coordinates
        valid_indices   = [idx for idx, word in enumerate(words_raw) if word.strip()]
        words   = [words_raw[idx] for idx in valid_indices]
        left    = [left[idx] for idx in valid_indices]
        top     = [top[idx] for idx in valid_indices]
        width   = [width[idx] for idx in valid_indices]
        height  = [height[idx]for idx in valid_indices]

        #3. construct 2d bouding boxes in (x0, y0, x1, y1) format
        actual_boxes = [
            [x,y, x+w, y+h] 
            for x,y,w,h in zip(left, top, width, height)
        ]

        #4. normalize coordinates into the range [0,1000]
        normalized_boxes = [
            self.normalize_box(box, image_width, image_height)
            for box in actual_boxes
        ]

        #fall back : output's normalizing is  wrong shape
        assert len(words) == len(normalized_boxes), (
            f"Shape mismatch: {len(words)} words vs {len(normalized_boxes)} boxes"
        )

        #5. deterministic fallback for blank pages or failed character detection
        if not words:
            words = ["document has no words/ error"]
            normalized_boxes = [[0,0,1000,1000]]
        return words, normalized_boxes
    
    def __call__(
        self, image: Union[Image.Image, torch.Tensor, str, Path]
    )-> Tuple[List[str], List[List[int]]]:
        return self.extract_words_and_boxes(image)

if __name__ == "__main__":
  try:
    ocr_engine = TesseractOCR()
    demo_image_path = "nhap/2072197187.tif"
    out_words, out_boxes = ocr_engine(demo_image_path)

    print(
        f"\n=== TỔNG CỘNG: {len(out_words)} TỪ | {len(out_boxes)} BOUNDING"
        " BOXES ==="
    )

    # 1. In toàn bộ văn bản ghép lại thành đoạn hoàn chỉnh
    print("\n--- TOÀN BỘ VĂN BẢN TRÍCH XUẤT ĐƯỢC ---")
    print(" ".join(out_words))

    # 2. In chi tiết từng từ đi kèm bounding box [x0, y0, x1, y1]
    print("\n--- CHI TIẾT TỪNG TỪ KÈM TOẠ ĐỘ BOX [x0, y0, x1, y1] ---")
    for idx, (w, b) in enumerate(zip(out_words, out_boxes), start=1):
      print(f"[{idx:03d}] {w:<25} -> Box: {b}")

    # 3. Hoặc nếu bạn muốn xem mộc cả 2 list nguyên bản
    print("\n--- RAW WORDS LIST ---")
    print(out_words)
    print("\n--- RAW BOXES LIST ---")
    print(out_boxes)

  except Exception as exc:
    print(f"Tesseract Execution Note: {exc}")
