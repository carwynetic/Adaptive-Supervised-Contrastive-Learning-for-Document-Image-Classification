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
        cmd = tesseract_cmd or os.getevn("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        
        self.lang = lang
        self.config = config
        self.device = "cuda" if torch.device.is_avaible() else "cpu"

