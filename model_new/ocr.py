"""
Module PP-OCRv6 module replacing the Microsoft Read API 
which used in the Original LayoutLMv3 Paper.
Using a local of PP-OCRv6 Det and Rec Modules
Input : 
    - image : PIL.image, torch.Tensor or image path
Output:
    - words = list[str]  : list of recognized words
    - boxes = List[list[int]] : list of bounding boxes, [x0,y0,x1,y1] normalized to [0,1000]
"""
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np 
from PIL import Image
from scipy.ndimage import find_objects, label
import torch
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor

# Import kien truc PP-OCRv6 thuc te tu third_party trong repo
from third_party.pp_ocrv6_small_det.configuration_pp_ocrv6_small_det import (
    PPOCRV6SmallDetConfig,
)
from third_party.pp_ocrv6_small_det.modeling_pp_ocrv6_small_det import (
    PPOCRV6SmallDetForObjectDetection,
)
from third_party.pp_ocrv6_small_rec.configuration_pp_ocrv6_small_rec import (
    PPOCRV6SmallRecConfig,
)
from third_party.pp_ocrv6_small_rec.image_processing_pp_ocrv6_small_rec import (
    PPOCRV6SmallRecImageProcessor,
)
from third_party.pp_ocrv6_small_rec.modeling_pp_ocrv6_small_rec import (
    PPOCRV6SmallRecForTextRecognition,
)

class DocumentOCR:
    def __init__(
        self,
        det_config_dir      : Optional[Union[str, Path]] = None,
        det_checkpoint_dir  : Optional[Union[str, Path]] = None,
        rec_config_dir      : Optional[Union[str, Path]] = None,
        rec_checkpoint_dir  : Optional[Union[str, Path]] = None,
        det_threshold       : float = 0.3,
        min_box_size        : int = 4,
        device              : torch.device | None  = None,
    ) -> None:
        """
        Args:
            det_config_dir      : folder configs/pp_ocrv6_small_det chua config.json va preprocessor_config.json
            det_checkpoint_dir  : folder checkpoints/pp_ocrv6_small_det chua model.safetensors
            rec_config_dir      : folder configs/pp_ocrv6_small_rec chua config.json va preprocessor_config.json
            rec_checkpoint_dir  : folder checkpoints/pp_ocrv6_small_rec chua model.safetensors
            det_threshold       : Nguong xac suat binarize ban do nhiet DBNet (default= 0.3)
            min_box_size        : Kich thuoc canh hop nho nhat de loai bo hat nhieu (mac dinh 4px)
            device              : Thiet bi tinh toan ('cpu' hoac 'cuda')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = torch.device(device)

        self.det_threshold  = det_threshold
        self.min_box_size   = min_box_size
        
        root_dir = Path(__file__).resolve().