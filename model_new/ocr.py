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
from threading import local
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
        
        #xác định thư mục gốc của project
        root_dir = Path(__file__).resolve().parents[1]
        self.det_config_dir = Path(det_config_dir) if not det_config_dir else root_dir/'config'/ "pp_ocrv6_small_det"
        self.rec_config_dir = Path(rec_config_dir) if not rec_config_dir else root_dir/'config'/ "pp_ocrv6_small_rec"
        self.det_checkpoint_dir = Path(det_checkpoint_dir) if det_checkpoint_dir else root_dir/"checkpoints" / "pp_ocrv6_small_det"
        self.rec_checkpoint_dir = Path(rec_checkpoint_dir) if rec_checkpoint_dir else root_dir/"checkpoints" / "pp_ocrv6_small_rec"

        #Intialize Detection
        self.det_processor = AutoImageProcessor.from_pretrained(
            self.det_config_dir, local_files_only=True
        )
        det_config = PPOCRV6SmallDetConfig.from_pretrained(
            self.det_config_dir, local_files_only=True
        )
        self.det_model = PPOCRV6SmallDetForObjectDetection.from_pretrained(
            self.det_checkpoint_dir, config = det_config, local_files_only=True
        ).to(self.device)
        self.det_model.eval()

        #Intialize Recognition
        self.rec_processor = AutoImageProcessor.from_pretrained(
            self.rec_config_dir, local_files_only =True
        )
        rec_config  = PPOCRV6SmallRecConfig.from_pretrained(
            self.rec_config_dir, local_files_only=True
        )
        self.rec_model = PPOCRV6SmallRecForTextRecognition.from_pretrained(
            self.rec_checkpoint_dir, config = rec_config, local_files_only=True
        )

    #adapter transfer image from multi-sources (tensor, .tif, pil.image, path) to PIL.image (RGB)   
    @staticmethod
    def _to_pil_image(image : Union[Image.Image, torch.Tensor, str, Path]) -> Image.Image:
        if isinstance(image, (str,Path)): #variable imae is str OR path?
            return Image.open(image).convert("RGB")


