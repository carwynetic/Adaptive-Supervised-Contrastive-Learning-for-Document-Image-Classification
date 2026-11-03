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
from multiprocessing.sharedctypes import Value
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
        if isinstance(image, torch.Ternsor):
            if image.ndim==3:
                return TF.to_pil_image(image).convert("RGB")
            if image.ndim==4 and image.shape[0]==1:
                return TF.to_pil_image(image.squeeze(0).convert("RGB"))
            raise ValueError(f"Tensor shape is not valid: {image.shape}")
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise TypeError(f"Type of Image do not supported : {type(image)}")
    
    @staticmethod
    def normalize_box(box: List[int], width: int, height:int) -> List[int]:
        x0,y0,x1,y1 = box
        norm_x0 = max(0, min(1000, int(1000 * (x0 / width))))
        norm_y0 = max(0, min(1000, int(1000 * (y0 / height))))
        norm_x1 = max(0, min(1000, int(1000 * (x1 / width))))
        norm_y1 = max(0, min(1000, int(1000 * (y1 / height))))
        return [norm_x0, norm_y0, norm_x1, norm_y1]
    
    def _extract_boxes_from_prob_map(
        self, 
        prob_map :torch.Tensor, 
        orig_width:int, orig_height:int)-> List[List[int]]:
        if prob_map.ndim==4:
            map_arr = prob_map[0,0].detach().cpu().float().numpy()
        elif prob_map.ndim==3:
            map_arr = prob_map[0].detach().cpu().float().numpy()
        else:
            map_arr = prob_map.detach().cpu().float().numpy()
        
        map_h, map_w = map_arr.shape
        binary_mask = map_arr > self.det_threshold
        
        
        #label() từ thư viện scipy.ndimage
        # hõ trợ gán nhãn connectted component nhanh
        # áp dụng điều kiện lọc  là binary_mask đã được định nghĩa ở trên
        # labeled_mask trả về mảng 2 chiều  
        #   - pixel = 0 : background
        #   - pixel = 1,2,3... id of components
        # _ : number of component, we don't use it 
        labeled_mask, _ = label(binary_mask)
        




