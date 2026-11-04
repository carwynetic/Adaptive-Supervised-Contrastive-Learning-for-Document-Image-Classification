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
from transformers.models.nougat.tokenization_nougat import normalize_list_like_lines
from networkx import normalized_cut_size
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
        self.det_config_dir = Path(det_config_dir) if det_config_dir else root_dir/'configs'/ "pp_ocrv6_small_det"
        self.rec_config_dir = Path(rec_config_dir) if rec_config_dir else root_dir/'configs'/ "pp_ocrv6_small_rec"
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
        self.rec_processor = PPOCRV6SmallRecImageProcessor.from_pretrained(
            self.rec_config_dir, local_files_only =True
        )
        rec_config  = PPOCRV6SmallRecConfig.from_pretrained(
            self.rec_config_dir, local_files_only=True
        )
        self.rec_model = PPOCRV6SmallRecForTextRecognition.from_pretrained(
            self.rec_checkpoint_dir, config = rec_config, local_files_only=True
        ).to(self.device)
        self.rec_model.eval()

    #adapter transfer image from multi-sources (tensor, .tif, pil.image, path) to PIL.image (RGB)   
    @staticmethod
    def _to_pil_image(image : Union[Image.Image, torch.Tensor, str, Path]) -> Image.Image:
        if isinstance(image, (str,Path)): #variable imae is str OR path?
            return Image.open(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            if image.ndim==3:
                return TF.to_pil_image(image).convert("RGB")
            if image.ndim==4 and image.shape[0]==1:
                return TF.to_pil_image(image.squeeze(0)).convert("RGB")
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
        slices = find_objects(labeled_mask)

        raw_boxes =[]
        scale_x = orig_width /float(map_w)
        scale_y = orig_height / float(map_h)

        for s in slices:
            if s is None:
                continue
            y_slice, x_slice = s
            box_w = x_slice.stop - x_slice.start
            box_h = y_slice.stop - y_slice.start

            if box_w < self.min_box_size or box_h <self.min_box_size:
                continue
            #avoiding detect chinese word
            if (box_w / max(1, box_h) > 20.0) or (box_h / max(1, box_w) > 20.0): 
                continue

            
            #add padding
            #mo rong chieu cao 2 huong , avoid slicing words
            pad_h = int(box_h * 0.7) + 3 
            pad_w = int(box_w * 0.2 ) +2

            y0_map = max(0, y_slice.start - pad_h)
            y1_map = min(map_h, y_slice.stop + pad_h)
            x0_map = max(0, x_slice.start - pad_w)
            x1_map = min(map_w, x_slice.stop + pad_w)

            x0 = int(x0_map * scale_x)
            y0 = int(y0_map * scale_y)
            x1 = int(x1_map * scale_x)
            y1 = int(y1_map * scale_y)

            x0 = max(0, min(orig_width, x0))
            y0 = max(0, min(orig_height, y0))
            x1 = max(x0 + 1, min(orig_width, x1))
            y1 = max(y0 + 1, min(orig_height, y1))

            raw_boxes.append([x0, y0, x1, y1])
        raw_boxes.sort(key = lambda b : (b[1], b[0]))
        return raw_boxes


    @torch.inference_mode()
    def extract_words_and_boxes(
        self, image : Union[Image.Image, torch.Tensor, str, Path]
    )-> Tuple[List[str], List[List[int]]]:
        pil_img = self._to_pil_image(image)
        width, height = pil_img.size

        #1. detection

        #configs\pp_ocrv6_small_det\preprocessor_config.json:
        #   input ipl image raw
        #   output: dict {"pixel_values": tensor_anh_kich_thuoc_(1, 3, H, W)}
        det_inputs = self.det_processor(images = pil_img, return_tensors="pt")
        det_inputs = {k : v.to(self.device) for k, v in det_inputs.items()}

        #trong modeling_pp_ocrv6_small_det.py:
        #   input  : pixel_values   là ảnh tensor (B , 3 , H, W), cụ thể lấy từ dict (output det_inputs)
        #   output của PPOCRV6SmallDetForObjectDetection sẽ là last_hidden_state is xác suất văn bản ~ pob_map
        det_outputs = self.det_model(pixel_values = det_inputs["pixel_values"])


        detected_boxes = self._extract_boxes_from_prob_map(
            det_outputs.last_hidden_state, width, height
        )

        #khoi tao danh sach rong
        words : List[str] =[]
        normalized_boxes : List[List[int]] =[]

        #2. Regconition tren tung crop
        if detected_boxes:
            # use method .crop((x0, y0, x1,y1)) of PIL to crop many images from 'pil_img' raw
            crop_images = [pil_img.crop(box) for box in detected_boxes]

            batch_size = 32
            for i in range(0, len(crop_images), batch_size):
                batch_crops = crop_images[i: i+batch_size]
                batch_boxes = detected_boxes[i : i+batch_size]

                #preprocessing of rec
                rec_inputs = self.rec_processor(images=batch_crops, return_tensors='pt')
                rec_inputs = {k : v.to(self.device) for k, v in rec_inputs.items()}
                rec_outputs = self.rec_model(pixel_values = rec_inputs["pixel_values"])

                decoded_list = self.rec_processor.post_process_text_recognition(rec_outputs)

                for decoded, box in zip(decoded_list, batch_boxes):
                    text_str = decoded.get("text", "").strip()
                    if text_str:
                        clean_text = "".join(ch for ch in text_str if ch.isascii()).strip()
                        if not clean_text:
                            continue
                        sub_tokens = clean_text.split()
                        norm_b = self.normalize_box(box, width, height)
                        for token in sub_tokens:
                            if any(c.isalnum() for c in token):
                                words.append(token)
                                normalized_boxes.append(norm_b)
        # 3. fall back document has no words
        if not words: 
            words = ["document has no words / error"]
            normalized_boxes = [[0,0,1000,1000]]

        return words, normalized_boxes

    def __call__(
        self, image : Union[Image.Image, torch.Tensor, str, Path]
    ) -> Tuple[List[str], List[List[int]]]:
        return self.extract_words_and_boxes(image)


if __name__ == "__main__":
    print("khoi tao DocumentOCR from checkpoints")
    ocr_engine = DocumentOCR()
    
    print("OCR 's Device :", ocr_engine.device)

    demo_image_path= "nhap/2072197187.tif"
    out_words, out_boxes = ocr_engine(demo_image_path)

    print(f"Number of words: {len(out_words)}")
    print(f"Number of  boxes: {len(out_boxes)}")
    
    # In ra 10 tu dau tien va hop toa do tuong ung
    print("\n--- 10 tu dau tien tim thay ---")
    for idx, (w, b) in enumerate(zip(out_words[:100], out_boxes[:100])):
        print(f"[{idx + 1}] Word: '{w}' | Box [0, 1000]: {b}")

    full_text = " ".join(out_words)
    print("\n--- Toan bo van ban trich xuat duoc ---")
    print(full_text)