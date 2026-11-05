"""
Module PP-OCRv6 module replacing the Microsoft Read API 
which used in the Original LayoutLMv3 Paper.
Using a local of PP-OCRv6 Det and Rec Modules combined with PP-DocLayoutV3.
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
from transformers.conversion_mapping import USER_REGISTERED_MAPPINGS

USER_REGISTERED_MAPPINGS.add("pp_doclayout_v3")
USER_REGISTERED_MAPPINGS.add("PPDocLayoutV3ForObjectDetection")

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
from third_party.pp_doclayout_v3 import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessor,
)


class DocumentOCR:
    def __init__(
        self,
        det_config_dir          : Optional[Union[str, Path]] = None,
        det_checkpoint_dir      : Optional[Union[str, Path]] = None,
        rec_config_dir          : Optional[Union[str, Path]] = None,
        rec_checkpoint_dir      : Optional[Union[str, Path]] = None,
        layout_config_dir       : Optional[Union[str, Path]] = None,
        layout_checkpoint_dir   : Optional[Union[str, Path]] = None,  
        layout_threshold        : float = 0.3,
        det_threshold           : float = 0.3,
        min_box_size            : int = 4,
        device                  : torch.device | None = None,
    ) -> None:
        """
        Args:
            det_config_dir        : folder configs/pp_ocrv6_small_det
            det_checkpoint_dir    : folder checkpoints/pp_ocrv6_small_det
            rec_config_dir        : folder configs/pp_ocrv6_small_rec
            rec_checkpoint_dir    : folder checkpoints/pp_ocrv6_small_rec
            layout_config_dir     : folder configs/pp_doclayout_v3
            layout_checkpoint_dir : folder checkpoints/pp_doclayout_v3
            layout_threshold      : nguong xac suat lay khoi bo cuc (default=0.3)
            det_threshold         : nguong binarize ban do nhiet DBNet (default=0.3)
            min_box_size          : kich thuoc canh hop nho nhat de loai bo hat nhieu (mac dinh 4px)
            device                : thiet bi tinh toan ('cpu' hoac 'cuda')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.det_threshold    = det_threshold
        self.min_box_size     = min_box_size
        self.layout_threshold = layout_threshold
        
        # Xac dinh thu muc goc cua project
        root_dir = Path(__file__).resolve().parents[1]
        self.det_config_dir        = Path(det_config_dir) if det_config_dir else root_dir / 'configs' / "pp_ocrv6_small_det"
        self.rec_config_dir        = Path(rec_config_dir) if rec_config_dir else root_dir / 'configs' / "pp_ocrv6_small_rec"
        self.det_checkpoint_dir    = Path(det_checkpoint_dir) if det_checkpoint_dir else root_dir / "checkpoints" / "pp_ocrv6_small_det"
        self.rec_checkpoint_dir    = Path(rec_checkpoint_dir) if rec_checkpoint_dir else root_dir / "checkpoints" / "pp_ocrv6_small_rec"
        self.layout_config_dir     = Path(layout_config_dir) if layout_config_dir else root_dir / "configs" / "pp_doclayout_v3"
        self.layout_checkpoint_dir = Path(layout_checkpoint_dir) if layout_checkpoint_dir else root_dir / "checkpoints" / "pp_doclayout_v3"
        
        # 1. Detection (PP-OCRv6 Det)
        self.det_processor = AutoImageProcessor.from_pretrained(
            self.det_config_dir, local_files_only=True
        )
        det_config = PPOCRV6SmallDetConfig.from_pretrained(
            self.det_config_dir, local_files_only=True
        )
        self.det_model = PPOCRV6SmallDetForObjectDetection.from_pretrained(
            self.det_checkpoint_dir, config=det_config, local_files_only=True
        ).to(self.device)
        self.det_model.eval()

        # 2. Recognition (PP-OCRv6 Rec)
        self.rec_processor = PPOCRV6SmallRecImageProcessor.from_pretrained(
            self.rec_config_dir, local_files_only=True
        )
        rec_config = PPOCRV6SmallRecConfig.from_pretrained(
            self.rec_config_dir, local_files_only=True
        )
        self.rec_model = PPOCRV6SmallRecForTextRecognition.from_pretrained(
            self.rec_checkpoint_dir, config=rec_config, local_files_only=True
        ).to(self.device)
        self.rec_model.eval()

        # 3. Layout Analysis (PP-DocLayoutV3)
        self.layout_processor = PPDocLayoutV3ImageProcessor.from_pretrained(
            self.layout_config_dir, local_files_only=True
        )
        self.layout_model = PPDocLayoutV3ForObjectDetection.from_pretrained(
            self.layout_checkpoint_dir,
            config=str(self.layout_config_dir),
            local_files_only=True
        ).to(self.device)
        self.layout_model.eval()

    # Adapter chuyen doi hinh anh tu nhieu nguon ve PIL.Image (RGB)
    @staticmethod
    def _to_pil_image(image: Union[Image.Image, torch.Tensor, str, Path]) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                return TF.to_pil_image(image).convert("RGB")
            if image.ndim == 4 and image.shape[0] == 1:
                return TF.to_pil_image(image.squeeze(0)).convert("RGB")
            raise ValueError(f"Tensor shape is not valid: {image.shape}")
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise TypeError(f"Type of Image not supported: {type(image)}")
    
    @staticmethod
    def normalize_box(box: List[int], width: int, height: int) -> List[int]:
        x0, y0, x1, y1 = box
        norm_x0 = max(0, min(1000, int(1000 * (x0 / width))))
        norm_y0 = max(0, min(1000, int(1000 * (y0 / height))))
        norm_x1 = max(0, min(1000, int(1000 * (x1 / width))))
        norm_y1 = max(0, min(1000, int(1000 * (y1 / height))))
        return [norm_x0, norm_y0, norm_x1, norm_y1]
    
    @torch.inference_mode()
    def extract_layout_blocks(self, pil_img: Image.Image) -> List[dict]:
        """
        Trich xuat cac khoi bo cuc vi mo bang PP-DocLayoutV3 va sap xep thu tu doc thong minh:
        - Tu dong phat hien tai lieu 1 cot hay da cot (multi-column) dua tren cac cap khoi song song.
        - Neu 1 cot: sap xep top-down tu nhien (CURRICULUM VITAE, Date of Birth, Education, ...).
        - Neu da cot: ap dung Dynamic Column Clustering doc cot trai -> cot phai.
        """
        w, h = pil_img.size
        processed = self.layout_processor(images=pil_img, return_tensors="pt")
        processed = {k: v.to(self.device) for k, v in processed.items()}
        
        outputs = self.layout_model(**processed)
        target_sizes = [(h, w)]
        results = self.layout_processor.post_process_object_detection(
            outputs, threshold=self.layout_threshold, target_sizes=target_sizes
        )[0]

        id2label = self.layout_model.config.id2label
        blocks = []
        for box, score, lbl_id in zip(results["boxes"], results["scores"], results["labels"]):
            # Ho tro ca key dang int lan str trong id2label
            lbl = id2label.get(lbl_id.item(), id2label.get(str(lbl_id.item()), ""))
            # Bo qua cac visual block khong chua text chinh (anh, bieu do, con dau)
            if lbl in ["image", "chart", "seal"]:
                continue
            b = [int(v.item()) for v in box]
            blocks.append({"box": b, "label": lbl, "score": score.item()})

        if not blocks:
            return []

        # 1. Kiem tra xem trang co thuc su la bo cuc Da cot (Multi-column) khong
        # Trang da cot khi va chi khi ton tai cac khoi van ban song song cung khoang Y nhung tach biet ve X
        text_blocks = [b for b in blocks if b["label"] in ["text", "content", "paragraph_title"]]
        parallel_col_pairs = 0
        for i in range(len(text_blocks)):
            for j in range(i + 1, len(text_blocks)):
                b1 = text_blocks[i]["box"]
                b2 = text_blocks[j]["box"]
                y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                min_h = min(b1[3] - b1[1], b2[3] - b2[1])
                if min_h > 0 and (y_overlap / min_h) > 0.5:
                    # Khoang cach giua 2 khoi theo chieu ngang (ranh cach nhau it nhat 25px)
                    if (b1[2] < b2[0] - 25) or (b2[2] < b1[0] - 25):
                        parallel_col_pairs += 1

        is_multi_column = (parallel_col_pairs >= 3)

        # 2. Neu la tai lieu 1 cot (Resume/CV, hoa don, don tu, thu tu):
        # Thu tu doc tu nhien tuyet doi la tu tren xuong duoi theo toa do Y
        if not is_multi_column:
            blocks.sort(key=lambda b: (b["box"][1], b["box"][0]))
            return blocks

        # 3. Neu la tai lieu Da cot (bao chi, tap chi khoa hoc chia 2-3 cot):
        headers = []
        tails = []
        body = []

        for b in blocks:
            bx0, by0, bx1, by1 = b["box"]
            bw = bx1 - bx0
            lbl = b["label"]

            if lbl in ["footer", "footnote"]:
                tails.append(b)
            elif lbl in ["header", "doc_title"] or bw > 0.45 * w or by1 < 0.20 * h:
                headers.append(b)
            else:
                body.append(b)

        headers.sort(key=lambda item: (item["box"][1], item["box"][0]))
        tails.sort(key=lambda item: (item["box"][1], item["box"][0]))
        body.sort(key=lambda item: item["box"][0])

        columns = []
        for blk in body:
            bx0, by0, bx1, by1 = blk["box"]
            bw = bx1 - bx0
            matched_col = None
            best_overlap = 0.0

            for col in columns:
                overlap = max(0, min(bx1, col["x1"]) - max(bx0, col["x0"]))
                min_w = min(bw, col["x1"] - col["x0"])
                overlap_ratio = overlap / float(max(1, min_w))

                if overlap_ratio > 0.40 and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    matched_col = col

            if matched_col is not None:
                matched_col["blocks"].append(blk)
                matched_col["x0"] = min(matched_col["x0"], bx0)
                matched_col["x1"] = max(matched_col["x1"], bx1)
            else:
                columns.append({"x0": bx0, "x1": bx1, "blocks": [blk]})

        columns.sort(key=lambda c: c["x0"])
        sorted_body = []
        for col in columns:
            col["blocks"].sort(key=lambda b: b["box"][1])
            sorted_body.extend(col["blocks"])

        return headers + sorted_body + tails

    @staticmethod
    def _assign_lines_to_layout_blocks(
        line_boxes: List[List[int]], layout_blocks: List[dict]
    ) -> List[List[int]]:
        """
        Anh xa tung dong chu vi mo vao dung khoi bo cuc vi mo chua no de sap xep thu tu doc.
        """
        if not layout_blocks:
            return line_boxes

        assigned = []
        used_indices = set()

        for blk in layout_blocks:
            bx0, by0, bx1, by1 = blk["box"]
            blk_lines = []
            for idx, lb in enumerate(line_boxes):
                if idx in used_indices:
                    continue
                # Tinh tam dong chu
                cx = (lb[0] + lb[2]) / 2.0
                cy = (lb[1] + lb[3]) / 2.0
                # Neu tam nam trong khoi bo cuc (dung sai 10px)
                if (bx0 - 10 <= cx <= bx1 + 10) and (by0 - 10 <= cy <= by1 + 10):
                    blk_lines.append((idx, lb))

            # Trong cung mot khoi, sap xep cac dong chu tu tren xuong duoi
            blk_lines.sort(key=lambda item: item[1][1])
            for idx, lb in blk_lines:
                used_indices.add(idx)
                assigned.append(lb)

        # Nhung dong chu nam ngoai khoi bo cuc (fallback)
        unassigned = [lb for idx, lb in enumerate(line_boxes) if idx not in used_indices]
        unassigned.sort(key=lambda b: (b[1], b[0]))
        return assigned + unassigned

    def _extract_boxes_from_prob_map(
        self, 
        prob_map : torch.Tensor, 
        orig_width: int, orig_height: int
    ) -> List[List[int]]:
        if prob_map.ndim == 4:
            map_arr = prob_map[0, 0].detach().cpu().float().numpy()
        elif prob_map.ndim == 3:
            map_arr = prob_map[0].detach().cpu().float().numpy()
        else:
            map_arr = prob_map.detach().cpu().float().numpy()
        
        map_h, map_w = map_arr.shape
        binary_mask = map_arr > self.det_threshold
        
        labeled_mask, _ = label(binary_mask)
        slices = find_objects(labeled_mask)

        raw_boxes = []
        scale_x = orig_width / float(map_w)
        scale_y = orig_height / float(map_h)

        for s in slices:
            if s is None:
                continue
            y_slice, x_slice = s
            box_w = x_slice.stop - x_slice.start
            box_h = y_slice.stop - y_slice.start

            if box_w < self.min_box_size or box_h < self.min_box_size:
                continue
            
            # Nguong loc aspect ratio (50.0 tranh xoa dong text cot bao co ti le ~ 25-35)
            if (box_w / max(1, box_h) > 50.0) or (box_h / max(1, box_w) > 30.0): 
                continue

            # Mo rong padding tranh cat mat net chu
            pad_h = int(box_h * 0.7) + 3 
            pad_w = int(box_w * 0.2) + 2

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

        return raw_boxes

    @torch.inference_mode()
    def extract_words_and_boxes(
        self, image : Union[Image.Image, torch.Tensor, str, Path]
    ) -> Tuple[List[str], List[List[int]]]:
        pil_img = self._to_pil_image(image)
        width, height = pil_img.size

        # 1. Detection bang PP-OCRv6 Det
        det_inputs = self.det_processor(images=pil_img, return_tensors="pt")
        det_inputs = {k: v.to(self.device) for k, v in det_inputs.items()}
        det_outputs = self.det_model(pixel_values=det_inputs["pixel_values"])

        detected_boxes = self._extract_boxes_from_prob_map(
            det_outputs.last_hidden_state, width, height
        )

        # 2. Anh xa vao bo cuc PP-DocLayoutV3 de sap xep thu tu doc
        if detected_boxes:
            layout_blocks = self.extract_layout_blocks(pil_img)
            detected_boxes = self._assign_lines_to_layout_blocks(detected_boxes, layout_blocks)

        words : List[str] = []
        normalized_boxes : List[List[int]] = []

        # 3. Recognition tren tung crop bang PP-OCRv6 Rec
        if detected_boxes:
            crop_images = [pil_img.crop(box) for box in detected_boxes]
            batch_size = 32
            for i in range(0, len(crop_images), batch_size):
                batch_crops = crop_images[i : i + batch_size]
                batch_boxes = detected_boxes[i : i + batch_size]

                rec_inputs = self.rec_processor(images=batch_crops, return_tensors='pt')
                rec_inputs = {k: v.to(self.device) for k, v in rec_inputs.items()}
                rec_outputs = self.rec_model(pixel_values=rec_inputs["pixel_values"])

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

        # 4. Fallback neu tai lieu khong co chu
        if not words: 
            words = ["document has no words / error"]
            normalized_boxes = [[0, 0, 1000, 1000]]

        return words, normalized_boxes

    def __call__(
        self, image : Union[Image.Image, torch.Tensor, str, Path]
    ) -> Tuple[List[str], List[List[int]]]:
        return self.extract_words_and_boxes(image)


if __name__ == "__main__":
    print("Khoi tao DocumentOCR from checkpoints...")
    ocr_engine = DocumentOCR()
    
    print("OCR's Device :", ocr_engine.device)

    demo_image_path = "nhap/50672317-2318.tif"
    out_words, out_boxes = ocr_engine(demo_image_path)

    print(f"Number of words: {len(out_words)}")
    print(f"Number of boxes: {len(out_boxes)}")
    
    # In ra 100 tu dau tien va hop toa do tuong ung
    print("\n--- 100 tu dau tien tim thay ---")
    for idx, (w, b) in enumerate(zip(out_words[:100], out_boxes[:100])):
        print(f"[{idx + 1:3d}] Word: '{w:20s}' | Box [0, 1000]: {b}")

    full_text = " ".join(out_words)
    print("\n--- Toan bo van ban trich xuat duoc ---")
    print(full_text)
