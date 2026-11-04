"""Check local model copies with official checkpoints, offline and on CPU.

Run from the project root: python -m third_party.smoke_test
"""

import gc
from pathlib import Path

import torch
from PIL import Image

from third_party.layoutlmv3 import LayoutLMv3Model, LayoutLMv3Processor
from third_party.pp_ocrv6_small_det import PPOCRV6SmallDetForObjectDetection
from third_party.pp_ocrv6_small_rec import (
    PPOCRV6SmallRecForTextRecognition,
    PPOCRV6SmallRecImageProcessor,
)
from third_party.pp_doclayout_v3 import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessor,
)
from transformers import AutoImageProcessor
from transformers.conversion_mapping import USER_REGISTERED_MAPPINGS

# Đăng ký weight mapping cho PP-DocLayoutV3 chạy dưới package cục bộ third_party
USER_REGISTERED_MAPPINGS.add("pp_doclayout_v3")
USER_REGISTERED_MAPPINGS.add("PPDocLayoutV3ForObjectDetection")

ROOT = Path(__file__).resolve().parents[1]


def load_model(model_class, folder):
    # Đọc config từ configs/ và load weights từ checkpoints/
    config_dir = ROOT / "configs" / folder
    checkpoint_dir = ROOT / "checkpoints" / folder

    model, info = model_class.from_pretrained(
        checkpoint_dir,
        config=str(config_dir),
        local_files_only=True,
        output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise RuntimeError(f"{folder}: {key}: {info[key]}")
    model.eval()
    print(f"Loaded {folder}: {model.__class__.__module__}", flush=True)
    return model


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    image = Image.new("RGB", (320, 224), "white")

    # 1. LayoutLMv3
    model = load_model(LayoutLMv3Model, "layoutlmv3-base")
    processor = LayoutLMv3Processor.from_pretrained(
        ROOT / "configs" / "layoutlmv3-base",
        apply_ocr=False,
        local_files_only=True,
    )
    inputs = processor(
        images=image,
        text=["hello", "world"],
        boxes=[[0, 0, 200, 100], [210, 0, 400, 100]],
        return_tensors="pt",
    )
    result = model(**inputs).last_hidden_state
    assert result.shape[0] == 1 and torch.isfinite(result).all()
    print(f"LayoutLMv3 forward OK: {tuple(result.shape)}", flush=True)
    del model, result, processor, inputs
    gc.collect()

    # 2. PP-OCRv6 Small Det
    model = load_model(PPOCRV6SmallDetForObjectDetection, "pp_ocrv6_small_det")
    processor = AutoImageProcessor.from_pretrained(
        ROOT / "configs" / "pp_ocrv6_small_det", local_files_only=True
    )
    processed = processor(images=image, return_tensors="pt")
    result = model(pixel_values=processed["pixel_values"]).last_hidden_state
    tensors = result if isinstance(result, (tuple, list)) else (result,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
    print(f"Small Det forward OK: {[tuple(t.shape) for t in tensors]}", flush=True)
    del model, result, tensors, processor, processed
    gc.collect()

    # 3. PP-OCRv6 Small Rec
    model = load_model(PPOCRV6SmallRecForTextRecognition, "pp_ocrv6_small_rec")
    processor = PPOCRV6SmallRecImageProcessor.from_pretrained(
        ROOT / "configs" / "pp_ocrv6_small_rec", local_files_only=True
    )
    processed = processor(images=image, return_tensors="pt")
    outputs = model(pixel_values=processed["pixel_values"])
    assert torch.isfinite(outputs.last_hidden_state).all()
    decoded = processor.post_process_text_recognition(outputs)
    assert len(decoded) == 1 and "text" in decoded[0]
    print(f"Small Rec forward/decode OK: {tuple(outputs.last_hidden_state.shape)}", flush=True)
    del model, outputs, processor, processed, decoded
    gc.collect()

    # 4. PP-DocLayoutV3
    model = load_model(PPDocLayoutV3ForObjectDetection, "pp_doclayout_v3")
    processor = PPDocLayoutV3ImageProcessor.from_pretrained(
        ROOT / "configs" / "pp_doclayout_v3", local_files_only=True
    )
    layout_image = Image.new("RGB", (800, 800), "white")
    processed = processor(images=layout_image, return_tensors="pt")
    outputs = model(**processed)
    assert torch.isfinite(outputs.logits).all()
    assert torch.isfinite(outputs.pred_boxes).all()

    target_sizes = [layout_image.size[::-1]]
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)
    assert len(results) == 1 and "boxes" in results[0]
    print(
        f"DocLayoutV3 forward/post-process OK: logits={tuple(outputs.logits.shape)}, boxes={tuple(outputs.pred_boxes.shape)}",
        flush=True,
    )
    del model, outputs, processor, processed, results, layout_image
    gc.collect()

    print("PASS: all four local models loaded and ran; this is not an OCR accuracy test.", flush=True)


if __name__ == "__main__":
    main()