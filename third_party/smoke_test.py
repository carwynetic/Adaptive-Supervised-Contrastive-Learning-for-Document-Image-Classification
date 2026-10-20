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
from transformers import AutoImageProcessor


ROOT = Path(__file__).resolve().parents[1]


def load_model(model_class, folder):
    model, info = model_class.from_pretrained(
        ROOT / "checkpoints" / folder,
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

    model = load_model(LayoutLMv3Model, "layoutlmv3-base")
    processor = LayoutLMv3Processor.from_pretrained(
        ROOT / "checkpoints" / "layoutlmv3-base",
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

    model = load_model(PPOCRV6SmallDetForObjectDetection, "pp_ocrv6_small_det")
    processor = AutoImageProcessor.from_pretrained(
        ROOT / "checkpoints" / "pp_ocrv6_small_det", local_files_only=True
    )
    # The detection processor lives in Transformers (PP-OCRv5 Server Det).
    processed = processor(images=image, return_tensors="pt")
    result = model(pixel_values=processed["pixel_values"]).last_hidden_state
    tensors = result if isinstance(result, (tuple, list)) else (result,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
    print(f"Small Det forward OK: {[tuple(t.shape) for t in tensors]}", flush=True)
    del model, result, tensors, processor, processed
    gc.collect()

    model = load_model(PPOCRV6SmallRecForTextRecognition, "pp_ocrv6_small_rec")
    processor = PPOCRV6SmallRecImageProcessor.from_pretrained(
        ROOT / "checkpoints" / "pp_ocrv6_small_rec", local_files_only=True
    )
    processed = processor(images=image, return_tensors="pt")
    outputs = model(pixel_values=processed["pixel_values"])
    assert torch.isfinite(outputs.last_hidden_state).all()
    decoded = processor.post_process_text_recognition(outputs)
    assert len(decoded) == 1 and "text" in decoded[0]
    print(f"Small Rec forward/decode OK: {tuple(outputs.last_hidden_state.shape)}", flush=True)
    print("PASS: all three local models loaded and ran; this is not an OCR accuracy test.", flush=True)


if __name__ == "__main__":
    main()
