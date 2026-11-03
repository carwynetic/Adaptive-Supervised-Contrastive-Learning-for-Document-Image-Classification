import os
import urllib.request
from pathlib import Path

ref_dir = Path("data_processing_ref")
ref_dir.mkdir(exist_ok=True)

files_to_download = [
    {
        "subfolder": "01_rvl_cdip_dataset",
        "filename": "huggingface_rvl_cdip.py",
        "url": "https://raw.githubusercontent.com/huggingface/datasets/2.4.0/datasets/rvl_cdip/rvl_cdip.py",
        "desc": "Hugging Face Datasets: RVL-CDIP parser & 16 class definitions"
    },
    {
        "subfolder": "01_rvl_cdip_dataset",
        "filename": "niels_rogge_rvl_cdip.ipynb",
        "url": "https://raw.githubusercontent.com/NielsRogge/Transformers-Tutorials/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb",
        "desc": "Niels Rogge Tutorial: RVL-CDIP TIFF to RGB & PyTorch DataLoader"
    },
    {
        "subfolder": "02_layoutlm_collator_processor",
        "filename": "microsoft_unilm_data_collator.py",
        "url": "https://raw.githubusercontent.com/microsoft/unilm/master/layoutlmft/layoutlmft/data/data_collator.py",
        "desc": "Microsoft UniLM: Dynamic padding for LayoutLM bounding boxes & input_ids"
    },
    {
        "subfolder": "02_layoutlm_collator_processor",
        "filename": "hf_processing_layoutlmv3.py",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/layoutlmv3/processing_layoutlmv3.py",
        "desc": "Hugging Face Transformers: LayoutLMv3Processor multimodal call"
    },
    {
        "subfolder": "02_layoutlm_collator_processor",
        "filename": "hf_image_processing_layoutlmv3.py",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py",
        "desc": "Hugging Face Transformers: LayoutLMv3ImageProcessor on-the-fly OCR & box normalization"
    },
    {
        "subfolder": "03_contrastive_learning_supcon",
        "filename": "supcontrast_main_supcon.py",
        "url": "https://raw.githubusercontent.com/HobbitLong/SupContrast/master/main_supcon.py",
        "desc": "HobbitLong/SupContrast: TwoCropTransform and contrastive DataLoader pipeline"
    },
    {
        "subfolder": "03_contrastive_learning_supcon",
        "filename": "supcontrast_losses.py",
        "url": "https://raw.githubusercontent.com/HobbitLong/SupContrast/master/losses.py",
        "desc": "HobbitLong/SupContrast: Official SupConLoss implementation"
    }
]

for item in files_to_download:
    folder = ref_dir / item["subfolder"]
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / item["filename"]
    print(f"Downloading {item['filename']}...")
    req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        content = resp.read()
        f.write(content)
        print(f" -> Saved {out_path} ({len(content)} bytes)")

print("\nAll reference files downloaded successfully!")
