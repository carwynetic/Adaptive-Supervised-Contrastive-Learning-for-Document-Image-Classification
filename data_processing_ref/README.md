# Bảng Đối Chiếu Nguồn Tham Khảo (Data Processing References)

Thư mục này chứa toàn bộ các file mã nguồn gốc chính thức từ các kho lưu trữ (repositories) lớn trên GitHub mà module `data_processing/` đã tham khảo, kế thừa và tích hợp.

---

## 1. Cấu trúc thư mục tham khảo

```
data_processing_ref/
├── 01_rvl_cdip_dataset/
│   ├── huggingface_rvl_cdip.py          # Nguồn từ Hugging Face Datasets
│   └── niels_rogge_rvl_cdip.ipynb       # Notebook mẫu từ tác giả Niels Rogge (Hugging Face)
├── 02_layoutlm_collator_processor/
│   ├── microsoft_unilm_data_collator.py # Data Collator chính thức của Microsoft UniLM
│   ├── hf_processing_layoutlmv3.py      # LayoutLMv3Processor từ Hugging Face Transformers
│   └── hf_image_processing_layoutlmv3.py# OCR Tesseract & Box Normalization từ Hugging Face
├── 03_contrastive_learning_supcon/
│   ├── supcontrast_main_supcon.py       # Code gốc Supervised Contrastive Learning (Yonglong Tian)
│   └── supcontrast_losses.py            # Hàm mất mát SupConLoss chuẩn NeurIPS 2020
├── download_refs.py                     # Script Python tự động tải lại các nguồn
└── README.md                            # Hướng dẫn đối chiếu chi tiết
```

---

## 2. Bảng Đối Chiếu Trực Tiếp Từng Module

| File trong `data_processing/` | File nguồn đối chiếu trong `data_processing_ref/` | Các thành phần đối chiếu chính |
| :--- | :--- | :--- |
| **`dataset.py`** | `01_rvl_cdip_dataset/huggingface_rvl_cdip.py`<br/>`01_rvl_cdip_dataset/niels_rogge_rvl_cdip.ipynb`<br/>`03_contrastive_learning_supcon/supcontrast_main_supcon.py` | • **16 Classes**: Đối chiếu mảng `_CLASSES` chuẩn trong `huggingface_rvl_cdip.py` với `RVLCDIPDataset.LABEL_NAMES`.<br/>• **Parse Labels**: Logic duyệt file `train.txt`, `val.txt`, `test.txt` từ `_generate_examples`.<br/>• **TIFF to RGB**: Cách chuyển đổi ảnh đen trắng 1 kênh TIFF sang tensor RGB 3 kênh từ Cell 6 & 16 của notebook Niels Rogge.<br/>• **Two-Crop SupCon**: Class `TwoCropTransform` sinh ra 2 biến thể ảnh từ cùng một mẫu trong `supcontrast_main_supcon.py`. |
| **`collator.py`** | `02_layoutlm_collator_processor/microsoft_unilm_data_collator.py`<br/>`02_layoutlm_collator_processor/hf_processing_layoutlmv3.py`<br/>`02_layoutlm_collator_processor/hf_image_processing_layoutlmv3.py` | • **Dynamic Sequence Padding**: Kế thừa kỹ thuật padding động theo độ dài tối đa của batch cho `input_ids`, `bbox`, `attention_mask` thay vì padding cứng cố định.<br/>• **Multimodal Processor**: Cơ chế gọi `LayoutLMv3Processor` nhận song song `images` + `text` + `boxes`.<br/>• **OCR on-the-fly**: Chuẩn hóa bounding boxes về thang đo [0, 1000] theo hàm `normalize_box` trong `image_processing_layoutlmv3.py`. |
| **`data_loader.py`** | `03_contrastive_learning_supcon/supcontrast_main_supcon.py`<br/>`01_rvl_cdip_dataset/niels_rogge_rvl_cdip.ipynb` | • **Cấu hình SupCon DataLoader**: `pin_memory=True`, `drop_last=True` để tránh batch lẻ làm suy biến ma trận tương đồng contrastive (hàm `set_loader(opt)` trong `main_supcon.py`). |
| **Hàm mất mát SupCon** | `03_contrastive_learning_supcon/supcontrast_losses.py` | • Đối chiếu công thức tính `SupConLoss` (SupCon và SimCLR) cho 2 views `features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)`. |

---

## 3. Link GitHub gốc để kiểm chứng trực tuyến

1. **Hugging Face Datasets - RVL-CDIP**: [datasets/rvl_cdip](https://github.com/huggingface/datasets/blob/2.4.0/datasets/rvl_cdip/rvl_cdip.py)
2. **Niels Rogge - LayoutLM RVL-CDIP Tutorial**: [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb)
3. **Microsoft UniLM - Data Collator**: [microsoft/unilm](https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/data/data_collator.py)
4. **Hugging Face Transformers - LayoutLMv3 Processor**: [processing_layoutlmv3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/processing_layoutlmv3.py)
5. **Hugging Face Transformers - LayoutLMv3 Image Processor**: [image_processing_layoutlmv3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py)
6. **HobbitLong / SupContrast**: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)
