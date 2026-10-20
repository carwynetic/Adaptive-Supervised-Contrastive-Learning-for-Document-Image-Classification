# Ba model cục bộ

```text
third_party/
├── layoutlmv3/
├── pp_ocrv6_small_det/
├── pp_ocrv6_small_rec/
├── smoke_test.py
├── requirements.txt
├── SOURCE_MANIFEST.json
└── LICENSE.transformers
```

Đây là bản sao source model từ `../transformers/src/transformers/models`.
Repository gốc được giữ nguyên. Các import dùng chung đã đổi sang
`from transformers...`; import nội bộ như `from .configuration_...` được giữ lại.
Không cần thêm repository gốc vào `PYTHONPATH` hoặc cài editable repository đó.
Runtime vẫn cần thư viện Transformers trong môi trường Python.

Chạy code từ thư mục gốc `G:\My Drive\DIC`, hoặc bảo đảm thư mục gốc project có
trong đường dẫn import Python. Không chạy trực tiếp các file `modeling_*.py`.

## Nạp checkpoint

```python
from pathlib import Path

from third_party.layoutlmv3 import LayoutLMv3Model
from third_party.pp_ocrv6_small_det import PPOCRV6SmallDetForObjectDetection
from third_party.pp_ocrv6_small_rec import PPOCRV6SmallRecForTextRecognition

root = Path(r"G:\My Drive\DIC")

layout = LayoutLMv3Model.from_pretrained(
    root / "checkpoints/layoutlmv3-base", local_files_only=True
)
det = PPOCRV6SmallDetForObjectDetection.from_pretrained(
    root / "checkpoints/pp_ocrv6_small_det", local_files_only=True
)
rec = PPOCRV6SmallRecForTextRecognition.from_pretrained(
    root / "checkpoints/pp_ocrv6_small_rec", local_files_only=True
)
```

`from_pretrained` tự đọc `config.json` và `model.safetensors`. Các class được
import từ `third_party`, nên thay đổi trong bản sao sẽ được dùng. Nếu import
`LayoutLMv3Model` từ `transformers`, bạn sẽ dùng bản thư viện thay vì bản sao.
Hai class OCR `For...` bao gồm cả backbone và head đã học; class `...Model`
chỉ là phần trích xuất đặc trưng.

## Dependency và phần đã lược bỏ

- Các tiện ích chung, AutoConfig và backbone PPLCNetV4 lấy từ Transformers.
- Image processor của Small Det lấy từ Transformers:
  `transformers.models.pp_ocrv5_server_det.image_processing_pp_ocrv5_server_det`.
- Image processor Small Rec nằm trong package cục bộ, có hàm giải mã CTC.
- Không copy source `modular_*.py`, script chuyển đổi, CI, tài liệu và test của
  repository. Những file modular phục vụ sinh code, không cần để chạy bản sao.
- Hai config OCR được chuyển từ API `SubConfigSpec` của source development sang
  API `sub_configs` của runtime 5.19.0. Thông số backbone vẫn đọc từ checkpoint.
- Các bản sao là module được duy trì riêng; không chạy công cụ sinh code của
  repo Transformers lên chúng. Thông báo generated trong bản sao đã được thay.

Các phiên bản runtime dùng để kiểm tra được ghi trong `requirements.txt`.
Nếu tạo môi trường khác: `python -m pip install -r third_party/requirements.txt`.
Torchvision đã được bổ sung vào môi trường hiện tại để chạy image processor OCR.
OCR detection hậu xử lý thành polygon còn cần OpenCV; smoke test bên dưới kiểm
tra nạp trọng số, tiền xử lý, forward và giải mã Rec, không kiểm tra polygon.

## Kiểm tra

```powershell
cd "G:\My Drive\DIC"
python -m third_party.smoke_test
```

Script chạy offline trên CPU, lần lượt nạp checkpoint chính thức cho ba class
cục bộ, kiểm tra không thiếu/thừa/sai kích thước trọng số và chạy forward với
ảnh giả. Đây là kiểm tra tích hợp, không đánh giá chất lượng OCR/phân loại.

