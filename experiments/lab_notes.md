# Nhật ký thực nghiệm (Research Lab Notes)

File này đi kèm với 2 bảng theo dõi dạng CSV (mở trực tiếp bằng Excel hoặc import vào Google Sheets):
- **Bảng tổng hợp (Sheet 1):** [experiment_log.csv](experiment_log.csv)
- **Bảng chi tiết theo từng Exit (Sheet 2):** [per_exit_details.csv](per_exit_details.csv)

---

## Quy ước đặt tên `Run_ID`
Dùng chuẩn: `R<STT>_<NhómExp>_<CấuHìnhNgắn>`
- `R01_Exp1_HamedCE_Pilot160` — Pilot local 160 mẫu/split, Hamed CE
- `R02_Exp2_SupConNoGW_Pilot160` — Pilot local 160 mẫu/split, SupCon-ST không groupwise
- `R03_Exp3_6Grp_S1` — Kaggle full 800/400/400, 6-Group Sequential Recompute (Stage 1)
- `R04_Exp3_6Grp_S2_BestS1` — Kaggle Stage 2 (frozen backbone từ `stage1_best.pt` Ep 36) + Eval chuẩn Hamed
- `R05_Exp1_HamedCE_Full800` — Baseline Hamed CE trên full 800/400/400 (để so sánh trực tiếp với R04)

---

## [2026-11-26] R01 & R02 — Pilot chứng minh Gradient Conflict (Local RTX 3050 6GB)

- **Mục tiêu:** Chứng minh định lượng hiện tượng xung đột gradient (Gradient Conflict) khi huấn luyện đồng thời nhiều exit trên LayoutLMv3 (160 train / 160 val / 160 test, 20 epochs, `bs=2`).
- **Kết quả chính (đã lưu tại `outputs/figures/summary_table.md`):**
  - **R01 (Hamed CE):** Conflict Rate `40.48% – 46.58%`, Max HSR `39.13%`, `landscape_min_dist = 0.5831`. Exit nông `L1 = 55.63%` vượt xa exit sâu `L12 = 38.75%` → chứng tỏ trọng số subgraph ưu tiên exit nông kéo lùi tầng sâu.
  - **R02 (SupCon-ST không groupwise):** Conflict Rate `41.93% – 45.34%`, Max HSR `34.78%`, `L1 = 41.88%` vs `L12 = 18.75%`. Xác nhận bỏ groupwise sẽ gây suy thoái biểu diễn tầng sâu dù dùng SupCon-ST.
- **Quyết định tiếp theo:** Chuyển sang chiến lược **6-Group Incremental Sequential Recompute** (kiểu A — backward tuần tự từng nhóm exit trên cùng 1 batch) và huấn luyện trên Kaggle T4 15GB với full subset `rvl-cdip-100-ocr` (`800/400/400`).

---

## [2026-11-27] R03 — `R03_Exp3_6Grp_S1` (Kaggle Stage 1 — 60 Epochs)

- **Notebook / Dataset Kaggle:** `chay_idea_bai_minh_1 (v1)` | Dataset: `/kaggle/input/rvl-cdip-100-ocr/rvl_cdip_100`
- **Git Commit:** `7c44912` (nhánh `Phuc`)
- **Cấu hình:** 6 exits `[0, 1, 4, 8, 10, 12]`, `epochs=60`, `bs=16 (P=8, K=2)`, `accum=1`, `AdamW (no LARS)`, `temp=0.1`, `smooth=0.1`, `seed=42`.
- **Quan sát quá trình train:**
  - Tốc độ ổn định: `10m50s – 10m59s/ep` (~11.25s/step), tổng thời gian **~10h55m** (nằm gọn trong 1 phiên Kaggle 12h).
  - VRAM ổn định tuyệt đối: **`8,076 MB / 15,360 MB`** (0 MB memory leak, 0 NaN).
  - Train loss giảm mượt: `3.4098` (Ep 1) → `1.8604` (Ep 60).
  - Val loss giảm mạnh từ `3.2394` (Ep 1) → chạm đáy **`2.5824` tại Ep 36** (`stage1_best.pt`), rồi đi ngang cực ổn định ở `2.5824 – 2.5901` suốt 24 epoch cuối (biến động chỉ `0.3%`, không collapse, không overfit).
- **Quyết định tiếp theo:**
  - Chốt dùng **`stage1_best.pt` (Ep 36)** làm backbone cố định cho **R04 (Stage 2 + Eval chuẩn Hamed)**.
  - Nhớ tải về / gắn Output Dataset cho `stage1_best.pt` và `stage1_final.pt` để không mất 11 giờ train.

---

## [YYYY-MM-DD] R04 — `R04_Exp3_6Grp_S2_BestS1` (Template chờ điền)

- **Notebook / Output Kaggle:** *(điền tên notebook + version)*
- **Khởi tạo từ:** `stage1_best.pt` (Ep 36 của R03)
- **Cấu hình Stage 2:** `20 epochs`, `bs=8`, `lr=3e-4`, frozen backbone, train 6 classifier heads `[0, 1, 4, 8, 10, 12]`.
- **Kết quả cần điền sau khi chạy xong:**
  - Best Val Acc (L12): `...%` (tại Epoch `...`)
  - Test Acc từng exit (`L0 / L1 / L4 / L8 / L10 / L12`): `... / ... / ... / ... / ... / ...`
  - Test ECE (trước / sau Temperature Scaling): `... / ...`
  - Early-Exit Heuristic: Acc `...%`, Params Saved `...%`, Avg Exit Layer `...`, Speedup `...x`
- **Nhận xét & bước tiếp theo:** *(So sánh trực tiếp với Baseline Hamed CE cùng split 800/400/400 — `R05_Exp1_HamedCE_Full800`)*
