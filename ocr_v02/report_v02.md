# Báo Cáo Thực Thi OCR Tiếng Việt v02 (Theo Kế Hoạch v02)

- **Run ID:** `v02_20261116` | **Thời gian hoàn tất:** `2026-11-17T10:45:19+07:00`
- **Trạng thái kiểm định:** `PASS`

## 1. Kiến trúc Routing Thực Nghiệm (100% OCR mới từ PDF gốc)
- **0 kế thừa text cũ (`ocr_output` / `ocr_v01`):** Toàn bộ **2479** leaf Việt/hỗn hợp được đọc mới trực tiếp từ 12 PDF gốc.
- **Nhóm A – 6 quyển ảnh nét (`1-bit` rõ / `giấy vàng`):** `qtkb`, `hvddc`, `hvxt`, `lthclc_q13_19_quan_chuc`, `lthclc_q26_28_khoa_muc`, `lthclc_q29_32_quoc_dung` (`1307` leaf) → **Tesseract 5 (`vie_best`, `PSM 3`)** + tiền xử lý từng profile.
- **Nhóm B – 6 quyển ảnh mờ / độ phân giải thấp (`75–100 DPI`):** `dia_chi_loai_q5`, `hvll_q1..q5` (`1172` leaf) → **VietOCR `vgg_transformer`** trên các dòng cắt giữ dấu (`segment_lines_preserve_diacritics`) + chạy đối chứng **Tesseract 5** để ghi nhận hàng đợi bất đồng `06_nhat_ky/review_queue.jsonl`.

## 2. Bảng Benchmark Thực Đo (E0 v01 vs E4 v02)
| Split | Cấu hình | Số mẫu | Micro CER | Macro CER | Median CER | P90 CER | Trang >5% | Trang >10% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev | E0_v01_clean | 8 | 1.12% | 1.22% | 1.31% | 2.69% | 0 | 0 |
| dev | E1_tesseract_psm3 | 8 | 4.43% | 4.02% | 2.07% | 10.61% | 3 | 1 |
| dev | E2_tesseract_psm7 | 8 | 6.32% | 6.01% | 5.30% | 11.35% | 5 | 3 |
| dev | E3_vietocr_transformer | 8 | 3.78% | 3.54% | 2.91% | 8.65% | 3 | 0 |
| dev | E4_routed_v02 | 8 | 3.61% | 3.33% | 2.07% | 8.65% | 2 | 0 |
| test | E0_v01_clean | 9 | 1.70% | 2.03% | 1.65% | 9.18% | 1 | 0 |
| test | E1_tesseract_psm3 | 9 | 4.50% | 4.18% | 2.07% | 12.52% | 3 | 1 |
| test | E2_tesseract_psm7 | 9 | 4.78% | 4.66% | 1.97% | 12.46% | 4 | 1 |
| test | E3_vietocr_transformer | 9 | 4.11% | 4.43% | 2.63% | 9.96% | 3 | 0 |
| test | E4_routed_v02 | 9 | 2.86% | 2.77% | 2.07% | 8.41% | 1 | 0 |

## 3. Chi tiết từng quyển bàn giao cho bước Hybrid Matching
| book_id | Engine chính | Tổng unit | Unit dóng hàng (body/trans/heading) | Unit gắn cờ needs_review |
|---|---|---:|---:|---:|
| `dia_chi_loai_q5` | `vietocr_vgg_transformer` | 754 | 754 | 37 |
| `lthclc_q26_28_khoa_muc` | `tesseract_best_psm3` | 1225 | 1225 | 314 |
| `lthclc_q29_32_quoc_dung` | `tesseract_best_psm3` | 1284 | 1284 | 231 |
| `lthclc_q13_19_quan_chuc` | `tesseract_best_psm3` | 2479 | 2479 | 230 |
| `hvll_q1` | `vietocr_vgg_transformer` | 1973 | 1961 | 26 |
| `hvddc` | `tesseract_best_psm3` | 2317 | 2314 | 1508 |
| `hvxt` | `tesseract_best_psm3` | 3030 | 2886 | 2808 |
| `hvll_q3` | `vietocr_vgg_transformer` | 3241 | 3180 | 16 |
| `qtkb` | `tesseract_best_psm3` | 7203 | 7194 | 5413 |
| `hvll_q2` | `vietocr_vgg_transformer` | 2890 | 2832 | 12 |
| `hvll_q5` | `vietocr_vgg_transformer` | 4079 | 4055 | 43 |
| `hvll_q4` | `vietocr_vgg_transformer` | 5512 | 5455 | 11 |
