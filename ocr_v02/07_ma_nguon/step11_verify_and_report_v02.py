"""Step 11 (v02): End-to-end deliverable verification, release_manifest.json,
closed-test comparison (v01 vs v02), and report_v02.md generator."""
import csv
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v02 import (
    BOOKS_META,
    DIR_DANH_GIA,
    DIR_NGUON,
    DIR_NHAT_KY,
    DIR_OCR_THO_VIET,
    DIR_RAW_TESS,
    DIR_RAW_VIETOCR,
    DIR_TEXT_PER_BOOK,
    DIR_TEXT_SACH,
    META_BY_ID,
    RUN_ID,
    V02_ROOT,
    now_vn_iso,
    sha256_file,
)


def main():
    results = {}

    # 1. Source manifest & page map
    manifest = list(csv.DictReader(open(os.path.join(DIR_NGUON, "source_manifest.csv"), encoding="utf-8-sig")))
    page_map = list(csv.DictReader(open(os.path.join(DIR_NGUON, "page_map.csv"), encoding="utf-8-sig")))
    viet_leaves = [r for r in page_map if r["page_type"] in ("viet", "mixed")]
    results["1_manifest_and_pagmap"] = {
        "pass": len(manifest) == 12 and len(page_map) >= 4589 and len(viet_leaves) >= 2460,
        "pdfs": len(manifest),
        "total_leaves": len(page_map),
        "viet_mixed_leaves": len(viet_leaves),
    }

    # 2. 100% Fresh OCR & Provenance check (zero legacy text)
    missing_raw = 0
    bad_provenance = 0
    src_sha = {m["book_id"]: m["sha256"] for m in manifest}
    engine_counts = {"tesseract_best_psm3": 0, "vietocr_vgg_transformer": 0}
    for r in viet_leaves:
        bid = r["book_id"]
        pnum = int(r["pdf_page"])
        lid = r["leaf_id"]
        stem = f"page_{pnum:04d}_{lid}" if lid in ("L", "R") else f"page_{pnum:04d}"
        can_p = os.path.join(DIR_OCR_THO_VIET, bid, f"{stem}.txt")
        prim = META_BY_ID[bid]["primary_engine"]
        raw_dir = DIR_RAW_VIETOCR if prim == "vietocr_vgg_transformer" else DIR_RAW_TESS
        json_p = os.path.join(raw_dir, bid, f"{stem}.json")
        if not (os.path.exists(can_p) and os.path.exists(json_p)):
            missing_raw += 1
            continue
        meta_j = json.load(open(json_p, encoding="utf-8"))
        if meta_j.get("source_pdf_sha256") != src_sha[bid] or meta_j.get("run_id") != RUN_ID:
            bad_provenance += 1
        else:
            engine_counts[prim] += 1

    results["2_fresh_ocr_provenance"] = {
        "pass": missing_raw == 0 and bad_provenance == 0,
        "missing_raw": missing_raw,
        "bad_provenance": bad_provenance,
        "engine_counts": engine_counts,
    }

    # 3. Clean files & NFC & structural checks
    viet_txt_p = os.path.join(DIR_TEXT_SACH, "viet.txt")
    viet_sents_p = os.path.join(DIR_TEXT_SACH, "viet_sentences.txt")
    sent_map_p = os.path.join(DIR_TEXT_SACH, "sentence_map.jsonl")

    viet_txt = open(viet_txt_p, encoding="utf-8").read()
    sents_lines = [l for l in open(viet_sents_p, encoding="utf-8").read().splitlines() if l.strip()]
    map_lines = [json.loads(l) for l in open(sent_map_p, encoding="utf-8")]

    is_nfc = (viet_txt == unicodedata.normalize("NFC", viet_txt)) and all(s == unicodedata.normalize("NFC", s) for s in sents_lines)
    has_sep_in_sents = any(s.startswith("===") for s in sents_lines)
    bad_roles_in_sents = [m["role"] for m in map_lines if m["role"] not in ("heading", "body", "translation")]
    bad_tokens = [tok for tok in ("O7KB", "OTKB") if tok in viet_txt]
    preserved_ok = ("tính nhiếp" in viet_txt) and ("30.350" in viet_txt)

    per_book_ok = all(
        os.path.exists(os.path.join(DIR_TEXT_PER_BOOK, f"{b['book_id']}_viet_clean.txt"))
        and os.path.exists(os.path.join(DIR_TEXT_PER_BOOK, f"{b['book_id']}_units.jsonl"))
        and os.path.exists(os.path.join(DIR_TEXT_PER_BOOK, f"{b['book_id']}_units.txt"))
        for b in BOOKS_META
    )

    results["3_clean_text_and_units"] = {
        "pass": is_nfc and not has_sep_in_sents and not bad_roles_in_sents and not bad_tokens and preserved_ok and per_book_ok and len(sents_lines) == len(map_lines),
        "is_nfc": is_nfc,
        "sentences_count": len(sents_lines),
        "sentence_map_count": len(map_lines),
        "has_sep_in_sents": has_sep_in_sents,
        "bad_roles_in_sents_count": len(bad_roles_in_sents),
        "bad_tokens": bad_tokens,
        "preserved_tinh_nhiep_and_30350": preserved_ok,
        "all_12_books_exported": per_book_ok,
    }

    # 4. Benchmark summary (compare automatic v02 against automatic baseline E1 & unedited E0 test pages)
    bench_rows = list(csv.DictReader(open(os.path.join(DIR_DANH_GIA, "benchmark.csv"), encoding="utf-8-sig")))
    per_sample_bench = list(csv.DictReader(open(os.path.join(DIR_DANH_GIA, "benchmark_per_sample.csv"), encoding="utf-8-sig")))
    test_e1 = next((r for r in bench_rows if r["split"] == "test" and r["config"] == "E1_tesseract_psm3"), None)
    test_e4 = next((r for r in bench_rows if r["split"] == "test" and r["config"] == "E4_routed_v02"), None)
    e1_cer = float(test_e1["micro_cer"]) if test_e1 else 1.0
    e4_cer = float(test_e4["micro_cer"]) if test_e4 else 1.0

    # Separate E0 automatic subset (exclude v01 manually proofread pages p0010_L & p0015_R)
    manual_v01_ids = {"dia_chi_loai_q5_p0010_L", "dia_chi_loai_q5_p0015_R"}
    e0_auto_rows = [r for r in per_sample_bench if r["split"] == "test" and r["config"] == "E0_v01_clean" and r["sample_id"] not in manual_v01_ids]
    e4_auto_rows = [r for r in per_sample_bench if r["split"] == "test" and r["config"] == "E4_routed_v02" and r["sample_id"] not in manual_v01_ids]
    e0_auto_cer = round(sum(int(r["S"]) + int(r["D"]) + int(r["I"]) for r in e0_auto_rows) / max(1, sum(int(r["N_chars"]) for r in e0_auto_rows)), 5)
    e4_auto_cer = round(sum(int(r["S"]) + int(r["D"]) + int(r["I"]) for r in e4_auto_rows) / max(1, sum(int(r["N_chars"]) for r in e4_auto_rows)), 5)

    results["4_benchmark_improvement"] = {
        "pass": (e4_cer < e1_cer) and (e4_auto_cer < e0_auto_cer),
        "test_e1_tesseract_micro_cer": e1_cer,
        "test_e4_routed_micro_cer": e4_cer,
        "test_e0_unedited_subset_cer": e0_auto_cer,
        "test_e4_unedited_subset_cer": e4_auto_cer,
    }

    # 5. Logs & release manifest
    req_logs = [
        os.path.join(DIR_NHAT_KY, "environment.txt"),
        os.path.join(DIR_NHAT_KY, "run_config.json"),
        os.path.join(DIR_NHAT_KY, "routing_table.json"),
        os.path.join(DIR_NHAT_KY, "edit_log.csv"),
        os.path.join(DIR_NHAT_KY, "excluded_regions.jsonl"),
        os.path.join(DIR_NHAT_KY, "review_queue.jsonl"),
        os.path.join(DIR_NHAT_KY, "step_run_log.jsonl"),
    ]
    missing_logs = [p for p in req_logs if not os.path.exists(p)]
    results["5_audit_logs"] = {
        "pass": len(missing_logs) == 0,
        "missing_logs": missing_logs,
    }

    overall_pass = all(v["pass"] for v in results.values())

    # Write release_manifest.json
    release = {
        "run_id": RUN_ID,
        "generated_at": now_vn_iso(),
        "overall_pass": overall_pass,
        "checks": results,
        "artifacts": {
            "viet_txt": {"path": viet_txt_p, "sha256": sha256_file(viet_txt_p), "bytes": os.path.getsize(viet_txt_p)},
            "viet_sentences_txt": {"path": viet_sents_p, "sha256": sha256_file(viet_sents_p), "bytes": os.path.getsize(viet_sents_p)},
            "sentence_map_jsonl": {"path": sent_map_p, "sha256": sha256_file(sent_map_p), "bytes": os.path.getsize(sent_map_p)},
        },
        "per_book_deliverables": [],
    }
    for b in BOOKS_META:
        bid = b["book_id"]
        up = os.path.join(DIR_TEXT_PER_BOOK, f"{bid}_units.jsonl")
        cp = os.path.join(DIR_TEXT_PER_BOOK, f"{bid}_viet_clean.txt")
        u_rows = [json.loads(l) for l in open(up, encoding="utf-8")]
        release["per_book_deliverables"].append({
            "book_id": bid,
            "primary_engine": b["primary_engine"],
            "clean_txt": cp,
            "units_jsonl": up,
            "total_units": len(u_rows),
            "align_units": sum(1 for u in u_rows if u["role"] in ("heading", "body", "translation")),
            "needs_review_units": sum(1 for u in u_rows if u["review_status"] == "needs_review"),
        })

    with open(os.path.join(DIR_NHAT_KY, "release_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(release, f, ensure_ascii=False, indent=2)

    # Generate concise, accurate report_v02.md
    per_sample_rows = list(csv.DictReader(open(os.path.join(DIR_DANH_GIA, "benchmark_per_sample.csv"), encoding="utf-8-sig")))
    md_lines = [
        "# Báo Cáo Thực Thi OCR Tiếng Việt v02 (Theo Kế Hoạch v02)",
        "",
        f"- **Run ID:** `{RUN_ID}` | **Thời gian hoàn tất:** `{now_vn_iso()}`",
        f"- **Trạng thái kiểm định:** `{'PASS' if overall_pass else 'FAIL'}`",
        "",
        "## 1. Kiến trúc Routing Thực Nghiệm (100% OCR mới từ PDF gốc)",
        "- **0 kế thừa text cũ (`ocr_output` / `ocr_v01`):** Toàn bộ **{}** leaf Việt/hỗn hợp được đọc mới trực tiếp từ 12 PDF gốc.".format(len(viet_leaves)),
        "- **Nhóm A – 6 quyển ảnh nét (`1-bit` rõ / `giấy vàng`):** `qtkb`, `hvddc`, `hvxt`, `lthclc_q13_19_quan_chuc`, `lthclc_q26_28_khoa_muc`, `lthclc_q29_32_quoc_dung` (`{}` leaf) → **Tesseract 5 (`vie_best`, `PSM 3`)** + tiền xử lý từng profile.".format(engine_counts["tesseract_best_psm3"]),
        "- **Nhóm B – 6 quyển ảnh mờ / độ phân giải thấp (`75–100 DPI`):** `dia_chi_loai_q5`, `hvll_q1..q5` (`{}` leaf) → **VietOCR `vgg_transformer`** trên các dòng cắt giữ dấu (`segment_lines_preserve_diacritics`) + chạy đối chứng **Tesseract 5** để ghi nhận hàng đợi bất đồng `06_nhat_ky/review_queue.jsonl`.".format(engine_counts["vietocr_vgg_transformer"]),
        "",
        "## 2. Bảng Benchmark Thực Đo (E0 v01 vs E4 v02)",
        "| Split | Cấu hình | Số mẫu | Micro CER | Macro CER | Median CER | P90 CER | Trang >5% | Trang >10% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in bench_rows:
        md_lines.append(
            f"| {r['split']} | {r['config']} | {r['n_samples']} | {float(r['micro_cer'])*100:.2f}% | "
            f"{float(r['macro_cer'])*100:.2f}% | {float(r['median_cer'])*100:.2f}% | {float(r['p90_cer'])*100:.2f}% | "
            f"{r['pages_gt_5pct']} | {r['pages_gt_10pct']} |"
        )

    md_lines.extend([
        "",
        "## 3. Chi tiết từng quyển bàn giao cho bước Hybrid Matching",
        "| book_id | Engine chính | Tổng unit | Unit dóng hàng (body/trans/heading) | Unit gắn cờ needs_review |",
        "|---|---|---:|---:|---:|",
    ])
    for pb in release["per_book_deliverables"]:
        md_lines.append(
            f"| `{pb['book_id']}` | `{pb['primary_engine']}` | {pb['total_units']} | {pb['align_units']} | {pb['needs_review_units']} |"
        )

    with open(os.path.join(V02_ROOT, "report_v02.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    for k, v in results.items():
        print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}: {v}")
    print("OVERALL_V02:", "PASS" if overall_pass else "FAIL")


if __name__ == "__main__":
    main()
