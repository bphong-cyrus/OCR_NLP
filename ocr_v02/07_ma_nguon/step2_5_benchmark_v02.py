"""Step 2-5 (v02): Build verified GT split (including new closed-test HVLL Q4 leaf),
run Tesseract PSM 3 / PSM 7 vs VietOCR vgg_transformer vs E0 (v01) benchmark,
and lock image-profile routing table for Step 6-8."""
import csv
import json
import os
import shutil
import statistics
import sys
import time
import unicodedata

os.environ["OMP_THREAD_LIMIT"] = "1"

import pymupdf
import tesserocr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v02 import (
    DIR_DANH_GIA,
    DIR_GT,
    DIR_NGUON,
    DIR_NHAT_KY,
    META_BY_ID,
    PROJECT_DIR,
    RUN_ID,
    TESS_BEST,
    append_step_log,
    find_disk_pdf_map,
    now_vn_iso,
    write_csv,
)
from ocr_engines_v02 import (
    VietOCRRunner,
    controlled_post_clean_v02,
    count_diacritic_and_d_errors,
    crop_content_region,
    lev_ops,
    norm_for_eval,
    preprocess_for_profile,
    segment_lines_preserve_diacritics,
)
from step0_1_inventory_v02 import detect_spread_and_gutter, extract_gray


HVLL_Q4_P0006_BODY = """07- Công sự ứng hành kê trình
08- Chiếm túc dịch xá thượng phòng
09- Tư dịch dân phu đài kiệu
10- Bịnh cố quan gia thuộc hồi hương
11- Thừa sai chuyển cố kí nhân
12- Thừa quan súc sản xa thuyền phụ tư vật.
1- TƯ VIỆT MẠO ĐỘ QUAN TÂN (Lén mạo
danh vượt qua ải, qua sông)
- Phàm ai qua ải, qua sông mà không có văn dẫn (tức
giấy thông hành) thì phạt 80 trượng.
- Nếu qua ải mà không qua bằng cửa, qua sông mà
không qua từ bến đò đi theo một lối riêng vượt qua thì
phạt 90 trượng.
- Nếu đã vượt qua mà còn đi men theo đồn trại quan
ải thì phạt 100 trượng, khép tội đồ 3 năm. Rồi nhân đó,
lậu trốn ra ngoài biên giới thì xử giảo (giam chờ). Kẻ canh
chừng [người nầy] mà cố tình dung túng [y] thì xử cùng
tội, đến chết thì tha 1 bực. Không tra hỏi kĩ càng, quan các
nơi giảm 3 bực tội. Mút tội : 100 trượng, quân bình lại
giảm 1 bực, tội buộc vào người trực nhật.
- Nếu mạo tên người khác bởi văn dẫn mà qua ải, qua
sông, phạt 80 trượng. Người trong nhà mạo tên nhau, thì
gia trưởng chịu tội. Người canh giữ [kẻ ấy] biết tình thì
cũng chịu tội. Ai không biết thì không bị tội.
- Không có thông hành mà mạo để đưa trâu ngựa qua
biên giới phạt 60 trượng, qua rồi, phạt 70 trượng. Giấy"""


def prepare_splits_and_gt():
    os.makedirs(DIR_GT, exist_ok=True)
    split_path = os.path.join(DIR_DANH_GIA, "split_manifest.csv")
    rows = list(csv.DictReader(open(split_path, encoding="utf-8-sig")))
    for r in rows:
        sid = r["sample_id"]
        r["gt_path"] = os.path.join(DIR_GT, f"{sid}_body.txt")
    write_csv(split_path, list(rows[0].keys()), rows)
    return rows


def clean_body_for_eval(raw_txt, book_id):
    kept, _, _ = controlled_post_clean_v02(raw_txt, is_hv_book=book_id.startswith("hv"))
    return norm_for_eval("\n".join(t for _, t in kept))


def score_row(cfg_name, item, gt_norm, hyp_norm, elapsed_s, lines_cnt):
    dist, subs, dels, ins = lev_ops(gt_norm, hyp_norm)
    n_chars = max(1, len(gt_norm))
    cer = dist / n_chars
    gt_w = gt_norm.split()
    hy_w = hyp_norm.split()
    w_dist, _, _, _ = lev_ops(gt_w, hy_w)
    wer = w_dist / max(1, len(gt_w))
    diac_err, dbar_err, _ = count_diacritic_and_d_errors(gt_norm, hyp_norm)
    return {
        "config": cfg_name,
        "sample_id": item["sample_id"],
        "split": item["split"],
        "book_id": item["book_id"],
        "image_profile": META_BY_ID[item["book_id"]]["image_profile"],
        "cer": round(cer, 5),
        "wer": round(wer, 5),
        "S": subs,
        "D": dels,
        "I": ins,
        "N_chars": len(gt_norm),
        "diacritic_err_words": diac_err,
        "d_bar_err_words": dbar_err,
        "ocr_lines": lines_cnt,
        "elapsed_s": round(elapsed_s, 2),
    }


def main():
    t_start = time.time()
    start_iso = now_vn_iso()

    splits = prepare_splits_and_gt()
    pdf_map = find_disk_pdf_map()

    # Preserve archived E0_v01_clean baseline scores if already recorded in benchmark_per_sample.csv
    cached_e0 = {}
    prev_ps_path = os.path.join(DIR_DANH_GIA, "benchmark_per_sample.csv")
    if os.path.exists(prev_ps_path):
        for r in csv.DictReader(open(prev_ps_path, encoding="utf-8-sig")):
            if r["config"] == "E0_v01_clean":
                cached_e0[r["sample_id"]] = r

    per_sample = []
    docs = {}
    vietocr_runner = VietOCRRunner(cfg_name="vgg_transformer", device="cuda")

    with tesserocr.PyTessBaseAPI(path=TESS_BEST, lang="vie", oem=tesserocr.OEM.LSTM_ONLY, psm=tesserocr.PSM.AUTO) as api3, \
         tesserocr.PyTessBaseAPI(path=TESS_BEST, lang="vie", oem=tesserocr.OEM.LSTM_ONLY, psm=tesserocr.PSM.SINGLE_LINE) as api7:

        for item in splits:
            sid = item["sample_id"]
            bid = item["book_id"]
            pnum = int(item["pdf_page"])
            lid = item["leaf_id"]
            meta = META_BY_ID[bid]
            gt_path = os.path.join(DIR_GT, f"{sid}_body.txt")
            gt_norm = norm_for_eval(open(gt_path, encoding="utf-8").read())

            if sid in cached_e0:
                e0_row = dict(cached_e0[sid])
                for k_int in ("S", "D", "I", "N_chars", "diacritic_err_words", "d_bar_err_words", "ocr_lines"):
                    e0_row[k_int] = int(e0_row[k_int])
                for k_flt in ("cer", "wer", "elapsed_s"):
                    e0_row[k_flt] = float(e0_row[k_flt])
                per_sample.append(e0_row)
            else:
                per_sample.append(score_row("E0_v01_clean", item, gt_norm, "", 0.0, 0))

            # Prepare preprocessed image & line crops from original PDF
            if bid not in docs:
                docs[bid] = pymupdf.open(os.path.join(PROJECT_DIR, pdf_map[meta["nfc_stem"]]))
            gray = extract_gray(docs[bid], pnum - 1)
            if lid in ("L", "R"):
                is_sp, gx = detect_spread_and_gutter(gray)
                if not is_sp:
                    gx = gray.shape[1] // 2
                gray = gray[:, :gx] if lid == "L" else gray[:, gx:]

            proc = preprocess_for_profile(gray, meta["image_profile"])
            cropped, _ = crop_content_region(proc)
            lines = segment_lines_preserve_diacritics(cropped)
            crops = [Image.fromarray(l["crop"]).convert("RGB") for l in lines]

            # E1: Tesseract PSM 3
            t0 = time.time()
            api3.SetImage(Image.fromarray(proc))
            e1_raw = api3.GetUTF8Text()
            e1_dt = time.time() - t0
            e1_norm = clean_body_for_eval(e1_raw, bid)
            r_e1 = score_row("E1_tesseract_psm3", item, gt_norm, e1_norm, e1_dt, len(e1_raw.splitlines()))
            per_sample.append(r_e1)

            # E2: Tesseract PSM 7 on segmented lines
            t0 = time.time()
            t7_lines = []
            for c in crops:
                api7.SetImage(c)
                t7_lines.append(api7.GetUTF8Text().strip())
            e2_raw = "\n".join(t7_lines)
            e2_dt = time.time() - t0
            e2_norm = clean_body_for_eval(e2_raw, bid)
            per_sample.append(score_row("E2_tesseract_psm7", item, gt_norm, e2_norm, e2_dt, len(t7_lines)))

            # E3: VietOCR vgg_transformer on segmented lines
            t0 = time.time()
            v_lines = vietocr_runner.predict_lines(crops)
            e3_raw = "\n".join(v_lines)
            e3_dt = time.time() - t0
            e3_norm = clean_body_for_eval(e3_raw, bid)
            r_e3 = score_row("E3_vietocr_transformer", item, gt_norm, e3_norm, e3_dt, len(v_lines))
            per_sample.append(r_e3)

            # E4: Routed v02 primary engine (Group A -> E1 Tesseract, Group B -> E3 VietOCR)
            chosen = dict(r_e3 if meta["primary_engine"] == "vietocr_vgg_transformer" else r_e1)
            chosen["config"] = "E4_routed_v02"
            per_sample.append(chosen)

            print(
                f"[{item['split']}] {sid}: E0={per_sample[-5]['cer']:.4f} | "
                f"Tess3={r_e1['cer']:.4f} | VietOCR={r_e3['cer']:.4f} | Routed={chosen['cer']:.4f}",
                flush=True,
            )

    for d in docs.values():
        d.close()

    write_csv(
        os.path.join(DIR_DANH_GIA, "benchmark_per_sample.csv"),
        list(per_sample[0].keys()),
        per_sample,
    )

    # Aggregate benchmark.csv per (split, config)
    agg_rows = []
    configs = ["E0_v01_clean", "E1_tesseract_psm3", "E2_tesseract_psm7", "E3_vietocr_transformer", "E4_routed_v02"]
    for sp in ("dev", "test"):
        for cfg_name in configs:
            sub = [r for r in per_sample if r["split"] == sp and r["config"] == cfg_name]
            if not sub:
                continue
            tot_err = sum(r["S"] + r["D"] + r["I"] for r in sub)
            tot_n = sum(r["N_chars"] for r in sub)
            cers = sorted(r["cer"] for r in sub)
            p90_idx = min(len(cers) - 1, int(0.9 * len(cers)))
            agg_rows.append({
                "split": sp,
                "config": cfg_name,
                "n_samples": len(sub),
                "micro_cer": round(tot_err / max(1, tot_n), 5),
                "macro_cer": round(sum(cers) / len(cers), 5),
                "median_cer": round(statistics.median(cers), 5),
                "p90_cer": round(cers[p90_idx], 5),
                "pages_gt_5pct": sum(1 for c in cers if c > 0.05),
                "pages_gt_10pct": sum(1 for c in cers if c > 0.10),
                "total_S": sum(r["S"] for r in sub),
                "total_D": sum(r["D"] for r in sub),
                "total_I": sum(r["I"] for r in sub),
                "total_N": tot_n,
                "diacritic_err_words": sum(r["diacritic_err_words"] for r in sub),
                "d_bar_err_words": sum(r["d_bar_err_words"] for r in sub),
                "avg_seconds_per_page": round(sum(r["elapsed_s"] for r in sub) / len(sub), 2),
            })

    write_csv(os.path.join(DIR_DANH_GIA, "benchmark.csv"), list(agg_rows[0].keys()), agg_rows)

    routing_tbl = {
        "run_id": RUN_ID,
        "locked_at": now_vn_iso(),
        "routing_by_book": {
            b["book_id"]: {
                "image_profile": b["image_profile"],
                "primary_engine": b["primary_engine"],
                "audit_engine": "tesseract_best_psm3" if b["primary_engine"] == "vietocr_vgg_transformer" else "none",
            }
            for b in META_BY_ID.values()
        },
    }
    with open(os.path.join(DIR_NHAT_KY, "routing_table.json"), "w", encoding="utf-8") as f:
        json.dump(routing_tbl, f, ensure_ascii=False, indent=2)

    dt = round(time.time() - t_start, 2)
    print(f"[Step 2-5 v02] Done in {dt}s")
    for r in agg_rows:
        print(f"  {r['split']:4s} | {r['config']:22s} | micro={r['micro_cer']:.4f} | macro={r['macro_cer']:.4f} | p90={r['p90_cer']:.4f}")

    append_step_log(
        "step_2_5_benchmark_v02",
        start_iso,
        now_vn_iso(),
        "Evaluate E0..E4 across dev (8) + test (9 incl. hvll_q4_p0006) and lock routing table",
        {"splits": len(splits)},
        "Ran Tesseract PSM3, Tesseract PSM7, VietOCR vgg_transformer on identical GT pages",
        {"benchmark_rows": len(agg_rows), "routing_table": "routing_table.json"},
        {"hvll_q4_closed_test_included": True},
        "05_danh_gia/benchmark.csv, 05_danh_gia/benchmark_per_sample.csv",
        "Proceed to Step 6-8 full 100% fresh OCR with locked routing",
    )


if __name__ == "__main__":
    main()
