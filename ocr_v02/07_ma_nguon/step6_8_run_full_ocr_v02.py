"""Step 6 & Step 8 (v02): 100% fresh OCR from original PDFs using locked routing table.
- Group A (6 crisp books): Tesseract 5 vie_best PSM 3 (CPU parallel workers)
- Group B (6 faint/low-res books: dia_chi_loai_q5, hvll_q1..q5):
  * Tesseract 5 vie_best PSM 3 (CPU audit pass) -> 02_ocr_tho/raw/tesseract/
  * VietOCR vgg_transformer on diacritic-preserving line crops (GPU batched) -> 02_ocr_tho/raw/vietocr/
  * Dual-engine disagreement & numeric audit -> 06_nhat_ky/review_queue.jsonl
- Strict provenance (.json sidecar with source_sha256, model_sha256, config_hash) and full resumability."""
import csv
import difflib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata

os.environ["OMP_THREAD_LIMIT"] = "1"

import pymupdf
import tesserocr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v02 import (
    BOOKS_META,
    DIR_NGUON,
    DIR_NHAT_KY,
    DIR_OCR_THO,
    DIR_OCR_THO_VIET,
    DIR_RAW_TESS,
    DIR_RAW_VIETOCR,
    META_BY_ID,
    PROJECT_DIR,
    RUN_ID,
    TESS_BEST,
    VIETOCR_TRANSFORMER_WEIGHTS,
    append_step_log,
    ensure_dirs,
    find_disk_pdf_map,
    now_vn_iso,
    sha256_file,
    sha256_text,
)
from ocr_engines_v02 import (
    VietOCRRunner,
    crop_content_region,
    norm_for_eval,
    preprocess_for_profile,
    segment_lines_preserve_diacritics,
)
from step0_1_inventory_v02 import detect_spread_and_gutter, extract_gray


_w_tess_api = None


def _init_tess_worker():
    global _w_tess_api
    _w_tess_api = tesserocr.PyTessBaseAPI(
        path=TESS_BEST, lang="vie", oem=tesserocr.OEM.LSTM_ONLY, psm=tesserocr.PSM.AUTO
    )


def _valid_conf_from_tsv(tsv_text):
    confs = []
    for line in tsv_text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 12 and parts[11].strip():
            try:
                c = float(parts[10])
                if c >= 0:
                    confs.append(c)
            except ValueError:
                pass
    return round(sum(confs) / len(confs), 2) if confs else 0.0, len(confs)


def _is_cached_ok(meta_path, expected_src_sha, expected_model_sha, expected_cfg_hash, txt_path):
    if not (os.path.exists(meta_path) and os.path.exists(txt_path)):
        return False
    try:
        m = json.load(open(meta_path, encoding="utf-8"))
        return (
            m.get("status") == "ok"
            and m.get("source_pdf_sha256") == expected_src_sha
            and m.get("model_sha256") == expected_model_sha
            and m.get("config_hash") == expected_cfg_hash
        )
    except Exception:
        return False


def _run_tess_leaf(task):
    bid, pnum, lid, pdf_path, src_sha, model_sha, cfg_hash, copy_to_canonical = task
    meta = META_BY_ID[bid]
    stem = f"page_{pnum:04d}_{lid}" if lid in ("L", "R") else f"page_{pnum:04d}"

    raw_dir = os.path.join(DIR_RAW_TESS, bid)
    os.makedirs(raw_dir, exist_ok=True)
    txt_path = os.path.join(raw_dir, f"{stem}.txt")
    tsv_path = os.path.join(raw_dir, f"{stem}.tsv")
    hocr_path = os.path.join(raw_dir, f"{stem}.hocr")
    json_path = os.path.join(raw_dir, f"{stem}.json")

    can_dir = os.path.join(DIR_OCR_THO_VIET, bid)
    os.makedirs(can_dir, exist_ok=True)
    can_txt = os.path.join(can_dir, f"{stem}.txt")

    if _is_cached_ok(json_path, src_sha, model_sha, cfg_hash, txt_path) and os.path.exists(tsv_path):
        if copy_to_canonical and not os.path.exists(can_txt):
            with open(txt_path, encoding="utf-8") as fi, open(can_txt, "w", encoding="utf-8") as fo:
                fo.write(fi.read())
        m = json.load(open(json_path, encoding="utf-8"))
        return (bid, pnum, lid, m.get("mean_conf", 0.0), True)

    t0 = time.time()
    doc = pymupdf.open(pdf_path)
    gray = extract_gray(doc, pnum - 1)
    doc.close()

    if lid in ("L", "R"):
        is_sp, gx = detect_spread_and_gutter(gray)
        if not is_sp:
            gx = gray.shape[1] // 2
        gray = gray[:, :gx] if lid == "L" else gray[:, gx:]

    proc = preprocess_for_profile(gray, meta["image_profile"])
    _w_tess_api.SetImage(Image.fromarray(proc))
    _w_tess_api.Recognize()
    txt = unicodedata.normalize("NFC", _w_tess_api.GetUTF8Text())
    tsv_str = _w_tess_api.GetTSVText(0)
    hocr_str = _w_tess_api.GetHOCRText(0)
    mean_c, w_cnt = _valid_conf_from_tsv(tsv_str)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(tsv_str)
    with open(hocr_path, "w", encoding="utf-8") as f:
        f.write(hocr_str)

    if copy_to_canonical:
        with open(can_txt, "w", encoding="utf-8") as f:
            f.write(txt)

    rec_meta = {
        "run_id": RUN_ID,
        "book_id": bid,
        "pdf_page": pnum,
        "leaf_id": lid,
        "engine": "tesseract_best_psm3",
        "image_profile": meta["image_profile"],
        "source_pdf_sha256": src_sha,
        "model_sha256": model_sha,
        "config_hash": cfg_hash,
        "mean_conf": mean_c,
        "word_count": w_cnt,
        "elapsed_s": round(time.time() - t0, 2),
        "timestamp": now_vn_iso(),
        "status": "ok",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec_meta, f, ensure_ascii=False, indent=2)

    return (bid, pnum, lid, mean_c, False)


def _norm_disagreement(a, b):
    na = norm_for_eval(a)
    nb = norm_for_eval(b)
    if not na and not nb:
        return 0.0
    sm = difflib.SequenceMatcher(a=na, b=nb, autojunk=False)
    # normalized edit distance proxy via 1 - ratio
    return round(1.0 - sm.ratio(), 4)


def run_group_b_vietocr(group_b_leaves, pdf_map, src_sha_by_book, model_sha, cfg_hash):
    """Run VietOCR vgg_transformer in multi-page GPU batches across Group B leaves with checkpointing."""
    pending = []
    docs = {}

    for r in group_b_leaves:
        bid = r["book_id"]
        pnum = int(r["pdf_page"])
        lid = r["leaf_id"]
        stem = f"page_{pnum:04d}_{lid}" if lid in ("L", "R") else f"page_{pnum:04d}"

        v_dir = os.path.join(DIR_RAW_VIETOCR, bid)
        c_dir = os.path.join(DIR_OCR_THO_VIET, bid)
        os.makedirs(v_dir, exist_ok=True)
        os.makedirs(c_dir, exist_ok=True)
        txt_p = os.path.join(v_dir, f"{stem}.txt")
        json_p = os.path.join(v_dir, f"{stem}.json")
        can_p = os.path.join(c_dir, f"{stem}.txt")

        if _is_cached_ok(json_p, src_sha_by_book[bid], model_sha, cfg_hash, txt_p):
            if not os.path.exists(can_p):
                with open(txt_p, encoding="utf-8") as fi, open(can_p, "w", encoding="utf-8") as fo:
                    fo.write(fi.read())
            continue
        pending.append((bid, pnum, lid, stem, txt_p, json_p, can_p))

    print(f"[Step 6-8 VietOCR] cached={len(group_b_leaves)-len(pending)} | pending={len(pending)}", flush=True)
    if not pending:
        return

    runner = VietOCRRunner(cfg_name="vgg_transformer", device="cuda")
    chunk_size = 40  # ~1200 line crops per GPU batched call (safe on 6GB VRAM, maximizes width bucket batching)
    done_cnt = 0

    for c_start in range(0, len(pending), chunk_size):
        chunk = pending[c_start:c_start + chunk_size]
        t0 = time.time()
        all_crops = []
        leaf_slices = []

        for bid, pnum, lid, stem, txt_p, json_p, can_p in chunk:
            meta = META_BY_ID[bid]
            if bid not in docs:
                docs[bid] = pymupdf.open(os.path.join(PROJECT_DIR, pdf_map[meta["nfc_stem"]]))
            gray = extract_gray(docs[bid], pnum - 1)
            if lid in ("L", "R"):
                is_sp, gx = detect_spread_and_gutter(gray)
                if not is_sp:
                    gx = gray.shape[1] // 2
                gray = gray[:, :gx] if lid == "L" else gray[:, gx:]

            proc = preprocess_for_profile(gray, meta["image_profile"])
            cropped, crop_box = crop_content_region(proc)
            lines = segment_lines_preserve_diacritics(cropped)
            s_idx = len(all_crops)
            for l in lines:
                all_crops.append(Image.fromarray(l["crop"]).convert("RGB"))
            e_idx = len(all_crops)
            leaf_slices.append((bid, pnum, lid, stem, txt_p, json_p, can_p, crop_box, lines, s_idx, e_idx, proc.shape))

        preds, probs = runner.predict_lines(all_crops, return_prob=True)
        dt = time.time() - t0
        per_leaf_dt = round(dt / max(1, len(chunk)), 2)

        for bid, pnum, lid, stem, txt_p, json_p, can_p, crop_box, lines, s_idx, e_idx, proc_shape in leaf_slices:
            l_preds = [unicodedata.normalize("NFC", p).strip() for p in preds[s_idx:e_idx]]
            l_probs = [round(float(pr), 4) if pr is not None else 0.0 for pr in probs[s_idx:e_idx]]
            raw_txt = "\n".join(l_preds) + ("\n" if l_preds else "")

            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(raw_txt)
            with open(can_p, "w", encoding="utf-8") as f:
                f.write(raw_txt)

            line_records = [
                {
                    "line_id": ln["line_id"],
                    "reading_order": ln["reading_order"],
                    "bbox_in_crop": ln["bbox"],
                    "prob": l_probs[idx],
                    "text": l_preds[idx],
                }
                for idx, ln in enumerate(lines)
            ]
            meta_obj = {
                "run_id": RUN_ID,
                "book_id": bid,
                "pdf_page": pnum,
                "leaf_id": lid,
                "engine": "vietocr_vgg_transformer",
                "image_profile": META_BY_ID[bid]["image_profile"],
                "source_pdf_sha256": src_sha_by_book[bid],
                "model_sha256": model_sha,
                "config_hash": cfg_hash,
                "proc_shape": list(proc_shape),
                "content_crop_box": list(crop_box),
                "line_count": len(lines),
                "mean_prob": round(sum(l_probs) / len(l_probs), 4) if l_probs else 0.0,
                "elapsed_s_est": per_leaf_dt,
                "timestamp": now_vn_iso(),
                "status": "ok" if len(l_preds) > 0 else "empty",
                "lines": line_records,
            }
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(meta_obj, f, ensure_ascii=False, indent=2)

        done_cnt += len(chunk)
        print(
            f"[Step 6-8 VietOCR] chunk {done_cnt}/{len(pending)} ({len(all_crops)} crops in {dt:.1f}s, {per_leaf_dt}s/leaf)",
            flush=True,
        )

    for d in docs.values():
        d.close()


def build_regions_line_map_and_review_queue(viet_leaves):
    """Create regions.jsonl, line_map.jsonl, and Step 6 dual-engine review_queue.jsonl."""
    regions_path = os.path.join(DIR_OCR_THO, "regions.jsonl")
    line_map_path = os.path.join(DIR_OCR_THO, "line_map.jsonl")
    rq_path = os.path.join(DIR_NHAT_KY, "review_queue.jsonl")

    reg_cnt = 0
    lm_cnt = 0
    rq_cnt = 0

    with open(regions_path, "w", encoding="utf-8") as fr, \
         open(line_map_path, "w", encoding="utf-8") as fl, \
         open(rq_path, "w", encoding="utf-8") as fq:

        for r in viet_leaves:
            bid = r["book_id"]
            pnum = int(r["pdf_page"])
            lid = r["leaf_id"]
            stem = f"page_{pnum:04d}_{lid}" if lid in ("L", "R") else f"page_{pnum:04d}"
            meta = META_BY_ID[bid]
            prim = meta["primary_engine"]

            fr.write(json.dumps({
                "book_id": bid,
                "pdf_page": pnum,
                "leaf_id": lid,
                "region_id": r["region_id"],
                "primary_engine": prim,
                "img_width": int(r["img_width"]),
                "img_height": int(r["img_height"]),
                "split_gutter_x": r["split_gutter_x"],
                "page_type": r["page_type"],
            }, ensure_ascii=False) + "\n")
            reg_cnt += 1

            v_json = os.path.join(DIR_RAW_VIETOCR, bid, f"{stem}.json")
            t_txt_p = os.path.join(DIR_RAW_TESS, bid, f"{stem}.txt")
            t_tsv_p = os.path.join(DIR_RAW_TESS, bid, f"{stem}.tsv")

            if prim == "vietocr_vgg_transformer" and os.path.exists(v_json):
                vm = json.load(open(v_json, encoding="utf-8"))
                v_full = open(os.path.join(DIR_RAW_VIETOCR, bid, f"{stem}.txt"), encoding="utf-8").read()
                for ln in vm.get("lines", []):
                    fl.write(json.dumps({
                        "book_id": bid,
                        "pdf_page": pnum,
                        "leaf_id": lid,
                        "line_id": ln["line_id"],
                        "reading_order": ln["reading_order"],
                        "bbox": ln["bbox_in_crop"],
                        "prob": ln["prob"],
                    }, ensure_ascii=False) + "\n")
                    lm_cnt += 1

                # Dual-engine disagreement check (VietOCR vs Tesseract)
                if os.path.exists(t_txt_p):
                    t_full = open(t_txt_p, encoding="utf-8").read()
                    disagree = _norm_disagreement(v_full, t_full)
                    nums_v = re.findall(r"\d+", v_full)
                    nums_t = re.findall(r"\d+", t_full)
                    if disagree > 0.14 or nums_v != nums_t or vm.get("mean_prob", 1.0) < 0.86:
                        fq.write(json.dumps({
                            "book_id": bid,
                            "pdf_page": pnum,
                            "leaf_id": lid,
                            "primary_engine": "vietocr_vgg_transformer",
                            "audit_engine": "tesseract_best_psm3",
                            "normalized_disagreement": disagree,
                            "vietocr_mean_prob": vm.get("mean_prob", 0.0),
                            "nums_mismatch": nums_v != nums_t,
                            "nums_vietocr": nums_v[:12],
                            "nums_tesseract": nums_t[:12],
                            "status": "needs_review" if (disagree > 0.22 or vm.get("mean_prob", 1.0) < 0.80) else "spot_check",
                        }, ensure_ascii=False) + "\n")
                        rq_cnt += 1
            elif os.path.exists(t_tsv_p):
                # Low-conf Tesseract words queue for Group A
                low_toks = []
                for line in open(t_tsv_p, encoding="utf-8"):
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 12 and parts[11].strip():
                        try:
                            c = float(parts[10])
                            if 0 <= c < 65.0:
                                low_toks.append({"token": parts[11].strip(), "conf": c, "bbox": parts[6:10]})
                        except ValueError:
                            pass
                if low_toks:
                    fq.write(json.dumps({
                        "book_id": bid,
                        "pdf_page": pnum,
                        "leaf_id": lid,
                        "primary_engine": "tesseract_best_psm3",
                        "low_conf_count": len(low_toks),
                        "samples": low_toks[:8],
                        "status": "needs_review" if len(low_toks) >= 8 else "spot_check",
                    }, ensure_ascii=False) + "\n")
                    rq_cnt += 1

    return reg_cnt, lm_cnt, rq_cnt


def main():
    t0 = time.time()
    start_iso = now_vn_iso()
    ensure_dirs()

    manifest = {r["book_id"]: r for r in csv.DictReader(open(os.path.join(DIR_NGUON, "source_manifest.csv"), encoding="utf-8-sig"))}
    page_map = list(csv.DictReader(open(os.path.join(DIR_NGUON, "page_map.csv"), encoding="utf-8-sig")))
    pdf_map = find_disk_pdf_map()

    tess_sha = sha256_file(os.path.join(TESS_BEST, "vie.traineddata"))
    vietocr_sha = sha256_file(VIETOCR_TRANSFORMER_WEIGHTS)
    cfg_hash = sha256_text(json.dumps(META_BY_ID, sort_keys=True, ensure_ascii=False))
    src_sha_by_book = {bid: m["sha256"] for bid, m in manifest.items()}

    viet_leaves = [r for r in page_map if r["page_type"] in ("viet", "mixed")]
    group_a = [r for r in viet_leaves if META_BY_ID[r["book_id"]]["primary_engine"] == "tesseract_best_psm3"]
    group_b = [r for r in viet_leaves if META_BY_ID[r["book_id"]]["primary_engine"] == "vietocr_vgg_transformer"]

    print(f"[Step 6-8 v02] Total viet/mixed={len(viet_leaves)} | Group A (Tesseract)={len(group_a)} | Group B (VietOCR+Tess audit)={len(group_b)}", flush=True)

    # 1. Run Tesseract across BOTH Group A (canonical) and Group B (audit engine for Step 6)
    tess_tasks = []
    for r in group_a:
        bid = r["book_id"]
        pdf_p = os.path.join(PROJECT_DIR, pdf_map[META_BY_ID[bid]["nfc_stem"]])
        tess_tasks.append((bid, int(r["pdf_page"]), r["leaf_id"], pdf_p, src_sha_by_book[bid], tess_sha, cfg_hash, True))
    for r in group_b:
        bid = r["book_id"]
        pdf_p = os.path.join(PROJECT_DIR, pdf_map[META_BY_ID[bid]["nfc_stem"]])
        tess_tasks.append((bid, int(r["pdf_page"]), r["leaf_id"], pdf_p, src_sha_by_book[bid], tess_sha, cfg_hash, False))

    print(f"[Step 6-8 Tesseract CPU pass] Running {len(tess_tasks)} leaves on 8 workers...", flush=True)
    t_tess = time.time()
    cached_t = 0
    with mp.Pool(processes=8, initializer=_init_tess_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_tess_leaf, tess_tasks, chunksize=8), 1):
            if res[4]:
                cached_t += 1
            if i % 250 == 0 or i == len(tess_tasks):
                print(f"  Tesseract pass {i}/{len(tess_tasks)} (cached={cached_t}, elapsed={time.time()-t_tess:.1f}s)", flush=True)

    # 2. Run VietOCR GPU batched across Group B (canonical)
    run_group_b_vietocr(group_b, pdf_map, src_sha_by_book, vietocr_sha, cfg_hash)

    # 3. Build regions, line_map, and Step 6 review_queue
    reg_cnt, lm_cnt, rq_cnt = build_regions_line_map_and_review_queue(viet_leaves)
    dt = round(time.time() - t0, 2)
    print(f"[Step 6-8 v02] Finished in {dt}s | regions={reg_cnt} | lines={lm_cnt} | review_queue={rq_cnt}", flush=True)

    append_step_log(
        "step_6_8_full_ocr_v02",
        start_iso,
        now_vn_iso(),
        "100% fresh OCR of all Viet/mixed leaves from source PDFs via routed Tesseract + VietOCR",
        {"viet_mixed_leaves": len(viet_leaves), "group_a_tess": len(group_a), "group_b_vietocr": len(group_b)},
        "Parallel 8-worker Tesseract PSM3 + GPU multi-page batched VietOCR vgg_transformer + dual-engine audit",
        {"regions": reg_cnt, "line_map_entries": lm_cnt, "review_queue_entries": rq_cnt},
        {"zero_legacy_text_imported": True, "all_raw_immutable_with_hashes": True},
        "02_ocr_tho/raw/, 02_ocr_tho/viet/, 06_nhat_ky/review_queue.jsonl",
        "Proceed to Step 9-10 clean, role separation, and alignment-ready units",
    )


if __name__ == "__main__":
    main()
