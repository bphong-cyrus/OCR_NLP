"""Step 0 & Step 1 (v02): Independent source audit, SHA-256 manifest, source quality profile,
fresh spread detection & PDF-only Viet/Han classification (NO legacy ocr_output dependency)."""
import ctypes
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import unicodedata

os.environ["OMP_THREAD_LIMIT"] = "1"

import cv2
import numpy as np
import openpyxl
import PIL
import pymupdf
import tesserocr
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v02 import (
    BOOKS_META,
    DIR_NGUON,
    DIR_NHAT_KY,
    META_BY_ID,
    PROJECT_DIR,
    RUN_ID,
    TESS_BEST,
    TESS_FAST,
    VIETOCR_SEQ2SEQ_WEIGHTS,
    VIETOCR_TRANSFORMER_WEIGHTS,
    VN_DIACRITICS,
    append_step_log,
    ensure_dirs,
    find_disk_pdf_map,
    now_vn_iso,
    sha256_file,
    write_csv,
)


def get_ram_gb():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return round(stat.ullTotalPhys / (1024 ** 3), 2), round(stat.ullAvailPhys / (1024 ** 3), 2)


def capture_environment():
    total_ram, avail_ram = get_ram_gb()
    best_model = os.path.join(TESS_BEST, "vie.traineddata")
    fast_model = os.path.join(TESS_FAST, "vie.traineddata")
    best_sha = sha256_file(best_model)
    fast_sha = sha256_file(fast_model)
    vietocr_tf_sha = sha256_file(VIETOCR_TRANSFORMER_WEIGHTS)
    tess_ver = tesserocr.tesseract_version().replace("\r\n", " | ").replace("\n", " | ")
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_ONLY"

    env_lines = [
        f"run_id: {RUN_ID}",
        f"timestamp_vn: {now_vn_iso()}",
        f"os: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"logical_cpus: {os.cpu_count()}",
        f"gpu: {cuda_name}",
        f"torch_version: {torch.__version__}",
        f"ram_total_gb: {total_ram}",
        f"ram_avail_gb_at_start: {avail_ram}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
        f"tesserocr_version: {tesserocr.__version__}",
        f"tesseract_engine: {tess_ver}",
        f"pymupdf_version: {pymupdf.__version__}",
        f"opencv_version: {cv2.__version__}",
        f"numpy_version: {np.__version__}",
        f"pillow_version: {PIL.__version__}",
        f"openpyxl_version: {openpyxl.__version__}",
        f"tessdata_best_vie_path: {best_model}",
        f"tessdata_best_vie_sha256: {best_sha}",
        f"tessdata_fast_vie_path: {fast_model}",
        f"tessdata_fast_vie_sha256: {fast_sha}",
        f"vietocr_vgg_transformer_weights: {VIETOCR_TRANSFORMER_WEIGHTS}",
        f"vietocr_vgg_transformer_sha256: {vietocr_tf_sha}",
    ]
    with open(os.path.join(DIR_NHAT_KY, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(env_lines) + "\n")

    cfg = {
        "run_id": RUN_ID,
        "created_at": now_vn_iso(),
        "timezone": "+07:00",
        "omp_thread_limit": 1,
        "workers_inventory": 8,
        "workers_tesseract": 8,
        "gpu": cuda_name,
        "tessdata_best_sha256": best_sha,
        "vietocr_vgg_transformer_sha256": vietocr_tf_sha,
        "project_dir": PROJECT_DIR,
    }
    with open(os.path.join(DIR_NHAT_KY, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def extract_gray(doc, page_idx):
    page = doc[page_idx]
    imgs = page.get_images()
    if imgs:
        xref = imgs[0][0]
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n > 1:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    pix = page.get_pixmap(dpi=250, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def detect_spread_and_gutter(gray):
    h, w = gray.shape
    if w <= h * 1.15:
        return False, None
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35)
    norm = cv2.divide(gray, bg, scale=255)
    ink = norm < 210
    x0, x1 = int(0.43 * w), int(0.57 * w)
    col_ink = ink[int(0.08 * h):int(0.92 * h), x0:x1].mean(axis=0)
    sm = np.convolve(col_ink, np.ones(11) / 11, mode="same")
    mn = float(sm.min())
    if mn >= 0.02:
        return False, None
    cands = np.where(sm <= mn + 0.005)[0]
    gutter_x = x0 + int(np.median(cands))
    return True, gutter_x


_probe_fast_api = None


def _init_probe_worker():
    global _probe_fast_api
    _probe_fast_api = tesserocr.PyTessBaseAPI(path=TESS_FAST, lang="vie", psm=tesserocr.PSM.SINGLE_BLOCK)


def probe_leaf_fast(norm_gray):
    h, w = norm_gray.shape
    ink_ratio = float((norm_gray < 205).mean())
    if ink_ratio < 0.0020 or h < 80 or w < 80:
        return False, 0, 0, 0, ink_ratio, ""

    s = min(1.0, 900.0 / w) if w > 900 else max(1.0, 900.0 / w)
    small = cv2.resize(
        norm_gray,
        (int(w * s), max(80, int(h * s))),
        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
    )
    grad = np.abs(np.diff(small.astype(np.int16), axis=1)).mean(axis=1)
    win = max(40, int(small.shape[0] * 0.30))
    scores = np.convolve(grad, np.ones(win) / win, mode="valid")
    y0 = int(np.argmax(scores))
    x_pad = max(10, int(small.shape[1] * 0.05))
    crop = small[y0:y0 + win, x_pad:small.shape[1] - x_pad]

    _probe_fast_api.SetImage(Image.fromarray(crop))
    txt = unicodedata.normalize("NFC", _probe_fast_api.GetUTF8Text())
    conf = int(_probe_fast_api.MeanTextConf())
    words = re.findall(r"[a-zA-ZÀ-ỹ]+", txt)
    vn_chars = sum(1 for c in txt if c in VN_DIACRITICS)
    is_vn = (conf >= 52 and len(words) >= 2 and (vn_chars >= 1 or conf >= 70)) or (conf >= 40 and vn_chars >= 12)
    return is_vn, conf, vn_chars, len(words), ink_ratio, txt.strip()[:120]


def _inspect_book_chunk(args):
    """Inspect all pages of 1 PDF from raw embedded images only (zero old ocr_output dependency)."""
    meta, pdf_path = args
    book_id = meta["book_id"]
    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count

    rows = []
    qual_rows = []
    for page_idx in range(total_pages):
        pnum = page_idx + 1
        page = doc[page_idx]
        imgs = page.get_images(full=True)
        if imgs:
            img_w, img_h, bpc, cs_n = imgs[0][2], imgs[0][3], imgs[0][4], imgs[0][5]
        else:
            r = page.rect
            img_w, img_h, bpc, cs_n = int(r.width), int(r.height), 8, "Vector/Fallback"

        is_landscape = img_w > img_h * 1.15
        gray = extract_gray(doc, page_idx)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35)
        norm = cv2.divide(gray, bg, scale=255)
        is_spread, gutter_x = detect_spread_and_gutter(gray) if (book_id == "dia_chi_loai_q5" or is_landscape) else (False, None)

        qual_rows.append({
            "book_id": book_id,
            "pdf_page": pnum,
            "img_width": img_w,
            "img_height": img_h,
            "bpc": bpc,
            "colorspace": cs_n,
            "is_spread": 1 if is_spread else 0,
            "image_profile": meta["image_profile"],
            "primary_engine": meta["primary_engine"],
        })

        if is_spread:
            leaves = [
                ("L", 0, gutter_x, norm[:, :gutter_x], gray[:, :gutter_x]),
                ("R", gutter_x, gray.shape[1], norm[:, gutter_x:], gray[:, gutter_x:]),
            ]
        else:
            leaves = [("S", 0, gray.shape[1], norm, gray)]

        for leaf_id, x0, x1, leaf_norm, leaf_gray in leaves:
            lh, lw = leaf_gray.shape
            ink_ratio = float((leaf_norm < 205).mean())

            if ink_ratio < 0.0020:
                page_type = "blank"
                reason = f"ink_ratio={ink_ratio:.4f} < 0.0020"
                probe_conf, probe_vn, sample = 0, 0, ""
            elif is_landscape and not is_spread:
                page_type = "han"
                reason = f"rotated_single_han_woodblock(w={lw},h={lh},ink={ink_ratio:.4f})"
                probe_conf, probe_vn, sample = 0, 0, ""
            else:
                is_vn, probe_conf, probe_vn, wcnt, _, sample = probe_leaf_fast(leaf_norm)
                if is_vn:
                    if book_id == "dia_chi_loai_q5" and pnum == 30 and leaf_id == "L":
                        page_type = "mixed"
                        reason = f"spread_split_half; viet_plus_han_poem(conf={probe_conf}, vn={probe_vn})"
                    elif book_id == "hvxt" and ("Nguyên" in sample or "Phiên âm" in sample):
                        page_type = "mixed"
                        reason = f"viet_plus_han_block(conf={probe_conf}, vn={probe_vn})"
                    else:
                        page_type = "viet"
                        reason = f"fresh_probe_viet(conf={probe_conf}, vn={probe_vn}, words={wcnt})"
                else:
                    page_type = "han"
                    reason = f"fresh_probe_non_viet(conf={probe_conf}, vn={probe_vn}, ink={ink_ratio:.4f})"

            rows.append({
                "book_id": book_id,
                "pdf_page": pnum,
                "leaf_id": leaf_id,
                "region_id": "full" if leaf_id == "S" else f"half_{leaf_id}",
                "img_width": lw,
                "img_height": lh,
                "split_gutter_x": "" if gutter_x is None else gutter_x,
                "ink_ratio": f"{ink_ratio:.4f}",
                "page_type": page_type,
                "probe_conf": probe_conf,
                "probe_vn_chars": probe_vn,
                "classification_reason": reason,
            })

    doc.close()
    return rows, qual_rows


def main():
    t0 = time.time()
    start_iso = now_vn_iso()
    ensure_dirs()
    cfg = capture_environment()

    disk_map = find_disk_pdf_map()
    manifest_rows = []
    book_tasks = []
    for meta in BOOKS_META:
        stem = meta["nfc_stem"]
        fname = disk_map.get(stem)
        if not fname:
            raise FileNotFoundError(f"Missing PDF for {stem}")
        pdf_path = os.path.join(PROJECT_DIR, fname)
        doc = pymupdf.open(pdf_path)
        n_pages = doc.page_count
        doc.close()
        sha = sha256_file(pdf_path)
        manifest_rows.append({
            "book_id": meta["book_id"],
            "title": meta["title"],
            "pdf_filename_nfc": fname,
            "source_path": pdf_path,
            "file_size_bytes": os.path.getsize(pdf_path),
            "sha256": sha,
            "total_pdf_pages": n_pages,
            "image_profile": meta["image_profile"],
            "primary_engine": meta["primary_engine"],
            "scope": meta["scope"],
            "notes": meta["notes"],
        })
        book_tasks.append((meta, pdf_path))

    write_csv(
        os.path.join(DIR_NGUON, "source_manifest.csv"),
        ["book_id", "title", "pdf_filename_nfc", "source_path", "file_size_bytes", "sha256", "total_pdf_pages", "image_profile", "primary_engine", "scope", "notes"],
        manifest_rows,
    )

    page_map_rows = []
    qual_rows = []
    with mp.Pool(processes=8, initializer=_init_probe_worker) as pool:
        for b_rows, b_qual in pool.map(_inspect_book_chunk, book_tasks):
            page_map_rows.extend(b_rows)
            qual_rows.extend(b_qual)

    write_csv(
        os.path.join(DIR_NGUON, "source_quality.csv"),
        ["book_id", "pdf_page", "img_width", "img_height", "bpc", "colorspace", "is_spread", "image_profile", "primary_engine"],
        qual_rows,
    )
    write_csv(
        os.path.join(DIR_NGUON, "page_map.csv"),
        ["book_id", "pdf_page", "leaf_id", "region_id", "img_width", "img_height", "split_gutter_x", "ink_ratio", "page_type", "probe_conf", "probe_vn_chars", "classification_reason"],
        page_map_rows,
    )

    from collections import Counter
    tc = Counter(r["page_type"] for r in page_map_rows)
    bc = Counter(r["book_id"] for r in page_map_rows if r["page_type"] in ("viet", "mixed"))
    dt = round(time.time() - t0, 2)
    print(f"[Step 0-1 v02] Done in {dt}s | total_leaves={len(page_map_rows)} | types={dict(tc)}")
    for k, v in sorted(bc.items()):
        print(f"  {k}: {v}")

    append_step_log(
        "step_0_1_inventory_v02",
        start_iso,
        now_vn_iso(),
        "Lock v02 workspace, SHA-256 source manifest, source_quality.csv, fresh PDF probe page_map.csv",
        {"pdf_count": len(manifest_rows)},
        "PyMuPDF image extract + adaptive spread split + fresh Tesseract FAST probe on 8 workers",
        {"total_leaves": len(page_map_rows), "page_types": dict(tc), "viet_mixed_per_book": dict(bc)},
        {"all_12_pdfs_hashed": True, "zero_legacy_ocr_dependency": True},
        "source_manifest.csv, source_quality.csv, page_map.csv",
        "Proceed to Step 2-5 dev/test GT & Tesseract vs VietOCR benchmark",
    )


if __name__ == "__main__":
    main()
