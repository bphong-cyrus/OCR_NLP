"""Shared configuration, metadata, routing table, PDF helpers, and step logger (+07:00) for ocr_v02."""
import csv
import datetime
import hashlib
import json
import os
import unicodedata

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))

V02_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(V02_ROOT)
PROJECT_DIR = os.path.join(PROJECT_PARENT, "data") if os.path.isdir(os.path.join(PROJECT_PARENT, "data")) else PROJECT_PARENT
DIR_MODELS = os.path.join(V02_ROOT, "08_models")
DIR_NGUON = os.path.join(V02_ROOT, "01_nguon")
DIR_OCR_THO = os.path.join(V02_ROOT, "02_ocr_tho")
DIR_RAW_TESS = os.path.join(DIR_OCR_THO, "raw", "tesseract")
DIR_RAW_VIETOCR = os.path.join(DIR_OCR_THO, "raw", "vietocr")
DIR_OCR_THO_VIET = os.path.join(DIR_OCR_THO, "viet")
DIR_TEXT_SACH = os.path.join(V02_ROOT, "03_text_sach")
DIR_TEXT_PER_PAGE = os.path.join(DIR_TEXT_SACH, "viet_per_page")
DIR_TEXT_PER_BOOK = os.path.join(DIR_TEXT_SACH, "viet_per_book")
DIR_DONG_HANG = os.path.join(V02_ROOT, "04_dong_hang")
DIR_DANH_GIA = os.path.join(V02_ROOT, "05_danh_gia")
DIR_GT = os.path.join(DIR_DANH_GIA, "ground_truth")
DIR_LINE_IMAGES = os.path.join(DIR_DANH_GIA, "line_images")
DIR_OVERLAY = os.path.join(DIR_DANH_GIA, "overlay")
DIR_NHAT_KY = os.path.join(V02_ROOT, "06_nhat_ky")
DIR_MA_NGUON = os.path.join(V02_ROOT, "07_ma_nguon")

TESS_BEST = os.path.join(DIR_MODELS, "tessdata", "best") + os.sep
TESS_FAST = os.path.join(DIR_MODELS, "tessdata", "fast") + os.sep
VIETOCR_TRANSFORMER_WEIGHTS = os.path.join(DIR_MODELS, "weights", "vgg_transformer.pth")
VIETOCR_SEQ2SEQ_WEIGHTS = os.path.join(DIR_MODELS, "weights", "vgg_seq2seq.pth")
RUN_ID = "v02_20261116"

VN_DIACRITICS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)

VN_VOWELS = set(
    "aăâeêioôơuưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "AĂÂEÊIOÔƠUƯYÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
)

BOOKS_META = [
    {
        "book_id": "dia_chi_loai_q5",
        "title": "Phương Đình Dư Địa Chí - Địa Chí Loại quyển thứ 5",
        "nfc_stem": "Địa Chí Loại q5",
        "scope": "Quyển thứ năm (tr. 217-275), 30 trang PDF ghép đôi -> 60 trang sách",
        "image_profile": "lowres_yellow_spread",
        "primary_engine": "vietocr_vgg_transformer",
        "notes": "Ảnh ghép 2 trang/tờ PDF (~75 DPI), chữ nhỏ và mất nét",
    },
    {
        "book_id": "lthclc_q26_28_khoa_muc",
        "title": "Lịch triều hiến chương loại chí - Quyển 26-28 Khoa mục chí",
        "nfc_stem": "Lịch triều hiến chương loại chí Quyển 26-28 Khoa mục chí",
        "scope": "Quyển 26-28 Khoa mục chí, 80 trang PDF",
        "image_profile": "highres_bitonal",
        "primary_engine": "tesseract_best_psm3",
        "notes": "Scan đen trắng độ phân giải cao (~250-310 DPI); có lớp text rác",
    },
    {
        "book_id": "lthclc_q29_32_quoc_dung",
        "title": "Lịch triều hiến chương loại chí - Quyển 29-32 Quốc dụng chí",
        "nfc_stem": "Lịch triều hiến chương loại chí Quyển 29-32 Quốc dụng chí",
        "scope": "Quyển 29-32 Quốc dụng chí, 85 trang PDF",
        "image_profile": "highres_bitonal",
        "primary_engine": "tesseract_best_psm3",
        "notes": "Scan đen trắng độ phân giải cao (~247-308 DPI); có lớp text rác",
    },
    {
        "book_id": "lthclc_q13_19_quan_chuc",
        "title": "Lịch triều hiến chương loại chí - Quyển 13-19 Quan chức chí",
        "nfc_stem": "Lịch triều hiến chương loại chí Quyển 13-19 Quan chức chí",
        "scope": "Quyển 13-19 Quan chức chí, 177 trang PDF",
        "image_profile": "highres_bitonal",
        "primary_engine": "tesseract_best_psm3",
        "notes": "Scan đen trắng độ phân giải cao (~253-315 DPI); có lớp text rác",
    },
    {
        "book_id": "hvll_q1",
        "title": "Hoàng Việt Luật Lệ - quyển 1",
        "nfc_stem": "Hoàng Việt Luật Lệ - quyển 1",
        "scope": "Quyển 1, 299 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "gray_faint_midcol",
        "primary_engine": "vietocr_vgg_transformer",
        "notes": "Ảnh xám 800x1169; chữ in nghiêng mảnh và cột chữ mờ nét",
    },
    {
        "book_id": "hvddc",
        "title": "Hoàng Việt Địa Dư Chí",
        "nfc_stem": "Hoàng Việt Địa Dư Chí",
        "scope": "Toàn tập, 439 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "color_yellow_vignette",
        "primary_engine": "tesseract_best_psm3",
        "notes": "Ảnh chụp màu 1200x1695 trên giấy vàng, tối 4 góc và hằn chữ mặt sau",
    },
    {
        "book_id": "hvxt",
        "title": "Hoàng Việt Xuân Thu",
        "nfc_stem": "Hoàng Việt Xuân Thu",
        "scope": "Toàn tập, 493 trang PDF (bản dịch Việt + thơ Nguyên âm/Phiên âm/Dịch + nguyên bản Hán)",
        "image_profile": "highres_bitonal",
        "primary_engine": "tesseract_best_psm3",
        "notes": "Scan đen trắng ~400 DPI (2684x4108), xen thơ Nguyên âm + Phiên âm + Dịch nghĩa",
    },
    {
        "book_id": "hvll_q3",
        "title": "Hoàng Việt Luật Lệ - quyển 3",
        "nfc_stem": "Hoàng Việt Luật Lệ - quyển 3",
        "scope": "Quyển 3, 511 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "gray_faint_midcol",
        "primary_engine": "vietocr_vgg_transformer",
        "notes": "Ảnh xám 800x1169; có cột chữ mờ và chữ in nghiêng mảnh",
    },
    {
        "book_id": "qtkb",
        "title": "Quần Thư Khảo Biện",
        "nfc_stem": "Quần Thư Khảo Biện",
        "scope": "Toàn tập, 534 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "lowres_bitonal",
        "primary_engine": "tesseract_best_psm3",
        "notes": "Scan 1-bit 904x1167 (~100 DPI), có chú thích (1), (2) cuối trang",
    },
    {
        "book_id": "hvll_q2",
        "title": "Hoàng Việt Luật Lệ - quyển 2",
        "nfc_stem": "Hoàng Việt Luật Lệ - quyển 2",
        "scope": "Quyển 2, 542 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "gray_faint_midcol",
        "primary_engine": "vietocr_vgg_transformer",
        "notes": "Ảnh xám 800x1169; có cột chữ mờ và chữ in nghiêng mảnh",
    },
    {
        "book_id": "hvll_q5",
        "title": "Hoàng Việt Luật Lệ - quyển 5",
        "nfc_stem": "Hoàng Việt Luật Lệ - quyển 5",
        "scope": "Quyển 5, 672 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "gray_faint_midcol",
        "primary_engine": "vietocr_vgg_transformer",
        "notes": "Ảnh xám 800x1169; có cột chữ mờ và chữ in nghiêng mảnh",
    },
    {
        "book_id": "hvll_q4",
        "title": "Hoàng Việt Luật Lệ - quyển 4",
        "nfc_stem": "Hoàng Việt Luật Lệ - quyển 4",
        "scope": "Quyển 4, 697 trang PDF (bản dịch Việt + nguyên bản Hán)",
        "image_profile": "gray_faint_midcol",
        "primary_engine": "vietocr_vgg_transformer",
        "notes": "Ảnh xám 800x1034; trang 579 là trang Hán xoay ngang (không phải ghép đôi)",
    },
]

META_BY_ID = {b["book_id"]: b for b in BOOKS_META}


def ensure_dirs():
    for d in [
        V02_ROOT, DIR_NGUON, DIR_OCR_THO, DIR_RAW_TESS, DIR_RAW_VIETOCR,
        DIR_OCR_THO_VIET, DIR_TEXT_SACH, DIR_TEXT_PER_PAGE, DIR_TEXT_PER_BOOK,
        DIR_DONG_HANG, DIR_DANH_GIA, DIR_GT, DIR_LINE_IMAGES, DIR_OVERLAY,
        DIR_NHAT_KY, DIR_MA_NGUON,
    ]:
        os.makedirs(d, exist_ok=True)


def find_disk_pdf_map():
    mapping = {}
    for fname in os.listdir(PROJECT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        stem = os.path.splitext(unicodedata.normalize("NFC", fname))[0]
        mapping[stem] = fname
    return mapping


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_vn_iso():
    return datetime.datetime.now(VN_TZ).isoformat(timespec="seconds")


def append_step_log(step_id, start_time, end_time, goal, inputs, operations, outputs, checks, evidence, decision):
    os.makedirs(DIR_NHAT_KY, exist_ok=True)
    record = {
        "run_id": RUN_ID,
        "step_id": step_id,
        "operator": "ocr_pipeline_v02_agent",
        "start_time": start_time,
        "end_time": end_time,
        "goal": goal,
        "inputs": inputs,
        "operations": operations,
        "outputs": outputs,
        "checks": checks,
        "evidence": evidence,
        "decision": decision,
    }
    log_jsonl = os.path.join(DIR_NHAT_KY, "step_run_log.jsonl")
    with open(log_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
