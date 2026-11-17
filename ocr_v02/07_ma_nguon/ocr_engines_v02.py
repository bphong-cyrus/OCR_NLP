"""Core preprocessing, layout/line segmentation, VietOCR GPU batch runner, controlled cleaning,
and exact evaluation metrics for ocr_v02."""
from collections import defaultdict
import os
import re
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common_v02 import VIETOCR_SEQ2SEQ_WEIGHTS, VIETOCR_TRANSFORMER_WEIGHTS, VN_DIACRITICS, VN_VOWELS


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def remove_cjk_and_empty_parens(s):
    s = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf]+", "", s)
    s = re.sub(r"\(\s*\)", "", s)
    return s


def preprocess_for_profile(gray, image_profile):
    """Diacritic-preserving preprocessing tailored per image quality group (Step 3)."""
    h, w = gray.shape
    if image_profile == "highres_bitonal":
        im = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=0.45))
        return np.array(im)

    if image_profile == "lowres_bitonal":
        scale = 3 if w < 1200 else 2
        im = Image.fromarray(gray).resize((w * scale, h * scale), Image.LANCZOS)
        return np.array(im.filter(ImageFilter.GaussianBlur(radius=1.0)))

    if image_profile == "gray_faint_midcol":
        scale = 3 if w < 1200 else 2
        big = np.array(Image.fromarray(gray).resize((w * scale, h * scale), Image.LANCZOS))
        bg = cv2.GaussianBlur(big, (0, 0), sigmaX=35 * scale)
        norm = cv2.divide(big, bg, scale=255)
        lut = np.array(
            [min(255, int(((i / 255.0) ** 1.75) * 255)) if i < 242 else 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(norm, lut)

    if image_profile == "color_yellow_vignette":
        scale = 2 if w < 1500 else 1
        big = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC) if scale > 1 else gray
        bg = cv2.GaussianBlur(big, (0, 0), sigmaX=35 * scale)
        clean = cv2.divide(big, bg, scale=255)
        clean = np.clip((clean.astype(np.float32) - 220) * (255.0 / 220.0) + 255.0, 0, 255).astype(np.uint8)
        return clean

    if image_profile == "lowres_yellow_spread":
        scale = 3 if w < 900 else 2
        big = np.array(Image.fromarray(gray).resize((w * scale, h * scale), Image.LANCZOS))
        bg = cv2.GaussianBlur(big, (0, 0), sigmaX=30 * scale)
        clean = cv2.divide(big, bg, scale=255)
        lut = np.array(
            [min(255, int(((i / 255.0) ** 1.65) * 255)) if i < 238 else 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(clean, lut)

    return gray


def deskew_image(proc_gray):
    h, w = proc_gray.shape
    s = min(1.0, 600.0 / w)
    small = cv2.resize(
        (proc_gray < 180).astype(np.uint8),
        (max(50, int(w * s)), max(50, int(h * s))),
        interpolation=cv2.INTER_NEAREST,
    )
    best_a, best_var = 0.0, -1.0
    cx, cy = small.shape[1] / 2.0, small.shape[0] / 2.0
    for a in np.linspace(-3.0, 3.0, 61):
        M = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        v = float(rot.sum(axis=1).var())
        if v > best_var:
            best_var, best_a = v, float(a)
    if abs(best_a) >= 0.15:
        M_full = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_a, 1.0)
        deskewed = cv2.warpAffine(proc_gray, M_full, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        return deskewed, round(best_a, 2)
    return proc_gray, 0.0


def crop_content_region(proc_gray):
    deskewed, angle = deskew_image(proc_gray)
    h, w = deskewed.shape
    ink = (deskewed < 185).astype(np.uint8)
    inner_x0 = int(0.05 * w)
    inner_x1 = int(0.95 * w)
    row_sum = ink[:, inner_x0:inner_x1].sum(axis=1)
    col_sum = ink[int(0.03 * h):int(0.97 * h), :].sum(axis=0)

    ys = np.where(row_sum >= 4)[0]
    xs = np.where(col_sum >= 6)[0]
    if len(ys) == 0 or len(xs) == 0:
        return deskewed, (0, 0, w, h, angle)

    pad_x = max(18, int(0.025 * w))
    pad_y = max(18, int(0.02 * h))
    x0 = max(0, int(xs[0]) - pad_x)
    x1 = min(w, int(xs[-1]) + 1 + pad_x)
    y0 = max(0, int(ys[0]) - pad_y)
    y1 = min(h, int(ys[-1]) + 1 + pad_y)
    cropped = deskewed[y0:y1, x0:x1]
    if cropped.shape[0] < 40 or cropped.shape[1] < 40:
        return deskewed, (0, 0, w, h, angle)
    return cropped, (int(x0), int(y0), int(x1), int(y1), angle)


def segment_lines_preserve_diacritics(proc_gray):
    """Segment horizontal text lines while strictly preserving Vietnamese upper/lower diacritics (Step 4)."""
    h, w = proc_gray.shape
    bin_inv = (proc_gray < 180).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)

    raw_comps = []
    for idx in range(1, num_labels):
        cx, cy, cw, ch, area = stats[idx]
        if area < 5:
            continue
        if cw > 0.85 * w and ch > 0.85 * h:
            continue
        if cw > 0.55 * w and ch < 14:
            continue
        if (cx < 0.06 * w or cx + cw > 0.94 * w) and ch > 0.08 * h and cw < 0.06 * w:
            continue
        if ch > 0.15 * h and cw < 0.05 * w:
            continue
        raw_comps.append((int(cx), int(cy), int(cx + cw), int(cy + ch), int(cw), int(ch), int(area)))

    if not raw_comps:
        return []

    heights = [c[5] for c in raw_comps if 8 <= c[5] <= 0.08 * h]
    if not heights:
        heights = [c[5] for c in raw_comps if c[5] >= 6]
    if not heights:
        return []
    med_h = float(np.median(heights))

    comps = [c for c in raw_comps if c[5] <= max(3.2 * med_h, 0.05 * h)]
    if not comps:
        comps = raw_comps

    body_comps = [c for c in comps if c[5] >= max(6, 0.50 * med_h)]
    small_comps = [c for c in comps if c[5] < max(6, 0.50 * med_h)]
    if not body_comps:
        body_comps = comps
        small_comps = []

    body_proj = np.zeros(h, dtype=np.int32)
    for _, y0, _, y1, cw, ch, _ in body_comps:
        if ch <= 2.3 * med_h:
            body_proj[y0:y1] += cw

    row_thresh = max(12, int(0.012 * w))
    in_band = body_proj >= row_thresh
    raw_bands = []
    start = None
    for y in range(h):
        if in_band[y] and start is None:
            start = y
        elif not in_band[y] and start is not None:
            raw_bands.append([start, y])
            start = None
    if start is not None:
        raw_bands.append([start, h])

    merged_bands = []
    for b in raw_bands:
        if merged_bands and (b[0] - merged_bands[-1][1]) <= max(2, int(0.15 * med_h)):
            merged_bands[-1][1] = b[1]
        else:
            merged_bands.append(b)

    def _split_tall_band(b0, b1):
        bh = b1 - b0
        if bh <= max(24, int(2.75 * med_h)):
            return [[b0, b1]]
        m0 = b0 + int(0.28 * bh)
        m1 = b1 - int(0.28 * bh)
        if m1 <= m0 + 2:
            return [[b0, b1]]
        band_peak = int(body_proj[b0:b1].max()) if b1 > b0 else 0
        sub_proj = body_proj[m0:m1]
        min_val = int(sub_proj.min())
        if band_peak > 0 and min_val > 0.25 * band_peak:
            return [[b0, b1]]
        split_y = m0 + int(np.argmin(sub_proj))
        return _split_tall_band(b0, split_y) + _split_tall_band(split_y + 1, b1)

    split_bands = []
    for b0, b1 in merged_bands:
        split_bands.extend(_split_tall_band(b0, b1))

    line_bands = [b for b in split_bands if (b[1] - b[0]) >= max(5, int(0.35 * med_h))]
    if not line_bands:
        return []

    lines_comps = [[] for _ in line_bands]
    for c in body_comps:
        cy_mid = 0.5 * (c[1] + c[3])
        best_i = min(range(len(line_bands)), key=lambda i: abs(cy_mid - 0.5 * (line_bands[i][0] + line_bands[i][1])))
        by0, by1 = line_bands[best_i]
        if c[1] <= by1 + 0.8 * med_h and c[3] >= by0 - 0.8 * med_h:
            lines_comps[best_i].append(c)

    for c in small_comps:
        _, cy0, _, cy1, _, _, _ = c
        cy_mid = 0.5 * (cy0 + cy1)
        best_i = None
        best_dist = 1e9
        for i, (by0, by1) in enumerate(line_bands):
            if cy0 >= by0 - 1.15 * med_h and cy1 <= by1 + 0.9 * med_h:
                dy = 0.0 if by0 <= cy_mid <= by1 else min(abs(cy_mid - by0), abs(cy_mid - by1))
                if cy_mid < by0:
                    dy *= 0.85
                if dy < best_dist:
                    best_dist = dy
                    best_i = i
        if best_i is not None:
            lines_comps[best_i].append(c)

    results = []
    order = 0
    edge_blob_w = max(20, int(0.035 * w))
    for i, lc in enumerate(lines_comps):
        if not lc:
            continue
        lc_sorted = sorted(lc, key=lambda c: c[0])
        if len(lc_sorted) >= 3:
            # Trim only extreme scanner edge specks separated by >14% of page width
            while len(lc_sorted) >= 3 and lc_sorted[0][4] < edge_blob_w and (lc_sorted[1][0] - lc_sorted[0][2]) > 0.14 * w:
                lc_sorted.pop(0)
            while len(lc_sorted) >= 3 and lc_sorted[-1][4] < edge_blob_w and (lc_sorted[-1][0] - lc_sorted[-2][2]) > 0.14 * w:
                lc_sorted.pop(-1)

        lx0 = min(c[0] for c in lc_sorted)
        ly0 = min(c[1] for c in lc_sorted)
        lx1 = max(c[2] for c in lc_sorted)
        ly1 = max(c[3] for c in lc_sorted)
        if (lx1 - lx0) < max(12, int(0.8 * med_h)) and sum(c[6] for c in lc_sorted) < 80:
            continue

        pad_top = max(5, int(0.18 * med_h))
        pad_bot = max(4, int(0.14 * med_h))
        pad_x = max(10, int(0.35 * med_h))

        if i > 0:
            prev_bot = line_bands[i - 1][1]
            safe_top = max(0, min(ly0 - 2, max(prev_bot + 1, ly0 - pad_top)))
        else:
            safe_top = max(0, ly0 - pad_top)
        if i + 1 < len(line_bands):
            next_top = line_bands[i + 1][0]
            safe_bot = min(h, max(ly1 + 2, min(next_top - 1, ly1 + pad_bot)))
        else:
            safe_bot = min(h, ly1 + pad_bot)

        sx0 = max(0, lx0 - pad_x)
        sx1 = min(w, lx1 + pad_x)
        crop = proc_gray[safe_top:safe_bot, sx0:sx1]
        if crop.size == 0:
            continue
        padded_crop = cv2.copyMakeBorder(crop, 10, 10, 14, 14, cv2.BORDER_CONSTANT, value=255)
        order += 1
        results.append({
            "line_id": f"line_{order:03d}",
            "reading_order": order,
            "bbox": [int(sx0), int(safe_top), int(sx1), int(safe_bot)],
            "line_height": int(safe_bot - safe_top),
            "crop": padded_crop,
        })
    return results


def draw_line_overlay(proc_gray, lines):
    rgb = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 150, 0), width=2)
        draw.text((max(2, x0 - 24), y0), f"#{ln['reading_order']}", fill=(200, 0, 0))
    return im


class VietOCRRunner:
    """Persistent GPU VietOCR predictor with multi-page width-bucketed batching."""

    def __init__(self, cfg_name="vgg_transformer", device="cuda"):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name(cfg_name)
        cfg["device"] = device
        wpath = VIETOCR_TRANSFORMER_WEIGHTS if cfg_name == "vgg_transformer" else VIETOCR_SEQ2SEQ_WEIGHTS
        if os.path.exists(wpath):
            cfg["weights"] = wpath
        cfg["predictor"]["beamsearch"] = False
        self.cfg_name = cfg_name
        self.weights_path = wpath
        self.rec = Predictor(cfg)
        self.rec.model.eval()
        # Warmup
        dummy = Image.new("RGB", (120, 32), color=(255, 255, 255))
        _ = self.rec.predict_batch([dummy])

    def predict_lines(self, pil_crops, return_prob=False):
        if not pil_crops:
            return ([], []) if return_prob else []
        return self.rec.predict_batch(pil_crops, return_prob=return_prob)


# ---------------- Safe Vietnamese Normalizer & Post-Cleaning ----------------

def _is_valid_vn_word(w):
    return len(w) >= 2 and any(c in VN_VOWELS for c in w)


def _clean_garbled_paren(m):
    inner = m.group(1).strip()
    if re.match(r"^[\d\.,\-]+$", inner):
        return m.group(0)
    words = re.findall(r"[a-zA-ZÀ-ỹ]+", inner)
    if any(_is_valid_vn_word(w) for w in words) or re.search(r"\d{4}", inner):
        return m.group(0)
    return ""


SAFE_VN_RULES = [
    (r"\bđươc\b", "được"),
    (r"\bĐươc\b", "Được"),
    (r"\bdược\b", "được"),
    (r"\bkhỏng\b", "không"),
    (r"\bkhong\s+(có|được|phải|thể|biết|thuộc|nên|cho|đồng|chắc|thế)\b", r"không \1"),
    (r"\bngừơi\b", "người"),
    (r"\btrưóc\b", "trước"),
    (r"\bchưong\b", "chương"),
    (r"\bhoác\b", "hoặc"),
    (r"\bhoạc\b", "hoặc"),
    (r"(?<![A-Za-z0-9])[ØO0]{1,2}[T7]KB\b", "QTKB"),
    (r"\b(?:nbà|nhì|nhả|chà|phì|ahà)\s+(Lê|Minh|Trần|Nguyễn|Đường|Tống|Hán|Lý|Đinh|Tây\s+Sơn)\b", r"nhà \1"),
    (r"\b(?:nbà|nhà)\s+(?:12|I2|1ê|Le|Lẻ)\b", "nhà Lê"),
    (r"\b(?:địt|đít|dịt)\b(?=\s+(?:\d|làm|là|thêm|riêng|chức|tổng|xã))", "đặt"),
    (r"\b(nhà\s+(?:Lê|Minh|Trần|Nguyễn)|lại|mới)\s+(?:địt|đít|dịt|đạt)\b", r"\1 đặt"),
    (r"\bv[nư]\s+theo\b", "vẫn theo"),
    (r"\b(?:tính|tỉnh)\s+(?:nhữp|nhiêp|sbiếp|chiếp|phiếp|nhấp)\b", "tính nhiếp"),
    (r"\b(?:kiệm|kiềm)\s+lý\b", "kiêm lý"),
    (r"\b(?:Hnyện|Huyền|Huyệu|Hưyện|buyện|buyệo|bưyện|huy;n|huyệp|hoyện|buyển|huyên)\b", "huyện"),
    (r"\b(?:Hnyện|Huyên|Buyện)\b", "Huyện"),
    (r"\bTh;ch\s+Osi\b", "Thanh Oai"),
    (r"\bĐỏ\s+động\s+cgis\b", "Đỗ Động cũ"),
    (r"(xã|thôn|trang|trại|phường)[,\.]\s*(?:tỏ|tổ|sở-|sử)\.", r"\1, sở."),
    (r"\bphai\s+(chịu|nộp|đóng|theo|xử|biết|làm|có|lấy|giữ|báo|tra|xét)\b", r"phải \1"),
    (r"\bphái\s+(chịu|nộp|đóng|theo|xử|biết|làm|có|lấy|giữ|báo|tra|xét)\b", r"phải \1"),
    (r"\bpham\s+(tội|tọi|tôi|phép|luật|người|roi)\b", r"phạm \1"),
    (r"\b(phạm|chịu|mắc|luận|xử|kết|bị|tạp)\s+(?:tôi|tọi)\b", r"\1 tội"),
    (r"\b(?:tôi|tọi)\s+(đồ|lưu|chết|dư|nặng|nhẹ|nhân|danh|trạng|sái|gốc)\b", r"tội \1"),
    (r"\bmôi\s+(trường\s+hợp|người|lần|khi|bực|năm|tháng|ngày|thứ|tên)\b", r"mỗi \1"),
    (r"\b(số|chịu|phạt|đánh|thêm|một trăm|100)\s+trương\b", r"\1 trượng"),
    (r"\bgiam\s+(bực|một\s+bực|hai\s+bực|1\s+năm|nhẹ|tội)\b", r"giảm \1"),
    (r"\btrần\s+trong\b", "trân trọng"),
    (r"\bcong\s+bô\b", "công bố"),
    (r"\bđịnh\s+đoat\b", "định đoạt"),
    (r"\btang\s+phuc\b", "tang phục"),
    (r"\bhon\s+nhan\b", "hôn nhân"),
    (r"\bbị\s+phat\b", "bị phạt"),
    (r"\bđặc\s+biết\b", "đặc biệt"),
    (r"\bđạc\s+biết\b", "đặc biệt"),
]

_COMPILED_V02 = [(re.compile(p, flags=re.UNICODE), r) for p, r in SAFE_VN_RULES]


def normalize_vietnamese_safe(text):
    s = unicodedata.normalize("NFC", text)
    s = re.sub(r"\(([^\(\)]{1,24})\)", _clean_garbled_paren, s)
    for rx, rep in _COMPILED_V02:
        s = rx.sub(rep, s)
    s = re.sub(r"([a-zà-ỹđ]):([a-zà-ỹđA-ZÀ-ỸĐ])", r"\1: \2", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,\.;!\?\)\]])", r"\1", s)
    return s.strip()


def controlled_post_clean_v02(raw_text, is_hv_book=False):
    """Controlled cleaning separating excluded regions (headers, page numbers, smudges) from body lines.
    Never drops all-caps book titles ('HOÀNG VIỆT LUẬT LỆ'). Preserves archaic forms."""
    text = unicodedata.normalize("NFC", raw_text)
    lines = text.splitlines()

    kept = []
    excluded = []
    edits = []

    weird_sym = set("$%#^|&*~<>§®©±÷×¤¦¨ª«¬¯°²³´µ¶¸¹º»¼½¾¿")

    non_empty_rank = 0
    for idx, orig_line in enumerate(lines, 1):
        s = orig_line.strip()
        if not s:
            continue

        if non_empty_rank <= 2 and len(s) <= 4 and not any(c in VN_DIACRITICS for c in s) and not re.match(r"^\d{1,4}$", s):
            excluded.append((idx, s, "top_edge_smudge"))
            continue

        if re.match(r"^[\W_0-9\-<>]{4,}$", s) and not re.match(r"^\d{1,4}$", s):
            excluded.append((idx, s, "separator_line"))
            continue

        non_empty_rank += 1

        # Running headers at top of page (preserve all-caps book titles without page numbers!)
        is_all_caps_title = (s == "HOÀNG VIỆT LUẬT LỆ" or s == "ĐỊA-CHÍ LOẠI" or s == "PHƯƠNG-ĐÌNH DƯ ĐỊA CHÍ")
        if non_empty_rank <= 2 and not is_all_caps_title and (
            re.match(r"^[\d%O]{1,4}\s+[A-ZÀ-ỸĐ][a-zA-ZÀ-ỸĐà-ỹ\-\s\.\,\':;]{4,}$", s)
            or re.match(r"^[A-ZÀ-ỸĐ][a-zA-ZÀ-ỸĐà-ỹ\-\s\.\,\':;]{4,}\s+\d{1,4}$", s)
            or re.search(
                r"(?:^\d{1,4}\s*PHƯƠNG[\.\-\s]*Đ[IỈĨ]NH\s+D[UƯ]|PHƯƠNG[\.\-\s]*Đ[IỈĨ]NH\s+D[UƯ]\s+ĐỊA\s+CH[IÍ]\s*\d{0,4}$"
                r"|Hoàng\s+Việt\s+luật\s+lệ|^\d{1,4}\s+Hoàng\s+Việt|Quan\s+chức\s+chí\s*\d{0,4}$|Khoa\s+mục\s+chí\s*\d{0,4}$|Quốc\s+dụng\s+chí\s*\d{0,4}$)",
                s,
            )
            or re.match(r"^\d{1,4}\s+(?:Hoàng Việt|Lịch triều)", s, re.I)
        ):
            excluded.append((idx, s, "running_header"))
            continue

        if (non_empty_rank <= 2 or idx >= len(lines) - 2) and re.match(r"^[\dIVXLCDM]{1,6}$", s):
            excluded.append((idx, s, "page_number"))
            continue

        words = re.findall(r"[a-zA-ZÀ-ỹ]+", s)
        valid_vn = [w for w in words if _is_valid_vn_word(w)]
        weird_cnt = sum(1 for c in s if c in weird_sym)
        if len(valid_vn) == 0 and (len(s) < 35 or weird_cnt >= 2) and not re.match(r"^\d{1,4}$", s):
            excluded.append((idx, s, "inline_han_or_noise"))
            continue
        if weird_cnt >= 3 and len(valid_vn) <= 1:
            excluded.append((idx, s, "high_weird_symbol_noise"))
            continue

        # Strip VietOCR border/artwork hallucinations (e.g. 0000000, repeated token >= 4x, or pure-ASCII English/caps gibberish)
        if re.search(r"0{5,}", s) and not any(c in VN_DIACRITICS for c in s):
            excluded.append((idx, s, "long_zero_border_noise"))
            continue
        if re.search(r"\b([A-Za-zÀ-ỹĐđ]+)(?:\s+\1){3,}\b", s, re.I):
            excluded.append((idx, s, "repeated_token_hallucination"))
            continue
        if not any(c in VN_DIACRITICS for c in s):
            ascii_words = re.findall(r"[A-Za-z]+", s)
            if ascii_words and all(ord(c) < 128 for c in s) and not re.match(r"^[\d\s\.\-\(\)/:,]+$", s):
                # English-style uppercase gibberish >=5 letters or English suffix hallucination, not Roman numerals
                has_eng_gib = any(
                    (len(w) >= 5 and w.isupper() and not re.match(r"^[IVXLCDM]+$", w))
                    or re.search(r"(?:tion|ally|ical|ized|less|ness|ing|mm|zz|xx)$", w, re.I)
                    for w in ascii_words
                )
                if has_eng_gib or s in ("Spire", "AZVA", "VM", "arr:", "e"):
                    excluded.append((idx, s, "ascii_artwork_hallucination"))
                    continue

        new_s = s
        new_s = re.sub(r"^[\|¬°©§†‡•·~_=\*\+\s]{1,3}(?=[A-ZÀ-ỸĐ0-9\"\(])", "", new_s)
        new_s = new_s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        new_s = new_s.translate(str.maketrans({"ö": "ô", "Ö": "Ô", "ä": "a", "Ä": "A", "ð": "đ", "Ð": "Đ"}))
        new_s = re.sub(r"[ \t]{2,}", " ", new_s)
        new_s = re.sub(r"\s+([,\.;!\?\)\]])", r"\1", new_s).strip()

        if new_s != s:
            edits.append((idx, s, new_s, "deterministic_char_spacing_fix"))
        kept.append((idx, new_s))

    return kept, excluded, edits


# ---------------- Evaluation Metrics ----------------

def norm_for_eval(s, strip_cjk=True):
    s = unicodedata.normalize("NFC", s)
    if strip_cjk:
        s = remove_cjk_and_empty_parens(s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def lev_ops(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            d_del = prow[j] + 1
            d_ins = row[j - 1] + 1
            d_sub = prow[j - 1] + cost
            v = d_del if d_del <= d_ins else d_ins
            if d_sub < v:
                v = d_sub
            row[j] = v

    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            if a[i - 1] != b[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return dp[n][m], subs, dels, ins


def count_diacritic_and_d_errors(gt_text, hyp_text):
    import difflib

    gt_w = gt_text.split()
    hy_w = hyp_text.split()
    sm = difflib.SequenceMatcher(
        a=[strip_accents(w.lower()) for w in gt_w],
        b=[strip_accents(w.lower()) for w in hy_w],
        autojunk=False,
    )
    diacritic_err_words = 0
    d_bar_err_words = 0
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            for gw, hw in zip(gt_w[a0:a1], hy_w[b0:b1]):
                if gw != hw:
                    if gw.replace("đ", "d").replace("Đ", "D") != hw.replace("đ", "d").replace("Đ", "D"):
                        if ("đ" in gw.lower()) != ("đ" in hw.lower()):
                            d_bar_err_words += 1
                    else:
                        diacritic_err_words += 1
    return diacritic_err_words, d_bar_err_words, len(gt_w)
