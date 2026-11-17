"""Step 9 & Step 10 (v02): Semantic region/role separation, header/footnote stripping without losing
body continuity, safe source-preserving cleaning, heading-aware paragraph merge, and
Hybrid-Matching-ready unit export (viet_clean.txt, *_units.jsonl, viet_sentences.txt)."""
import csv
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v02 import (
    BOOKS_META,
    DIR_NGUON,
    DIR_NHAT_KY,
    DIR_OCR_THO,
    DIR_OCR_THO_VIET,
    DIR_TEXT_PER_BOOK,
    DIR_TEXT_PER_PAGE,
    DIR_TEXT_SACH,
    META_BY_ID,
    RUN_ID,
    append_step_log,
    ensure_dirs,
    now_vn_iso,
    write_csv,
)
from ocr_engines_v02 import controlled_post_clean_v02, normalize_vietnamese_safe


ABBREV_TAIL_RE = re.compile(
    r"(?:\bQTKB|\btr|\bct|\bT|\bQ|\bN|\bP\.Đ|\bTP|\bGS|\bTS|\bThS|\bNXB|"
    r"^(?:[IVX]+|\d{1,3}))\.$"
)

HEADING_PREFIX_RE = re.compile(
    r"^(?:QUYỂN\s+|TỈNH\s+[A-ZÀ-ỸĐ]|PHỦ\s+[A-ZÀ-ỸĐ]|ĐIỀU\s+LỆ\s*:|"
    r"QUAN\s+TẤN\s*:|CỨU\s+MỤC\s*:|BƯU\s+DỊCH\s*:|ĐỊA-?CHÍ\s+LOẠI|"
    r"HOÀNG\s+VIỆT\s+LUẬT\s+LỆ|TỔNG\s+MỤC|MỤC\s+LỤC)"
)


def is_heading_line(s):
    s = s.strip()
    if not s or len(s) > 75:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.85:
        # Avoid false trigger on short lowercase-like or 1-word sentences ending with period
        if not s.endswith(".") or len(s.split()) <= 8:
            return True
    if HEADING_PREFIX_RE.match(s) and len(s.split()) <= 10:
        return True
    if re.match(r"^Mục\s+Lục\s+Quyển", s, re.I):
        return True
    return False


def classify_page_roles(book_id, pnum, kept_lines, raw_line_map=None):
    """Classify each kept line into semantic role:
    'toc', 'footnote', 'han_original', 'transliteration', 'translation', 'heading', or 'body'."""
    total = len(kept_lines)
    if total == 0:
        return []

    full_all = "\n".join(t for _, t in kept_lines)
    full_head = "\n".join(t for _, t in kept_lines[:4])
    is_frontmatter_page = (
        pnum <= 3
        and bool(re.search(r"(?:Chịu\s+trách\s+nhiệm\s+xuất\s+bản|Cục\s+Xuất\s+bản|In\s+xong\s+nộp\s+lưu|Luật\s+Gia\s+Long|NHÀ\s+XUẤT\s+BẢN|BẢN\s+DỊCH\s+của)", full_all, re.I))
    )
    is_toc_page = bool(re.search(r"(?:^\s*TỔNG\s+MỤC\b|^\s*MỤC\s+LỤC\b|Mục\s+Lục\s+Quyển)", full_head, re.I | re.M))

    # Detect hvxt poem mode transitions
    hvxt_mode = "body"
    # Detect footnote zone at bottom of page (only in prose/commentary books like qtkb, lthclc, hvddc)
    footnote_start_idx = total + 1
    if book_id in ("qtkb", "lthclc_q13_19_quan_chuc", "lthclc_q26_28_khoa_muc", "lthclc_q29_32_quoc_dung", "hvddc"):
        for idx in range(int(total * 0.55), total):
            _, t = kept_lines[idx]
            if re.match(r"^(?:\(\d+\)|\[\d+\]|Xem\s+(?:ct|chú\s+thích)\b)", t):
                footnote_start_idx = idx
                break

    classified = []
    for idx, (line_no, txt) in enumerate(kept_lines):
        s = txt.strip()
        if idx >= footnote_start_idx:
            classified.append((line_no, s, "footnote"))
            continue

        if is_frontmatter_page:
            classified.append((line_no, s, "frontmatter"))
            continue

        if is_toc_page:
            # Lines on TOC page (except if main article 1- ... starts mid-page like hvll_q4 p6!)
            if re.match(r"^1[-\.\s]+[A-ZÀ-ỸĐ]{2,}", s) and len(s.split()) >= 4:
                is_toc_page = False
            else:
                classified.append((line_no, s, "toc"))
                continue

        if book_id == "hvxt":
            if re.match(r"^Nguyên[\s\-]?âm\s*[:\.]?", s, re.I):
                hvxt_mode = "han_original"
                classified.append((line_no, s, "han_original"))
                continue
            elif re.match(r"^Phiên[\s\-]?âm\s*[:\.]?", s, re.I):
                hvxt_mode = "transliteration"
                classified.append((line_no, s, "transliteration"))
                continue
            elif re.match(r"^Dịch(?:\s+nghĩa|\s+thơ)?\s*[:\.]?", s, re.I):
                hvxt_mode = "translation"
                classified.append((line_no, s, "translation"))
                continue
            elif is_heading_line(s):
                hvxt_mode = "body"
                classified.append((line_no, s, "heading"))
                continue

        if hvxt_mode in ("han_original", "transliteration", "translation"):
            classified.append((line_no, s, hvxt_mode))
        elif is_heading_line(s):
            classified.append((line_no, s, "heading"))
        else:
            classified.append((line_no, s, "body"))

    return classified


def merge_lines_to_paragraphs(tagged_lines):
    """Merge wrapped lines within same role while strictly breaking on headings and role transitions."""
    paras = []
    cur_text = ""
    cur_lids = []
    cur_role = None

    def flush():
        nonlocal cur_text, cur_lids, cur_role
        if cur_text.strip():
            paras.append({"text": cur_text.strip(), "source_line_ids": cur_lids, "role": cur_role})
        cur_text = ""
        cur_lids = []
        cur_role = None

    for lid_str, line_txt, role in tagged_lines:
        s = line_txt.strip()
        if not s:
            flush()
            continue

        if role == "heading":
            flush()
            paras.append({"text": s, "source_line_ids": [lid_str], "role": "heading"})
            continue

        if cur_role is not None and role != cur_role:
            flush()

        if not cur_text:
            cur_text = s
            cur_lids = [lid_str]
            cur_role = role
            continue

        is_new_para_marker = bool(
            re.match(r"^(?:[-•\+]|\d+[\)\.\-]|Huyện\s+|Phủ\s+|Tỉnh\s+|\[Giải thích\])", s)
        )
        prev_terminal = cur_text[-1] in ".?!:\""
        if is_new_para_marker or (prev_terminal and len(cur_text) > 70 and not ABBREV_TAIL_RE.search(cur_text)):
            flush()
            cur_text = s
            cur_lids = [lid_str]
            cur_role = role
        else:
            if cur_text.endswith("-") and len(cur_text) >= 2 and cur_text[-2].isalpha() and s[0].isalpha():
                cur_text = cur_text[:-1] + s
            else:
                cur_text = cur_text + " " + s
            cur_lids.append(lid_str)

    flush()
    return paras


def split_sentences_v02(paragraph_text):
    """Context-aware Vietnamese sentence splitter preventing fragmentation on 'như: cha. anh. con.' or abbreviations."""
    txt = unicodedata.normalize("NFC", paragraph_text).strip()
    if not txt:
        return []

    raw_parts = re.split(r"(?<=[\.\?!;])\s+", txt)
    sentences = []
    buf = []

    for i, part in enumerate(raw_parts):
        p = part.strip()
        if not p:
            continue
        buf.append(p)
        joined_so_far = " ".join(buf)

        # Keep abbreviations/numbered prefixes attached
        if ABBREV_TAIL_RE.search(p) or re.match(r"^(?:\d+|[IVXLCDM]+|Điều\s+\d+|Khoản\s+\d+)\.$", p):
            continue
        if re.search(r"(?:\btr\.|\bQ\.|\btp\.|\b[A-ZÀ-Ỹ]\.)$", p):
            continue

        nxt = raw_parts[i + 1].strip() if i + 1 < len(raw_parts) else ""
        # Prevent fragmenting short list items after "như:" (e.g., "như: cha. anh. em, cháu. con.")
        if ("như:" in joined_so_far or "gồm:" in joined_so_far) and len(p.split()) <= 3 and nxt and len(nxt.split()) <= 3:
            continue
        # Keep short clauses joined across ';' unless buffer gets long
        if p.endswith(";") and nxt and (nxt[0].islower() or len(joined_so_far) < 150):
            continue

        sent = joined_so_far.strip()
        if sent:
            sentences.append(sent)
        buf = []

    if buf:
        rem = " ".join(buf).strip()
        if rem:
            sentences.append(rem)
    return sentences


def make_match_text(source_text):
    """Optional auxiliary matching form (does NOT overwrite source_text; Plan v02 Section 3)."""
    s = unicodedata.normalize("NFC", source_text)
    s = re.sub(r"\s*\(\d+\)\s*", " ", s)
    s = re.sub(r"\s*\[\d+\]\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    t_start = time.time()
    start_iso = now_vn_iso()
    ensure_dirs()

    page_map = list(csv.DictReader(open(os.path.join(DIR_NGUON, "page_map.csv"), encoding="utf-8-sig")))
    rq_map = {}
    rq_path = os.path.join(DIR_NHAT_KY, "review_queue.jsonl")
    if os.path.exists(rq_path):
        for line in open(rq_path, encoding="utf-8"):
            obj = json.loads(line)
            rq_map[(obj["book_id"], int(obj["pdf_page"]), obj["leaf_id"])] = obj

    edit_rows = []
    excluded_records = []
    edit_seq = 0

    book_order = {b["book_id"]: i for i, b in enumerate(BOOKS_META)}
    page_map_sorted = sorted(page_map, key=lambda r: (book_order[r["book_id"]], int(r["pdf_page"]), r["leaf_id"]))

    by_book_paras = {}
    for r in page_map_sorted:
        bid = r["book_id"]
        pnum = int(r["pdf_page"])
        lid = r["leaf_id"]
        if r["page_type"] not in ("viet", "mixed"):
            continue

        stem = f"page_{pnum:04d}_{lid}" if lid in ("L", "R") else f"page_{pnum:04d}"
        raw_path = os.path.join(DIR_OCR_THO_VIET, bid, f"{stem}.txt")
        if not os.path.exists(raw_path):
            continue
        raw_txt = unicodedata.normalize("NFC", open(raw_path, encoding="utf-8").read())

        kept_lines, excl_list, auto_edits = controlled_post_clean_v02(raw_txt, is_hv_book=bid.startswith("hv"))
        for line_no, orig_txt, excl_reason in excl_list:
            excluded_records.append({
                "book_id": bid,
                "pdf_page": pnum,
                "leaf_id": lid,
                "line_id": f"{bid}:p{pnum:04d}:{lid}:L{line_no:03d}",
                "role": excl_reason,
                "original_text": orig_txt,
            })

        for line_no, before_txt, after_txt, reason in auto_edits:
            edit_seq += 1
            edit_rows.append({
                "edit_id": f"E{edit_seq:06d}",
                "run_id": RUN_ID,
                "book_id": bid,
                "pdf_page": pnum,
                "leaf_id": lid,
                "line_id": f"{bid}:p{pnum:04d}:{lid}:L{line_no:03d}",
                "text_before": before_txt,
                "text_after": after_txt,
                "error_type": "char_spacing_v02",
                "reason": reason,
                "editor": "controlled_post_clean_v02",
                "timestamp": now_vn_iso(),
            })

        normalized_kept = []
        for line_no, txt in kept_lines:
            norm_txt = normalize_vietnamese_safe(txt)
            if norm_txt != txt:
                edit_seq += 1
                edit_rows.append({
                    "edit_id": f"E{edit_seq:06d}",
                    "run_id": RUN_ID,
                    "book_id": bid,
                    "pdf_page": pnum,
                    "leaf_id": lid,
                    "line_id": f"{bid}:p{pnum:04d}:{lid}:L{line_no:03d}",
                    "text_before": txt,
                    "text_after": norm_txt,
                    "error_type": "chuan_hoa_tieng_viet_v02",
                    "reason": "Safe deterministic Vietnamese OCR normalization",
                    "editor": "normalize_vietnamese_safe",
                    "timestamp": now_vn_iso(),
                })
            normalized_kept.append((line_no, norm_txt))

        classified = classify_page_roles(bid, pnum, normalized_kept)

        body_for_page_file = []
        tagged_for_merge = []
        for line_no, txt, role in classified:
            lid_str = f"{bid}:p{pnum:04d}:{lid}:L{line_no:03d}"
            if role == "footnote":
                excluded_records.append({
                    "book_id": bid,
                    "pdf_page": pnum,
                    "leaf_id": lid,
                    "line_id": lid_str,
                    "role": "footnote",
                    "original_text": txt,
                })
                continue
            body_for_page_file.append(txt)
            tagged_for_merge.append((lid_str, txt, role))

        book_page_dir = os.path.join(DIR_TEXT_PER_PAGE, bid)
        os.makedirs(book_page_dir, exist_ok=True)
        with open(os.path.join(book_page_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            page_str = "\n".join(body_for_page_file).strip()
            f.write(page_str + ("\n" if page_str else ""))

        if tagged_for_merge:
            paras = merge_lines_to_paragraphs(tagged_for_merge)
            by_book_paras.setdefault(bid, []).append((pnum, lid, paras))

    # Build per-book clean files, units.jsonl, and alignment-ready viet_sentences.txt
    all_book_clean_blocks = []
    sentences_out = []
    sentence_map_rows = []
    total_units = 0
    total_align_units = 0

    for meta in BOOKS_META:
        bid = meta["book_id"]
        page_paras_list = by_book_paras.get(bid, [])
        if not page_paras_list:
            continue

        merged_book_paras = []
        for pnum, lid, paras in page_paras_list:
            for idx_p, p_obj in enumerate(paras):
                txt = p_obj["text"]
                role = p_obj["role"]
                if (
                    idx_p == 0
                    and merged_book_paras
                    and merged_book_paras[-1]["role"] == role
                    and role in ("body", "translation")
                    and merged_book_paras[-1]["text"][-1] not in ".?!:\""
                    and txt
                    and txt[0].islower()
                ):
                    merged_book_paras[-1]["text"] = merged_book_paras[-1]["text"] + " " + txt
                    merged_book_paras[-1]["source_line_ids"].extend(p_obj["source_line_ids"])
                    merged_book_paras[-1]["pages"].append((pnum, lid))
                else:
                    merged_book_paras.append({
                        "text": txt,
                        "role": role,
                        "source_line_ids": list(p_obj["source_line_ids"]),
                        "pages": [(pnum, lid)],
                    })

        # Clean prose per-book (excluding toc/han_original/transliteration from main matching stream, keep heading/body/translation in clean prose)
        clean_book_paras = [p["text"] for p in merged_book_paras if p["role"] in ("heading", "body", "translation")]
        clean_book_txt = "\n\n".join(clean_book_paras).strip() + "\n"
        with open(os.path.join(DIR_TEXT_PER_BOOK, f"{bid}_viet_clean.txt"), "w", encoding="utf-8") as f:
            f.write(clean_book_txt)
        all_book_clean_blocks.append(f"=== {bid} | {meta['title']} ===\n\n{clean_book_txt.strip()}")

        # Export step 10 units (.jsonl + .txt) per book
        units_jsonl_p = os.path.join(DIR_TEXT_PER_BOOK, f"{bid}_units.jsonl")
        units_txt_p = os.path.join(DIR_TEXT_PER_BOOK, f"{bid}_units.txt")
        unit_order = 0
        book_align_lines = []
        with open(units_jsonl_p, "w", encoding="utf-8") as fu:
            for p_obj in merged_book_paras:
                role = p_obj["role"]
                p_sents = [p_obj["text"]] if role in ("heading", "toc", "han_original", "transliteration", "frontmatter") else split_sentences_v02(p_obj["text"])
                # Group source_refs by page
                refs_by_page = {}
                for lid_str in p_obj["source_line_ids"]:
                    parts = lid_str.split(":")
                    p_n = int(parts[1][1:])
                    lf = parts[2]
                    ln = parts[3]
                    refs_by_page.setdefault((p_n, lf), []).append(ln)
                source_refs = [
                    {"pdf_page": pn, "leaf_id": lf, "line_ids": lns}
                    for (pn, lf), lns in refs_by_page.items()
                ]
                has_rq = any(rq_map.get((bid, pn, lf), {}).get("status") == "needs_review" for pn, lf in refs_by_page)

                for s in p_sents:
                    unit_order += 1
                    total_units += 1
                    uid = f"{bid}_u{unit_order:05d}"
                    flags = []
                    if len(s.split()) <= 3:
                        flags.append("short_unit_le_3_tokens")
                    if has_rq:
                        flags.append("ocr_page_needs_review")
                    u_rec = {
                        "unit_id": uid,
                        "book_id": bid,
                        "chapter_id": "main",
                        "order": unit_order,
                        "role": role,
                        "source_refs": source_refs,
                        "source_text": s,
                        "match_text": make_match_text(s),
                        "review_status": "needs_review" if has_rq else "auto_pass",
                        "uncertainty_flags": flags,
                        "ocr_run_id": RUN_ID,
                    }
                    fu.write(json.dumps(u_rec, ensure_ascii=False) + "\n")

                    # Include alignment candidates ONLY from body/translation/heading (not toc, han_original, transliteration)
                    if role in ("heading", "body", "translation"):
                        total_align_units += 1
                        book_align_lines.append(s)
                        sentences_out.append(s)
                        sentence_map_rows.append(u_rec)

        with open(units_txt_p, "w", encoding="utf-8") as f:
            f.write("\n".join(book_align_lines) + ("\n" if book_align_lines else ""))

    with open(os.path.join(DIR_TEXT_SACH, "viet.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_book_clean_blocks).strip() + "\n")

    with open(os.path.join(DIR_TEXT_SACH, "viet_sentences.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sentences_out) + ("\n" if sentences_out else ""))

    with open(os.path.join(DIR_TEXT_SACH, "sentence_map.jsonl"), "w", encoding="utf-8") as f:
        for row in sentence_map_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        os.path.join(DIR_NHAT_KY, "edit_log.csv"),
        ["edit_id", "run_id", "book_id", "pdf_page", "leaf_id", "line_id", "text_before", "text_after", "error_type", "reason", "editor", "timestamp"],
        edit_rows,
    )

    for excl_p in [os.path.join(DIR_OCR_THO, "excluded_regions.jsonl"), os.path.join(DIR_NHAT_KY, "excluded_regions.jsonl")]:
        with open(excl_p, "w", encoding="utf-8") as f:
            for rec in excluded_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dt = round(time.time() - t_start, 2)
    print(
        f"[Step 9-10 v02] Done in {dt}s | total_units={total_units} | align_units={total_align_units} | "
        f"edits={len(edit_rows)} | excluded={len(excluded_records)}",
        flush=True,
    )

    append_step_log(
        "step_9_10_clean_and_units_v02",
        start_iso,
        now_vn_iso(),
        "Role-separated cleaning (excluding headers/footnotes/toc/transliteration from matching stream) and unit export",
        {"raw_dir": DIR_OCR_THO_VIET},
        "Controlled clean + safe normalizer + role classifier + heading-safe merge + sentence splitter",
        {"total_units": total_units, "align_units": total_align_units, "edits": len(edit_rows), "excluded": len(excluded_records)},
        {"nfc_only": True, "zero_cross_book_merges": True, "no_separator_lines_in_sentences": True},
        "03_text_sach/viet_per_book/*_units.jsonl, 03_text_sach/viet_sentences.txt, 06_nhat_ky/edit_log.csv",
        "Proceed to Step 11 independent verification and v02 report",
    )


if __name__ == "__main__":
    main()
