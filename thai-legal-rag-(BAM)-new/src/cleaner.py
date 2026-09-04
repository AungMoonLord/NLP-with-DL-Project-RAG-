"""STEP 1.5 — ทำความสะอาดข้อความที่ได้จากการแปลง PDF

ปัญหาที่พบในคลังเอกสารจริง และวิธีรับมือ:

  (A) หมายเหตุริมหน้ากระดาษปนเข้ามากลางเนื้อความ  -> ลบได้ (ความมั่นใจสูง)
  (B) บรรทัดถูกตัดกลางประโยคตามความกว้างหน้ากระดาษ -> ต่อกลับได้ (ความมั่นใจสูง)
  (C) ช่องว่างแทรกกลางคำ เช่น "แสดงต นตาม"        -> แก้ได้ด้วยพจนานุกรม (ความมั่นใจกลาง)
  (D) สระ/วรรณยุกต์หายไป เช่น "ขอบังคับ"           -> แก้อัตโนมัติไม่ได้ ต้องมีรายการที่คนตรวจ

ทำไม (D) แก้อัตโนมัติไม่ได้:
  "ขอบังคับ" ตัดคำได้เป็น ["ขอ", "บังคับ"] ซึ่ง *เป็นคำจริงทั้งคู่ในพจนานุกรม*
  เครื่องจึงไม่มีทางรู้ว่ามันผิด ต้องใช้ความรู้ว่าเอกสารนี้พูดถึง "ข้อบังคับ"
  จึงต้องให้มนุษย์ยืนยันผ่านไฟล์ data/corrections.txt

หลักการออกแบบ: ทุกการแก้ไขถูกบันทึกลง report เสมอ ไม่มีการแก้แบบเงียบ ๆ
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .thai_utils import PYTHAINLP_AVAILABLE, normalize_text, word_tokenize

# ---------- พจนานุกรมไทย ----------
if PYTHAINLP_AVAILABLE:
    from pythainlp.corpus import thai_words
    THAI_DICT = set(thai_words())
else:  # pragma: no cover
    THAI_DICT = set()

THAI_CHAR = r"\u0E00-\u0E7F"
MAIYAMOK = "\u0E46"          # ๆ
RE_THAI = re.compile(f"[{THAI_CHAR}]")

# บรรทัดที่เป็น "จุดเริ่มบล็อกใหม่" — ห้ามนำไปต่อท้ายบรรทัดก่อนหน้า
RE_BLOCK_START = re.compile(
    r"^\s*(?:มาตรา\s*[0-9]|ข้อ\s*[0-9]|\([0-9ก-ฮ]+\)|[\u201C\"']|\*|"
    r"(?:ภาค|ลักษณะ|หมวด|ส่วนที่|บทที่|บรรพ)\s)"
)
RE_SENTENCE_END = re.compile(r"[\.\:\;\?\!\u0E2F]\s*$")   # . : ; ? ! ฯ


@dataclass
class CleanReport:
    doc_id: str = ""
    margin_blocks_removed: List[str] = field(default_factory=list)
    lines_rejoined: int = 0
    spaces_fixed: List[Tuple[str, str]] = field(default_factory=list)
    corrections_applied: Counter = field(default_factory=Counter)
    chars_before: int = 0
    chars_after: int = 0


# ==========================================================================
# (A) ลบบล็อกหมายเหตุริมหน้ากระดาษ
# ==========================================================================
# คำที่บ่งบอกว่าบรรทัดนี้เป็น "ส่วนหนึ่งของประโยค" ไม่ใช่หัวข้อริมกระดาษ
# หมายเหตุริมกระดาษเป็นวลีนาม (การ.../ความ.../หน้าที่...) แทบไม่ขึ้นต้นด้วยคำเหล่านี้
SENTENCE_PREFIXES = ("เว้นแต่", "ทั้งนี้", "ต้อง", "ให้", "จึง", "แล้ว", "ก็",
                     "มิให้", "ไม่", "จะ", "อาจ", "ย่อม", "และให้", "ซึ่ง",
                     "โดย", "เพื่อ", "แต่")
SENTENCE_SUFFIXES = ("นั้น", "นี้", "ก็ได้", "ได้", "ไว้", "แล้ว", "ด้วย", "ก่อน")


def _looks_like_sentence_part(line: str) -> bool:
    """บรรทัดนี้น่าจะเป็นหางประโยคจริง ไม่ใช่หัวข้อริมกระดาษ

    ⚠️ กฎนี้มีไว้เพื่อ 'เอียงไปทางเก็บไว้' โดยตั้งใจ
       การเผลอลบตัวบทกฎหมายทิ้ง เสียหายกว่าการเก็บ noise ไว้หนึ่งบรรทัดมาก
       ต้นทุนของการเก็บผิด = มี noise เล็กน้อยใน chunk
       ต้นทุนของการลบผิด = ข้อความกฎหมายหายไปถาวร ค้นไม่เจอตลอดกาล
    """
    s = line.strip()
    return s.startswith(SENTENCE_PREFIXES) or s.endswith(SENTENCE_SUFFIXES)


def _is_fragment(line: str, max_len: int) -> bool:
    """บรรทัดที่ 'ดูเหมือนเศษ' — สั้น ไม่จบประโยค ไม่ใช่หัวข้อ ไม่ใช่รายการ"""
    s = line.strip()
    if not s:
        return False
    if re.fullmatch(r"[0-9]{1,4}", s):        # เลขหน้า
        return True
    if len(s) > max_len:
        return False
    if RE_BLOCK_START.match(s):
        return False
    if RE_SENTENCE_END.search(s):
        return False
    if not RE_THAI.search(s):
        return False
    return True


def strip_margin_blocks(text: str, min_run: int = 3, max_len: int = 42,
                        report: CleanReport | None = None) -> str:
    """ลบ 'แถบ' ของบรรทัดสั้นติดกันตั้งแต่ min_run บรรทัดขึ้นไป

    เหตุผลที่ใช้เกณฑ์ 'ติดกันหลายบรรทัด' แทน 'บรรทัดสั้น' เฉย ๆ:
      บรรทัดสุดท้ายของย่อหน้าปกติก็สั้นได้ แต่จะไม่มีบรรทัดสั้นตามมาติด ๆ กัน 3 บรรทัด
      ส่วนหมายเหตุริมกระดาษจะถูกดึงมาเป็นกลุ่มก้อนเสมอ (มักมีเลขหน้าคั่นอยู่ตรงกลาง)
    """
    lines = text.split("\n")
    flags = [_is_fragment(l, max_len) for l in lines]
    keep = [True] * len(lines)

    i = 0
    while i < len(lines):
        if flags[i]:
            j = i
            while j < len(lines) and flags[j]:
                j += 1

            # หด "ขอบ" ของก้อนเข้ามา ถ้าบรรทัดริมสุดดูเหมือนหางประโยคจริง
            # (บรรทัดสุดท้ายของย่อหน้ามักสั้น เลยถูกดูดเข้าก้อนหมายเหตุโดยบังเอิญ)
            a, b = i, j
            while a < b and _looks_like_sentence_part(lines[a]):
                a += 1
            while b > a and _looks_like_sentence_part(lines[b - 1]):
                b -= 1

            run = lines[a:b]
            thai_lines = [l for l in run if RE_THAI.search(l)]
            if len(run) >= min_run and thai_lines:
                for k in range(a, b):
                    keep[k] = False
                if report is not None:
                    report.margin_blocks_removed.append(" | ".join(s.strip() for s in run))
            i = j
        else:
            i += 1

    return "\n".join(l for l, k in zip(lines, keep) if k)


# ==========================================================================
# (B) ต่อบรรทัดที่ถูกตัดกลางประโยค
# ==========================================================================
def rejoin_wrapped_lines(text: str, report: CleanReport | None = None) -> str:
    """ต่อบรรทัดที่ถูกตัดตามความกว้างหน้ากระดาษกลับเป็นย่อหน้าเดียว

    ภาษาไทยไม่มีช่องว่างระหว่างคำ จึงต่อแบบไม่ใส่ช่องว่าง
    (ถ้าใส่ช่องว่างจะสร้างปัญหา (C) ขึ้นมาเองโดยไม่ตั้งใจ)
    """
    lines = [l.rstrip() for l in text.split("\n")]
    out: List[str] = []
    joined = 0

    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if (out and out[-1].strip()
                and not RE_BLOCK_START.match(s)
                and not RE_SENTENCE_END.search(out[-1])
                and RE_THAI.search(out[-1][-1:])          # บรรทัดก่อนจบด้วยอักษรไทย
                and RE_THAI.search(s[:1])):               # บรรทัดนี้เริ่มด้วยอักษรไทย
            out[-1] = out[-1].rstrip() + s
            joined += 1
        else:
            out.append(s)

    if report is not None:
        report.lines_rejoined += joined
    return "\n".join(out)


# ==========================================================================
# (C) ลบช่องว่างที่แทรกกลางคำ
# ==========================================================================
def _span_token(left: str, right: str) -> str | None:
    """หา token ที่ 'คร่อม' รอยต่อระหว่าง left กับ right พอดี

    ⚠️ ต้องตัดคำแบบ keep_whitespace=True เท่านั้น
       เพราะถ้า tokenizer ทิ้งช่องว่างออก ผลรวมความยาว token จะไม่ตรงกับ
       ตำแหน่งตัวอักษรในสตริงเดิม -> คำนวณรอยต่อผิด -> ตัดสินใจ merge ผิด
       (บั๊กนี้ทำให้ระบบลบช่องว่างคั่นวลีที่ถูกต้อง เช่น "ทั้งนี้ เว้นแต่")
    """
    if not PYTHAINLP_AVAILABLE:
        return None
    from pythainlp.tokenize import word_tokenize as _wt
    joined = left + right
    boundary = len(left)
    pos = 0
    for t in _wt(joined, engine="newmm", keep_whitespace=True):
        if pos < boundary < pos + len(t):
            return t
        pos += len(t)
    return None


def _orphan_count(tokens: List[str]) -> int:
    """นับ token ที่น่าจะเป็น 'เศษคำ' — สั้นและไม่อยู่ในพจนานุกรม

    ตัด token ตัวแรกกับตัวสุดท้ายทิ้งเสมอ เพราะมันถูก 'หน้าต่าง' ตัดขาดกลางคำ
    ไม่ใช่เศษคำจริง ถ้านับรวมจะทำให้ตัดสินใจผิด
    """
    core = tokens[1:-1] if len(tokens) > 2 else []
    return sum(1 for t in core
               if RE_THAI.search(t) and len(t) <= 2 and t not in THAI_DICT)


def fix_intraword_spaces(text: str, window: int = 25,
                         report: CleanReport | None = None) -> str:
    """ลบช่องว่างเฉพาะจุดที่ 'การลบทำให้การตัดคำดีขึ้น'

    ⚠️ ข้อควรระวังสำคัญ: ภาษาไทยใช้ช่องว่างคั่นวลีอย่างถูกต้องอยู่แล้ว
    การลบช่องว่างทั้งหมดจะทำลายข้อความ จึงต้องมีเกณฑ์ตัดสินที่วัดได้:

      ลบช่องว่าง ก็ต่อเมื่อ (1) การลบทำให้จำนวน 'เศษคำ' ลดลง  หรือ
                          (2) การลบทำให้เกิดคำในพจนานุกรมยาว >= 3 ตัวอักษร
                              ที่คร่อมตำแหน่งช่องว่างนั้นพอดี

    ตัวอย่าง:
      "แสดงต นตาม" -> ตัดคำได้ [แสดง, ต, น, ตาม] มีเศษคำ 2 ตัว
                      ลบช่องว่างแล้วได้ [แสดง, ตน, ตาม] เศษคำ 0 -> ลบ ✅
      "ประธานรัฐสภา กำหนด" -> ทั้งสองแบบไม่มีเศษคำ และไม่เกิดคำใหม่ -> ไม่ลบ ✅
    """
    if not THAI_DICT:
        return text

    result = []
    i = 0
    fixed = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if (ch == " " and 0 < i < n - 1
                and RE_THAI.match(text[i - 1]) and RE_THAI.match(text[i + 1])
                # ห้ามแตะช่องว่างรอบไม้ยมก: อักขรวิธีไทยกำหนดให้ "ใด ๆ" มีเว้นวรรค
                # ถ้าลบจะได้ "ใดๆ" ซึ่งผิด และทำให้ tokenizer ตัดคำเพี้ยน
                and text[i - 1] != MAIYAMOK and text[i + 1] != MAIYAMOK):
            left = text[max(0, i - window):i]
            right = text[i + 1:i + 1 + window]
            with_space = word_tokenize(left + " " + right)
            without = word_tokenize(left + right)

            merge = _orphan_count(without) < _orphan_count(with_space)
            if not merge:
                # ตรวจว่ามี token ที่คร่อมรอยต่อและเป็นคำจริงหรือไม่
                span = _span_token(left, right)
                merge = bool(span and len(span) >= 3 and span in THAI_DICT)

            if merge:
                fixed += 1
                if report is not None and len(report.spaces_fixed) < 50:
                    report.spaces_fixed.append(
                        (left[-12:] + " " + right[:12], left[-12:] + right[:12]))
                i += 1          # ข้ามช่องว่างไป = ลบทิ้ง
                continue
        result.append(ch)
        i += 1

    if report is not None and fixed:
        report.corrections_applied["_intraword_spaces"] = fixed
    return "".join(result)


# ==========================================================================
# (D) รายการแก้คำที่มนุษย์ยืนยันแล้ว
# ==========================================================================
def load_corrections(path: str | Path) -> Dict[str, str]:
    """อ่านไฟล์ corrections.txt รูปแบบ  คำผิด<TAB>คำถูก  บรรทัดละคู่"""
    path = Path(path)
    if not path.exists():
        return {}
    mapping = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) >= 2:
            mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def apply_corrections(text: str, mapping: Dict[str, str],
                      report: CleanReport | None = None) -> str:
    for wrong, right in mapping.items():
        if wrong and wrong in text:
            count = text.count(wrong)
            text = text.replace(wrong, right)
            if report is not None:
                report.corrections_applied[f"{wrong} -> {right}"] += count
    return text


# ==========================================================================
def normalize_maiyamok(text: str) -> str:
    """จัดระยะไม้ยมกให้ถูกอักขรวิธี: มีเว้นวรรคทั้งหน้าและหลัง

    หลังจากต่อบรรทัด อาจเกิด "เรื่องอื่น ๆในกรณี" ซึ่งติดกับคำถัดไป
    """
    text = re.sub(r"\s*" + MAIYAMOK + r"\s*", " " + MAIYAMOK + " ", text)
    return re.sub(r"[ ]{2,}", " ", text)


def clean_document(text: str, corrections: Dict[str, str] | None = None,
                   doc_id: str = "", min_run: int = 3, max_len: int = 42,
                   do_margin: bool = True, do_rejoin: bool = True,
                   do_spaces: bool = True) -> Tuple[str, CleanReport]:
    rep = CleanReport(doc_id=doc_id, chars_before=len(text))
    text = normalize_text(text)
    if do_margin:
        text = strip_margin_blocks(text, min_run, max_len, rep)
    if do_rejoin:
        text = rejoin_wrapped_lines(text, rep)
    if do_spaces:
        text = fix_intraword_spaces(text, report=rep)
    text = normalize_maiyamok(text)
    if corrections:
        text = apply_corrections(text, corrections, rep)
    rep.chars_after = len(text)
    return text, rep


def find_unknown_tokens(text: str, top_n: int = 200) -> Counter:
    """หา token ที่ไม่อยู่ในพจนานุกรม — ใช้เป็นตัวช่วยสร้าง corrections.txt

    ไม่ได้แก้อะไร แค่ชี้เป้าให้คนไปตรวจ
    """
    if not THAI_DICT:
        return Counter()
    c = Counter()
    for t in word_tokenize(text):
        t = t.strip()
        if len(t) >= 3 and RE_THAI.search(t) and t not in THAI_DICT:
            c[t] += 1
    return Counter(dict(c.most_common(top_n)))


# ==========================================================================
# ตัวช่วยหาคำที่ "สระ/วรรณยุกต์หาย" โดยอาศัยพจนานุกรม pythainlp
# ==========================================================================
TONE_MARKS = "่้๊๋"          # ไม้เอก โท ตรี จัตวา
EXTRA_MARKS = "ิีึืุูัํ็ำ์"    # สระบน/ล่าง + สระอำ + การันต์ ที่หายบ่อยเวลาแปลง PDF

# บางกรณีต้อง "แทนที่" ไม่ใช่ "แทรก"
# เช่น สระอำ (ำ = นิคหิต + สระอา) ถ้านิคหิตหล่นหาย จะเหลือแค่ สระอา (า)
# "กาหนด" ต้องแก้เป็น "กำหนด" โดยเปลี่ยน า -> ำ ไม่ใช่เติมตัวใหม่
SUBSTITUTIONS = {"า": "ำ", "ั": "ำ", "เ": "แ"}


def _tone_insert_candidates(word: str, marks: str) -> List[str]:
    """สร้างคำที่เป็นไปได้ ด้วยการ 'ใส่เครื่องหมายกลับ' หรือ 'แทนที่' ทีละตำแหน่ง

    หลักการ: PDF extractor มักทำเครื่องหมายที่ลอยอยู่เหนือ/ใต้ตัวอักษรหล่นหาย
    เราจึงลองซ่อมทุกตำแหน่งที่เป็นไปได้ แล้วถามพจนานุกรมว่าได้คำจริงไหม
    """
    out = []
    for i in range(1, len(word) + 1):          # แบบแทรก
        for m in marks:
            cand = word[:i] + m + word[i:]
            if cand in THAI_DICT:
                out.append(cand)
    for i, ch in enumerate(word):              # แบบแทนที่
        if ch in SUBSTITUTIONS:
            cand = word[:i] + SUBSTITUTIONS[ch] + word[i + 1:]
            if cand in THAI_DICT:
                out.append(cand)
    return out


def suggest_corrections(text: str, min_count: int = 3,
                        max_suggestions: int = 300, verbose: bool = False) -> List[Tuple[str, str, int, str]]:
    """เสนอคู่ (คำผิด, คำถูก, ความถี่, เหตุผล) ให้มนุษย์ตรวจ — ไม่แก้ให้เอง

    ตรวจสองรูปแบบ:
      1. token เดี่ยวที่ไม่มีในพจนานุกรม  เช่น "แกไข" -> "แก้ไข"
      2. คู่ token ที่ *เป็นคำจริงทั้งคู่* แต่พอรวมกันแล้วใส่วรรณยุกต์ได้เป็นคำเดียว
         เช่น "ขอ"+"บังคับ" -> "ข้อบังคับ"
         (รูปแบบที่ 2 คือเหตุผลที่ auto-correct ไม่ปลอดภัย ต้องให้คนตัดสิน)

    เงื่อนไขกันเสียงรบกวน: ต้องมีผลลัพธ์ที่เป็นไปได้ *เพียงคำเดียว* เท่านั้น
    ถ้ามีหลายคำแปลว่ากำกวม ให้ข้ามไป ไม่เสนอ
    """
    if not THAI_DICT:
        return []

    tokens = [t.strip() for t in word_tokenize(text) if t.strip()]
    single, pair = Counter(), Counter()

    for i, t in enumerate(tokens):
        if not RE_THAI.search(t):
            continue
        if len(t) >= 3 and t not in THAI_DICT:
            single[t] += 1
        # ดูหน้าต่าง 2 และ 3 โทเค็น เพราะคำที่สระหายมักถูกซอยเป็นหลายชิ้น
        # เช่น "กาหนด" ถูกตัดเป็น ["กา","หน","ด"] -> ถ้าดูแค่คู่จะจับไม่ได้
        for w in (2, 3):
            if i + w > len(tokens):
                continue
            group = tokens[i:i + w]
            # ⚡ ตัวกรองเร็ว: คำที่สระหายจะถูกซอยเป็นชิ้นสั้น ๆ เสมอ
            #    ถ้าไม่มีชิ้นไหนสั้นเลย แปลว่าตัดคำได้สวยอยู่แล้ว ไม่ต้องตรวจ
            #    ตัวกรองนี้ตัดงานทิ้งได้ >90% บนคลังจริง
            if min(len(g) for g in group) > 3:
                continue
            if not all(RE_THAI.search(g) for g in group):
                continue
            merged = "".join(group)
            if 4 <= len(merged) <= 20 and merged not in THAI_DICT:
                pair[merged] += 1

        # กัน Counter บวมจนกินหน่วยความจำบนคลังขนาดใหญ่
        if len(pair) > 400_000:
            pair = Counter({k: v for k, v in pair.items() if v >= min_count})

    if verbose:
        print(f"      โทเค็น {len(tokens):,} | ผู้สมัคร {len(single)+len(pair):,} รายการ")
    results: List[Tuple[str, str, int, str]] = []
    cands_pool = [(w, c) for w, c in single.most_common() if c >= min_count]
    cands_pool += [(w, c) for w, c in pair.most_common() if c >= min_count]
    for word, count in cands_pool:
        cands = _tone_insert_candidates(word, TONE_MARKS + EXTRA_MARKS)
        cands = sorted(set(cands))
        if len(cands) == 1:      # ต้องไม่กำกวม
            reason = "unknown token" if word in single else "adjacent pair"
            results.append((word, cands[0], count, reason))
        if len(results) >= max_suggestions:
            break
    return results
