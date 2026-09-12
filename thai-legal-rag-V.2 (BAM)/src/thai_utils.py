"""Thai-specific text utilities.

ทำไมต้องมีไฟล์นี้: ภาษาไทย "ไม่มีช่องว่างระหว่างคำ"
=> tokenizer แบบ .split() ที่ใช้กับภาษาอังกฤษพังทันที
=> กระทบทั้ง BM25, ROUGE และการนับความยาวประโยค
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

try:
    from pythainlp.tokenize import word_tokenize as _pythai_word_tokenize
    from pythainlp.tokenize import sent_tokenize as _pythai_sent_tokenize
    from pythainlp.util import normalize as _pythai_normalize
    from pythainlp.corpus.common import thai_stopwords as _thai_stopwords
    PYTHAINLP_AVAILABLE = True
    _STOPWORDS = set(_thai_stopwords())
except Exception:  # pragma: no cover - fallback ให้รันได้แม้ไม่มี pythainlp
    PYTHAINLP_AVAILABLE = False
    _STOPWORDS = set()

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
_THAI_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(THAI_DIGITS)}

# เก็บเลขไทยไว้ในรูป arabic เพราะกฎหมายไทยเขียน "มาตรา ๓๒๖"
# แต่ผู้ใช้พิมพ์ค้นหาว่า "มาตรา 326" -> ถ้าไม่ normalize BM25 จะ match ไม่เจอเลย


def thai_digits_to_arabic(text: str) -> str:
    return text.translate(_THAI_DIGIT_MAP)


def recompose_sara_am(text: str) -> str:
    """แก้ผลข้างเคียงของ unicodedata.normalize('NFKC', ...)

    ⚠️ BUG ที่เจอจริงตอนพัฒนา (ดู AI_AUDIT § Error Catch):
       NFKC จะ "แตก" สระอำ U+0E33 ออกเป็น นิคหิต U+0E4D + สระอา U+0E32
       ผลคือคำว่า "ชำระ" กลายเป็นลำดับอักขระ 5 ตัวที่ไม่ตรงกับพจนานุกรมของ newmm
       -> การตัดคำพัง -> BM25 หา term ไม่เจอ -> เสียคะแนน recall โดยไม่มี error ฟ้อง
       จึงต้องประกอบกลับทุกครั้งหลัง NFKC
    """
    return text.replace("\u0e4d\u0e32", "\u0e33")


def normalize_text(text: str, to_arabic_digits: bool = True) -> str:
    """normalize รูปแบบตัวอักษรก่อนทำอย่างอื่นเสมอ

    ลำดับสำคัญ: NFKC (จัดการ full-width/ligature) -> ประกอบสระอำกลับ ->
    pythainlp.normalize (จัดเรียงสระ/ลบสระซ้ำ) -> แปลงเลขไทย -> ยุบช่องว่าง
    """
    text = unicodedata.normalize("NFKC", text)
    text = recompose_sara_am(text)
    if PYTHAINLP_AVAILABLE:
        text = _pythai_normalize(text)   # ลบสระซ้ำ/จัดลำดับวรรณยุกต์ผิดตำแหน่ง
    if to_arabic_digits:
        text = thai_digits_to_arabic(text)
    text = text.replace("\u200b", "")             # zero-width space
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_THAI_RUN = re.compile(r"[\u0E00-\u0E7F]+")


def word_tokenize(text: str, engine: str = "newmm") -> List[str]:
    """ตัดคำไทย. newmm = dictionary-based maximal matching (เร็ว, พอสำหรับ BM25)

    ถ้าไม่มี pythainlp จะถอยไปใช้ character n-gram แทน (ไม่ใช่ whitespace split!)
    เพราะภาษาไทยไม่มีช่องว่างระหว่างคำ การ split(' ') จะได้ token เดียวทั้งประโยค
    ทำให้ BM25 และ ROUGE-L ให้ค่าเป็น 0 หรือ 1 เท่านั้น (ไร้ความหมาย)
    """
    if PYTHAINLP_AVAILABLE:
        return _pythai_word_tokenize(text, engine=engine, keep_whitespace=False)

    tokens: List[str] = []
    pos = 0
    for m in _THAI_RUN.finditer(text):
        head = text[pos:m.start()]
        tokens += [t for t in re.split(r"[\s]+|([^\w])", head) if t and not t.isspace()]
        run = m.group(0)                      # fallback: character bigram บนช่วงอักษรไทย
        tokens += [run[i:i + 2] for i in range(0, len(run) - 1)] or [run]
        pos = m.end()
    tail = text[pos:]
    tokens += [t for t in re.split(r"[\s]+|([^\w])", tail) if t and not t.isspace()]
    return tokens


_SENT_ENGINE = ["crfcut"]      # เก็บสถานะไว้ ไม่ต้องลองใหม่ทุกครั้งที่เรียก


def sent_tokenize(text: str) -> List[str]:
    """ตัดประโยค. กฎหมายไทยมักใช้ '\\n' และ ' ' แทนจุด full stop

    crfcut เป็นโมเดล CRF ที่เทรนมาตัดประโยคไทยโดยเฉพาะ แม่นกว่าการใช้ regex มาก
    แต่ต้องพึ่ง pycrfsuite ถ้าไม่มีจะถอยไปใช้ regex แทน (ผลแย่ลงแต่ระบบไม่ล้ม)
    """
    if PYTHAINLP_AVAILABLE and _SENT_ENGINE[0]:
        try:
            sents = _pythai_sent_tokenize(text, engine=_SENT_ENGINE[0])
            return [s.strip() for s in sents if s and s.strip()]
        except Exception as e:
            print(f"[thai_utils] ใช้ crfcut ไม่ได้ ({type(e).__name__}) "
                  f"-> ถอยไปใช้การตัดด้วย regex "
                  f"(ติดตั้ง python-crfsuite เพื่อผลที่ดีกว่า)")
            _SENT_ENGINE[0] = None
    sents = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [s.strip() for s in sents if s and s.strip()]


def tokenize_for_bm25(text: str, remove_stopwords: bool = True) -> List[str]:
    """BM25 tokenizer.

    หมายเหตุการออกแบบ: เราตัด stopword ออก แต่ *ไม่* ตัดตัวเลขและคำว่า 'มาตรา'
    เพราะเป็น high-signal term ในโดเมนกฎหมาย
    """
    text = normalize_text(text.lower())
    tokens = word_tokenize(text)
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if re.fullmatch(r"[^\w\u0E00-\u0E7F]+", t):   # เครื่องหมายวรรคตอนล้วน
            continue
        if remove_stopwords and t in _STOPWORDS and not t.isdigit():
            continue
        out.append(t)
    return out


def count_thai_words(text: str) -> int:
    return len(word_tokenize(text))
