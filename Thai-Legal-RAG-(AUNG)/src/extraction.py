"""PDF → clean text pipeline สำหรับเอกสารกฎหมายไทย

กลยุทธ์: ลอง text layer ก่อน (เร็ว) → ตรวจคุณภาพ → ถ้าไม่ผ่านค่อย OCR (ช้าแต่ชัวร์)
เหตุผลที่ต้องมี quality check: PDF ไทยจำนวนมากมี text layer ที่ 'พัง'
(สระลอย/ตัวอักษรสลับ) ซึ่ง extract ได้โดยไม่ error แต่ embedding จะเสียทั้งหมด
"""
import io
import json
import os
import re
import unicodedata

import fitz  # PyMuPDF
from PIL import Image
from pythainlp.util import normalize as thai_normalize

try:
    import pytesseract
    HAS_TESSERACT = True
    # ระบุ path ไปยังตัว tesseract.exe ในเครื่อง Windows
    if os.name == "nt":  # เฉพาะบน Windows
        tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
except ImportError:
    HAS_TESSERACT = False

# ---------- Quality heuristics ----------
THAI_CHAR = re.compile(r"[\u0E00-\u0E7F]")
# สระ/วรรณยุกต์ที่ "ลอย" (ตามหลังช่องว่าง) = สัญญาณ text layer พัง
FLOATING_MARKS = re.compile(r"\s[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]")


def quality_score(text: str) -> dict:
    """คืน metrics สำหรับตัดสินว่า text layer ใช้ได้ไหม"""
    if not text or len(text) < 50:
        return {"ok": False, "reason": "too_short", "thai_ratio": 0.0}

    no_space = re.sub(r"\s", "", text)
    thai_ratio = len(THAI_CHAR.findall(no_space)) / max(len(no_space), 1)
    floating = len(FLOATING_MARKS.findall(text))
    replacement = text.count("\ufffd")  # อักขระที่ decode ไม่ได้

    ok = (
        thai_ratio > 0.5          # เอกสารไทยควรมีอักษรไทยเกินครึ่ง
        and floating < len(text) / 500   # สระลอยเกิน ~0.2% = layer พัง
        and replacement < 5
    )
    reason = "ok" if ok else (
        "low_thai_ratio" if thai_ratio <= 0.5
        else "floating_marks" if floating >= len(text) / 500
        else "replacement_chars"
    )
    return {"ok": ok, "reason": reason, "thai_ratio": round(thai_ratio, 3),
            "floating_marks": floating}


# ---------- Cleaning / Normalization ----------
def clean_text(text: str) -> str:
    """Normalize ให้ทุกไฟล์มี 'schema กลาง' เดียวกันก่อนเข้า chunker"""
    # 1) Unicode normalization (สระ/วรรณยุกต์บางไฟล์ encode ต่างกัน)
    text = unicodedata.normalize("NFC", text)
    # 2) ลบ zero-width & soft hyphen ที่ PDF ชอบแทรก
    text = re.sub(r"[\u200b\u200c\u200d\u00ad\ufeff]", "", text)
    # 3) แปลงเลขหน้า/หัวกระดาษท้ายกระดาษที่พบบ่อยในราชกิจจาฯ
    text = re.sub(r"^\s*หน้า\s*[\d๐-๙]+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*เล่ม\s*[\d๐-๙]+.*ราชกิจจานุเบกษา.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-–]?\s*[\d๐-๙]+\s*[-–]?\s*$", "", text, flags=re.MULTILINE)
    # 4) รวมบรรทัดที่ถูกตัดกลางประโยค (PDF ตัดบรรทัดตาม layout ไม่ใช่ตามความหมาย)
    #    แต่รักษา newline หน้า "มาตรา" ไว้ให้ chunker ใช้แยก section
    text = re.sub(r"\n(?!\s*(มาตรา|ข้อ|หมวด|ส่วนที่|บรรพ|ลักษณะ)\s)", " ", text)
    # 5) ยุบช่องว่างซ้ำ
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 6) ช่วยแก้สระซ้อน/สระลอย
    text = thai_normalize(text)
    return text.strip()


# ---------- Extractors ----------
def extract_text_layer(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        # sort=True เรียง block ตามตำแหน่งบนหน้า (แก้ปัญหาลำดับข้อความสลับ)
        pages.append(page.get_text("text", sort=True))
    doc.close()
    return "\n".join(pages)


def extract_ocr(pdf_path: str, dpi: int = 300) -> str:
    """Render แต่ละหน้าเป็นภาพแล้ว OCR — ใช้เมื่อ text layer ใช้ไม่ได้"""
    assert HAS_TESSERACT, "ต้องติดตั้ง pytesseract + tesseract-ocr-tha ก่อน"
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        # tha+eng เพราะเอกสารกฎหมายมักมีเลขอารบิก/ชื่อเฉพาะภาษาอังกฤษปน
        pages.append(pytesseract.image_to_string(img, lang="tha+eng"))
    doc.close()
    return "\n".join(pages)


def extract_pdf(pdf_path: str) -> tuple[str, dict]:
    """Entry point: คืน (cleaned_text, report)"""
    raw = extract_text_layer(pdf_path)
    q = quality_score(raw)
    method = "text_layer"

    if not q["ok"] and HAS_TESSERACT:
        raw_ocr = extract_ocr(pdf_path)
        q_ocr = quality_score(raw_ocr)
        #if q_ocr["thai_ratio"] > q["thai_ratio"]:   # ใช้อันที่ดีกว่า
        if q_ocr["ok"] or len(raw.strip()) < 50:  #ถ้า OCR ผ่านเกณฑ์ หรือเดิม text layer มีข้อความน้อยมากๆ ให้เลือก OCR
            raw, q, method = raw_ocr, q_ocr, "ocr"

    cleaned = clean_text(raw)
    report = {
        "file": os.path.basename(pdf_path),
        "method": method,
        "quality": q,
        "n_chars": len(cleaned),
        "n_matra": len(re.findall(r"มาตรา\s*[\d๐-๙]+", cleaned)),
    }
    return cleaned, report
