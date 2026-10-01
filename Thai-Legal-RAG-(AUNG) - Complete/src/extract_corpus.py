import sys; sys.path.insert(0, ".")
import json
import os
from tqdm import tqdm
from src.extraction import extract_pdf
from pathlib import Path

# หา Root directory ของ thai-legal-rag โดยอิงจากไฟล์นี้
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = "data/pdfs"           # วาง PDF 80 ไฟล์ตรงนี้
OUT_DIR = "data/documents"      # .txt ที่สะอาดแล้ว (chunker อ่านต่อจากตรงนี้)
os.makedirs(OUT_DIR, exist_ok=True)

reports = []
for fname in tqdm(sorted(os.listdir(PDF_DIR))):
    if not fname.lower().endswith(".pdf"):
        continue
    try:
        #text, report = extract_pdf(os.path.join(PDF_DIR, fname))
        # แปลงเป็น Absolute Path และใส่ prefix \\?\ ปลดล็อกข้อจำกัด 260 ตัวอักษรบน Windows
        full_pdf_path = os.path.abspath(os.path.join(PDF_DIR, fname))
        if os.name == "nt" and not full_pdf_path.startswith("\\\\?\\"):
            full_pdf_path = "\\\\?\\" + full_pdf_path

        text, report = extract_pdf(full_pdf_path)
        out_name = fname[:-4] + ".txt"
        with open(os.path.join(OUT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(text)
        reports.append(report)
    except Exception as e:
        reports.append({"file": fname, "method": "FAILED", "error": str(e)})

with open("data/extraction_report.json", "w", encoding="utf-8") as f:
    json.dump(reports, f, ensure_ascii=False, indent=2)

# ---- สรุปให้เห็นภาพรวมทันที ----
ocr_files = [r["file"] for r in reports if r.get("method") == "ocr"]
failed = [r["file"] for r in reports if r.get("method") == "FAILED"]
suspicious = [r["file"] for r in reports
              if r.get("quality") and not r["quality"]["ok"]]
no_matra = [r["file"] for r in reports if r.get("n_matra", 1) == 0]

print(f"\n✅ สำเร็จ: {len(reports) - len(failed)}/{len(reports)}")
print(f"🔍 ใช้ OCR: {len(ocr_files)} ไฟล์ → {ocr_files[:5]}")
print(f"⚠️ คุณภาพน่าสงสัย (ต้องเปิดดูเอง!): {suspicious}")
print(f"⚠️ ไม่เจอคำว่า 'มาตรา' (schema ต่าง?): {no_matra}")
print(f"❌ พัง: {failed}")
