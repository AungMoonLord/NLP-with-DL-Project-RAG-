"""ตรวจหาหน้าที่เป็นภาพสแกน (ดึงข้อความไม่ออก) ในคลังเอกสาร

    python scripts/check_scanned.py
    python scripts/check_scanned.py --dir data/raw --threshold 80

ทำไมต้องตรวจก่อนตัดสินใจทำ OCR:
  OCR ภาษาไทยมีต้นทุนสูง (ต้องติดตั้ง Tesseract + โมเดลภาษาไทย, ประมวลผลช้า)
  และคุณภาพไม่ดีเท่าข้อความที่ฝังมาในไฟล์อยู่แล้ว
  ถ้าหน้าที่มีปัญหาน้อยกว่า 5% ของทั้งหมด การทำ OCR อาจไม่คุ้ม
  เทียบกับการบันทึกไว้เป็นข้อจำกัดของระบบ

วิธีตรวจ: นับจำนวนตัวอักษรที่ดึงได้ต่อหน้า
  หน้าข้อความปกติของกฎหมายไทย = 800-2,500 ตัวอักษร
  หน้าที่เป็นภาพสแกน = เกือบ 0 (อาจมีเลขหน้าหลุดมาบ้าง)
  หน้าที่มีแต่ตาราง/รูป = อยู่ระหว่างกลาง ต้องดูด้วยตา
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_pdf(path: Path, threshold: int):
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ต้องติดตั้ง pypdf ก่อน: pip install pypdf")
        sys.exit(1)
    reader = PdfReader(str(path))
    counts = []
    for page in reader.pages:
        try:
            counts.append(len((page.extract_text() or "").strip()))
        except Exception:
            counts.append(0)
    empty = [i + 1 for i, c in enumerate(counts) if c < threshold]
    return counts, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw")
    ap.add_argument("--threshold", type=int, default=80,
                    help="หน้าที่มีตัวอักษรน้อยกว่านี้ ถือว่าน่าจะเป็นภาพสแกน")
    args = ap.parse_args()

    root = ROOT / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    others = [p for p in files if p.suffix.lower() in {".txt", ".md", ".json", ".jsonl"}]

    print(f"[check] โฟลเดอร์: {root}")
    print(f"[check] PDF {len(pdfs)} ไฟล์ | ไฟล์ข้อความ {len(others)} ไฟล์\n")

    if others and not pdfs:
        print("คลังนี้เป็นไฟล์ข้อความล้วน ไม่มี PDF ให้ตรวจ")
        print("ถ้าคุณแปลง PDF เป็น .txt มาก่อนหน้านี้ หน้าที่สแกนจะกลายเป็นช่องว่าง")
        print("ให้ดูรายงานด้านล่างว่าไฟล์ไหนสั้นผิดปกติเมื่อเทียบกับไฟล์อื่น\n")
        sizes = sorted(((len(p.read_text(encoding="utf-8", errors="ignore")), p)
                        for p in others))
        for n, p in sizes[:10]:
            print(f"  {n:>8,} ตัวอักษร  {p.name[:60]}")
        median = sizes[len(sizes) // 2][0]
        print(f"\n  ค่ากลางของคลัง: {median:,} ตัวอักษร")
        suspicious = [p.name for n, p in sizes if n < median * 0.15]
        if suspicious:
            print(f"  ⚠️ ไฟล์ที่สั้นผิดปกติ (<15% ของค่ากลาง): {suspicious[:5]}")
        else:
            print("  ✅ ไม่มีไฟล์ที่สั้นผิดปกติ")
        return

    total_pages = total_empty = 0
    problem_files = []
    for p in pdfs:
        counts, empty = check_pdf(p, args.threshold)
        total_pages += len(counts)
        total_empty += len(empty)
        if empty:
            problem_files.append((p.name, len(counts), empty))
            ratio = len(empty) / max(len(counts), 1)
            flag = "🔴" if ratio > 0.5 else "🟡"
            print(f"{flag} {p.name[:55]:55s} {len(empty):>3d}/{len(counts):>3d} หน้าไม่มีข้อความ")
            print(f"     หน้าที่มีปัญหา: {empty[:15]}{' ...' if len(empty) > 15 else ''}")

    print(f"\n{'='*70}")
    if total_pages:
        pct = total_empty / total_pages * 100
        print(f"สรุป: {total_empty:,} จาก {total_pages:,} หน้า ({pct:.1f}%) ดึงข้อความไม่ออก")
        print(f"      ไฟล์ที่มีปัญหา: {len(problem_files)} จาก {len(pdfs)} ไฟล์")
        print()
        if pct == 0:
            print("✅ ทุกหน้ามีข้อความฝังอยู่แล้ว ไม่ต้องทำ OCR")
        elif pct < 5:
            print("🟡 มีปัญหาน้อยกว่า 5% — แนะนำให้บันทึกเป็นข้อจำกัดของระบบ")
            print("   แล้วเอาเวลาไปทำ ablation กับ evaluation ซึ่งมีคะแนนมากกว่า")
        else:
            print("🔴 มีปัญหามากกว่า 5% — ควรทำ OCR เฉพาะไฟล์ที่มีปัญหา")
            print("   ดูวิธีในหัวข้อ OCR ของ README")


if __name__ == "__main__":
    main()
