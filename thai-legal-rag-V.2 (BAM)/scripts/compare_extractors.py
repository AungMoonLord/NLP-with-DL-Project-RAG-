"""เทียบคุณภาพการสกัดข้อความ: pypdf vs PyMuPDF

    python scripts/compare_extractors.py            # เทียบ 10 ไฟล์แรก
    python scripts/compare_extractors.py --all      # เทียบทั้งคลัง (ช้ากว่า)
    python scripts/compare_extractors.py --sample 3 # ดูข้อความจริงเทียบกัน 3 ไฟล์

สมมติฐานที่ทดสอบ:
  pypdf  ดึงข้อความตามลำดับที่อยู่ในไฟล์ PDF ซึ่งไม่จำเป็นต้องตรงกับสายตาคน
         -> หมายเหตุริมกระดาษถูกแทรกกลางประโยคเนื้อหา
  PyMuPDF ด้วย sort=True เรียง block ตามตำแหน่ง (บน->ล่าง, ซ้าย->ขวา) ก่อนดึง
         -> ควรได้ลำดับข้อความตรงกับที่คนอ่าน

วัดด้วย 3 ตัวชี้วัด:
  - สระลอย ต่อ 1,000 ตัวอักษร  (ยิ่งน้อยยิ่งดี)
  - จำนวน "บล็อกบรรทัดสั้นติดกัน" ที่ cleaner ต้องลบ (ยิ่งน้อยยิ่งดี = noise น้อยแต่แรก)
  - จำนวนตัวอักษรที่ดึงได้ (ควรใกล้เคียงกัน ถ้าต่างมากแปลว่าตัวหนึ่งดึงตกหล่น)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cleaner import CleanReport, strip_margin_blocks
from src.loader import _read_pdf
from src.thai_utils import normalize_text
from scripts.text_quality import score


def margin_blocks(text: str) -> int:
    rep = CleanReport()
    strip_margin_blocks(text, report=rep)
    return len(rep.margin_blocks_removed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=0,
                    help="แสดงข้อความจริงเทียบกันกี่ไฟล์")
    args = ap.parse_args()

    pdfs = sorted(p for p in (ROOT / args.dir).rglob("*.pdf"))
    if not pdfs:
        print(f"ไม่พบไฟล์ PDF ใน {args.dir}")
        return
    if not args.all:
        pdfs = pdfs[:10]

    print(f"[compare] เทียบ {len(pdfs)} ไฟล์\n")
    totals = {"pypdf": [0, 0, 0], "pymupdf": [0, 0, 0]}   # chars, floating, blocks
    fails = {"pypdf": 0, "pymupdf": 0}

    for i, p in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {p.name[:55]}")
        for eng in ("pypdf", "pymupdf"):
            try:
                text = normalize_text(_read_pdf(p, engine=eng))
            except Exception as e:
                print(f"        {eng}: ล้มเหลว ({type(e).__name__})")
                fails[eng] += 1
                continue
            q = score(text)
            b = margin_blocks(text)
            totals[eng][0] += q["n_chars"]
            totals[eng][1] += q["floating"]
            totals[eng][2] += b
            print(f"        {eng:9s} {q['n_chars']:>7,} ตัวอักษร | "
                  f"สระลอย {q['floating_per_1k']:>5.2f}/1k | บล็อก noise {b:>3d}")

    print("\n" + "=" * 68)
    print(f"{'ตัวชี้วัด':<32}{'pypdf':>16}{'PyMuPDF':>16}")
    print("-" * 68)
    rows = [
        ("ตัวอักษรที่ดึงได้", 0, "{:,}"),
        ("สระลอย (จุด)", 1, "{:,}"),
        ("บล็อก noise ที่ต้องลบ", 2, "{:,}"),
    ]
    for label, idx, fmt in rows:
        a, b = totals["pypdf"][idx], totals["pymupdf"][idx]
        print(f"{label:<32}{fmt.format(a):>16}{fmt.format(b):>16}")

    a_chars = max(totals["pypdf"][0], 1)
    b_chars = max(totals["pymupdf"][0], 1)
    a_f = totals["pypdf"][1] / (a_chars / 1000)
    b_f = totals["pymupdf"][1] / (b_chars / 1000)
    print(f"{'สระลอยต่อ 1,000 ตัวอักษร':<32}{a_f:>16.2f}{b_f:>16.2f}")
    print("=" * 68)

    if totals["pymupdf"][2] < totals["pypdf"][2]:
        d = (1 - totals["pymupdf"][2] / max(totals["pypdf"][2], 1)) * 100
        print(f"\n✅ PyMuPDF สร้าง noise น้อยกว่า {d:.0f}% "
              f"-> คุ้มที่จะเปลี่ยนและ ingest ใหม่")
    elif totals["pymupdf"][2] > totals["pypdf"][2]:
        print("\n🟡 PyMuPDF ไม่ได้ดีกว่าในคลังนี้ -> ใช้ pypdf ต่อไปได้")
    else:
        print("\n🟡 ผลเท่ากัน -> ไม่ต้องเปลี่ยน")

    for k, v in fails.items():
        if v:
            print(f"⚠️ {k} อ่านไม่ได้ {v} ไฟล์")

    if args.sample:
        print("\n" + "=" * 68)
        print("ตัวอย่างข้อความจริง (อ่านด้วยตาว่าอันไหนลำดับถูกกว่า)")
        for p in pdfs[: args.sample]:
            print(f"\n--- {p.name[:60]} ---")
            for eng in ("pypdf", "pymupdf"):
                try:
                    t = normalize_text(_read_pdf(p, engine=eng))
                except Exception:
                    continue
                print(f"\n[{eng}]")
                print(t[600:1400])


if __name__ == "__main__":
    main()
