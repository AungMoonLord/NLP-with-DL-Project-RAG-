"""วัด "ความเสียหายของข้อความ" ที่ได้จากการแปลง PDF เป็นตัวเลข

    python scripts/text_quality.py                  # ตรวจ data/raw
    python scripts/text_quality.py --dir data/clean # ตรวจหลังทำความสะอาด
    python scripts/text_quality.py --compare        # เทียบ raw กับ clean

ทำไมต้องมี: เดิมเรารู้ว่าข้อความจาก PDF มีปัญหา แต่ไม่เคยวัดว่า "แค่ไหน"
ทำให้พิสูจน์ไม่ได้ว่า cleaner ช่วยจริงหรือไม่ สคริปต์นี้ให้ตัวเลข 3 ตัว:

  1. thai_ratio     สัดส่วนอักษรไทยต่ออักขระทั้งหมด
                    เอกสารกฎหมายไทยควรสูงกว่า 0.5 ถ้าต่ำแปลว่า extract เพี้ยน

  2. floating_marks สระ/วรรณยุกต์ที่ "ลอย" อยู่หลังช่องว่าง เช่น "ครอบง ำใด"
                    เป็นลายเซ็นเฉพาะของ PDF ที่ text layer พัง
                    เพราะในภาษาไทยที่ถูกต้อง สระบน-ล่างต้องเกาะพยัญชนะเสมอ
                    ไม่มีทางขึ้นต้นหลังช่องว่างได้

  3. replacement    อักขระ U+FFFD ที่ decode ไม่ออก

แนวคิด floating_marks ปรับมาจาก quality check ของทีมอื่นในชั้นเรียน
(ดูบันทึกใน AI_AUDIT — เป็นแนวคิดที่เราไม่ได้คิดเอง)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THAI_CHAR = re.compile(r"[\u0E00-\u0E7F]")
# สระบน/ล่าง + วรรณยุกต์ ที่ตามหลังช่องว่าง = ลอย = extract พัง
FLOATING_MARKS = re.compile(r"\s[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]")


def score(text: str) -> dict:
    if not text or len(text) < 50:
        return {"ok": False, "reason": "too_short", "thai_ratio": 0.0,
                "floating": 0, "floating_per_1k": 0.0, "replacement": 0,
                "n_chars": len(text or "")}

    no_space = re.sub(r"\s", "", text)
    thai_ratio = len(THAI_CHAR.findall(no_space)) / max(len(no_space), 1)
    floating = len(FLOATING_MARKS.findall(text))
    replacement = text.count("\ufffd")
    per_1k = floating / (len(text) / 1000)

    ok = thai_ratio > 0.5 and per_1k < 2.0 and replacement < 5
    reason = ("ok" if ok else
              "low_thai_ratio" if thai_ratio <= 0.5 else
              "floating_marks" if per_1k >= 2.0 else "replacement_chars")
    return {"ok": ok, "reason": reason, "thai_ratio": round(thai_ratio, 3),
            "floating": floating, "floating_per_1k": round(per_1k, 2),
            "replacement": replacement, "n_chars": len(text)}


def scan(d: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            out.append((p.name, score(p.read_text(encoding="utf-8", errors="ignore"))))
    return out


def summarize(rows, label):
    if not rows:
        print(f"[quality] ไม่พบไฟล์ .txt ใน {label}")
        return None
    n = len(rows)
    bad = [r for _, r in rows if not r["ok"]]
    avg_ratio = sum(r["thai_ratio"] for _, r in rows) / n
    total_float = sum(r["floating"] for _, r in rows)
    total_chars = sum(r["n_chars"] for _, r in rows)
    per_1k = total_float / max(total_chars / 1000, 1)
    print(f"\n=== {label} ===")
    print(f"  ไฟล์ {n} | ไม่ผ่านเกณฑ์ {len(bad)}")
    print(f"  thai_ratio เฉลี่ย   {avg_ratio:.3f}  (ควร > 0.5)")
    print(f"  สระลอยรวม          {total_float:,} จุด = {per_1k:.2f} ต่อ 1,000 ตัวอักษร")
    worst = sorted(rows, key=lambda x: -x[1]["floating_per_1k"])[:5]
    print("  ไฟล์ที่สระลอยหนาแน่นที่สุด:")
    for name, r in worst:
        print(f"    {r['floating_per_1k']:>6.2f}/1k  ratio={r['thai_ratio']:.2f}  {name[:52]}")
    return {"per_1k": per_1k, "avg_ratio": avg_ratio, "bad": len(bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw")
    ap.add_argument("--compare", action="store_true",
                    help="เทียบ data/raw กับ data/clean")
    args = ap.parse_args()

    if args.compare:
        a = summarize(scan(ROOT / "data/raw"), "ก่อนทำความสะอาด (data/raw)")
        b = summarize(scan(ROOT / "data/clean"), "หลังทำความสะอาด (data/clean)")
        if a and b:
            d = (a["per_1k"] - b["per_1k"]) / max(a["per_1k"], 1e-9) * 100
            print(f"\n>>> สระลอยลดลง {d:.1f}%  |  ไฟล์ที่ไม่ผ่านเกณฑ์ "
                  f"{a['bad']} -> {b['bad']}")
            print(">>> ตัวเลขคู่นี้ใช้ใส่สไลด์ได้ เป็นหลักฐานว่า cleaner ทำงานจริง")
    else:
        summarize(scan(ROOT / args.dir), args.dir)


if __name__ == "__main__":
    main()
