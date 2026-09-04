"""ทำความสะอาดคลังเอกสารทั้งชุด

    python scripts/clean_corpus.py --dry-run     # ดูว่าจะแก้อะไรบ้าง ไม่เขียนไฟล์
    python scripts/clean_corpus.py               # เขียนผลลง data/clean/
    python scripts/clean_corpus.py --no-spaces   # ปิดการลบช่องว่างกลางคำ

ปรัชญา: ไม่แก้อะไรแบบเงียบ ๆ ทุกการเปลี่ยนแปลงถูกบันทึกลง
        data/processed/cleaning_report.md เพื่อให้ทีมตรวจสอบย้อนหลังได้
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cleaner import (clean_document, find_unknown_tokens,
                         load_corrections, suggest_corrections)
from src.config import Config
from src.loader import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--no-rejoin", action="store_true")
    ap.add_argument("--no-spaces", action="store_true")
    ap.add_argument("--min-run", type=int, default=3,
                    help="ต้องมีบรรทัดสั้นติดกันกี่บรรทัดจึงถือว่าเป็นหมายเหตุริมกระดาษ")
    ap.add_argument("--min-count", type=int, default=3,
                    help="คำต้องพบกี่ครั้งจึงจะถูกเสนอให้แก้")
    ap.add_argument("--skip-suggest", action="store_true",
                    help="ข้ามขั้นหาคำที่สระหาย (ขั้นนี้ช้าที่สุด)")
    ap.add_argument("--max-len", type=int, default=42,
                    help="บรรทัดยาวไม่เกินกี่ตัวอักษรจึงนับว่า 'สั้น'")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    docs = load_corpus(cfg.path(cfg.raw_dir))
    corrections = load_corrections(ROOT / "data/corrections.txt")
    print(f"[clean] โหลดรายการแก้คำจาก corrections.txt: {len(corrections)} รายการ")

    out_dir = ROOT / args.out
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    reports, unknown_all = [], Counter()
    cleaned_texts = []
    total_before = total_after = 0

    for i, d in enumerate(docs, 1):
        text, rep = clean_document(
            d.text, corrections=corrections, doc_id=d.doc_id,
            min_run=args.min_run, max_len=args.max_len,
            do_margin=not args.no_margin, do_rejoin=not args.no_rejoin,
            do_spaces=not args.no_spaces)
        reports.append(rep)
        total_before += rep.chars_before
        total_after += rep.chars_after
        unknown_all.update(find_unknown_tokens(text, top_n=100))
        cleaned_texts.append(text)
        if not args.dry_run:
            (out_dir / f"{d.doc_id}.txt").write_text(text, encoding="utf-8")
        print(f"  [{i}/{len(docs)}] {d.doc_id[:45]:45s} "
              f"{rep.chars_before:>7d} -> {rep.chars_after:>7d} "
              f"| หมายเหตุ {len(rep.margin_blocks_removed):>3d} บล็อก "
              f"| ต่อบรรทัด {rep.lines_rejoined:>4d} "
              f"| ช่องว่าง {rep.corrections_applied.get('_intraword_spaces', 0):>4d}")

    # ---------- รายงาน ----------
    proc = ROOT / cfg.processed_dir
    proc.mkdir(parents=True, exist_ok=True)

    n_margin = sum(len(r.margin_blocks_removed) for r in reports)
    n_join = sum(r.lines_rejoined for r in reports)
    n_space = sum(r.corrections_applied.get("_intraword_spaces", 0) for r in reports)
    n_corr = sum(v for r in reports for k, v in r.corrections_applied.items()
                 if not k.startswith("_"))
    pct = (total_before - total_after) / max(total_before, 1) * 100

    lines = [
        "# Cleaning Report", "",
        f"- เอกสาร: {len(docs)}",
        f"- ขนาดรวม: {total_before:,} -> {total_after:,} ตัวอักษร (ลดลง {pct:.1f}%)",
        f"- ลบบล็อกหมายเหตุริมกระดาษ: {n_margin} บล็อก",
        f"- ต่อบรรทัดที่ถูกตัดกลางประโยค: {n_join} ครั้ง",
        f"- ลบช่องว่างแทรกกลางคำ: {n_space} จุด",
        f"- แก้คำตามรายการที่มนุษย์ยืนยัน: {n_corr} ครั้ง",
        "",
        "> ⚠️ ถ้า 'ลดลง' เกิน ~15% ให้สงสัยว่าลบเนื้อหาจริงทิ้ง",
        "> ลองเพิ่ม --min-run เป็น 4-5 หรือลด --max-len แล้วรันใหม่",
        "", "## ตัวอย่างบล็อกที่ถูกลบ (ตรวจว่าไม่ใช่เนื้อหาจริง)", "",
    ]
    shown = 0
    for r in reports:
        for b in r.margin_blocks_removed:
            if shown >= 25:
                break
            lines.append(f"- `{r.doc_id[:30]}` : {b[:160]}")
            shown += 1
        if shown >= 25:
            break

    lines += ["", "## ตัวอย่างช่องว่างที่ถูกลบ", ""]
    shown = 0
    for r in reports:
        for a, b in r.spaces_fixed:
            if shown >= 25:
                break
            lines.append(f"- `{a}` → `{b}`")
            shown += 1
        if shown >= 25:
            break

    applied = Counter()
    for r in reports:
        for k, v in r.corrections_applied.items():
            if not k.startswith("_"):
                applied[k] += v
    if applied:
        lines += ["", "## รายการแก้คำที่ถูกใช้จริง", ""]
        lines += [f"- {k} ({v} ครั้ง)" for k, v in applied.most_common()]

    (proc / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")

    # คำที่ไม่อยู่ในพจนานุกรม -> ตัวช่วยสร้าง corrections.txt
    unk = ["# คำที่ไม่พบในพจนานุกรมไทย เรียงตามความถี่",
           "# ตรวจดูว่าคำไหนสะกดผิดจาก PDF แล้วเพิ่มลง data/corrections.txt",
           "# รูปแบบ:  คำผิด<TAB>คำถูก", ""]
    unk += [f"{w}\t{c}" for w, c in unknown_all.most_common(300)]
    (proc / "unknown_tokens.txt").write_text("\n".join(unk), encoding="utf-8")

    # ---------- เสนอคู่แก้คำอัตโนมัติ (ให้มนุษย์ตรวจก่อนใช้) ----------
    if args.skip_suggest:
        print("\n[clean] ข้ามขั้นหาคำที่สระ/วรรณยุกต์หาย (--skip-suggest)")
        sugg = []
    else:
        print("\n[clean] กำลังหาคำที่สระ/วรรณยุกต์น่าจะหายไป (ทีละเอกสาร) ...")
        from collections import Counter as _C
        agg = _C()
        for k, t in enumerate(cleaned_texts, 1):
            print(f"      [{k}/{len(cleaned_texts)}]", end="\r", flush=True)
            for w, c, n, r in suggest_corrections(t, min_count=args.min_count):
                agg[(w, c, r)] += n
        sugg = [(w, c, n, r) for (w, c, r), n in agg.most_common()
                if n >= args.min_count]
        print(" " * 40, end="\r")
    cand_lines = [
        "# คู่แก้คำที่ระบบเสนอ — ยังไม่ถูกนำไปใช้",
        "# วิธีใช้: อ่านทุกบรรทัด ลบบรรทัดที่ผิด แล้ว copy ที่เหลือไปต่อท้าย data/corrections.txt",
        "#",
        "# วิธีที่ระบบใช้หา: ลองเติม/แทนที่วรรณยุกต์และสระทุกตำแหน่ง",
        "# แล้วถามพจนานุกรม pythainlp ว่าได้คำจริงหรือไม่",
        "# เสนอเฉพาะกรณีที่ได้คำตอบ 'เพียงคำเดียว' เท่านั้น (ไม่กำกวม)",
        "#",
        "# คำผิด\tคำถูก\t# ความถี่, ประเภท", "",
    ]
    for w, c, n, reason in sugg:
        cand_lines.append(f"{w}\t{c}\t# {n} ครั้ง, {reason}")
    (proc / "corrections_candidates.tsv").write_text("\n".join(cand_lines), encoding="utf-8")
    print(f"[clean] เสนอคู่แก้คำ {len(sugg)} รายการ -> {proc/'corrections_candidates.tsv'}")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}สรุป: "
          f"{total_before:,} -> {total_after:,} ตัวอักษร (ลดลง {pct:.1f}%)")
    print(f"  หมายเหตุ {n_margin} บล็อก | ต่อบรรทัด {n_join} | ช่องว่าง {n_space} | แก้คำ {n_corr}")
    print(f"[clean] รายงาน -> {proc/'cleaning_report.md'}")
    print(f"[clean] คำน่าสงสัย -> {proc/'unknown_tokens.txt'}")
    if not args.dry_run:
        print(f"[clean] ไฟล์สะอาด -> {out_dir}")
        print("\nขั้นถัดไป: แก้ raw_dir ใน config เป็น data/clean แล้ว ingest ใหม่")


if __name__ == "__main__":
    main()
