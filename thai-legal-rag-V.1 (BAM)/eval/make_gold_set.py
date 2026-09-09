"""ตัวช่วยสร้างชุดทดสอบ (gold set)

⚠️ สำคัญมากสำหรับการ defend:
   ไฟล์นี้สร้างแค่ "ร่าง" คำถาม-คำตอบจาก chunk จริงในคลังเอกสาร
   ทีมต้องอ่านและแก้ด้วยมือทุกข้อ ก่อนใช้เป็นเฉลย
   ถ้าใช้ LLM สร้างเฉลยแล้วเอา LLM ตัวเดิมมาตรวจ = วัดตัวเองกับตัวเอง (self-preference bias)
   ผลที่ได้จะไม่มีความหมายทางวิชาการ และเป็นจุดที่กรรมการชอบถาม

ข้อแนะนำองค์ประกอบของชุดทดสอบ (อย่างน้อย 40-60 ข้อ):
   - single-hop  60%  ตอบได้จากมาตราเดียว
   - multi-hop   25%  ต้องรวมข้อมูลจาก 2 มาตราขึ้นไป
   - unanswerable 15% คำตอบไม่มีในคลัง -> ทดสอบว่าระบบกล้าตอบว่า "ไม่พบข้อมูล" ไหม
     (ข้อนี้คือสิ่งที่ทำให้ failure analysis ในสไลด์ที่ 9 มีเนื้อหา)

ใช้งาน:
    python eval/make_gold_set.py --n 40 --out data/qa_gold.draft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.chunker import load_chunks
from src.generator import build_generator

DRAFT_PROMPT = """จากข้อความกฎหมายไทยต่อไปนี้ จงสร้างคำถาม 1 ข้อที่ผู้ใช้ทั่วไปน่าจะถาม
และคำตอบที่ถูกต้องซึ่งอ้างอิงจากข้อความนี้เท่านั้น

ข้อความ ({citation}):
{text}

เงื่อนไข:
- คำถามต้องตอบได้จากข้อความนี้เท่านั้น และต้องไม่ยกคำในข้อความมาทั้งประโยค
- คำตอบยาวไม่เกิน 3 ประโยค และอ้างเลขมาตราถ้ามี
ตอบเป็น JSON เท่านั้น: {{"question": "...", "reference_answer": "..."}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-tokens", type=int, default=120)
    ap.add_argument("--out", default="data/qa_gold.draft.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    chunks = load_chunks(ROOT / cfg.processed_dir / f"chunks_{cfg.index_tag}.jsonl")
    pool = [c for c in chunks if c.n_tokens >= args.min_tokens]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.n]

    gen = build_generator(cfg.generator)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(pool, 1):
            print(f"[gold] {i}/{len(pool)} {c.citation[:60]}")
            raw = gen.generate("ตอบเป็น JSON เท่านั้น",
                               DRAFT_PROMPT.format(citation=c.citation, text=c.text[:3000]))
            try:
                data = json.loads(raw.strip().strip("`").replace("json\n", "", 1))
            except Exception:
                print("   ! parse ไม่ผ่าน ข้ามข้อนี้")
                continue
            f.write(json.dumps({
                "question": data["question"],
                "reference_answer": data["reference_answer"],
                "gold_chunk_ids": [c.chunk_id],
                "difficulty": "single-hop",
                "source_citation": c.citation,
                "VERIFIED_BY_HUMAN": False,   # ต้องแก้เป็นชื่อคนตรวจก่อนใช้จริง
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n[gold] เขียนร่าง {written} ข้อ -> {out_path}")
    print("[gold] ขั้นตอนถัดไป: อ่านทุกข้อ แก้คำถาม/เฉลยให้ถูกต้อง เปลี่ยน VERIFIED_BY_HUMAN")
    print("[gold] แล้วเติมข้อ multi-hop และ unanswerable ด้วยมือ ก่อน rename เป็น data/qa_gold.jsonl")


if __name__ == "__main__":
    main()
