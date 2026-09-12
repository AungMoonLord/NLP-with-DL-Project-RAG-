"""เดโมสำหรับนำเสนอ (4:00–7:00 ของสไลด์)

ใช้งาน:
    python scripts/demo.py                                # รัน 3 คำถามตัวอย่าง
    python scripts/demo.py --interactive                  # โหมดถาม-ตอบสด
    python scripts/demo.py -q "อายุความคดีอาญาคือกี่ปี" --explain

--explain จะแสดง "ก่อน/หลัง rerank" ให้เห็นว่า cross-encoder เปลี่ยนอันดับอย่างไร
เป็นภาพที่ใช้ตอบคำถามกรรมการได้ดีที่สุด
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.pipeline import RAGPipeline

DEFAULT_QUESTIONS = [
    "การผิดนัดชำระหนี้ทำให้ลูกหนี้ต้องรับผิดอย่างไรบ้าง",
    "สัญญาที่ทำโดยสำคัญผิดในสาระสำคัญมีผลทางกฎหมายอย่างไร",
    "ถ้าอยากจดทะเบียนบริษัทบนดาวอังคารต้องทำอย่างไร",  # คำถามนอกคลัง -> ต้องตอบว่าไม่พบข้อมูล
]


def show(res, explain: bool = False):
    print("\n" + "=" * 78)
    print(res.pretty())
    if explain:
        tr = res.trace
        print("\n--- retrieval trace ---")
        print(f"โหมด: {tr.get('mode')} | rerank: {tr.get('reranked')} | "
              f"อันดับเปลี่ยนหลัง rerank: {tr.get('rank_changed')}")
        print(f"ก่อน rerank : {tr.get('stage1_top_ids')}")
        print(f"หลัง rerank : {tr.get('final_top_ids')}")
        for i, r in enumerate(res.retrieved, 1):
            d = r.debug or {}
            print(f"  [S{i}] {r.chunk.chunk_id}")
            print(f"       score={r.score:.4f} | prev_rank={d.get('prev_rank')} | "
                  f"dense={d.get('dense_score')} bm25={d.get('bm25_score')}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("-q", "--question", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    print(f"[demo] โหลดระบบ ... (retrieval={cfg.retrieval.mode}, "
          f"rerank={cfg.retrieval.use_reranker}, generator={cfg.generator.backend})")
    pipe = RAGPipeline.from_config(cfg)
    print("[demo] พร้อมใช้งาน\n")

    if args.interactive:
        print("พิมพ์คำถาม (พิมพ์ 'exit' เพื่อออก)")
        while True:
            try:
                q = input("\n❓ > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit", "q"}:
                break
            show(pipe.answer(q, top_k=args.top_k), args.explain)
        return

    questions = [args.question] if args.question else DEFAULT_QUESTIONS
    for q in questions:
        show(pipe.answer(q, top_k=args.top_k), args.explain)


if __name__ == "__main__":
    main()
