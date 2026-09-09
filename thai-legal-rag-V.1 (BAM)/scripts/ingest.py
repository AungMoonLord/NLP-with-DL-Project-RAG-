"""สร้าง index ทั้งหมดจากเอกสารดิบใน data/raw/

ใช้งาน:
    python scripts/ingest.py                                  # ใช้ค่า default
    python scripts/ingest.py --config configs/variant_d_chunk1024.yaml
    python scripts/ingest.py --stats-only                     # ดูสถิติคลังก่อน ไม่สร้าง index

หมายเหตุ: ชื่อไฟล์ index ผูกกับ (embedding model + chunk size + overlap)
ดังนั้น variant ที่ใช้ chunk คนละขนาดต้อง ingest ใหม่ ระบบจะไม่ทับกัน
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import ThaiLegalChunker, build_token_counter, save_chunks
from src.config import Config
from src.embedder import DenseIndex, Embedder
from src.loader import corpus_stats, load_corpus
from src.sparse import BM25Index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    raw_dir = cfg.path(cfg.raw_dir)
    proc_dir = cfg.path(cfg.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) โหลดคลังเอกสาร ----------
    print("\n[1/4] โหลดคลังเอกสาร")
    docs = load_corpus(raw_dir)
    stats = corpus_stats(docs)
    print(f"      สถิติคลัง: {json.dumps(stats, ensure_ascii=False, indent=6)}")
    if len(docs) < 50:
        print(f"      ⚠️  โจทย์กำหนดอย่างน้อย 50 เอกสาร ตอนนี้มี {len(docs)}")
    (proc_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.stats_only:
        return

    # ---------- 2) chunking ----------
    print("\n[2/4] Chunking")
    counter = build_token_counter(cfg.embedding.model_name)
    chunker = ThaiLegalChunker(cfg.chunk, counter)
    chunks = chunker.chunk_corpus(docs)
    chunk_path = proc_dir / f"chunks_{cfg.index_tag}.jsonl"
    save_chunks(chunks, chunk_path)

    # ตัวอย่าง chunk ไว้ใส่สไลด์ / ตรวจด้วยตา
    with open(proc_dir / f"sample_chunks_{cfg.index_tag}.txt", "w", encoding="utf-8") as f:
        for c in chunks[:15]:
            f.write(f"### {c.chunk_id} | {c.citation} | {c.n_tokens} tokens\n{c.text}\n\n")

    texts = [c.embed_text(cfg.chunk.prepend_breadcrumb) for c in chunks]

    # ---------- 3) dense index ----------
    if not args.skip_dense:
        print("\n[3/4] สร้าง dense index")
        embedder = Embedder(cfg.embedding)
        vecs = embedder.encode_passages(texts)
        print(f"      เวกเตอร์: {vecs.shape} (dim={vecs.shape[1]})")
        DenseIndex(vecs, [c.chunk_id for c in chunks]).save(
            proc_dir / f"dense_{cfg.index_tag}.npz")
    else:
        print("\n[3/4] ข้าม dense index")

    # ---------- 4) BM25 index ----------
    if not args.skip_bm25:
        print("\n[4/4] สร้าง BM25 index")
        BM25Index.build([c.chunk_id for c in chunks], texts).save(
            proc_dir / f"bm25_{cfg.index_tag}.pkl")
    else:
        print("\n[4/4] ข้าม BM25 index")

    print(f"\n✅ Ingest เสร็จ | index_tag = {cfg.index_tag}")


if __name__ == "__main__":
    main()
