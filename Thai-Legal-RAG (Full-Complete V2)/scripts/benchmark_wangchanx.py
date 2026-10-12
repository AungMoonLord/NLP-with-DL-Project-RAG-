"""WangchanX-Legal-ThaiCCL-RAG Benchmark Runner (All-in-One)
สคริปต์สำหรับทดสอบระบบ RAG กับ WangchanX Benchmark โดยเฉพาะ
ขั้นตอนการทำงาน:
1. ดึงข้อสอบและสกัด Corpus กฎหมายแพ่งและพาณิชย์จาก Hugging Face อัตโนมัติ
2. สร้าง Index (FAISS Dense + BM25) เก็บไว้ที่ artifacts/wangchanx/
3. รันวัดผลความแม่นยำการค้นคืน (Hit@1, Hit@3, Hit@5, MRR) และ Latency
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# กำหนด Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.indexer import Indexer
from src.metrics import bootstrap_ci, mean
from src.retrieval import Retriever

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("กรุณาติดตั้ง datasets ก่อน: pip install datasets")


def extract_context_text(ctx: Any) -> str:
    """ดึงข้อความตัวบทกฎหมายออกจาก Context Object"""
    if isinstance(ctx, dict):
        return (
            ctx.get("context")
            or ctx.get("text")
            or ctx.get("section_text")
            or ctx.get("content")
            or ""
        ).strip()
    elif isinstance(ctx, str):
        return ctx.strip()
    return ""


def clean_text_for_match(text: str) -> str:
    """ทำความสะอาดข้อความเพื่อใช้ในการ Matching"""
    return re.sub(r"\s+", "", text).lower()


def build_wangchanx_corpus_and_index(dataset, index_dir: Path):
    """สกัดคลังตัวบทกฎหมายจาก WangchanX และสร้าง Index (FAISS + BM25)"""
    print(f"\n[*] กำลังสกัดคลังตัวบทกฎหมายจาก WangchanX Dataset...")
    unique_passages = {}

    # รวม Contexts จากทั้ง train และ test เพื่อให้ได้ Corpus ครอบคลุมกฎหมายทั้งหมด
    for split_name in dataset.keys():
        for row in dataset[split_name]:
            pos_contexts = row.get("positive_contexts") or []
            if not isinstance(pos_contexts, list):
                pos_contexts = [pos_contexts]

            for ctx in pos_contexts:
                text = extract_context_text(ctx)
                if text and text not in unique_passages:
                    law_title = "CCL_Law"
                    if isinstance(ctx, dict):
                        law_title = ctx.get("law_title") or ctx.get("law_code") or "CCL_Law"
                    unique_passages[text] = {
                        "chunk_id": f"ccl_{len(unique_passages)}",
                        "doc_id": law_title,
                        "text": text,
                        "metadata": {"law_title": law_title}
                    }

    chunks = list(unique_passages.values())
    print(f"[✓] สกัดตัวบทกฎหมายเอกเทศได้ทั้งหมด: {len(chunks)} Chunks")

    # บันทึก chunks.json
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    print(f"[*] เริ่มต้นการสร้าง Index (E5 Embeddings + BM25 Okapi)...")
    indexer = Indexer()
    indexer.build(chunks, out_dir=str(index_dir))
    print(f"[✓] สร้าง Index สำหรับ WangchanX สำเร็จที่: {index_dir}\n")


def is_chunk_hit(retrieved_text: str, gold_texts: List[str]) -> bool:
    """ตรวจสอบว่า Chunk ที่ระบบดึงมา ตรงกับเฉลยตัวบทกฎหมายหรือไม่"""
    clean_ret = clean_text_for_match(retrieved_text)
    if not clean_ret:
        return False

    for gold in gold_texts:
        clean_gold = clean_text_for_match(gold)
        if not clean_gold:
            continue
        # ตรวจสอบว่ามีข้อความตรงกันอย่างน้อย 50 ตัวอักษร หรือครอบคลุมกัน
        if clean_gold in clean_ret or clean_ret in clean_gold:
            return True
        sample_length = min(len(clean_gold), 50)
        if sample_length >= 20 and clean_gold[:sample_length] in clean_ret:
            return True
    return False


def run_wangchanx_eval(samples: int = 30, modes: List[str] = ["dense", "bm25", "hybrid", "hybrid_rerank"]):
    wangchanx_index_dir = ROOT_DIR / "artifacts" / "wangchanx"
    output_dir = ROOT_DIR / "artifacts" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" เริ่มต้นกระบวนการเตรียม WangchanX-Legal-ThaiCCL-RAG Benchmark")
    print("=" * 65)

    # 1. โหลด Dataset
    print("[*] กำลังดาวน์โหลด/โหลด Dataset จาก Hugging Face...")
    ds = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG")

    # 2. ตรวจสอบและสร้าง Index หากยังไม่มี
    if not (wangchanx_index_dir / "dense.faiss").exists() or not (wangchanx_index_dir / "bm25.pkl").exists():
        build_wangchanx_corpus_and_index(ds, wangchanx_index_dir)
    else:
        print(f"[✓] พบ Index ของ WangchanX อยู่แล้วที่: {wangchanx_index_dir}")

    # 3. เตรียมชุดคำถามทดสอบ (ใช้ Split 'test')
    test_rows = list(ds["test"])
    if samples > 0:
        test_rows = test_rows[:samples]

    items = []
    for idx, row in enumerate(test_rows):
        q = row.get("question") or row.get("query")
        pos_contexts = row.get("positive_contexts") or []
        if not isinstance(pos_contexts, list):
            pos_contexts = [pos_contexts]

        gold_texts = [extract_context_text(c) for c in pos_contexts if extract_context_text(c)]
        items.append({
            "id": f"test_{idx}",
            "question": q,
            "gold_texts": gold_texts,
            "reference_answer": row.get("answer") or row.get("positive_answer", "")
        })

    print(f"\n[✓] เตรียมคำถามทดสอบเรียบร้อย: จำนวน {len(items)} ข้อ")
    print(f"[*] โหมด Retrieval ที่จะทดสอบ: {modes}")

    # โหลด Retriever โดยชี้ไปยัง WangchanX Index
    retriever = Retriever(index_dir=wangchanx_index_dir)

    eval_k_values = [1, 3, 5]
    all_results = {}

    # 4. ลูปทดสอบแต่ละ Variant
    for mode in modes:
        print(f"\n>>> [Variant: {mode.upper()}] กำลังประเมินผล Retrieval...")
        per_q: Dict[str, List[float]] = {
            "mrr": [],
            "hit@1": [],
            "hit@3": [],
            "hit@5": [],
            "latency_ms": []
        }

        for item in tqdm(items, desc=f"Eval {mode}"):
            q = item["question"]
            gold = item["gold_texts"]

            t0 = time.perf_counter()
            hits = retriever.retrieve(q, mode=mode, k=5)
            latency = (time.perf_counter() - t0) * 1000
            per_q["latency_ms"].append(latency)

            # ตรวจสอบ Hit@1, Hit@3, Hit@5
            for k_val in eval_k_values:
                hit_flag = any(is_chunk_hit(h.get("text", ""), gold) for h in hits[:k_val])
                per_q[f"hit@{k_val}"].append(1.0 if hit_flag else 0.0)

            # คำนวณ MRR
            first_rank = 0
            for rank_idx, h in enumerate(hits, 1):
                if is_chunk_hit(h.get("text", ""), gold):
                    first_rank = rank_idx
                    break
            per_q["mrr"].append(1.0 / first_rank if first_rank > 0 else 0.0)

        # สรุปผลทางสถิติ
        summary = {"n_samples": len(items), "latency_ms_avg": round(mean(per_q["latency_ms"]), 2)}
        for metric in ["hit@1", "hit@3", "hit@5", "mrr"]:
            lo, hi = bootstrap_ci(per_q[metric], n_boot=1000)
            summary[metric] = {
                "mean": round(mean(per_q[metric]), 4),
                "ci95": [round(lo, 4), round(hi, 4)]
            }
        all_results[mode] = summary

    # 5. แสดงผลตาราง Markdown
    headers = ["| Variant | Hit@1 | Hit@3 | Hit@5 | MRR | Latency (ms) |"]
    sep = ["|---|---|---|---|---|---|"]
    md_lines = headers + sep

    for mode in modes:
        res = all_results[mode]
        h1 = res["hit@1"]["mean"]
        h3 = res["hit@3"]["mean"]
        h5 = res["hit@5"]["mean"]
        mrr_val = res["mrr"]["mean"]
        lat = res["latency_ms_avg"]
        md_lines.append(f"| `{mode}` | {h1:.4f} | {h3:.4f} | {h5:.4f} | {mrr_val:.4f} | {lat:.2f} |")

    table_md = "\n".join(md_lines)
    print("\n" + "=" * 65)
    print(" สรุปผลคะแนนระบบบน Benchmark: WANGCHANX-LEGAL-THAICCL-RAG")
    print("=" * 65)
    print(table_md)
    print("=" * 65 + "\n")

    # เซฟผลลัพธ์
    table_file = output_dir / "wangchanx_results_table.md"
    table_file.write_text(f"# WangchanX Benchmark Evaluation Results\n\n{table_md}\n", encoding="utf-8")
    
    json_file = output_dir / "wangchanx_results_detail.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[✓] บันทึกผลลัพธ์ไว้ที่: {table_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WangchanX Benchmark Runner")
    parser.add_argument("--samples", type=int, default=30, help="จำนวนตัวอย่างคำถามที่ต้องการทดสอบ (default: 30)")
    parser.add_argument("--mode", type=str, default="all", choices=["dense", "bm25", "hybrid", "hybrid_rerank", "all"])
    args = parser.parse_args()

    selected_modes = ["dense", "bm25", "hybrid", "hybrid_rerank"] if args.mode == "all" else [args.mode]
    run_wangchanx_eval(samples=args.samples, modes=selected_modes)
    