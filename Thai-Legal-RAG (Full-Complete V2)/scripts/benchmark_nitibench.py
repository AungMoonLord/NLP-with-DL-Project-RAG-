"""NitiBench (VISAI-AI/nitibench) Benchmark Runner - Robust Version
สคริปต์ทดสอบ RAG กับ NitiBench
รองรับทั้งข้อสอบแบบมี Context และแบบ Open-book Retrieval โดยใช้ Index กฎหมายในระบบ
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# ปรับ Path เข้าหา Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.metrics import bootstrap_ci, mean
from src.retrieval import Retriever

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("กรุณาติดตั้ง datasets ก่อน: pip install datasets")


def extract_legal_keywords(text: str) -> set[str]:
    """ดึงเลขมาตราและคำสำคัญทางกฎหมายจากคำถามและเฉลยเพื่อใช้วัด Retrieval Hit"""
    keywords = set()
    if not text:
        return keywords
    # ดึง มาตรา xxx
    matras = re.findall(r"มาตรา\s*\d+", text)
    for m in matras:
        keywords.add(re.sub(r"\s+", " ", m).strip().lower())
    # ดึงชื่อ พ.ร.บ. / ประมวลกฎหมาย
    laws = re.findall(
        r"(?:พระราชบัญญัติ|พ\.ร\.บ\.|ประมวลกฎหมายแพ่งและพาณิชย์|ประมวลกฎหมายอาญา|ป\.พ\.พ\.|ป\.อ\.)[^\s,()]*",
        text,
    )
    for law in laws:
        clean_l = law.strip().lower()
        if len(clean_l) >= 4:
            keywords.add(clean_l)
    return keywords


def is_chunk_relevant(chunk_text: str, doc_id: str, keywords: set[str], reference_text: str) -> bool:
    """ตรวจสอบว่า Chunk ที่ดึงขึ้นมาตรงกับข้อกฎหมายที่เกี่ยวข้องหรือไม่"""
    c_text = chunk_text.lower()
    d_id = doc_id.lower()

    # 1. เช็คว่ามีเลขมาตราหรือชื่อกฎหมายที่โจทย์/เฉลยถามถึงหรือไม่
    for kw in keywords:
        if kw in c_text or kw in d_id:
            return True

    # 2. เช็คการซ้ำกันของข้อความเฉลย (ถ้ามี)
    if reference_text:
        ref_clean = re.sub(r"\s+", "", reference_text.lower())
        c_clean = re.sub(r"\s+", "", c_text)
        if len(ref_clean) >= 20 and (ref_clean[:30] in c_clean or c_clean[:30] in ref_clean):
            return True

    return False


def run_nitibench_eval(
    samples: int = 30,
    modes: List[str] = ["dense", "bm25", "hybrid", "hybrid_rerank"],
    use_wangchanx_index: bool = False,
):
    output_dir = ROOT_DIR / "artifacts" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # เลือกว่าจะใช้ Index กฎหมายชุดไหน
    if use_wangchanx_index and (ROOT_DIR / "artifacts" / "wangchanx" / "dense.faiss").exists():
        target_index_dir = ROOT_DIR / "artifacts" / "wangchanx"
        print(f"[*] ใช้งาน Index จาก WangchanX (กฎหมายแพ่งและพาณิชย์): {target_index_dir}")
    else:
        target_index_dir = Path(getattr(config, "ARTIFACTS_DIR", getattr(config, "INDEX_DIR", ROOT_DIR / "artifacts")))
        print(f"[*] ใช้งาน Index กฎหมายหลักของระบบ: {target_index_dir}")

    if not (target_index_dir / "dense.faiss").exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ dense.faiss ที่ {target_index_dir} กรุณารัน python scripts/build_index.py ก่อน"
        )

    print("\n" + "=" * 65)
    print(" เริ่มต้นการทดสอบ Benchmark: VISAI-AI/nitibench")
    print("=" * 65)

    # 1. โหลด Dataset จาก Hugging Face
    print("[*] กำลังดาวน์โหลด/โหลด Dataset จาก Hugging Face: VISAI-AI/nitibench...")
    try:
        ds = load_dataset("VISAI-AI/nitibench")
    except Exception as e:
        print(f"[!] เกิดข้อผิดพลาดในการโหลด NitiBench: {e}")
        return

    # แสดงโครงสร้างคอลัมน์ของ Dataset
    split_target = "test" if "test" in ds else list(ds.keys())[0]
    first_item = ds[split_target][0]
    print(f"[✓] เชื่อมต่อ Dataset สำเร็จ (Split: {split_target})")
    print(f"[*] โครงสร้างคอลัมน์ที่มี: {list(first_item.keys())}")

    # 2. แปลงคำถามให้อยู่ใน Format มาตรฐาน
    test_rows = list(ds[split_target])
    if samples > 0:
        test_rows = test_rows[:samples]

    items = []
    for idx, row in enumerate(test_rows):
        # ดึงคำถามจากคอลัมน์ที่มี
        q = (
            row.get("question")
            or row.get("prompt")
            or row.get("instruction")
            or row.get("input")
            or ""
        )
        # ดึงคำตอบ/คำอธิบาย
        ref_ans = str(
            row.get("answer")
            or row.get("output")
            or row.get("explanation")
            or row.get("ground_truth")
            or ""
        )

        if not q.strip():
            continue

        # สกัดเลขมาตราและคีย์เวิร์ดทางกฎหมายที่ปรากฏในคำถามและเฉลย
        keywords = extract_legal_keywords(q) | extract_legal_keywords(ref_ans)

        items.append({
            "id": row.get("id", f"niti_{idx}"),
            "question": q,
            "reference_answer": ref_ans,
            "keywords": keywords,
        })

    print(f"[✓] เตรียมคำถามทดสอบเรียบร้อย: จำนวน {len(items)} ข้อ")
    print(f"[*] โหมด Retrieval ที่จะทดสอบ: {modes}")

    # 3. โหลด Retriever
    retriever = Retriever(index_dir=target_index_dir)

    eval_k_values = [1, 3, 5]
    all_results = {}

    # 4. ประเมินผล Retrieval
    for mode in modes:
        print(f"\n>>> [Variant: {mode.upper()}] กำลังประเมินผล Retrieval...")
        per_q: Dict[str, List[float]] = {
            "mrr": [],
            "hit@1": [],
            "hit@3": [],
            "hit@5": [],
            "latency_ms": [],
        }

        for item in tqdm(items, desc=f"Eval {mode}"):
            q = item["question"]
            kws = item["keywords"]
            ref_ans = item["reference_answer"]

            t0 = time.perf_counter()
            hits = retriever.retrieve(q, mode=mode, k=5)
            latency = (time.perf_counter() - t0) * 1000
            per_q["latency_ms"].append(latency)

            # คำนวณ Hit@k
            for k_val in eval_k_values:
                is_hit = False
                for h in hits[:k_val]:
                    txt = h.get("text", "")
                    doc = str(h.get("doc_id", ""))
                    if is_chunk_relevant(txt, doc, kws, ref_ans):
                        is_hit = True
                        break
                per_q[f"hit@{k_val}"].append(1.0 if is_hit else 0.0)

            # คำนวณ MRR
            first_rank = 0
            for rank_idx, h in enumerate(hits, 1):
                txt = h.get("text", "")
                doc = str(h.get("doc_id", ""))
                if is_chunk_relevant(txt, doc, kws, ref_ans):
                    first_rank = rank_idx
                    break
            per_q["mrr"].append(1.0 / first_rank if first_rank > 0 else 0.0)

        # สรุปผลตัวชี้วัด
        summary = {
            "n_samples": len(items),
            "latency_ms_avg": round(mean(per_q["latency_ms"]), 2),
        }
        for metric in ["hit@1", "hit@3", "hit@5", "mrr"]:
            lo, hi = bootstrap_ci(per_q[metric], n_boot=1000)
            summary[metric] = {
                "mean": round(mean(per_q[metric]), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
        all_results[mode] = summary

    # 5. แสดงผลตารางสรุปผล
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
    print(" สรุปผลคะแนนระบบบน Benchmark: VISAI-AI/NITIBENCH")
    print("=" * 65)
    print(table_md)
    print("=" * 65 + "\n")

    # บันทึกไฟล์ผลลัพธ์
    table_file = output_dir / "nitibench_results_table.md"
    table_file.write_text(f"# NitiBench Benchmark Evaluation Results\n\n{table_md}\n", encoding="utf-8")

    json_file = output_dir / "nitibench_results_detail.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[✓] บันทึกผลลัพธ์ตารางไว้ที่: {table_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NitiBench Benchmark Runner")
    parser.add_argument("--samples", type=int, default=30, help="จำนวนข้อที่ต้องการทดสอบ")
    parser.add_argument("--mode", type=str, default="all", choices=["dense", "bm25", "hybrid", "hybrid_rerank", "all"])
    parser.add_argument(
        "--use_wangchanx_corpus",
        action="store_true",
        help="ใช้คลังประมวลกฎหมายแพ่งของ WangchanX ในการตอบแทนคลังเดิม 80 ฉบับ",
    )
    args = parser.parse_args()

    selected_modes = ["dense", "bm25", "hybrid", "hybrid_rerank"] if args.mode == "all" else [args.mode]
    run_nitibench_eval(
        samples=args.samples,
        modes=selected_modes,
        use_wangchanx_index=args.use_wangchanx_corpus,
    )
