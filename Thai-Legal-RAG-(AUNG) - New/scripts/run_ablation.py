# # """Ablation: เปรียบเทียบ 3 variants บน testset เดียวกัน
# # - A: dense-only
# # - B: dense + BM25 hybrid (RRF)
# # - C: hybrid + cross-encoder reranker
# # วัดด้วย Recall@5 และ MRR (retrieval-level → แยก effect ของ retriever ออกจาก LLM)
# # """
# # import sys; sys.path.insert(0, ".")
# # import json
# # from src.retrieval import Retriever
# # from src.evaluation import recall_at_k, mrr
# # import config

# # with open(config.TESTSET_PATH, encoding="utf-8") as f:
# #     testset = json.load(f)

# # retriever = Retriever(use_reranker=True)
# # variants = ["dense", "hybrid", "hybrid_rerank"]
# # results = {v: {"recall@5": [], "mrr": []} for v in variants}

# # # for item in testset:
# # #     for v in variants:
# # #         retrieved = retriever.retrieve(item["question"], mode=v, k=5)
# # #         doc_ids = [c["doc_id"] for c in retrieved]
# # #         results[v]["recall@5"].append(recall_at_k(doc_ids, item["relevant_docs"]))
# # #         results[v]["mrr"].append(mrr(doc_ids, item["relevant_docs"]))

# # for item in testset:
# #     # ดึงเอกสารเป้าหมาย: ถ้าระบุ relevant_docs ไว้ใช้ตัวนั้น ถ้าไม่มีให้ดึงจาก source_doc
# #     target_docs = item.get("relevant_docs")
# #     if not target_docs:
# #         source = item.get("source_doc", "")
# #         # หุ้มเป็น list เพื่อให้ฟังก์ชัน evaluation นำไป match ได้
# #         target_docs = [source] if source else []

# #     for v in variants:
# #         retrieved = retriever.retrieve(item["question"], mode=v, k=5)
# #         # ตรวจสอบว่า retriever คืน doc_id หรือ file/source
# #         doc_ids = [c.get("doc_id") or c.get("file") or c.get("source", "") for c in retrieved]
        
# #         results[v]["recall@5"].append(recall_at_k(doc_ids, target_docs))
# #         results[v]["mrr"].append(mrr(doc_ids, target_docs))
        
# # summary = {
# #     v: {m: round(sum(vals) / len(vals), 4) for m, vals in metrics.items()}
# #     for v, metrics in results.items()
# # }

# # print(f"{'Variant':<16}{'Recall@5':<12}{'MRR':<8}")
# # for v, m in summary.items():
# #     print(f"{v:<16}{m['recall@5']:<12}{m['mrr']:<8}")

# # with open("artifacts/ablation_results.json", "w", encoding="utf-8") as f:
# #     json.dump(summary, f, ensure_ascii=False, indent=2)


# """Ablation Study: เปรียบเทียบ 3 Retrieval Variants บน testset เดียวกัน
# - Variant A: dense-only (intfloat/multilingual-e5-base)
# - Variant B: dense + BM25 hybrid (Reciprocal Rank Fusion: RRF)
# - Variant C: hybrid + cross-encoder reranker (BAAI/bge-reranker-v2-m3)

# วัดผลในระดับ Information Retrieval: Recall@5, MRR@5, nDCG@5, Hit@5
# (แยกประสิทธิภาพของ Retriever ออกจากผลกระทบของ LLM Generator)
# """
# import sys
# sys.path.insert(0, ".")

# import json
# from pathlib import Path
# from tqdm import tqdm
# from src.retrieval import Retriever
# from src.metrics import (
#     gold_docs_of,
#     retrieved_docs_of,
#     recall_at_k,
#     mrr,
#     ndcg_at_k,
#     hit_at_k,
#     mean,
#     bootstrap_ci,
#     paired_bootstrap_p
# )
# import config

# # โหลด testset
# with open(config.TESTSET_PATH, encoding="utf-8") as f:
#     testset = json.load(f)

# # กำหนดค่าตัวแปร
# K = getattr(config, "TOP_K_FINAL", 5)
# retriever = Retriever(use_reranker=True)
# variants = ["dense", "hybrid", "hybrid_rerank"]

# # โครงสร้างเก็บผลลัพธ์รายข้อสำหรับคำนวณสถิติ
# raw_metrics = {
#     v: {
#         "recall@5": [],
#         "mrr@5": [],
#         "ndcg@5": [],
#         "hit@5": []
#     }
#     for v in variants
# }

# print(f"กำลังรัน Ablation Study บนคำถาม {len(testset)} ข้อ (Top-k = {K})...\n")

# for item in tqdm(testset, desc="Evaluating Variants"):
#     # ดึง gold labels และ normalize ชื่อเอกสารอัตโนมัติ
#     gold_set = gold_docs_of(item)
#     if not gold_set:
#         continue

#     for v in variants:
#         # retrieve คืน list ของ dict chunks
#         retrieved_chunks = retriever.retrieve(item["question"], mode=v, k=K)
#         # ดึงรายชื่อเอกสารที่ normalize แล้ว
#         ret_docs = retrieved_docs_of(retrieved_chunks)

#         # คำนวณ metric แต่ละตัวโดยส่งค่า K กำกับ
#         raw_metrics[v]["recall@5"].append(recall_at_k(ret_docs, gold_set, k=K))
#         raw_metrics[v]["mrr@5"].append(mrr(ret_docs, gold_set, k=K))
#         raw_metrics[v]["ndcg@5"].append(ndcg_at_k(ret_docs, gold_set, k=K))
#         raw_metrics[v]["hit@5"].append(hit_at_k(ret_docs, gold_set, k=K))

# # สรุปผลลัพธ์เฉลี่ยและช่วงความเชื่อมั่น (95% Bootstrap CI)
# summary = {}
# for v in variants:
#     summary[v] = {}
#     for m, vals in raw_metrics[v].items():
#         avg_val = mean(vals)
#         ci_low, ci_high = bootstrap_ci(vals, n_boot=2000, seed=42)
#         summary[v][m] = round(avg_val, 4) if avg_val == avg_val else 0.0
#         summary[v][f"{m}_ci95"] = [round(ci_low, 4), round(ci_high, 4)]

# # คำนวณ Statistical Significance (p-value เทียบ hybrid_rerank กับ dense)
# p_val_mrr = paired_bootstrap_p(
#     raw_metrics["hybrid_rerank"]["mrr@5"],
#     raw_metrics["dense"]["mrr@5"]
# )
# summary["statistical_significance"] = {
#     "test": "Paired Bootstrap Test (hybrid_rerank vs dense)",
#     "mrr@5_p_value": round(p_val_mrr, 4) if p_val_mrr == p_val_mrr else None,
#     "significant_at_0.05": bool(p_val_mrr < 0.05) if p_val_mrr == p_val_mrr else False
# }

# # พิมพ์ตารางผลลัพธ์สรุปออกทางหน้าจอ
# print("\n" + "=" * 56)
# print(f"{'Variant':<16}{'Recall@5':<12}{'MRR@5':<10}{'nDCG@5':<10}{'Hit@5':<8}")
# print("-" * 56)
# for v in variants:
#     m = summary[v]
#     print(f"{v:<16}{m['recall@5']:<12.4f}{m['mrr@5']:<10.4f}{m['ndcg@5']:<10.4f}{m['hit@5']:<8.4f}")
# print("=" * 56)

# if summary["statistical_significance"]["mrr@5_p_value"] is not None:
#     print(f"Paired Bootstrap p-value (MRR@5: Rerank vs Dense): {summary['statistical_significance']['mrr@5_p_value']}")
# print()

# # บันทึกไฟล์ผลลัพธ์ลง artifacts/ablation_results.json
# out_path = Path("artifacts/ablation_results.json")
# out_path.parent.mkdir(parents=True, exist_ok=True)
# with open(out_path, "w", encoding="utf-8") as f:
#     json.dump(summary, f, ensure_ascii=False, indent=2)

# print(f"บันทึกผลการทดลองเรียบร้อยที่: {out_path}")


"""Ablation Study: เปรียบเทียบ 4 Retrieval Variants บนชุดข้อมูลทดสอบ
- Variant 1: dense (intfloat/multilingual-e5-base)
- Variant 2: bm25 (Sparse lexical search ด้วย PyThaiNLP)
- Variant 3: hybrid (Dense + BM25 รวมด้วย RRF)
- Variant 4: hybrid_rerank (Hybrid + BAAI/bge-reranker-v2-m3)

วัดผลทั้ง Retrieval Metrics (Recall, Hit, nDCG, MRR), Latency, และ Statistical Significance
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ปรับ path ให้มองเห็น root directory เสมอ
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.retrieval import Retriever
from src.metrics import (
    gold_docs_of,
    retrieved_docs_of,
    recall_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    mean,
    bootstrap_ci,
    paired_bootstrap_p,
)

VARIANTS = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def main():
    # 1. จัดการ Path และโหลด Testset อย่างปลอดภัย
    testset_path = getattr(config, "TESTSET_PATH", ROOT_DIR / "data" / "testset.json")
    with open(testset_path, encoding="utf-8") as f:
        raw_testset = json.load(f)

    if isinstance(raw_testset, dict):
        testset = raw_testset.get("questions") or raw_testset.get("data") or []
    else:
        testset = raw_testset

    # กรองเฉพาะข้อที่มี ground truth
    items = [t for t in testset if not t.get("adversarial")]
    items = [t for t in items if gold_docs_of(t)]

    if not items:
        raise SystemExit("❌ ไม่พบข้อคำถามที่มี gold doc — โปรดตรวจสอบ testset.json")

    print(f"✓ พร้อมประเมิน {len(items)} คำถาม × {len(VARIANTS)} variants\n")

    # 2. ดึงค่าคอนฟิกพร้อม Fallback ป้องกันตัวแปรหาย
    eval_k_values = getattr(config, "EVAL_K_VALUES", [1, 3, 5])
    max_k = getattr(config, "MAX_EVAL_K", max(eval_k_values))
    rerank_pool = getattr(config, "RERANK_CANDIDATES", getattr(config, "TOP_K_RETRIEVE", 30))
    n_boot = getattr(config, "N_BOOTSTRAP", 2000)
    seed = getattr(config, "RANDOM_SEED", 42)

    artifacts_dir = Path(getattr(config, "ARTIFACTS_DIR", getattr(config, "INDEX_DIR", ROOT_DIR / "artifacts")))
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    r = Retriever(index_dir=artifacts_dir)
    per_q: dict[str, dict[str, list]] = {v: {} for v in VARIANTS}
    diagnostics = {v: {"latency": [], "rank_shift": [], "n_cand": []} for v in VARIANTS}

    # 3. รันประเมินผลแต่ละคำถาม
    for n, item in enumerate(items, 1):
        q = item.get("question") or item.get("query")
        gold = gold_docs_of(item)

        for v in VARIANTS:
            t0 = time.perf_counter()
            hits, dbg = r.retrieve(
                q,
                mode=v,
                k=max_k,
                retrieve_k=rerank_pool,
                return_debug=True,
            )
            elapsed = time.perf_counter() - t0

            diagnostics[v]["latency"].append(elapsed)
            diagnostics[v]["rank_shift"].append(dbg.get("rank_shift", 0))
            diagnostics[v]["n_cand"].append(dbg.get("n_candidates", dbg.get("n_returned", 0)))

            docs = retrieved_docs_of(hits)
            d = per_q[v]

            for k_val in eval_k_values:
                d.setdefault(f"recall@{k_val}", []).append(recall_at_k(docs, gold, k_val))
                d.setdefault(f"hit@{k_val}", []).append(hit_at_k(docs, gold, k_val))
                d.setdefault(f"ndcg@{k_val}", []).append(ndcg_at_k(docs, gold, k_val))

            d.setdefault("mrr", []).append(mrr(docs, gold, max_k))

        print(f"  ความคืบหน้า: [{n}/{len(items)}] คำถามเรียบร้อย", end="\r")

    print("\nประมวลผลเสร็จสิ้น กำลังสรุปสถิติ...")

    # 4. รวมผลคะแนนและสถิติ Bootstrap CI
    results = {}
    for v in VARIANTS:
        agg = {}
        for m, vals in per_q[v].items():
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            agg[m] = {"mean": round(mean(vals), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        agg["latency_ms_avg"] = round(1000 * mean(diagnostics[v]["latency"]), 2)
        agg["avg_candidates"] = round(mean(diagnostics[v]["n_cand"]), 1)
        if v == "hybrid_rerank":
            agg["avg_rank_shift"] = round(mean(diagnostics[v]["rank_shift"]), 2)
        results[v] = agg

    # 5. คำนวณความต่างอย่างมีนัยสำคัญทางสถิติ (เทียบกับ Dense-only)
    sig = {}
    target_sig_metrics = [f"recall@{k}" for k in eval_k_values if f"recall@{k}" in per_q["dense"]]
    target_sig_metrics.append("mrr")

    for v in VARIANTS:
        if v == "dense":
            continue
        sig[v] = {
            m: round(paired_bootstrap_p(per_q[v][m], per_q["dense"][m], n_boot=n_boot, seed=seed), 4)
            for m in target_sig_metrics
            if m in per_q[v] and m in per_q["dense"]
        }

    # 6. บันทึกผลลัพธ์ JSON
    out = {
        "n_questions": len(items),
        "k_values": eval_k_values,
        "config": {
            "TOP_K_RETRIEVE": getattr(config, "TOP_K_RETRIEVE", 30),
            "RERANK_CANDIDATES": rerank_pool,
            "TOP_K_FINAL": getattr(config, "TOP_K_FINAL", 5),
            "RRF_K": getattr(config, "RRF_K", 60),
        },
        "results": results,
        "p_values_vs_dense": sig,
        "per_question": per_q,
    }

    out_json = artifacts_dir / "ablation_results.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. จัดทำและบันทึกตาราง Markdown สำหรับใช้ในสไลด์พรีเซนต์
    disp_cols = [f"recall@{k}" for k in eval_k_values] + ["mrr"]
    header = ["| Variant |"] + [f" {c.upper()} |" for c in disp_cols] + [" Latency (ms) |"]
    sep = ["|---"] * (len(disp_cols) + 2) + ["|"]

    md_lines = ["".join(header), "".join(sep)]
    for v in VARIANTS:
        row_str = f"| `{v}` |"
        for c in disp_cols:
            val = results[v].get(c, {}).get("mean", 0.0)
            row_str += f" {val:.4f} |"
        row_str += f" {results[v]['latency_ms_avg']} |"
        md_lines.append(row_str)

    md_table = "\n".join(md_lines)
    (artifacts_dir / "ablation_table.md").write_text(md_table, encoding="utf-8")

    print("\n" + "=" * 60)
    print(md_table)
    print("=" * 60)

    rs = results["hybrid_rerank"].get("avg_rank_shift", 0)
    print(f"🔧 Cross-Encoder Reranker สลับอันดับเฉลี่ย: {rs} ตำแหน่ง/คำถาม")
    print(f"📁 บันทึกผลลัพธ์ที่: {out_json}")
    print(f"📋 บันทึกตาราง Markdown ที่: {artifacts_dir / 'ablation_table.md'}\n")


if __name__ == "__main__":
    main()

