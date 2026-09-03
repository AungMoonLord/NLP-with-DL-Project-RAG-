"""Ablation: เปรียบเทียบ 3 variants บน testset เดียวกัน
- A: dense-only
- B: dense + BM25 hybrid (RRF)
- C: hybrid + cross-encoder reranker
วัดด้วย Recall@5 และ MRR (retrieval-level → แยก effect ของ retriever ออกจาก LLM)
"""
import sys; sys.path.insert(0, ".")
import json
from src.retrieval import Retriever
from src.evaluation import recall_at_k, mrr
import config

with open(config.TESTSET_PATH, encoding="utf-8") as f:
    testset = json.load(f)

retriever = Retriever(use_reranker=True)
variants = ["dense", "hybrid", "hybrid_rerank"]
results = {v: {"recall@5": [], "mrr": []} for v in variants}

# for item in testset:
#     for v in variants:
#         retrieved = retriever.retrieve(item["question"], mode=v, k=5)
#         doc_ids = [c["doc_id"] for c in retrieved]
#         results[v]["recall@5"].append(recall_at_k(doc_ids, item["relevant_docs"]))
#         results[v]["mrr"].append(mrr(doc_ids, item["relevant_docs"]))

for item in testset:
    # ดึงเอกสารเป้าหมาย: ถ้าระบุ relevant_docs ไว้ใช้ตัวนั้น ถ้าไม่มีให้ดึงจาก source_doc
    target_docs = item.get("relevant_docs")
    if not target_docs:
        source = item.get("source_doc", "")
        # หุ้มเป็น list เพื่อให้ฟังก์ชัน evaluation นำไป match ได้
        target_docs = [source] if source else []

    for v in variants:
        retrieved = retriever.retrieve(item["question"], mode=v, k=5)
        # ตรวจสอบว่า retriever คืน doc_id หรือ file/source
        doc_ids = [c.get("doc_id") or c.get("file") or c.get("source", "") for c in retrieved]
        
        results[v]["recall@5"].append(recall_at_k(doc_ids, target_docs))
        results[v]["mrr"].append(mrr(doc_ids, target_docs))
        
summary = {
    v: {m: round(sum(vals) / len(vals), 4) for m, vals in metrics.items()}
    for v, metrics in results.items()
}

print(f"{'Variant':<16}{'Recall@5':<12}{'MRR':<8}")
for v, m in summary.items():
    print(f"{v:<16}{m['recall@5']:<12}{m['mrr']:<8}")

with open("artifacts/ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
