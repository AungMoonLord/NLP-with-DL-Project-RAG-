"""Smoke Test & System Sanity Check
สคริปต์ตรวจความสมบูรณ์ของระบบแบบครบวงจร (Configuration, Metrics, Testset, และ Live Retriever)
ก่อนเริ่มรันสคริปต์ประเมินผลจริง (scripts/run_ablation.py และ scripts/run_eval.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.metrics import norm_doc, gold_docs_of, recall_at_k, hit_at_k, mrr, ndcg_at_k

PASS, FAIL = "✅", "❌"
errors = []


def check(name: str, cond: bool):
    print(f"{PASS if cond else FAIL} {name}")
    if not cond:
        errors.append(name)


print("── 1. Configuration Sanity ──")
top_k_final = getattr(config, "TOP_K_FINAL", 5)
top_k_retrieve = getattr(config, "TOP_K_RETRIEVE", 30)
rerank_cands = getattr(config, "RERANK_CANDIDATES", top_k_retrieve)
eval_k_vals = getattr(config, "EVAL_K_VALUES", [1, 3, 5])
max_eval_k = getattr(config, "MAX_EVAL_K", max(eval_k_vals))

check("Candidate Pool > TOP_K_FINAL", rerank_cands > top_k_final)
check("TOP_K_RETRIEVE >= MAX_EVAL_K", top_k_retrieve >= max_eval_k)
check("Endpoint URL Configured", bool(getattr(config, "LLM_BASE_URL", "")))
check("Model Configured", bool(getattr(config, "LLM_MODEL", "")))

print("\n── 2. Information Retrieval Metrics ──")
check("norm_doc ตัดนามสกุลและพาธ", norm_doc("data/docs/A.TXT") == "a")
check("gold จาก source_doc", gold_docs_of({"source_doc": "civil.txt"}) == {"civil"})
check("gold จาก relevant_docs", gold_docs_of({"relevant_docs": ["a.txt", "b"]}) == {"a", "b"})

sample_ret, sample_gold = ["x", "a", "y", "b", "z"], {"a", "b"}
check("recall@5 = 1.0", recall_at_k(sample_ret, sample_gold, 5) == 1.0)
check("recall@1 = 0.0", recall_at_k(sample_ret, sample_gold, 1) == 0.0)
check("hit@2 = 1.0", hit_at_k(sample_ret, sample_gold, 2) == 1.0)
check("mrr = 0.5", abs(mrr(sample_ret, sample_gold, 5) - 0.5) < 1e-9)
check("ndcg@5 อยู่ใน [0, 1]", 0 <= ndcg_at_k(sample_ret, sample_gold, 5) <= 1)
check("gold ว่าง คืนค่า 0.0 ปลอดภัย", recall_at_k(sample_ret, set(), 5) == 0.0)

print("\n── 3. Testset & Corpus Sync ──")
testset_path = getattr(config, "TESTSET_PATH", ROOT_DIR / "data" / "testset.json")
artifacts_dir = Path(getattr(config, "ARTIFACTS_DIR", getattr(config, "INDEX_DIR", ROOT_DIR / "artifacts")))

with open(testset_path, encoding="utf-8") as f:
    ts = json.load(f)
ts = ts.get("questions", ts) if isinstance(ts, dict) else ts

adv = [t for t in ts if t.get("adversarial")]
normal = [t for t in ts if not t.get("adversarial")]

check(f"มี Adversarial Questions >= 5 ข้อ (พบ {len(adv)} ข้อ)", len(adv) >= 5)
check("คำถามปกติทุกข้อมี Gold Docs ระบุครบ", all(gold_docs_of(t) for t in normal))

with open(artifacts_dir / "chunks.json", encoding="utf-8") as f:
    chunk_data = json.load(f)
chunk_docs = {norm_doc(c.get("doc_id") or c.get("file") or str(c.get("chunk_id", "")).rsplit("::", 1)[0]) for c in chunk_data}

orphan = [t.get("id") for t in normal if not (gold_docs_of(t) & chunk_docs)]
check(f"เอกสารอ้างอิงทุกข้อมีอยู่ใน Corpus (Orphans: {orphan[:3]})", not orphan)

print("\n── 4. Live Retriever Tests ──")
from src.retrieval import Retriever

r = Retriever(index_dir=artifacts_dir)
sample_query = normal[0].get("question") or normal[0].get("query")

for m in ("dense", "bm25", "hybrid"):
    hits = r.retrieve(sample_query, mode=m, k=5)
    check(f"Retriever mode='{m}' คืนค่า 5 Chunks", len(hits) == 5)

hits, dbg = r.retrieve(
    sample_query,
    mode="hybrid_rerank",
    k=5,
    retrieve_k=rerank_cands,
    return_debug=True,
)
check(f"Candidate ป้อนเข้า Reranker > 5 (ได้ {dbg.get('n_candidates', len(hits))})", dbg.get("n_candidates", 0) > 5)
check("ผลลัพธ์สุดท้าย hybrid_rerank = 5", len(hits) == 5)
check("ทุก Chunk มี rerank_score", all("rerank_score" in h for h in hits))
check(
    "คะแนน rerank_score เรียงจากมากไปน้อย",
    all(hits[i]["rerank_score"] >= hits[i + 1]["rerank_score"] for i in range(len(hits) - 1)),
)

check("Query ว่างเปล่าไม่ Crash", r.retrieve("   ", mode="hybrid") == [])

try:
    r.retrieve(sample_query, mode="invalid_mode")
    check("Mode ผิดต้อง Raise ValueError", False)
except ValueError:
    check("Mode ผิดต้อง Raise ValueError", True)

print("\n" + ("=" * 50))
if not errors:
    print(f"{PASS} ผ่านการทดสอบทั้งหมด ระบบพร้อมรัน run_ablation.py และ run_eval.py")
else:
    print(f"{FAIL} พบปัญหา {len(errors)} รายการที่ต้องแก้ไขก่อนรันจริง: {errors}")

sys.exit(1 if errors else 0)

