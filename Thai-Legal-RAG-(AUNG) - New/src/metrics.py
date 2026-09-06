"""Information Retrieval Metrics & Statistical Significance Harness
โมดูลสำหรับคำนวณ Retrieval Metrics (Recall@k, MRR@k, nDCG@k, Hit@k)
พร้อมระบบแก้ปัญหา Document ID String Mismatch และ Bootstrap Statistical Tests
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def norm_doc(x) -> str:
    """Normalize ชื่อเอกสารเพื่อแก้ปัญหา String Mismatch
    ตัด path ทั้งแบบ Windows (\) และ POSIX (/), ตัดนามสกุลไฟล์ (.txt, .pdf ฯลฯ), และตัด whitespace
    """
    if x is None:
        return ""
    s = str(x).strip().replace("\\", "/").split("/")[-1].lower()
    for ext in (".txt", ".pdf", ".json", ".docx"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s.strip()


def gold_docs_of(item: dict) -> set[str]:
    """ดึงรายชื่อ Ground Truth Document ID จาก testset.json
    รองรับทุกรูปแบบ Key ทั้ง relevant_docs (list), source_doc (str), doc_id ฯลฯ
    """
    for key in ("relevant_docs", "gold_docs", "relevant_doc_ids"):
        v = item.get(key)
        if v:
            v = [v] if isinstance(v, str) else v
            out = {norm_doc(d) for d in v if norm_doc(d)}
            if out:
                return out
    for key in ("source_doc", "doc_id", "source"):
        v = item.get(key)
        if v and norm_doc(v):
            return {norm_doc(v)}
    return set()


def retrieved_docs_of(hits: Sequence[dict]) -> list[str]:
    """ดึงรายชื่อ Document ID จาก Chunks ที่ Retriever ดึงขึ้นมา พร้อม Normalize"""
    docs = []
    for h in hits:
        # รองรับทั้ง doc_id, file, source หรือ parse จาก chunk_id (เช่น "doc_name::chunk_0")
        raw_id = h.get("doc_id") or h.get("file") or h.get("source")
        if not raw_id and "chunk_id" in h:
            raw_id = str(h["chunk_id"]).rsplit("::", 1)[0]
        docs.append(norm_doc(raw_id))
    return docs


# ==========================================
# 1. Retrieval Metrics
# ==========================================

def recall_at_k(ret: Sequence[str], gold: set[str], k: int = 5) -> float:
    """สัดส่วนของเอกสารที่ถูกต้อง (Gold Docs) ที่ถูกค้นพบใน Top-K"""
    if not gold:
        return 0.0
    pool = ret[:k]
    found = set(pool) & gold
    return len(found) / len(gold)


def hit_at_k(ret: Sequence[str], gold: set[str], k: int = 5) -> float:
    """ตรวจสอบว่าเจอเอกสารที่ถูกต้องอย่างน้อย 1 รายการใน Top-K หรือไม่ (Hit Rate)"""
    if not gold:
        return 0.0
    return 1.0 if set(ret[:k]) & gold else 0.0


def mrr(ret: Sequence[str], gold: set[str], k: int | None = 5) -> float:
    """Mean Reciprocal Rank (MRR): คำนวณส่วนกลับของลำดับแรกที่พบเอกสารถูกต้อง (1/Rank)"""
    if not gold:
        return 0.0
    pool = ret[:k] if k is not None else ret
    for rank, d in enumerate(pool, start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ret: Sequence[str], gold: set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain (nDCG@K) สำหรับวัด Ranking Quality"""
    if not gold:
        return 0.0
    pool = ret[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(pool, start=1) if d in gold)
    # Ideal DCG: สมมติว่าเอกสารที่ถูกต้องทั้งหมดขึ้นมาอยู่อันดับบนสุด
    ideal_hits = min(len(gold), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / ideal if ideal > 0.0 else 0.0


# ==========================================
# 2. สถิติเชิงอนุมาน (Statistical Tests)
# ==========================================

def mean(v: Sequence[float]) -> float:
    """คำนวณค่าเฉลี่ย โดยกรองค่า NaN ออกอัตโนมัติ"""
    vals = [x for x in v if x == x and x is not None]
    return sum(vals) / len(vals) if vals else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """คำนวณ 95% Bootstrap Confidence Interval ของค่าเฉลี่ย"""
    vals = [x for x in values if x == x and x is not None]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    boot_means = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo_idx = max(0, int((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (boot_means[lo_idx], boot_means[hi_idx])


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42) -> float:
    """Paired Bootstrap Hypothesis Test
    H0: mean(a) == mean(b) — ทดสอบว่า Variant C (a) เหนือกว่า Variant A (b) อย่างมีนัยสำคัญหรือไม่
    """
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y and x is not None and y is not None]
    if not pairs:
        return 1.0
    
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    obs = sum(diffs) / n
    if obs == 0:
        return 1.0

    centered = [d - obs for d in diffs]
    rng = random.Random(seed)
    cnt = sum(
        1 for _ in range(n_boot)
        if abs(sum(centered[rng.randrange(n)] for _ in range(n)) / n) >= abs(obs)
    )
    return (cnt + 1) / (n_boot + 1)

