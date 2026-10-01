# import json
# import os
# import pickle
# import faiss
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from src.indexer import thai_tokenize
# import config


# class Retriever:
#     def __init__(self, index_dir: str = config.INDEX_DIR, use_reranker: bool = False):
#         self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
#         self.index = faiss.read_index(os.path.join(index_dir, "dense.faiss"))
#         with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
#             self.bm25 = pickle.load(f)
#         with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
#             self.chunks = json.load(f)
#         self.reranker = CrossEncoder(config.RERANKER_MODEL, device="cpu") if use_reranker else None

#     # ---------- Variant A: Dense-only ----------
#     def dense_search(self, query: str, k: int = config.TOP_K_RETRIEVE) -> list[tuple[int, float]]:
#         q = self.model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
#         scores, idxs = self.index.search(q, k)
#         return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

#     def bm25_search(self, query: str, k: int = config.TOP_K_RETRIEVE) -> list[tuple[int, float]]:
#         scores = self.bm25.get_scores(thai_tokenize(query))
#         top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
#         return [(i, float(scores[i])) for i in top]

#     # ---------- Variant B: Hybrid ด้วย Reciprocal Rank Fusion ----------
#     def hybrid_search(self, query: str, k: int = config.TOP_K_RETRIEVE) -> list[tuple[int, float]]:
#         """RRF: score(d) = Σ 1/(RRF_K + rank_i(d))
#         ข้อดี: ไม่ต้อง normalize score ข้าม retrieval สองระบบที่ scale ต่างกัน"""
#         rrf: dict[int, float] = {}
#         for results in (self.dense_search(query, k), self.bm25_search(query, k)):
#             for rank, (idx, _) in enumerate(results):
#                 rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
#         merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
#         return merged

#     # ---------- Variant C: + Cross-encoder rerank ----------
#     def rerank(self, query: str, candidates: list[tuple[int, float]],
#                k: int = config.TOP_K_FINAL) -> list[tuple[int, float]]:
#         pairs = [(query, self.chunks[i]["text"]) for i, _ in candidates]
#         scores = self.reranker.predict(pairs)
#         order = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
#         return [(idx, float(s)) for (idx, _), s in order[:k]]

#     # ---------- Unified entry point ----------
#     def retrieve(self, query: str, mode: str = "hybrid",
#                  k: int = config.TOP_K_FINAL) -> list[dict]:
#         if mode == "dense":
#             results = self.dense_search(query)[:k]
#         elif mode == "bm25":
#             results = self.bm25_search(query)[:k]
#         elif mode == "hybrid":
#             results = self.hybrid_search(query)[:k]
#         elif mode == "hybrid_rerank":
#             assert self.reranker, "init Retriever with use_reranker=True"
#             results = self.rerank(query, self.hybrid_search(query))
#         else:
#             raise ValueError(f"unknown mode: {mode}")
#         return [{**self.chunks[i], "score": s} for i, s in results]

"""Retrieval Pipeline: Dense, BM25, Hybrid (RRF), and Cross-Encoder Reranking
รองรับการดึงข้อมูลแบบ 2-Stage Retrieval พร้อมระบบ Candidate Pool สำหรับ Reranker
"""
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.indexer import thai_tokenize
import config

VALID_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class Retriever:
    _reranker = None
    _lock = threading.Lock()

    def __init__(self, index_dir: str | Path | None = None, use_reranker: bool = False):
        # รองรับทั้ง index_dir เดิม และ artifacts_dir
        target_dir = index_dir or getattr(config, "ARTIFACTS_DIR", getattr(config, "INDEX_DIR", "artifacts"))
        art = Path(target_dir)

        # 1. โหลด Chunks
        with open(art / "chunks.json", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        # 2. โหลด FAISS Dense Index
        self.index = faiss.read_index(str(art / "dense.faiss"))

        # 3. โหลด BM25 Model (รองรับทั้ง object ตรงๆ และ dictionary wrapper)
        with open(art / "bm25.pkl", "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            self.bm25 = obj.get("bm25") or obj.get("model")
            if self.bm25 is None:
                corpus = obj.get("tokenized") or obj.get("corpus")
                if corpus is None:
                    raise ValueError("bm25.pkl ไม่มีทั้ง model และ corpus")
                self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = obj

        # Data Guard: ตรวจสอบความสอดคล้องของ FAISS Index กับ chunks.json
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"index ntotal={self.index.ntotal} != chunks={len(self.chunks)} "
                "-> artifacts ไม่ sync กัน ให้รัน build_index.py ใหม่"
            )

        # 4. โหลด Dense Embedding Model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    # ---------------- Helper Methods ----------------
    def _hit(self, idx: int, score: float, rank: int) -> dict:
        c = dict(self.chunks[idx])
        c["score"] = float(score)
        c["rank"] = rank
        c["_idx"] = int(idx)
        c.setdefault("doc_id", str(c.get("chunk_id", "")).rsplit("::", 1)[0])
        return c

    @classmethod
    def _get_reranker(cls) -> CrossEncoder:
        with cls._lock:
            if cls._reranker is None:
                cls._reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device="cpu")
        return cls._reranker

    # ---------------- Search Methods ----------------
    def _dense_search(self, query: str, k: int) -> list[dict]:
        k = max(0, min(int(k), self.index.ntotal))
        if k == 0:
            return []
        prefix = getattr(config, "E5_QUERY_PREFIX", "query: ")
        q = self.model.encode(
            [prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, ids = self.index.search(q, k)
        return [
            self._hit(i, s, r)
            for r, (i, s) in enumerate(zip(ids[0], scores[0]), 1)
            if i >= 0
        ]

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        toks = thai_tokenize(query)
        if not toks:
            return []
        scores = np.asarray(self.bm25.get_scores(toks), dtype="float64")
        n_docs = scores.shape[0]
        k = max(0, min(int(k), n_docs))
        if k == 0:
            return []
        
        # ป้องกัน Index Out of Bounds และจัดเรียงอันดับคะแนนคงที่
        if k < n_docs:
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top = top_idx[np.argsort(-scores[top_idx], kind="stable")]
        else:
            top = np.argsort(-scores, kind="stable")
            
        return [self._hit(int(i), float(scores[i]), r) for r, i in enumerate(top, 1)]

    @staticmethod
    def _rrf(rank_lists: list[list[dict]], k_const: int = 60) -> list[dict]:
        fused: dict[Any, dict] = {}
        for lst in rank_lists:
            for rank, hit in enumerate(lst, 1):
                cid = hit["chunk_id"]
                e = fused.setdefault(cid, {"hit": hit, "score": 0.0})
                e["score"] += 1.0 / (k_const + rank)
        out = []
        for cid, e in fused.items():
            h = dict(e["hit"])
            h["score"] = e["score"]
            out.append(h)
        out.sort(key=lambda x: (-x["score"], str(x["chunk_id"])))
        for r, h in enumerate(out, 1):
            h["rank"] = r
        return out

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates:
            return []
        ce = self._get_reranker()
        pairs = [(query, str(c.get("text", ""))) for c in candidates]
        batch_size = getattr(config, "RERANK_BATCH_SIZE", 16)
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        out = []
        for c, s in zip(candidates, scores):
            d = dict(c)
            d["pre_rerank_rank"] = c.get("rank")
            d["rerank_score"] = float(s)
            out.append(d)
        out.sort(key=lambda x: (-x["rerank_score"], str(x["chunk_id"])))
        for r, d in enumerate(out, 1):
            d["rank"] = r
        return out[:top_n]

    # ---------------- Public API ----------------
    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        k: int | None = None,
        retrieve_k: int | None = None,
        return_debug: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        if mode not in VALID_MODES:
            raise ValueError(f"mode ต้องเป็นหนึ่งใน {VALID_MODES}, ได้ {mode!r}")
        if not str(query).strip():
            return ([], {"error": "empty query"}) if return_debug else []

        final_k = int(k if k is not None else getattr(config, "TOP_K_FINAL", 5))
        
        # จัดการขนาด Candidate Pool ก่อน Rerank หรือ Fusion
        if retrieve_k is None:
            default_pool = getattr(config, "TOP_K_RETRIEVE", 30)
            retrieve_k = getattr(config, "RERANK_CANDIDATES", default_pool) if mode == "hybrid_rerank" else default_pool
        
        pool = max(int(retrieve_k), final_k)
        rrf_k = getattr(config, "RRF_K", 60)
        dbg: dict[str, Any] = {"mode": mode, "pool": pool, "final_k": final_k}

        if mode == "dense":
            results = self._dense_search(query, pool)[:final_k]
        elif mode == "bm25":
            results = self._bm25_search(query, pool)[:final_k]
        elif mode == "hybrid":
            d = self._dense_search(query, pool)
            b = self._bm25_search(query, pool)
            dbg.update(n_dense=len(d), n_bm25=len(b))
            results = self._rrf([d, b], rrf_k)[:final_k]
        else:  # hybrid_rerank
            d = self._dense_search(query, pool)
            b = self._bm25_search(query, pool)
            cands = self._rrf([d, b], rrf_k)[:pool]
            dbg.update(n_dense=len(d), n_bm25=len(b), n_candidates=len(cands))
            results = self.rerank(query, cands, final_k)

        dbg["n_returned"] = len(results)
        return (results, dbg) if return_debug else results

    