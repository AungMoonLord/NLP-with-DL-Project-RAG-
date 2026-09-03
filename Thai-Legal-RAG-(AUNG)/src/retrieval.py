import json
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.indexer import thai_tokenize
import config


class Retriever:
    def __init__(self, index_dir: str = config.INDEX_DIR, use_reranker: bool = False):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
        self.index = faiss.read_index(os.path.join(index_dir, "dense.faiss"))
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.reranker = CrossEncoder(config.RERANKER_MODEL, device="cpu") if use_reranker else None

    # ---------- Variant A: Dense-only ----------
    def dense_search(self, query: str, k: int = config.TOP_K_RETRIEVE) -> list[tuple[int, float]]:
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def bm25_search(self, query: str, k: int = config.TOP_K_RETRIEVE) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(thai_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in top]

    # ---------- Variant B: Hybrid ด้วย Reciprocal Rank Fusion ----------
    def hybrid_search(self, query: str, k: int = config.TOP_K_RETRIEVE) -> list[tuple[int, float]]:
        """RRF: score(d) = Σ 1/(RRF_K + rank_i(d))
        ข้อดี: ไม่ต้อง normalize score ข้าม retrieval สองระบบที่ scale ต่างกัน"""
        rrf: dict[int, float] = {}
        for results in (self.dense_search(query, k), self.bm25_search(query, k)):
            for rank, (idx, _) in enumerate(results):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return merged

    # ---------- Variant C: + Cross-encoder rerank ----------
    def rerank(self, query: str, candidates: list[tuple[int, float]],
               k: int = config.TOP_K_FINAL) -> list[tuple[int, float]]:
        pairs = [(query, self.chunks[i]["text"]) for i, _ in candidates]
        scores = self.reranker.predict(pairs)
        order = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(s)) for (idx, _), s in order[:k]]

    # ---------- Unified entry point ----------
    def retrieve(self, query: str, mode: str = "hybrid",
                 k: int = config.TOP_K_FINAL) -> list[dict]:
        if mode == "dense":
            results = self.dense_search(query)[:k]
        elif mode == "bm25":
            results = self.bm25_search(query)[:k]
        elif mode == "hybrid":
            results = self.hybrid_search(query)[:k]
        elif mode == "hybrid_rerank":
            assert self.reranker, "init Retriever with use_reranker=True"
            results = self.rerank(query, self.hybrid_search(query))
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [{**self.chunks[i], "score": s} for i, s in results]
