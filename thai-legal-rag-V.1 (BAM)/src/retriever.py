"""STEP 5 — Retrieval: dense / BM25 / hybrid (RRF) / + cross-encoder reranking

สถาปัตยกรรมแบบ 2 ขั้น (two-stage retrieval)
   ขั้นที่ 1 (recall-oriented) : ดึง candidate 20 อันด้วย bi-encoder + BM25 แล้ว fuse ด้วย RRF
   ขั้นที่ 2 (precision-oriented): cross-encoder ให้คะแนนคู่ (query, chunk) ใหม่ แล้วเลือก top-5

เหตุผลที่ต้องแยก 2 ขั้น:
   bi-encoder เข้ารหัส query กับ document แยกกัน -> คำนวณล่วงหน้าได้ เร็วมาก แต่ไม่เห็น
   ปฏิสัมพันธ์ระดับคำระหว่างคู่ (no cross-attention)
   cross-encoder ป้อน [query, doc] เข้าโมเดลพร้อมกัน -> แม่นกว่ามาก แต่ต้องรันทีละคู่
   => ใช้ของถูกกรองให้เหลือน้อย แล้วค่อยใช้ของแพงจัดอันดับ
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .chunker import Chunk
from .config import RetrievalConfig
from .embedder import DenseIndex, Embedder
from .sparse import BM25Index


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    source: str                 # "dense" | "bm25" | "rrf" | "rerank"
    debug: Dict = None


# --------------------------------------------------------------------------
def reciprocal_rank_fusion(rank_lists: Dict[str, List[str]], k: int = 60) -> List[tuple]:
    """RRF: score(d) = ผลรวมของ 1 / (k + rank_i(d)) จากทุกระบบค้นหา

    ทำไมต้อง RRF แทนการถ่วงน้ำหนักคะแนนดิบ (weighted score fusion)?
      - คะแนน BM25 (0 ถึง ~30, ไม่มีขอบเขตแน่นอน) กับ cosine (-1..1) อยู่คนละสเกล
        การเอามาบวกกันตรง ๆ ต้อง normalize ซึ่งอ่อนไหวต่อ outlier มาก
      - RRF ใช้เฉพาะ "อันดับ" ไม่ใช่ "คะแนน" -> ไม่ต้องจูนน้ำหนัก และทนต่อสเกลที่ต่างกัน
      - k=60 คือค่าที่เสนอไว้ในเปเปอร์ต้นฉบับ (Cormack et al., 2009) ทำหน้าที่ลดอิทธิพล
        ของอันดับ 1-2 ไม่ให้ครอบงำผลรวม (1/61 vs 1/62 ต่างกันน้อย = ให้ความสำคัญกับ
        การที่เอกสารติดอันดับใน "หลายระบบ" มากกว่าติดอันดับสูงใน "ระบบเดียว")
    """
    scores: Dict[str, float] = {}
    contrib: Dict[str, Dict[str, int]] = {}
    for system, ids in rank_lists.items():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            contrib.setdefault(cid, {})[system] = rank
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(cid, s, contrib[cid]) for cid, s in ranked]


# --------------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 8):
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[rerank] โหลด cross-encoder {self.model_name} ...")
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
        return self._model

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.embed_text()) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        order = np.argsort(-np.asarray(scores))
        out = []
        for new_rank, i in enumerate(order[:top_k], start=1):
            c = candidates[i]
            # ส่งต่อ debug ของขั้นที่ 1 ไปด้วย ไม่งั้นคะแนน dense/bm25 จะหายหลัง rerank
            # ทำให้อธิบายไม่ได้ว่า chunk นี้ถูกดึงมาเพราะระบบไหน
            debug = dict(c.debug or {})
            debug.update({"prev_rank": c.rank, "prev_score": c.score,
                          "prev_source": c.source})
            out.append(RetrievedChunk(chunk=c.chunk, score=float(scores[i]),
                                      rank=new_rank, source="rerank", debug=debug))
        return out


# --------------------------------------------------------------------------
class Retriever:
    def __init__(self, cfg: RetrievalConfig, chunks: List[Chunk],
                 embedder: Optional[Embedder] = None,
                 dense_index: Optional[DenseIndex] = None,
                 bm25_index: Optional[BM25Index] = None,
                 reranker: Optional[CrossEncoderReranker] = None):
        self.cfg = cfg
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.last_trace: Dict = {}

    # ---------- ขั้นที่ 1 ----------
    def _dense_search(self, query: str, top_k: int) -> List[str]:
        qv = self.embedder.encode_queries([query])
        scores, idx = self.dense_index.search(qv, top_k)
        self._dense_scores = {self.dense_index.ids[i]: float(s)
                              for s, i in zip(scores[0], idx[0])}
        return [self.dense_index.ids[i] for i in idx[0]]

    def _bm25_search(self, query: str, top_k: int) -> List[str]:
        scores, idx = self.bm25_index.search(query, top_k)
        self._bm25_scores = {self.bm25_index.ids[i]: float(s) for s, i in zip(scores, idx)}
        return [self.bm25_index.ids[i] for i in idx]

    # ---------- entry point ----------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = top_k or self.cfg.top_k_final
        t0 = time.time()
        self._dense_scores, self._bm25_scores = {}, {}
        rank_lists: Dict[str, List[str]] = {}

        if self.cfg.mode in ("dense", "hybrid"):
            rank_lists["dense"] = self._dense_search(query, self.cfg.top_k_dense)
        if self.cfg.mode in ("bm25", "hybrid"):
            rank_lists["bm25"] = self._bm25_search(query, self.cfg.top_k_bm25)

        if self.cfg.mode == "hybrid":
            fused = reciprocal_rank_fusion(rank_lists, k=self.cfg.rrf_k)
            candidates = [
                RetrievedChunk(self.by_id[cid], score, r, "rrf",
                               {"ranks": ranks,
                                "dense_score": self._dense_scores.get(cid),
                                "bm25_score": self._bm25_scores.get(cid)})
                for r, (cid, score, ranks) in enumerate(fused, start=1)
            ]
        else:
            system = self.cfg.mode
            score_map = self._dense_scores if system == "dense" else self._bm25_scores
            candidates = [
                RetrievedChunk(self.by_id[cid], score_map.get(cid, 0.0), r, system, {})
                for r, cid in enumerate(rank_lists[system], start=1)
            ]

        stage1_time = time.time() - t0
        stage1_ids = [c.chunk.chunk_id for c in candidates[:top_k]]

        # ---------- ขั้นที่ 2 ----------
        if self.cfg.use_reranker and self.reranker is not None:
            pool = candidates[: self.cfg.rerank_candidates]
            results = self.reranker.rerank(query, pool, top_k)
        else:
            results = candidates[:top_k]

        self.last_trace = {
            "query": query,
            "mode": self.cfg.mode,
            "reranked": bool(self.cfg.use_reranker and self.reranker is not None),
            "stage1_time_s": round(stage1_time, 4),
            "total_time_s": round(time.time() - t0, 4),
            "stage1_top_ids": stage1_ids,
            "final_top_ids": [c.chunk.chunk_id for c in results],
            "rank_changed": stage1_ids != [c.chunk.chunk_id for c in results],
        }
        return results


# --------------------------------------------------------------------------
def build_retriever(cfg, chunks: List[Chunk], artifacts_dir) -> Retriever:
    """ประกอบ retriever จาก index ที่ ingest ไว้แล้ว"""
    from pathlib import Path
    artifacts_dir = Path(artifacts_dir)
    rcfg = cfg.retrieval

    embedder = dense_index = bm25_index = reranker = None
    if rcfg.mode in ("dense", "hybrid"):
        embedder = Embedder(cfg.embedding)
        dense_index = DenseIndex.load(artifacts_dir / f"dense_{cfg.index_tag}.npz")
    if rcfg.mode in ("bm25", "hybrid"):
        bm25_index = BM25Index.load(artifacts_dir / f"bm25_{cfg.index_tag}.pkl")
    if rcfg.use_reranker:
        reranker = CrossEncoderReranker(rcfg.reranker_model, device=cfg.embedding.device)

    return Retriever(rcfg, chunks, embedder, dense_index, bm25_index, reranker)
