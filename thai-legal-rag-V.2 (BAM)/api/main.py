"""FastAPI backend สำหรับหน้าเว็บทดสอบระบบ

    python -m uvicorn api.main:app --reload --port 8000

ทำไมต้องมี API แยก แทนที่จะรัน demo.py:
  demo.py เปิด-ปิดโปรแกรมทุกครั้ง -> ต้องโหลดโมเดล 2.3 GB ใหม่ทุกคำถาม (ช้า ~45 วินาที)
  server โหลดครั้งเดียวตอนเปิด แล้วค้างไว้ในหน่วยความจำ -> คำถามถัดไปเหลือไม่กี่วินาที

จุดเด่นที่ใช้ตอนนำเสนอ: สลับโหมด retrieval ได้ทันทีจากหน้าเว็บ
(dense / hybrid / +reranker) เพื่อโชว์ผลของ ablation แบบสด ๆ
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Config
from src.pipeline import RAGPipeline

app = FastAPI(title="Thai Legal RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

STATE: dict = {"pipeline": None, "config": None, "error": None, "ready": False}
LOCK = threading.Lock()      # โมเดลไม่ thread-safe และเราสลับ config ระหว่างคำขอ


# ---------- schema ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    mode: Optional[str] = Field(default=None, pattern="^(dense|bm25|hybrid)$")
    use_reranker: Optional[bool] = None
    generate: bool = True


class Source(BaseModel):
    label: str
    citation: str
    title: str
    breadcrumb: str
    section_no: Optional[str]
    text: str
    chunk_id: str
    score: float
    rank: int
    prev_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    found_by: List[str] = []


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    mode: str
    reranked: bool
    timings: dict
    rank_changed: bool


# ---------- lifecycle ----------
@app.on_event("startup")
def load_pipeline():
    def _load():
        try:
            t0 = time.time()
            cfg = Config.load()
            # บังคับโหลดทุกองค์ประกอบ เพื่อให้สลับโหมดได้ทันทีโดยไม่ต้องโหลดใหม่
            cfg.retrieval.mode = "hybrid"
            cfg.retrieval.use_reranker = True
            pipe = RAGPipeline.from_config(cfg)
            print("[api] อุ่นเครื่องโมเดล ...")
            pipe.answer("ทดสอบระบบ", generate=False)   # โหลด weight เข้าหน่วยความจำ
            STATE.update(pipeline=pipe, config=cfg, ready=True)
            print(f"[api] พร้อมใช้งานใน {time.time()-t0:.1f} วินาที")
        except Exception as e:
            STATE["error"] = f"{type(e).__name__}: {e}"
            print(f"[api] โหลดไม่สำเร็จ: {STATE['error']}")

    threading.Thread(target=_load, daemon=True).start()


@app.get("/api/health")
def health():
    cfg: Config = STATE.get("config")
    chunks = STATE["pipeline"].retriever.chunks if STATE["ready"] else []
    docs = {c.doc_id for c in chunks}
    return {
        "ready": STATE["ready"],
        "error": STATE["error"],
        "corpus": {"documents": len(docs), "chunks": len(chunks)},
        "config": None if not cfg else {
            "embedding_model": cfg.embedding.model_name,
            "reranker_model": cfg.retrieval.reranker_model,
            "generator": cfg.generator.model,
            "chunk_size": cfg.chunk.max_tokens,
            "overlap": cfg.chunk.overlap_tokens,
            "top_k": cfg.retrieval.top_k_final,
        },
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if STATE["error"]:
        raise HTTPException(503, f"ระบบโหลดไม่สำเร็จ: {STATE['error']}")
    if not STATE["ready"]:
        raise HTTPException(503, "ระบบกำลังโหลดโมเดล กรุณารอสักครู่")

    pipe: RAGPipeline = STATE["pipeline"]
    rcfg = pipe.retriever.cfg

    with LOCK:
        saved = (rcfg.mode, rcfg.use_reranker)
        try:
            if req.mode:
                rcfg.mode = req.mode
            if req.use_reranker is not None:
                rcfg.use_reranker = req.use_reranker
            res = pipe.answer(req.question, top_k=req.top_k, generate=req.generate)
            mode, reranked = rcfg.mode, bool(rcfg.use_reranker)
        finally:
            rcfg.mode, rcfg.use_reranker = saved

    sources = []
    for i, r in enumerate(res.retrieved, start=1):
        d = r.debug or {}
        ranks = d.get("ranks") or {}
        sources.append(Source(
            label=f"S{i}",
            citation=r.chunk.citation,
            title=r.chunk.title,
            breadcrumb=r.chunk.breadcrumb,
            section_no=r.chunk.section_no,
            text=r.chunk.text,
            chunk_id=r.chunk.chunk_id,
            score=round(r.score, 4),
            rank=i,
            prev_rank=d.get("prev_rank"),
            dense_score=d.get("dense_score"),
            bm25_score=d.get("bm25_score"),
            found_by=sorted(ranks.keys()) if ranks else [r.source],
        ))

    return AskResponse(
        question=res.question, answer=res.answer, sources=sources,
        mode=mode, reranked=reranked,
        timings={k: round(v, 2) for k, v in res.timings.items()},
        rank_changed=bool(res.trace.get("rank_changed")),
    )
