"""STEP 7 — RAG pipeline (ประกอบทุกชิ้นเข้าด้วยกัน)

Query -> [normalize] -> Retrieval(stage1: dense+BM25+RRF) -> [stage2: rerank]
      -> Context assembly (budget-aware) -> LLM -> Answer + citations
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .chunker import Chunk, load_chunks
from .config import Config
from .generator import BaseGenerator, build_generator
from .prompts import format_context
from .retriever import Retriever, RetrievedChunk, build_retriever
from .thai_utils import normalize_text


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: List[str]
    citations: List[str]
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    timings: Dict = field(default_factory=dict)
    trace: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "citations": self.citations,
            "retrieved_ids": [r.chunk.chunk_id for r in self.retrieved],
            "retrieved_scores": [r.score for r in self.retrieved],
            "timings": self.timings,
            "trace": self.trace,
        }

    def pretty(self) -> str:
        lines = [f"❓ คำถาม: {self.question}", "", f"💬 คำตอบ:\n{self.answer}", "",
                 "📚 แหล่งอ้างอิงที่ใช้:"]
        for i, r in enumerate(self.retrieved, 1):
            lines.append(f"  [S{i}] {r.chunk.citation}  (score={r.score:.4f})")
        lines.append(f"\n⏱  retrieval {self.timings.get('retrieval_s', 0):.2f}s | "
                     f"generation {self.timings.get('generation_s', 0):.2f}s")
        return "\n".join(lines)


class RAGPipeline:
    def __init__(self, cfg: Config, retriever: Retriever, generator: BaseGenerator,
                 context_char_budget: int = 12000):
        self.cfg = cfg
        self.retriever = retriever
        self.generator = generator
        # จำกัดขนาด context เพื่อคุมต้นทุน token และกัน "lost in the middle"
        # (โมเดลมีแนวโน้มมองข้ามข้อมูลที่อยู่กลาง context ที่ยาวมาก)
        self.context_char_budget = context_char_budget

    @classmethod
    def from_config(cls, cfg: Config) -> "RAGPipeline":
        proc = cfg.path(cfg.processed_dir)
        chunks = load_chunks(proc / f"chunks_{cfg.index_tag}.jsonl")
        retriever = build_retriever(cfg, chunks, proc)
        generator = build_generator(cfg.generator)
        return cls(cfg, retriever, generator)

    def _assemble(self, retrieved: List[RetrievedChunk]):
        kept, used = [], 0
        for r in retrieved:
            n = len(r.chunk.text)
            if used + n > self.context_char_budget and kept:
                break
            kept.append(r)
            used += n
        return kept

    def answer(self, question: str, top_k: Optional[int] = None,
               generate: bool = True) -> RAGResult:
        q = normalize_text(question)

        t0 = time.time()
        retrieved = self.retriever.retrieve(q, top_k=top_k)
        t_retrieval = time.time() - t0

        retrieved = self._assemble(retrieved)
        context = format_context(retrieved)

        t1 = time.time()
        answer = self.generator.answer(q, context) if generate else ""
        t_gen = time.time() - t1

        return RAGResult(
            question=question,
            answer=answer,
            contexts=[r.chunk.text for r in retrieved],
            citations=[r.chunk.citation for r in retrieved],
            retrieved=retrieved,
            timings={"retrieval_s": t_retrieval, "generation_s": t_gen,
                     "total_s": t_retrieval + t_gen},
            trace=dict(self.retriever.last_trace),
        )

    def batch_answer(self, questions: List[str], generate: bool = True,
                     verbose: bool = True) -> List[RAGResult]:
        out = []
        for i, q in enumerate(questions, 1):
            if verbose:
                print(f"  [{i}/{len(questions)}] {q[:60]}...")
            out.append(self.answer(q, generate=generate))
        return out
