"""STEP 4 (Extension) — BM25 sparse retrieval

ทำไมต้องมี BM25 ทั้งที่มี dense แล้ว?
  dense retrieval เก่งเรื่อง "ความหมายใกล้เคียง" แต่พลาดเรื่อง "การตรงคำเป๊ะ ๆ"
  โดเมนกฎหมายเต็มไปด้วย exact term: เลขมาตรา, ชื่อกฎหมาย, ศัพท์เฉพาะ ("โมฆียะ", "ลาภมิควรได้")
  ซึ่งอาจไม่มีอยู่ในชุดเทรนของ embedding model -> vector มันเบลอ
  BM25 จับได้ทันทีเพราะทำ lexical matching ตรง ๆ
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .thai_utils import tokenize_for_bm25


class BM25Index:
    def __init__(self, ids: List[str], corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.ids = ids
        self.corpus_tokens = corpus_tokens
        self.k1, self.b = k1, b
        # b=0.75 ค่ามาตรฐาน (length normalization).
        # เอกสารกฎหมายมีความยาวมาตราต่างกันมาก -> ยังอยากได้ normalization ไว้
        self.bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)

    @classmethod
    def build(cls, ids: List[str], texts: List[str], **kw) -> "BM25Index":
        print(f"[bm25] กำลังตัดคำไทย {len(texts)} chunks ...")
        toks = [tokenize_for_bm25(t) for t in texts]
        return cls(ids, toks, **kw)

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = tokenize_for_bm25(query)
        scores = np.asarray(self.bm25.get_scores(q), dtype="float32")
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return scores[idx], idx

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "tokens": self.corpus_tokens,
                         "k1": self.k1, "b": self.b}, f)
        print(f"[bm25] บันทึก index -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["ids"], d["tokens"], k1=d["k1"], b=d["b"])
