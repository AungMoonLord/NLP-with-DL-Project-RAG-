"""STEP 3 — Dense embedding + vector index

ประเด็นเทคนิคที่ถูกถามบ่อยในการ defend:
1) ทำไมต้อง normalize เวกเตอร์  -> เพื่อให้ dot product เท่ากับ cosine similarity
   (ถ้าไม่ normalize แล้วใช้ dot product = ให้คะแนน chunk ที่ยาวกว่าโดยอัตโนมัติ)
2) ทำไม query กับ passage ใช้ prefix ต่างกัน -> โมเดลตระกูล E5 ถูกเทรนแบบ asymmetric
   ถ้าลืมใส่ "query: " / "passage: " ประสิทธิภาพตกได้หลายจุด
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import EmbeddingConfig


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[embedder] โหลดโมเดล {self.cfg.model_name} บน {self.cfg.device} ...")
            self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
            self._model.max_seq_length = self.cfg.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        texts = [self.cfg.passage_prefix + t for t in texts]
        return self._encode(texts, show_progress)

    def encode_queries(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        texts = [self.cfg.query_prefix + t for t in texts]
        return self._encode(texts, show_progress)

    def _encode(self, texts: List[str], show_progress: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        return vecs.astype("float32")


class DenseIndex:
    """Flat (exact) index.

    ที่คลังขนาด ~80 เอกสาร / ไม่กี่พัน chunk การค้นแบบ exact เร็วพออยู่แล้ว (มิลลิวินาที)
    ไม่จำเป็นต้องใช้ ANN (HNSW/IVF) ซึ่งแลกความแม่นยำกับความเร็ว
    -> เป็นเหตุผลที่ต้องตอบได้ว่า "ทำไมไม่ใช้ FAISS HNSW"
    """

    def __init__(self, vectors: Optional[np.ndarray] = None, ids: Optional[List[str]] = None):
        self.vectors = vectors
        self.ids = ids or []
        self._faiss_index = None
        if vectors is not None:
            self._try_build_faiss()

    def _try_build_faiss(self):
        try:
            import faiss
            index = faiss.IndexFlatIP(self.vectors.shape[1])  # IP + normalized = cosine
            index.add(self.vectors)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None   # fallback เป็น numpy matmul

    def search(self, query_vecs: np.ndarray, top_k: int = 10):
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(query_vecs, top_k)
        else:
            sims = query_vecs @ self.vectors.T          # (nq, N)
            idx = np.argsort(-sims, axis=1)[:, :top_k]
            scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, ids=np.array(self.ids, dtype=object))
        print(f"[dense] บันทึก index {self.vectors.shape} -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        data = np.load(path, allow_pickle=True)
        return cls(vectors=data["vectors"], ids=list(data["ids"]))
