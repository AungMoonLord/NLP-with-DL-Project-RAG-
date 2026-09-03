import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
import config


def thai_tokenize(text: str) -> list[str]:
    """ตัดคำภาษาไทยด้วย newmm — จำเป็นเพราะ BM25 ต้องการ word-level tokens
    (ภาษาไทยไม่มีช่องว่างคั่นคำ ใช้ split() ตรงๆ ไม่ได้)"""
    return [t for t in word_tokenize(text, engine="newmm") if t.strip()]


class Indexer:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    def build(self, chunks: list[dict], out_dir: str = config.INDEX_DIR):
        os.makedirs(out_dir, exist_ok=True)
        texts = [c["text"] for c in chunks]

        # ---- Dense (E5 ต้องใส่ prefix "passage: " ตอน index) ----
        embeddings = self.model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=16,
            normalize_embeddings=True,   # normalize → inner product = cosine similarity
            show_progress_bar=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(out_dir, "dense.faiss"))

        # ---- BM25 (สำหรับ hybrid variant) ----
        tokenized = [thai_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25, f)

        with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=1)

        print(f"Indexed {len(chunks)} chunks → {out_dir}")
