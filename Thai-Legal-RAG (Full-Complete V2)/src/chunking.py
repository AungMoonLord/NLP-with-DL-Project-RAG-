"""Token-aware chunking โดยใช้ tokenizer ตัวเดียวกับ embedding model
เพื่อรับประกันว่า chunk ไม่เกิน max sequence length ของโมเดลจริงๆ
(นับ token จริง ไม่ใช่นับตัวอักษร — สำคัญมากสำหรับภาษาไทยที่ 1 คำ ≠ 1 token)
"""
import os
import re
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer
import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    n_tokens: int
    metadata: dict = field(default_factory=dict)


class ThaiLegalChunker:
    def __init__(self, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        self.chunk_size = chunk_size
        self.overlap = overlap
        # แยกตาม "มาตรา" ก่อน (structure-aware) แล้วค่อย pack เป็น chunk
        #self.section_pattern = re.compile(r"(?=มาตรา\s*[\d๐-๙]+)")
        self.section_pattern = re.compile(r"(?=(?:มาตรา|ข้อ)\s*[\d๐-๙]+)")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_long_text(self, text: str) -> list[str]:
        """sliding window ระดับ token สำหรับข้อความที่ยาวเกิน chunk_size"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        step = self.chunk_size - self.overlap
        pieces = []
        for start in range(0, len(ids), step):
            window = ids[start:start + self.chunk_size]
            pieces.append(self.tokenizer.decode(window, skip_special_tokens=True))
            if start + self.chunk_size >= len(ids):
                break
        return pieces

    def chunk_document(self, doc_id: str, text: str) -> list[Chunk]:
        # 1) แยกตามมาตรา (ถ้าเจอ) — ไม่เจอก็ถือทั้งเอกสารเป็น section เดียว
        sections = [s.strip() for s in self.section_pattern.split(text) if s.strip()]
        chunks, buffer, buffer_tokens = [], [], 0

        def flush():
            nonlocal buffer, buffer_tokens
            if buffer:
                merged = "\n".join(buffer)
                chunks.append(merged)
                buffer, buffer_tokens = [], 0

        # 2) pack มาตราสั้นๆ รวมกันจนใกล้ chunk_size (ลดจำนวน chunk ที่สั้นเกิน)
        for sec in sections:
            n = self._count_tokens(sec)
            if n > self.chunk_size:
                flush()
                chunks.extend(self._split_long_text(sec))
            elif buffer_tokens + n > self.chunk_size:
                flush()
                buffer, buffer_tokens = [sec], n
            else:
                buffer.append(sec)
                buffer_tokens += n
        flush()

        return [
            Chunk(
                chunk_id=f"{doc_id}::chunk_{i}",
                doc_id=doc_id,
                text=c,
                n_tokens=self._count_tokens(c),
            )
            for i, c in enumerate(chunks)
        ]

    def chunk_corpus(self, docs_dir: str = config.DOCS_DIR) -> list[Chunk]:
        all_chunks = []
        for fname in sorted(os.listdir(docs_dir)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(docs_dir, fname), encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(self.chunk_document(fname.removesuffix(".txt"), text))
        return all_chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
