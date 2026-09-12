"""STEP 1 — Corpus loading.

รับไฟล์ดิบจาก data/raw/ (รองรับ .txt .md .json .jsonl .pdf)
คืนค่าเป็น list ของ Document ที่มี metadata ครบ
metadata สำคัญมากในโดเมนกฎหมาย เพราะคำตอบต้องอ้างอิงได้ว่า "มาจากกฎหมายฉบับไหน มาตราอะไร"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

from .thai_utils import normalize_text

TEXT_SUFFIXES = {".txt", ".md"}
JSON_SUFFIXES = {".json", ".jsonl"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source_path: str
    meta: Dict = None

    def to_dict(self):
        return asdict(self)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ImportError("ต้องติดตั้ง pypdf เพื่ออ่านไฟล์ PDF: pip install pypdf") from e
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_json(path: Path) -> List[Dict]:
    records = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    return records


def load_corpus(raw_dir: str | Path, verbose: bool = True) -> List[Document]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลังเอกสาร: {raw_dir}")

    docs: List[Document] = []
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())

    for path in files:
        suffix = path.suffix.lower()
        rel = str(path.relative_to(raw_dir))
        try:
            if suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(
                    doc_id=path.stem, title=path.stem,
                    text=normalize_text(text), source_path=rel, meta={"format": suffix},
                ))
            elif suffix in PDF_SUFFIXES:
                text = _read_pdf(path)
                docs.append(Document(
                    doc_id=path.stem, title=path.stem,
                    text=normalize_text(text), source_path=rel, meta={"format": "pdf"},
                ))
            elif suffix in JSON_SUFFIXES:
                for i, rec in enumerate(_read_json(path)):
                    text = rec.get("text") or rec.get("content") or ""
                    if not text.strip():
                        continue
                    docs.append(Document(
                        doc_id=str(rec.get("id", f"{path.stem}-{i}")),
                        title=rec.get("title", path.stem),
                        text=normalize_text(text),
                        source_path=rel,
                        meta={k: v for k, v in rec.items()
                              if k not in {"text", "content", "id", "title"}},
                    ))
        except Exception as e:  # เอกสารเสียหนึ่งไฟล์ ไม่ควรทำให้ทั้ง pipeline ล้ม
            print(f"[loader] ข้ามไฟล์ {rel}: {e}")

    # กรองเอกสารว่าง/สั้นผิดปกติ (สแกน PDF ที่ extract ไม่ออกจะเหลือไม่กี่ตัวอักษร)
    kept, dropped = [], []
    for d in docs:
        (kept if len(d.text) >= 200 else dropped).append(d)

    if verbose:
        print(f"[loader] โหลดสำเร็จ {len(kept)} เอกสาร (ข้าม {len(dropped)} เอกสารที่สั้น/ว่าง)")
        if dropped:
            print(f"[loader] ตรวจสอบไฟล์เหล่านี้ด้วย: {[d.source_path for d in dropped][:5]}")
    return kept


def corpus_stats(docs: Iterable[Document]) -> Dict:
    from .thai_utils import count_thai_words
    lengths = [count_thai_words(d.text) for d in docs]
    n = len(lengths) or 1
    return {
        "n_documents": len(lengths),
        "total_words": sum(lengths),
        "avg_words_per_doc": sum(lengths) / n,
        "min_words": min(lengths, default=0),
        "max_words": max(lengths, default=0),
    }
