"""STEP 2 — Token-aware, structure-aware chunking สำหรับเอกสารกฎหมายไทย

กลยุทธ์ 3 ชั้น (สำคัญ: ต้องอธิบายได้ในการนำเสนอ)
  ชั้นที่ 1  Structural split : ตัดตามขอบเขต "มาตรา" / "ข้อ" เพราะเป็นหน่วยความหมาย
                                ที่เล็กที่สุดที่ยัง "สมบูรณ์ทางกฎหมาย" ในตัวเอง
  ชั้นที่ 2  Token-aware split: มาตราไหนยาวเกิน max_tokens ค่อยซอยต่อด้วยประโยค
                                + overlap (นับ token ด้วย tokenizer ของ embedding model จริง
                                ไม่ใช่จำนวนคำ เพราะ subword ของภาษาไทยพองตัว ~1.8-2.5 เท่า)
  ชั้นที่ 3  Merge + breadcrumb: รวมมาตราสั้น ๆ ที่อยู่ในหมวดเดียวกัน และแปะ
                                "ชื่อกฎหมาย > ลักษณะ > หมวด" ไว้หัว chunk ตอน embed
                                เพื่อแก้ปัญหา chunk กำพร้าบริบท (context-less chunk)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import ChunkConfig
from .loader import Document
from .thai_utils import sent_tokenize, normalize_text, count_thai_words

# ---------- regex ของโครงสร้างกฎหมายไทย ----------
# หมายเหตุ: normalize เลขไทย -> อารบิกไปแล้วใน loader
RE_SECTION = re.compile(r"^\s*(มาตรา|ข้อ)\s*([0-9]+(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ))?(?:/[0-9]+)?)")
RE_HEADING = re.compile(r"^\s*(ภาค|ลักษณะ|หมวด|ส่วนที่|บทที่|บรรพ)\s*([0-9]+|[ก-ฮ]+)?\s*(.*)$")
HEADING_LEVEL = {"บรรพ": 0, "ภาค": 1, "ลักษณะ": 2, "หมวด": 3, "ส่วนที่": 4, "บทที่": 4}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str                 # ข้อความจริงที่จะส่งให้ LLM อ่าน
    breadcrumb: str           # ลักษณะ x > หมวด y
    section_no: Optional[str] # "มาตรา 326"
    n_tokens: int
    n_tokens_embed: int = 0   # token ที่โมเดลเห็นจริง = เนื้อหา + header
    part_idx: int = 0         # ถ้ามาตราถูกซอย จะเป็น 0,1,2...
    n_parts: int = 1
    source_path: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        bits = [self.title]
        if self.breadcrumb:
            bits.append(self.breadcrumb)
        if self.section_no:
            bits.append(self.section_no)
        s = " > ".join(bits)
        return f"{s} (ส่วนที่ {self.part_idx + 1}/{self.n_parts})" if self.n_parts > 1 else s

    def embed_text(self, prepend_breadcrumb: bool = True) -> str:
        """ข้อความที่ใช้ 'ตอน embed' ต่างจากข้อความที่ 'ตอนแสดงผล' ได้"""
        if not prepend_breadcrumb:
            return self.text
        header = " > ".join([b for b in [self.title, self.breadcrumb, self.section_no] if b])
        return f"{header}\n{self.text}"

    def to_dict(self):
        return asdict(self)


# ---------- token counter ----------
def build_token_counter(model_name: str) -> Callable[[str], int]:
    """ใช้ tokenizer ของ embedding model จริงในการนับ token

    ทำไมสำคัญ: multilingual-e5-base ใช้ XLM-R SentencePiece
    ข้อความไทย 1 คำ ≈ 1.8-2.5 subword tokens
    ถ้านับเป็น 'คำ' แล้วตั้ง 512 จะได้ chunk ที่ยาวเกิน max_seq_length ของโมเดล
    -> โดน truncate เงียบ ๆ ท้าย chunk หายไปโดยไม่มี error (bug ที่เจอบ่อยมาก)
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))
        return _count
    except Exception as e:
        print(f"[chunker] โหลด tokenizer ไม่ได้ ({e}) -> ใช้ค่าประมาณ 2.2 token/คำ")

        def _approx(text: str) -> int:
            return int(count_thai_words(text) * 2.2)
        return _approx


class ThaiLegalChunker:
    def __init__(self, cfg: ChunkConfig, token_counter: Callable[[str], int]):
        self.cfg = cfg
        self.count = token_counter
        self._dropped = 0

    # ---------- ชั้นที่ 1: ตัดตามโครงสร้าง ----------
    def _split_structural(self, text: str) -> List[Dict]:
        """คืน list ของ {section_no, breadcrumb, text}"""
        lines = text.split("\n")
        headings: Dict[int, str] = {}
        units: List[Dict] = []
        cur: Optional[Dict] = None

        def breadcrumb() -> str:
            return " > ".join(headings[k] for k in sorted(headings) if headings[k])

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if cur:
                    cur["lines"].append("")
                continue

            # ลบบรรทัดที่มีแต่ตัวเลข = เลขหน้าที่ติดมาจากการแปลง PDF
            # (ในกฎหมายไทยไม่มีบรรทัดเนื้อหาที่เป็นตัวเลขล้วนโดด ๆ)
            if self.cfg.strip_page_numbers and re.fullmatch(r"[0-9]{1,4}", stripped):
                continue

            m_head = RE_HEADING.match(stripped)
            m_sec = RE_SECTION.match(stripped)

            # หัวข้อ (หมวด/ลักษณะ) — ไม่ใช่ตัวเนื้อหา แต่เป็นบริบท
            if m_head and not m_sec and len(stripped) < 120:
                kind = m_head.group(1)
                level = HEADING_LEVEL.get(kind, 5)
                headings[level] = stripped
                for lv in list(headings):          # เข้าหมวดใหม่ -> ล้างหัวข้อย่อยเดิม
                    if lv > level:
                        headings.pop(lv)
                continue

            # เริ่มมาตราใหม่ -> ปิดมาตราเดิม
            if m_sec:
                if cur:
                    units.append(cur)
                cur = {
                    "section_no": f"{m_sec.group(1)} {m_sec.group(2)}".strip(),
                    "breadcrumb": breadcrumb(),
                    "lines": [stripped],
                }
            else:
                if cur is None:   # ข้อความก่อนมาตราแรก (คำปรารภ/ชื่อกฎหมาย)
                    cur = {"section_no": None, "breadcrumb": breadcrumb(), "lines": []}
                cur["lines"].append(stripped)

        if cur:
            units.append(cur)

        out = []
        for u in units:
            t = "\n".join(u["lines"]).strip()
            if t:
                out.append({"section_no": u["section_no"],
                            "breadcrumb": u["breadcrumb"], "text": t})
        return out

    # ---------- ชั้นที่ 2: ซอยหน่วยที่ยาวเกิน ----------
    def _split_long_unit(self, text: str, max_t: Optional[int] = None) -> List[str]:
        max_t = max_t if max_t is not None else self.cfg.max_tokens
        ov = min(self.cfg.overlap_tokens, max_t // 4)
        if self.count(text) <= max_t:
            return [text]

        sents = sent_tokenize(text)
        parts: List[str] = []
        buf: List[str] = []
        buf_tokens = 0

        for s in sents:
            st = self.count(s)
            if st > max_t:      # ประโยคเดียวยาวเกิน -> ตัดดิบตามอักขระ (กรณีหายาก)
                if buf:
                    parts.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                ratio = max_t / st
                step = max(1, int(len(s) * ratio))
                for i in range(0, len(s), step):
                    parts.append(s[i:i + step])
                continue

            if buf_tokens + st > max_t:
                parts.append(" ".join(buf))
                # สร้าง overlap: ถอยหลังเก็บประโยคท้าย ๆ จนครบ overlap_tokens
                back, back_tokens = [], 0
                for prev in reversed(buf):
                    pt = self.count(prev)
                    if back_tokens + pt > ov:
                        break
                    back.insert(0, prev)
                    back_tokens += pt
                # 🐛 BUG FIX: ถ้า overlap + ประโยคใหม่ ยังเกินงบอยู่ ต้องยอมทิ้ง overlap
                # ไม่งั้นจะได้ chunk ขนาด (overlap + ประโยคยาว) > max_tokens
                # ซึ่งจะถูก embedding model ตัดท้ายทิ้งเงียบ ๆ โดยไม่มี error
                if back_tokens + st > max_t:
                    back, back_tokens = [], 0
                buf, buf_tokens = back, back_tokens
            buf.append(s)
            buf_tokens += st

        if buf:
            parts.append(" ".join(buf))
        parts = [p.strip() for p in parts if p.strip()]

        # ✅ ตรวจซ้ำด้วยการนับ "ข้อความที่ต่อกันแล้ว" จริง ๆ
        # ทำไมจำเป็น: ตอนซอย เรานับ token ของแต่ละประโยคแยกกันแล้วบวก
        # แต่ SentencePiece ตัด subword ตามบริบท พอเอาประโยคมาต่อกัน
        # ตัวอักษรตรงรอยต่ออาจรวมเป็น subword คนละตัว -> จำนวนจริงคลาดเคลื่อน 2-5 token
        # ผลคือได้ chunk 516 token ทั้งที่ตั้งงบไว้ 512 แล้วโดน truncate ท้ายทิ้งเงียบ ๆ
        verified: List[str] = []
        for p in parts:
            if self.count(p) <= max_t:
                verified.append(p)
                continue
            # ยังเกินอยู่ -> ตัดท้ายทีละประโยคจนพอดีงบ
            words = p.split(" ")
            while len(words) > 1 and self.count(" ".join(words)) > max_t:
                words = words[:-1]
            verified.append(" ".join(words))
        return verified

    # ---------- ชั้นที่ 3: รวมหน่วยสั้น ----------
    @staticmethod
    def _section_label(first: Optional[str], last: Optional[str]) -> Optional[str]:
        """สร้าง citation ช่วงมาตรา เช่น ('ข้อ 17','ข้อ 22') -> 'ข้อ 17-22'"""
        if not first:
            return last
        if not last or last == first:
            return first
        head = first.split()[0]
        return f"{head} {first.split()[-1]}-{last.split()[-1]}"

    def _merge_short(self, units: List[Dict], budget: int) -> List[Dict]:
        merged: List[Dict] = []
        for u in units:
            u = dict(u)
            u.setdefault("section_last", u["section_no"])
            n = self.count(u["text"])
            if (merged and n < self.cfg.min_tokens
                    and merged[-1]["breadcrumb"] == u["breadcrumb"]
                    and self.count(merged[-1]["text"]) + n <= budget):
                merged[-1]["text"] += "\n" + u["text"]
                # เก็บ "ตัวแรก" กับ "ตัวสุดท้าย" ไว้ ไม่ต่อสตริงไปเรื่อย ๆ
                if u["section_no"]:
                    merged[-1]["section_last"] = u["section_no"]
                    if not merged[-1]["section_no"]:
                        merged[-1]["section_no"] = u["section_no"]
            else:
                merged.append(u)
        return merged

    # ---------- entry point ----------
    def chunk_document(self, doc: Document) -> List[Chunk]:
        text = normalize_text(doc.text)
        title = normalize_text(doc.title)
        units = (self._split_structural(text) if self.cfg.respect_law_structure
                 else [{"section_no": None, "breadcrumb": "", "text": text}])

        # งบ token ที่เหลือให้ "เนื้อหา" หลังหักหัว breadcrumb ที่จะแปะตอน embed
        # ชื่อกฎหมายไทยมักยาวมาก (30-80 token) ถ้าไม่หัก เนื้อหาท้าย chunk จะถูกตัดทิ้ง
        def content_budget(u: Dict) -> int:
            if not (self.cfg.reserve_header_tokens and self.cfg.prepend_breadcrumb):
                return self.cfg.max_tokens
            header = " > ".join([b for b in [title, u.get("breadcrumb"),
                                             u.get("section_no")] if b])
            reserve = self.count(header) + 12      # +12 เผื่อ special tokens + คลาดเคลื่อนของ subword
            return max(64, self.cfg.max_tokens - reserve)

        if self.cfg.merge_short_sections and units:
            units = self._merge_short(units, budget=min(content_budget(u) for u in units))

        chunks: List[Chunk] = []
        dropped = 0
        for ui, u in enumerate(units):
            parts = self._split_long_unit(u["text"], max_t=content_budget(u))
            label = self._section_label(u.get("section_no"), u.get("section_last"))
            for pi, ptext in enumerate(parts):
                n_tok = self.count(ptext)
                if n_tok < self.cfg.drop_below_tokens:
                    dropped += 1        # เศษจาก PDF เช่น เลขหน้า/หัวกระดาษเดี่ยว ๆ
                    continue
                c = Chunk(
                    chunk_id=f"{doc.doc_id}::u{ui:04d}::p{pi}",
                    doc_id=doc.doc_id,
                    title=title,
                    text=ptext,
                    breadcrumb=u["breadcrumb"],
                    section_no=label,
                    n_tokens=n_tok,
                    part_idx=pi,
                    n_parts=len(parts),
                    source_path=doc.source_path,
                    meta=dict(doc.meta or {}),
                )
                c.n_tokens_embed = self.count(c.embed_text(self.cfg.prepend_breadcrumb))
                chunks.append(c)
        self._dropped += dropped
        return chunks

    def chunk_corpus(self, docs: List[Document], verbose: bool = True) -> List[Chunk]:
        self._dropped = 0
        all_chunks: List[Chunk] = []
        for d in docs:
            all_chunks.extend(self.chunk_document(d))
        if verbose and all_chunks:
            lens = sorted(c.n_tokens for c in all_chunks)
            emb = [c.n_tokens_embed for c in all_chunks]
            over = sum(1 for x in emb if x > self.cfg.max_tokens)
            print(f"[chunker] {len(docs)} เอกสาร -> {len(all_chunks)} chunks "
                  f"(ทิ้งเศษ {self._dropped} chunk)")
            print(f"[chunker] token เนื้อหา: เฉลี่ย {sum(lens)/len(lens):.0f} | "
                  f"median {lens[len(lens)//2]} | min {lens[0]} | max {lens[-1]}")
            print(f"[chunker] token รวม header (ที่โมเดลเห็นจริง): max {max(emb)} | "
                  f"เกิน {self.cfg.max_tokens}: {over} chunk"
                  + ("  ✅" if over == 0 else "  ⚠️ จะถูก truncate"))
        return all_chunks


def save_chunks(chunks: List[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"[chunker] บันทึก {len(chunks)} chunks -> {path}")


def load_chunks(path: str | Path) -> List[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
