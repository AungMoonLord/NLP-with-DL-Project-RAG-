"""โหลดชุดทดสอบ รองรับทั้ง template ของทีม (JSON array) และรูปแบบ JSONL

ทำไมต้องมีไฟล์นี้ — และทำไมมันสำคัญกว่าที่คิด:

  ถ้าวัดผลโดยเทียบแค่ "ชื่อเอกสาร" (source_doc) ระบบจะดึง chunk 5 อัน
  จากกฎหมายฉบับเดียวกันหมด -> ชื่อเอกสารตรงทุกอัน -> Recall เต็มตลอด
  -> reranker สลับอันดับ chunk ยังไงก็ไม่มีผลต่อตัวเลข
  -> ablation จะได้ค่าเท่ากันทุก variant และสรุปอะไรไม่ได้เลย

  ทางแก้: template ของทีมมี field "section" ระบุมาตราไว้ เช่น "มาตรา 8 และมาตรา 15"
  ไฟล์นี้จะแปลงมันเป็น chunk_id จริงในคลัง ทำให้วัดที่ระดับ chunk ได้
  ซึ่งเป็นระดับเดียวที่มองเห็นผลของ reranker

รูปแบบที่รองรับ:
  {"question":..., "ground_truth":..., "source_doc":..., "section":"มาตรา 8 และมาตรา 15"}
  {"question":..., "reference_answer":..., "gold_chunk_ids":[...], "difficulty":...}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.chunker import Chunk
from src.thai_utils import thai_digits_to_arabic, normalize_text

RE_SECTION_NUM = re.compile(r"(?:มาตรา|ข้อ)\s*([0-9]+(?:/[0-9]+)?)")
RE_RANGE = re.compile(r"([0-9]+)\s*-\s*([0-9]+)")


def _norm_key(s: str) -> str:
    """ทำให้ชื่อเอกสารเทียบกันได้ ตัดช่องว่าง/เลขไทย/นามสกุลไฟล์ทิ้ง"""
    s = thai_digits_to_arabic(normalize_text(s or ""))
    s = re.sub(r"\.(pdf|txt|md)$", "", s, flags=re.I)
    return re.sub(r"\s+", "", s)


def parse_sections(text: str) -> List[str]:
    """'มาตรา 8 และมาตรา 15' -> ['8','15']"""
    return RE_SECTION_NUM.findall(thai_digits_to_arabic(text or ""))


def _chunk_covers(chunk: Chunk, num: str) -> bool:
    """chunk นี้ครอบคลุมมาตราเลขนี้หรือไม่

    section_no อาจเป็น 'มาตรา 8' หรือช่วง 'มาตรา 8-12' (เกิดจากการ merge มาตราสั้น)
    """
    sec = thai_digits_to_arabic(chunk.section_no or "")
    if not sec:
        return False
    m = RE_RANGE.search(sec)
    if m:
        try:
            lo, hi = int(m.group(1)), int(m.group(2))
            return lo <= int(num.split("/")[0]) <= hi
        except ValueError:
            return False
    nums = RE_SECTION_NUM.findall(sec) or re.findall(r"[0-9]+(?:/[0-9]+)?", sec)
    return num in nums


def resolve_gold_chunks(item: Dict, chunks: List[Chunk],
                        by_doc: Dict[str, List[Chunk]]) -> List[str]:
    """หา chunk_id ที่ตรงกับ source_doc + section ของข้อนี้"""
    doc_key = _norm_key(item.get("source_doc", ""))
    if not doc_key:
        return []

    # จับคู่เอกสาร: ตรงเป๊ะก่อน ถ้าไม่เจอค่อยดูว่าเป็นส่วนหนึ่งของกัน
    cands = by_doc.get(doc_key)
    if cands is None:
        for k, v in by_doc.items():
            if doc_key in k or k in doc_key:
                cands = v
                break
    if not cands:
        return []

    nums = parse_sections(item.get("section", ""))
    if not nums:
        return []       # ไม่ระบุมาตรา -> ปล่อยว่าง ดีกว่าเดา

    hits = [c.chunk_id for c in cands if any(_chunk_covers(c, n) for n in nums)]
    return hits


def load_gold(path: str | Path, chunks: Optional[List[Chunk]] = None,
              verbose: bool = True) -> List[Dict]:
    """คืนรายการข้อสอบในรูปแบบมาตรฐานของ pipeline"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")

    if path.suffix.lower() == ".jsonl":
        items = [json.loads(l) for l in raw.splitlines() if l.strip()]
    else:
        data = json.loads(raw)
        items = data if isinstance(data, list) else data.get("items", [])

    by_doc: Dict[str, List[Chunk]] = {}
    if chunks:
        for c in chunks:
            by_doc.setdefault(_norm_key(c.doc_id), []).append(c)

    out, resolved, unresolved = [], 0, []
    for i, it in enumerate(items, 1):
        gold_ids = it.get("gold_chunk_ids") or []
        if not gold_ids and chunks and it.get("source_doc"):
            gold_ids = resolve_gold_chunks(it, chunks, by_doc)

        ref = (it.get("reference_answer") or it.get("ground_truth") or "").strip()
        difficulty = it.get("difficulty")
        if not difficulty:
            # เดาจากเนื้อหา: ระบุหลายมาตรา = multi-hop, เฉลยบอกว่าไม่พบ = unanswerable
            n_sec = len(parse_sections(it.get("section", "")))
            difficulty = ("unanswerable" if "ไม่พบข้อมูล" in ref
                          else "multi-hop" if n_sec >= 2 else "single-hop")

        if gold_ids:
            resolved += 1
        elif difficulty != "unanswerable":
            unresolved.append((it.get("id", i), it.get("source_doc", "")[:40],
                               it.get("section", "")))

        out.append({
            "id": it.get("id", i),
            "question": it["question"],
            "reference_answer": ref,
            "gold_chunk_ids": gold_ids,
            "gold_doc": it.get("source_doc", ""),
            "section": it.get("section", ""),
            "difficulty": difficulty,
        })

    if verbose:
        n_un = sum(1 for x in out if x["difficulty"] == "unanswerable")
        print(f"[gold] โหลด {len(out)} ข้อจาก {path.name}")
        if chunks:
            print(f"[gold] จับคู่มาตรา -> chunk สำเร็จ {resolved}/{len(out) - n_un} ข้อ"
                  f"  (ข้อ unanswerable {n_un} ข้อไม่ต้องจับคู่)")
        if unresolved:
            print(f"[gold] ⚠️ จับคู่ไม่ได้ {len(unresolved)} ข้อ "
                  f"(จะถูกข้ามตอนวัด retrieval metrics):")
            for _id, doc, sec in unresolved[:8]:
                print(f"        ข้อ {_id}: '{doc}' {sec}")
            if len(unresolved) > 8:
                print(f"        ... และอีก {len(unresolved) - 8} ข้อ")
        if n_un == 0:
            print("[gold] ⚠️ ไม่มีข้อ unanswerable เลย -> วัด hallucination ไม่ได้")
    return out
