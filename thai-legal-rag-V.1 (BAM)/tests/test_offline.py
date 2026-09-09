"""ทดสอบส่วนที่รันได้โดยไม่ต้องโหลดโมเดล (ใช้ตรวจ logic ก่อน ingest จริง)

    python tests/test_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import ThaiLegalChunker
from src.config import ChunkConfig, Config
from src.loader import Document
from src.retriever import reciprocal_rank_fusion
from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k, rouge_l, _lcs_length

SAMPLE = """ประมวลกฎหมายแพ่งและพาณิชย์

บรรพ 2 หนี้
ลักษณะ 1 บทเบ็ดเสร็จทั่วไป
หมวด 2 ผลแห่งหนี้

มาตรา ๒๐๓ ถ้าเวลาอันจะพึงชำระหนี้นั้นมิได้กำหนดลงไว้ หรือจะอนุมานจากพฤติการณ์
ทั้งปวงก็ไม่ได้ไซร้ ท่านว่าเจ้าหนี้ย่อมจะเรียกให้ชำระหนี้ได้โดยพลัน และฝ่ายลูกหนี้
ก็ย่อมจะชำระหนี้ของตนได้โดยพลันดุจกัน

มาตรา ๒๐๔ ถ้าหนี้ถึงกำหนดชำระแล้ว และภายหลังแต่นั้นเจ้าหนี้ได้ให้คำเตือนลูกหนี้แล้ว
ลูกหนี้ยังไม่ชำระหนี้ไซร้ ลูกหนี้ได้ชื่อว่าผิดนัดเพราะเขาเตือนแล้ว

มาตรา ๒๐๕ ตราบใดการชำระหนี้ยังมิได้กระทำลงเพราะพฤติการณ์อันใดอันหนึ่งซึ่งลูกหนี้
ไม่ต้องรับผิดชอบ ตราบนั้นลูกหนี้ยังหาได้ชื่อว่าผิดนัดไม่

หมวด 3 ลูกหนี้และเจ้าหนี้หลายคน

มาตรา ๒๙๐ ถ้าการชำระหนี้เป็นการอันจะแบ่งกันชำระได้ และมีบุคคลหลายคนเป็นลูกหนี้
ก็ดี มีบุคคลหลายคนเป็นเจ้าหนี้ก็ดี เมื่อกรณีเป็นที่สงสัย ท่านว่าลูกหนี้แต่ละคน
จะต้องรับผิดเพียงเป็นส่วนเท่า ๆ กัน และเจ้าหนี้แต่ละคนก็ชอบที่จะได้รับแต่เพียง
เป็นส่วนเท่า ๆ กัน
"""


def approx_counter(text: str) -> int:
    """token counter ปลอมสำหรับทดสอบ: ~1 token ต่อ 2 ตัวอักษรไทย"""
    return max(1, len(text) // 2)


def test_chunker():
    print("\n=== TEST: chunker ===")
    doc = Document(doc_id="ccc", title="ประมวลกฎหมายแพ่งและพาณิชย์",
                   text=SAMPLE, source_path="ccc.txt", meta={})
    cfg = ChunkConfig(max_tokens=200, min_tokens=40, overlap_tokens=20)
    chunks = ThaiLegalChunker(cfg, approx_counter).chunk_document(doc)
    for c in chunks:
        print(f"  {c.chunk_id} | {c.n_tokens:>4} tok | {c.citation}")
        print(f"      {c.text[:70].replace(chr(10),' ')}...")
    assert len(chunks) >= 3, "ควรตัดได้อย่างน้อย 3 chunk"
    assert any(c.section_no and "204" in c.section_no for c in chunks), "ควรจับมาตรา 204 ได้"
    assert any("หมวด" in c.breadcrumb for c in chunks), "ควรมี breadcrumb หมวด"
    # เลขไทยต้องถูกแปลงเป็นอารบิกแล้ว
    assert all("๒" not in (c.section_no or "") for c in chunks), "เลขไทยควรถูก normalize"
    print("  ✅ ผ่าน")


def test_rrf():
    print("\n=== TEST: RRF ===")
    dense = ["c1", "c2", "c3", "c4"]
    bm25 = ["c9", "c3", "c1", "c7"]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)
    for cid, s, ranks in fused[:4]:
        print(f"  {cid}: {s:.5f}  {ranks}")
    top = [c for c, _, _ in fused]
    # c1: 1/61 + 1/63 ; c3: 1/63 + 1/62 ; c2: 1/62 เท่านั้น
    assert top[0] == "c1" and top[1] == "c3", f"ลำดับผิด: {top[:3]}"
    assert top.index("c3") < top.index("c2"), "เอกสารที่ติดทั้ง 2 ระบบต้องมาก่อน"
    print("  ✅ ผ่าน — เอกสารที่ปรากฏในทั้งสองระบบถูกดันขึ้นอันดับต้น")


def test_retrieval_metrics():
    print("\n=== TEST: retrieval metrics ===")
    retrieved = ["a", "b", "c", "d", "e"]
    gold = ["c", "z"]
    r5 = recall_at_k(retrieved, gold, 5)
    m5 = mrr_at_k(retrieved, gold, 5)
    n5 = ndcg_at_k(retrieved, gold, 5)
    print(f"  recall@5={r5:.3f} mrr@5={m5:.3f} ndcg@5={n5:.3f}")
    assert abs(r5 - 0.5) < 1e-9
    assert abs(m5 - 1 / 3) < 1e-9
    assert 0 < n5 < 1
    assert recall_at_k(retrieved, gold, 2) == 0.0
    print("  ✅ ผ่าน")


def test_rouge_thai():
    print("\n=== TEST: ROUGE-L ภาษาไทย ===")
    pred = "ลูกหนี้ผิดนัดเมื่อเจ้าหนี้เตือนแล้วไม่ชำระหนี้"
    ref = "เมื่อเจ้าหนี้เตือนแล้วลูกหนี้ยังไม่ชำระหนี้ ถือว่าลูกหนี้ผิดนัด"
    s = rouge_l(pred, ref)
    print(f"  {s}")
    assert 0 < s["rouge_l_f"] < 1, "ควรได้ค่ากลาง ๆ ไม่ใช่ 0 หรือ 1"
    # แสดงให้เห็นว่าถ้าใช้ whitespace split จะพัง
    naive = _lcs_length(pred.split(), ref.split())
    print(f"  เทียบ: ตัดคำด้วย whitespace จะได้ LCS = {naive} (จาก {len(pred.split())} token)")
    print("  ✅ ผ่าน")


if __name__ == "__main__":
    test_chunker()
    test_rrf()
    test_retrieval_metrics()
    test_rouge_thai()
    print("\n🎉 ผ่านทุกการทดสอบ (ส่วนที่ไม่ต้องใช้โมเดล)")
