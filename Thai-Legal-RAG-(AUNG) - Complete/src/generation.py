# from openai import OpenAI
# import config

# client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

# SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตอบคำถามด้านกฎหมายไทย จงตอบโดยอ้างอิงจาก "บริบท" ที่ให้เท่านั้น
# กติกา:
# 1. อ้างอิงเลขมาตราทุกครั้งที่เป็นไปได้
# 2. ถ้าบริบทไม่มีข้อมูลเพียงพอ ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่มี" ห้ามเดา
# 3. ตอบเป็นภาษาไทย กระชับ ตรงประเด็น"""


# def generate_answer(query: str, contexts: list[dict]) -> str:
#     context_block = "\n\n---\n\n".join(
#         f"[แหล่งที่มา: {c['doc_id']}]\n{c['text']}" for c in contexts
#     )
#     resp = client.chat.completions.create(
#         model=config.LLM_MODEL,
#         temperature=0.1,   # งานกฎหมายต้องการความ deterministic สูง
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": f"บริบท:\n{context_block}\n\nคำถาม: {query}"},
#         ],
#     )
#     return resp.choices[0].message.content

"""Generation Module with Abstention Handling for Thai Legal Q&A
ระบบสร้างคำตอบทางกฎหมาย พร้อมกลไกปฏิเสธการตอบ (Abstain) เพื่อป้องกัน Hallucination
"""
from __future__ import annotations

from openai import OpenAI
import config

# ใช้ Endpoint และ Key ของคลาสตาม config
client = OpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY
)

# ดึงข้อความปฏิเสธมาตรฐานจาก config
ABSTAIN_TEXT = getattr(config, "ABSTAIN_TEXT", "ไม่พบข้อมูลในเอกสารที่มี")

SYSTEM_PROMPT = f"""คุณเป็นผู้ช่วยตอบคำถามกฎหมายไทย ตอบโดยอ้างอิง "บริบทที่ให้มาเท่านั้น"

กฎเหล็ก:
1. ห้ามใช้ความรู้นอกบริบท ห้ามเดา ห้ามแต่งเลขมาตราหรือชื่อพระราชบัญญัติ
2. ถ้าบริบทไม่มีข้อมูลเพียงพอ ให้ตอบเพียงว่า "{ABSTAIN_TEXT}" แล้วหยุดทันที
3. ถ้าตอบได้บางส่วน ให้ตอบเฉพาะส่วนที่มีหลักฐาน แล้วระบุชัดว่าส่วนใดไม่พบข้อมูล
4. อ้างอิงเลขมาตราและชื่อเอกสารที่ปรากฏในบริบทเสมอ
5. ตอบเป็นภาษาไทย กระชับ ตรงประเด็น
"""

ABSTAIN_MARKERS = (
    "ไม่พบข้อมูล",
    "ไม่มีข้อมูล",
    "ไม่สามารถตอบ",
    "ไม่ปรากฏใน",
    "บริบทไม่ได้ระบุ",
    "ไม่ปรากฏข้อมูล"
)


def is_abstention(answer: str) -> bool:
    """ตรวจสอบว่าคำตอบเป็นการปฏิเสธ (Abstain) หรือไม่ ตรวจจับเฉพาะช่วง 120 ตัวอักษรแรก"""
    a = str(answer or "").strip()
    if not a:
        return True
    head = a[:120]
    return any(m in head for m in ABSTAIN_MARKERS)


def build_context(contexts: list[dict]) -> str:
    """จัดรูปแบบ Context Block ให้มีป้ายกำกับชัดเจน เพื่อช่วยให้ LLM สังเคราะห์คำตอบได้แม่นยำ"""
    parts = []
    for i, c in enumerate(contexts, 1):
        doc = c.get("doc_id") or c.get("file") or "เอกสารอ้างอิง"
        text = c.get("text", "").strip()
        parts.append(f"[บริบท {i} | แหล่งที่มา: {doc}]\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, contexts: list[dict]) -> str:
    """สร้างคำตอบจากบริบทที่ได้รับ หากไม่มีบริบทจะ Abstain ทันทีโดยไม่ยิง API"""
    if not contexts:
        return ABSTAIN_TEXT

    model_name = getattr(config, "LLM_MODEL", getattr(config, "GEN_MODEL", "gemma-4-E4B-it"))

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.0,  # ใช้ 0.0 เพื่อความแม่นยำสูงสุด ลดการสุ่มแต่งคำตอบ
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"บริบท:\n{build_context(contexts)}\n\nคำถาม: {query}"},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def generate_answer_stream(query: str, contexts: list[dict]):
    """สร้างคำตอบแบบ Streaming คืนค่าทีละ Token (Yield Chunks)"""
    if not contexts:
        yield ABSTAIN_TEXT
        return

    model_name = getattr(config, "LLM_MODEL", getattr(config, "GEN_MODEL", "gemma-4-E4B-it"))

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        stream=True,  # ★ เปิด Streaming จาก OpenAI Endpoint
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"บริบท:\n{build_context(contexts)}\n\nคำถาม: {query}"},
        ],
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content