from openai import OpenAI
import config

client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตอบคำถามด้านกฎหมายไทย จงตอบโดยอ้างอิงจาก "บริบท" ที่ให้เท่านั้น
กติกา:
1. อ้างอิงเลขมาตราทุกครั้งที่เป็นไปได้
2. ถ้าบริบทไม่มีข้อมูลเพียงพอ ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่มี" ห้ามเดา
3. ตอบเป็นภาษาไทย กระชับ ตรงประเด็น"""


def generate_answer(query: str, contexts: list[dict]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[แหล่งที่มา: {c['doc_id']}]\n{c['text']}" for c in contexts
    )
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0.1,   # งานกฎหมายต้องการความ deterministic สูง
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"บริบท:\n{context_block}\n\nคำถาม: {query}"},
        ],
    )
    return resp.choices[0].message.content
