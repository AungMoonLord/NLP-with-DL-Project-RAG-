# """Evaluation harness — 3 ระดับ:
# 1. Retrieval metrics (Recall@k, MRR) → วัดว่า retriever หา chunk ถูกไหม (ใช้ทำ ablation)
# 2. BERTScore (xlm-roberta) → วัด semantic similarity ระหว่างคำตอบกับ ground truth
# 3. LLM-as-Judge → วัด faithfulness (ตอบตรงกับ context) และ relevance (ตอบตรงคำถาม)
# """
# import json
# import re
# from bert_score import score as bertscore
# from openai import OpenAI
# import config

# judge_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


# # ---------- 1. Retrieval metrics ----------
# def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
#     hits = sum(1 for d in relevant_doc_ids if d in retrieved_doc_ids)
#     return hits / len(relevant_doc_ids) if relevant_doc_ids else 0.0


# def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
#     for rank, d in enumerate(retrieved_doc_ids, start=1):
#         if d in relevant_doc_ids:
#             return 1.0 / rank
#     return 0.0


# # ---------- 2. BERTScore ----------
# def compute_bertscore(candidates: list[str], references: list[str]) -> dict:
#     # ภาษาไทยต้องใช้ multilingual model — ค่า default (roberta-large) ใช้ไม่ได้
#     P, R, F1 = bertscore(
#         candidates, references,
#         model_type="xlm-roberta-large", lang="th", rescale_with_baseline=False,
#     )
#     return {
#         "precision": float(P.mean()),
#         "recall": float(R.mean()),
#         "f1": float(F1.mean()),
#     }


# # ---------- 3. LLM-as-Judge ----------
# JUDGE_PROMPT = """คุณเป็นกรรมการประเมินระบบตอบคำถามกฎหมาย ให้คะแนน 1-5 ในสองมิติ:

# **Faithfulness**: คำตอบอิงจากบริบทที่ให้เท่านั้นหรือไม่ (5 = ทุกข้อความมีที่มาจากบริบท, 1 = แต่งขึ้นเอง)
# **Relevance**: คำตอบตรงคำถามหรือไม่ (5 = ตรงประเด็นครบถ้วน, 1 = ไม่เกี่ยวข้อง)

# คำถาม: {query}
# บริบทที่ระบบดึงมา: {context}
# คำตอบของระบบ: {answer}

# ตอบเป็น JSON เท่านั้น: {{"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<สั้นๆ>"}}"""


# def llm_judge(query: str, context: str, answer: str) -> dict:
#     resp = judge_client.chat.completions.create(
#         model=config.JUDGE_MODEL,
#         temperature=0,
#         messages=[{"role": "user", "content": JUDGE_PROMPT.format(
#             query=query, context=context[:6000], answer=answer)}],
#     )
#     text = resp.choices[0].message.content
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     try:
#         return json.loads(match.group())
#     except (AttributeError, json.JSONDecodeError):
#         return {"faithfulness": None, "relevance": None, "reason": "parse_error"}


"""Evaluation Harness for Thai Legal RAG Pipeline
ประกอบด้วย:
1. BERTScore (xlm-roberta-large) สำหรับวัด Semantic Similarity
2. LLM-as-Judge แบบเข้มงวด (Claim-based Verification + Abstention Handling)
"""
from __future__ import annotations

import json
import re
from bert_score import score as bertscore
from openai import OpenAI
import config

# กำหนด Client เชื่อมต่อ Endpoint ของคลาส
judge_client = OpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY
)

# =======================================================
# 1. BERTScore (Mandatory Metric 1)
# =======================================================
def compute_bertscore(candidates: list[str], references: list[str]) -> dict:
    """คำนวณ BERTScore ด้วย xlm-roberta-large สำหรับภาษาไทย"""
    P, R, F1 = bertscore(
        candidates,
        references,
        model_type="xlm-roberta-large",
        lang="th",
        rescale_with_baseline=False,
    )
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


# =======================================================
# 2. LLM-as-Judge (Mandatory Metric 2 - เข้มงวดพิเศษ)
# =======================================================
JUDGE_PROMPT = """คุณเป็นกรรมการตรวจสอบระบบตอบคำถามกฎหมายไทย ที่เข้มงวดมาก

ขั้นตอนบังคับ:
1. แตกคำตอบออกเป็น "ข้อกล่าวอ้าง (claim)" ทีละประโยค
2. ตรวจทีละ claim ว่ามีข้อความรองรับในบริบทหรือไม่
3. รวบรวม claim ที่ "ไม่มีหลักฐานในบริบท" ไว้ใน unsupported_claims

เกณฑ์ Faithfulness (เข้มงวด — ห้ามให้ 5 ถ้าไม่เข้าเงื่อนไขเป๊ะ):
5 = ทุก claim มีหลักฐานตรงตัวในบริบท และเลขมาตรา/ชื่อกฎหมายถูกต้องครบ
4 = claim หลักถูกทั้งหมด แต่มีการสรุปความเกินบริบทเล็กน้อย (ไม่ใช่ข้อเท็จจริงใหม่)
3 = มี 1 claim ที่ไม่มีหลักฐานรองรับ
2 = มีหลายจุดไม่มีหลักฐาน หรืออ้างเลขมาตราผิด/ไม่มีในบริบท
1 = เนื้อหาส่วนใหญ่แต่งขึ้นเอง

เกณฑ์ Relevance:
5 = ตอบตรงประเด็น ครบทุกส่วนของคำถาม
4 = ตอบตรงประเด็นครบถ้วน แต่มีส่วนขยายที่ไม่จำเป็น
3 = ตอบได้เพียงบางส่วนของคำถาม
2 = แตะประเด็นคำถามแต่ไม่ตอบสาระสำคัญ
1 = ไม่เกี่ยวข้องกับคำถาม

กรณีพิเศษ: ถ้าคำตอบคือการปฏิเสธ (เช่น "ไม่พบข้อมูลในเอกสารที่มี")
- ถ้าบริบท "ไม่มี" ข้อมูลคำตอบจริง -> faithfulness = 5, relevance = 5 (ปฏิเสธถูกต้อง)
- ถ้าบริบท "มี" ข้อมูลคำตอบอยู่แล้ว แต่ระบบหาไม่เจอ -> faithfulness = 5, relevance = 1 (ปฏิเสธผิดพลาด)

คำถาม: {query}
---บริบท---
{context}
---คำตอบของระบบ---
{answer}

ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบายอื่นนอกเหนือจาก JSON:
{{"faithfulness": <1-5>, "relevance": <1-5>, "unsupported_claims": ["..."], "is_abstention": <true|false>, "reason": "<สั้นๆ>"}}"""


def _parse_json(txt: str) -> dict:
    """ทำความสะอาดข้อความและตัด markdown backticks เพื่อแปลงเป็น JSON"""
    t = str(txt).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _clamp(v, lo=1, hi=5):
    """จำกัดคะแนนให้อยู่ในช่วง 1 - 5"""
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def llm_judge(query: str, context: str, answer: str) -> dict:
    """ประเมินคุณภาพคำตอบด้วย LLM-as-Judge พร้อมคืนค่าความสมบูรณ์และเหตุผล"""
    model_name = getattr(config, "JUDGE_MODEL", "gemma-4-E4B-it")
    temp = getattr(config, "JUDGE_TEMPERATURE", 0.0)

    try:
        resp = judge_client.chat.completions.create(
            model=model_name,
            temperature=temp,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    query=query,
                    context=context[:6000],  # ป้องกัน Context Window ล้น
                    answer=answer
                )
            }],
        )
        raw = _parse_json(resp.choices[0].message.content)
    except Exception as e:
        return {
            "faithfulness": None,
            "relevance": None,
            "unsupported_claims": [],
            "is_abstention": None,
            "reason": f"judge_error: {e}",
            "ok": False
        }

    return {
        "faithfulness": _clamp(raw.get("faithfulness")),
        "relevance": _clamp(raw.get("relevance")),
        "unsupported_claims": raw.get("unsupported_claims") or [],
        "is_abstention": bool(raw.get("is_abstention", False)),
        "reason": str(raw.get("reason", ""))[:300],
        "ok": True,
    }

