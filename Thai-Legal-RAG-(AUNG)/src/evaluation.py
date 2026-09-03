"""Evaluation harness — 3 ระดับ:
1. Retrieval metrics (Recall@k, MRR) → วัดว่า retriever หา chunk ถูกไหม (ใช้ทำ ablation)
2. BERTScore (xlm-roberta) → วัด semantic similarity ระหว่างคำตอบกับ ground truth
3. LLM-as-Judge → วัด faithfulness (ตอบตรงกับ context) และ relevance (ตอบตรงคำถาม)
"""
import json
import re
from bert_score import score as bertscore
from openai import OpenAI
import config

judge_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


# ---------- 1. Retrieval metrics ----------
def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    hits = sum(1 for d in relevant_doc_ids if d in retrieved_doc_ids)
    return hits / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    for rank, d in enumerate(retrieved_doc_ids, start=1):
        if d in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


# ---------- 2. BERTScore ----------
def compute_bertscore(candidates: list[str], references: list[str]) -> dict:
    # ภาษาไทยต้องใช้ multilingual model — ค่า default (roberta-large) ใช้ไม่ได้
    P, R, F1 = bertscore(
        candidates, references,
        model_type="xlm-roberta-large", lang="th", rescale_with_baseline=False,
    )
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


# ---------- 3. LLM-as-Judge ----------
JUDGE_PROMPT = """คุณเป็นกรรมการประเมินระบบตอบคำถามกฎหมาย ให้คะแนน 1-5 ในสองมิติ:

**Faithfulness**: คำตอบอิงจากบริบทที่ให้เท่านั้นหรือไม่ (5 = ทุกข้อความมีที่มาจากบริบท, 1 = แต่งขึ้นเอง)
**Relevance**: คำตอบตรงคำถามหรือไม่ (5 = ตรงประเด็นครบถ้วน, 1 = ไม่เกี่ยวข้อง)

คำถาม: {query}
บริบทที่ระบบดึงมา: {context}
คำตอบของระบบ: {answer}

ตอบเป็น JSON เท่านั้น: {{"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<สั้นๆ>"}}"""


def llm_judge(query: str, context: str, answer: str) -> dict:
    resp = judge_client.chat.completions.create(
        model=config.JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            query=query, context=context[:6000], answer=answer)}],
    )
    text = resp.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group())
    except (AttributeError, json.JSONDecodeError):
        return {"faithfulness": None, "relevance": None, "reason": "parse_error"}
