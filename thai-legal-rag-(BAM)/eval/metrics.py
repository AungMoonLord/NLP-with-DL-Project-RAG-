"""STEP 8 — Evaluation harness

แบ่งเมตริกเป็น 2 กลุ่ม เพราะมันวัดคนละอย่าง (ประเด็นนี้มักถูกถามในการ defend):

A) Retrieval metrics — วัด "ขั้นค้นคืน" อย่างเดียว ไม่เกี่ยวกับ LLM
   Recall@k, MRR@k, nDCG@k  ->  ใช้ตัดสิน ablation ได้ตรงที่สุด เพราะไม่มี noise จาก generator

B) Generation metrics — วัด "คำตอบสุดท้าย"
   1. BERTScore    : ความคล้ายเชิงความหมายกับเฉลย (ใช้ contextual embedding, ไม่ใช่ n-gram)
   2. ROUGE-L      : ความซ้อนทับเชิงลำดับคำ (LCS) — สำคัญมากที่ต้องตัดคำไทยก่อน
   3. LLM-as-Judge : ความถูกต้องเชิงความหมายที่เมตริกอัตโนมัติจับไม่ได้
   4. RAGAS Faithfulness     : คำตอบยึดโยงกับ context แค่ไหน (วัด hallucination)
   5. RAGAS Answer Relevance : คำตอบตรงคำถามแค่ไหน (ไม่ต้องใช้เฉลย)
"""
from __future__ import annotations

import json
import math
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.thai_utils import word_tokenize, normalize_text


# ==========================================================================
# A) RETRIEVAL METRICS
# ==========================================================================
def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """สัดส่วนของ chunk เฉลยที่ถูกดึงมาได้ใน top-k"""
    if not gold_ids:
        return float("nan")
    top = set(retrieved_ids[:k])
    return len(top & set(gold_ids)) / len(set(gold_ids))


def hit_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """ส่วนกลับของอันดับแรกที่เจอเอกสารถูก — ลงโทษระบบที่เอาของถูกไว้อันดับท้าย"""
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """binary relevance nDCG — ให้เครดิตแบบลดหลั่นตามตำแหน่ง (log2 discount)"""
    gold = set(gold_ids)
    dcg = sum(1.0 / math.log2(i + 1)
              for i, cid in enumerate(retrieved_ids[:k], start=1) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else float("nan")


def evaluate_retrieval(results: List[Dict], ks: List[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """results: [{"retrieved_ids": [...], "gold_chunk_ids": [...]}, ...]"""
    scored = [r for r in results if r.get("gold_chunk_ids")]
    if not scored:
        return {}
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(r["retrieved_ids"], r["gold_chunk_ids"], k) for r in scored]))
        out[f"hit@{k}"] = float(np.mean([hit_at_k(r["retrieved_ids"], r["gold_chunk_ids"], k) for r in scored]))
        out[f"mrr@{k}"] = float(np.mean([mrr_at_k(r["retrieved_ids"], r["gold_chunk_ids"], k) for r in scored]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(r["retrieved_ids"], r["gold_chunk_ids"], k) for r in scored]))
    out["n_eval"] = len(scored)
    return out


# ==========================================================================
# B1) ROUGE-L (ตัดคำไทยเอง)
# ==========================================================================
def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> Dict[str, float]:
    """ROUGE-L สำหรับภาษาไทย

    ⚠️ จุดที่พังบ่อย: ไลบรารี rouge_score ตัดคำด้วย whitespace
    ภาษาไทยไม่มีเว้นวรรค -> ทั้งประโยคกลายเป็น 1 token -> คะแนนเป็น 0 หรือ 1 เท่านั้น
    เราจึงตัดคำด้วย pythainlp (newmm) ก่อนเสมอ
    """
    p = word_tokenize(normalize_text(prediction))
    r = word_tokenize(normalize_text(reference))
    if not p or not r:
        return {"rouge_l_p": 0.0, "rouge_l_r": 0.0, "rouge_l_f": 0.0}
    lcs = _lcs_length(p, r)
    prec, rec = lcs / len(p), lcs / len(r)
    if prec + rec == 0:
        f = 0.0
    else:
        f = ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec)
    return {"rouge_l_p": prec, "rouge_l_r": rec, "rouge_l_f": f}


# ==========================================================================
# B2) BERTScore
# ==========================================================================
def bertscore(predictions: List[str], references: List[str],
              model_type: str = "xlm-roberta-large", lang: str = "th",
              rescale: bool = False) -> Dict[str, float]:
    """BERTScore = greedy matching ระหว่าง contextual embedding ของแต่ละ token

    เลือก xlm-roberta-large เพราะ:
      - เป็น multilingual ที่เห็นภาษาไทยตอน pretrain (bert-base-multilingual ก็ได้ แต่เบากว่า/แม่นน้อยกว่า)
      - BERTScore ใช้ tokenizer ของโมเดลเอง จึงไม่ติดปัญหาการเว้นวรรคแบบ ROUGE
    ข้อควรระวัง: ค่าดิบของ BERTScore ภาษาไทยมักสูง (~0.75+) แม้คำตอบผิด
      -> ต้องดูเป็น "ค่าเปรียบเทียบระหว่าง variant" ไม่ใช่ค่าสัมบูรณ์
    """
    from bert_score import score as _bs
    P, R, F = _bs(predictions, references, model_type=model_type, lang=lang,
                  rescale_with_baseline=rescale, verbose=False)
    return {"bertscore_p": float(P.mean()), "bertscore_r": float(R.mean()),
            "bertscore_f1": float(F.mean()),
            "bertscore_f1_per_item": [float(x) for x in F]}


# ==========================================================================
# B3-B5) LLM-based metrics
# ==========================================================================
def _parse_json(text: str, fallback):
    """LLM ชอบห่อ JSON ด้วย ```json ... ``` -> ต้องลอกออกก่อน"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return fallback


class LLMEvaluator:
    """รวมเมตริกที่ต้องใช้ LLM: LLM-as-Judge, RAGAS Faithfulness, Answer Relevance"""

    def __init__(self, generator, embedder=None):
        self.gen = generator
        self.embedder = embedder

    # ---- LLM-as-Judge ----
    def judge(self, question: str, prediction: str, reference: str) -> Dict:
        from src.prompts import LLM_JUDGE_PROMPT
        raw = self.gen.generate(
            "คุณเป็นผู้ตรวจข้อสอบที่เข้มงวดและเป็นกลาง ตอบเป็น JSON เท่านั้น",
            LLM_JUDGE_PROMPT.format(question=question, prediction=prediction,
                                    reference=reference))
        data = _parse_json(raw, {"score": None, "reason": "parse_failed"})
        return {"llm_judge_score": data.get("score"), "llm_judge_reason": data.get("reason")}

    # ---- RAGAS Faithfulness ----
    def faithfulness(self, answer: str, contexts: List[str]) -> Dict:
        """faithfulness = (จำนวน claim ที่อนุมานได้จาก context) / (จำนวน claim ทั้งหมด)

        วัด hallucination โดยตรง และ *ไม่ต้องใช้เฉลย* -> ใช้กับคำถามจริงในระบบ production ได้
        """
        from src.prompts import FAITHFULNESS_EXTRACT_PROMPT, FAITHFULNESS_VERDICT_PROMPT
        claims = _parse_json(
            self.gen.generate("ตอบเป็น JSON array เท่านั้น",
                              FAITHFULNESS_EXTRACT_PROMPT.format(answer=answer)), [])
        if not claims:
            return {"faithfulness": float("nan"), "n_claims": 0}
        ctx = "\n\n".join(contexts)
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        verdicts = _parse_json(
            self.gen.generate("ตอบเป็น JSON array ของ 0/1 เท่านั้น",
                              FAITHFULNESS_VERDICT_PROMPT.format(context=ctx, claims=numbered)), [])
        verdicts = [v for v in verdicts if isinstance(v, (int, float))][:len(claims)]
        if not verdicts:
            return {"faithfulness": float("nan"), "n_claims": len(claims)}
        return {"faithfulness": float(np.mean(verdicts)), "n_claims": len(claims),
                "n_supported": int(sum(verdicts))}

    # ---- RAGAS Answer Relevance ----
    def answer_relevance(self, question: str, answer: str, n: int = 3) -> Dict:
        """ให้ LLM 'ย้อนสร้างคำถาม' จากคำตอบ n ข้อ แล้ววัด cosine กับคำถามจริง

        ตรรกะ: ถ้าคำตอบตรงประเด็น เราควรเดาคำถามเดิมกลับมาได้ใกล้เคียง
        คำตอบที่วกวน/ตอบไม่ตรงคำถาม จะได้คำถามย้อนกลับที่ห่างจากคำถามเดิม
        """
        from src.prompts import ANSWER_RELEVANCE_PROMPT
        gen_qs = _parse_json(
            self.gen.generate("ตอบเป็น JSON array ของ string เท่านั้น",
                              ANSWER_RELEVANCE_PROMPT.format(answer=answer, n=n)), [])
        gen_qs = [q for q in gen_qs if isinstance(q, str)]
        if not gen_qs or self.embedder is None:
            return {"answer_relevance": float("nan")}
        qv = self.embedder.encode_queries([question])[0]
        gv = self.embedder.encode_queries(gen_qs)
        sims = gv @ qv     # normalize แล้ว -> dot = cosine
        return {"answer_relevance": float(np.mean(sims)),
                "generated_questions": gen_qs}


# ==========================================================================
def aggregate(rows: List[Dict], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))
                and not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_n"] = len(vals)
    return out
