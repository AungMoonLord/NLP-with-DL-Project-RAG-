"""Evaluation Harness for Thai Legal RAG Pipeline

ประกอบด้วย 5 ส่วนหลัก:

1. BERTScore
   - Semantic similarity ระหว่างคำตอบของระบบกับ reference answer

2. LLM-as-Judge
   - Faithfulness
   - Relevance
   - Claim-based verification
   - Unsupported claims
   - Abstention handling

3. run_pipeline_on_testset()
   - รัน Full RAG Pipeline บน testset
   - เก็บข้อมูลสำหรับวิเคราะห์ภายหลัง

4. summarize()
   - แยก Answerable / Adversarial
   - คำนวณ Answer Rate
   - BERTScore
   - Faithfulness / Relevance
   - False Abstention
   - Correct Abstention

5. collect_failures()
   - วิเคราะห์ failure cases
   - แยก Retriever failure / Generator failure
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from tqdm import tqdm
from bert_score import score as bertscore
from openai import OpenAI

import config

from src.generation import is_abstention, build_context
from src.metrics import mean, bootstrap_ci


# ================================================================
# LLM-as-Judge Client
# ================================================================

judge_client = OpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,
)


# ================================================================
# 1. BERTScore
# ================================================================

def compute_bertscore(
    candidates: list[str],
    references: list[str],
) -> dict:
    """คำนวณ BERTScore ด้วย xlm-roberta-large สำหรับภาษาไทย."""

    if not candidates or not references:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
        }

    if len(candidates) != len(references):
        raise ValueError(
            "BERTScore: candidates และ references ต้องมีจำนวนเท่ากัน"
        )

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


# ================================================================
# 2. LLM-as-Judge
# ================================================================

JUDGE_PROMPT = """คุณเป็นกรรมการตรวจสอบระบบตอบคำถามกฎหมายไทยที่เข้มงวดมาก

ให้ประเมินคำตอบของระบบใน 2 มิติ:
1. Faithfulness
2. Relevance

ขั้นตอนบังคับ:

1. แตกคำตอบออกเป็น "ข้อกล่าวอ้าง (claim)" ทีละประเด็น
2. ตรวจทีละ claim ว่ามีข้อความรองรับอยู่ในบริบทหรือไม่
3. รวบรวม claim ที่ "ไม่มีหลักฐานในบริบท" ไว้ใน unsupported_claims
4. ตรวจสอบเลขมาตรา ชื่อกฎหมาย และข้อเท็จจริงที่อ้างในคำตอบกับบริบท

==============================
FAITHFULNESS
==============================

5 = ทุก claim มีหลักฐานตรงหรืออนุมานได้โดยตรงจากบริบท
    และเลขมาตรา/ชื่อกฎหมายถูกต้อง

4 = claim หลักถูกต้องทั้งหมด
    แต่มีการสรุปความเกินบริบทเล็กน้อย
    โดยไม่ได้เพิ่มข้อเท็จจริงสำคัญใหม่

3 = มีอย่างน้อย 1 claim ที่ไม่มีหลักฐานรองรับชัดเจน

2 = มีหลาย claim ไม่มีหลักฐาน
    หรือมีเลขมาตรา/ชื่อกฎหมายผิด
    หรืออ้างข้อมูลที่ไม่มีในบริบทอย่างมีนัยสำคัญ

1 = เนื้อหาส่วนใหญ่ไม่มีหลักฐานจากบริบท
    หรือเป็นการแต่งข้อมูลขึ้นเอง

==============================
RELEVANCE
==============================

5 = ตอบตรงคำถามและครอบคลุมสาระสำคัญทั้งหมด

4 = ตอบตรงคำถามครบถ้วน
    แต่มีส่วนขยายที่ไม่จำเป็นเล็กน้อย

3 = ตอบคำถามได้เพียงบางส่วน

2 = แตะประเด็นของคำถาม
    แต่ไม่ได้ตอบสาระสำคัญ

1 = ไม่เกี่ยวข้องกับคำถาม

==============================
ABSTENTION
==============================

ถ้าคำตอบเป็นการปฏิเสธ เช่น:

"ไม่พบข้อมูลในเอกสารที่มี"
"ไม่สามารถตอบจากเอกสารที่ให้มาได้"

ให้ตรวจบริบทก่อน

ถ้าบริบทไม่มีข้อมูลที่สามารถตอบคำถามได้:
    faithfulness = 5
    relevance = 5

ถ้าบริบทมีข้อมูลที่ตอบคำถามได้:
    faithfulness = 5
    relevance = 1

==============================
INPUT
==============================

คำถาม:
{query}

--- บริบทที่ระบบดึงมา ---
{context}

--- คำตอบของระบบ ---
{answer}

==============================
OUTPUT
==============================

ตอบ JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON

{{
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "unsupported_claims": ["..."],
    "is_abstention": <true|false>,
    "reason": "<สั้นๆ>"
}}
"""


def _parse_json(text: str) -> dict:
    """
    แปลง output ของ LLM ให้เป็น dict
    รองรับทั้ง JSON ปกติและ JSON ที่ถูกครอบด้วย ```json
    """

    if not text:
        return {}

    t = str(text).strip()

    # เอา markdown code fence ออก
    t = re.sub(
        r"^```(?:json)?\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\s*```$",
        "",
        t,
    )

    t = t.strip()

    # ลอง parse ตรง ๆ ก่อน
    try:
        result = json.loads(t)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    # ถ้ามีข้อความอื่นปนมา ให้หา JSON object
    match = re.search(
        r"\{.*\}",
        t,
        flags=re.DOTALL,
    )

    if match:
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def _clamp(
    value: Any,
    lo: int = 1,
    hi: int = 5,
) -> Optional[int]:
    """บังคับคะแนนให้อยู่ในช่วง 1-5."""

    try:
        value = float(value)
        value = round(value)
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return None


def llm_judge(
    query: str,
    context: str,
    answer: str,
) -> dict:
    """
    ประเมินคำตอบด้วย LLM-as-Judge

    คืนค่า:
        faithfulness
        relevance
        unsupported_claims
        is_abstention
        reason
        ok
    """

    model_name = getattr(
        config,
        "JUDGE_MODEL",
        "gemma-4-E4B-it",
    )

    temperature = getattr(
        config,
        "JUDGE_TEMPERATURE",
        0.0,
    )

    try:
        response = judge_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        query=query,
                        context=context[:6000],
                        answer=answer,
                    ),
                }
            ],
        )

        raw_text = response.choices[0].message.content or ""
        raw = _parse_json(raw_text)

    except Exception as e:
        return {
            "faithfulness": None,
            "relevance": None,
            "unsupported_claims": [],
            "is_abstention": None,
            "reason": f"judge_error: {e}",
            "ok": False,
        }

    unsupported = raw.get("unsupported_claims")

    if not isinstance(unsupported, list):
        unsupported = []

    return {
        "faithfulness": _clamp(
            raw.get("faithfulness")
        ),
        "relevance": _clamp(
            raw.get("relevance")
        ),
        "unsupported_claims": unsupported,
        "is_abstention": bool(
            raw.get("is_abstention", False)
        ),
        "reason": str(
            raw.get("reason", "")
        )[:300],
        "ok": True,
    }


# ================================================================
# Safe Helpers
# ================================================================

def safe_mean(
    values: list[float],
) -> Optional[float]:
    """คืน None ถ้าไม่มีข้อมูล."""

    return mean(values) if values else None


def safe_round(
    value: Optional[float],
    nd: int = 3,
) -> Optional[float]:
    """Round แบบปลอดภัย."""

    return round(value, nd) if value is not None else None


def safe_ci(
    values: list[float],
    n_boot: int = 2000,
    seed: int = 42,
):
    """
    Bootstrap CI แบบปลอดภัย

    ถ้ามีข้อมูลน้อยกว่า 2 ตัวอย่าง จะคืน None
    """

    if not values or len(values) < 2:
        return None

    lo, hi = bootstrap_ci(
        values,
        n_boot=n_boot,
        seed=seed,
    )

    return [
        round(lo, 3),
        round(hi, 3),
    ]


def score_distribution(
    values: list[float],
) -> dict:
    """แจกแจงคะแนน 1-5."""

    distribution = {
        str(score): 0
        for score in (1, 2, 3, 4, 5)
    }

    invalid = 0

    for value in values:
        try:
            bucket = int(round(float(value)))
        except (TypeError, ValueError):
            invalid += 1
            continue

        if 1 <= bucket <= 5:
            distribution[str(bucket)] += 1
        else:
            invalid += 1

    distribution["invalid"] = invalid

    assert (
        sum(distribution.values()) == len(values)
    ), "distribution mismatch"

    return distribution


# ================================================================
# 3. Run Full RAG Pipeline
# ================================================================

def run_pipeline_on_testset(
    pipeline,
    testset: list[dict],
    judge_fn: Optional[Callable] = None,
    desc: str = "Evaluating",
) -> list[dict]:
    """
    รัน RAG pipeline ทีละข้อ

    เก็บ:
    - question
    - answer
    - reference
    - adversarial
    - abstained
    - gold_doc_id
    - retrieved_docs
    - retrieval_hit
    - context preview
    - retrieval scores
    - judge
    """

    rows = []

    for item in tqdm(
        testset,
        desc=desc,
    ):
        q = (
            item.get("question")
            or item.get("query")
            or ""
        )

        ref = (
            item.get("reference_answer")
            or item.get("ground_truth")
            or ""
        )

        adversarial = bool(
            item.get("adversarial", False)
        )

        gold_doc = item.get("gold_doc_id")

        # -----------------------------
        # Run RAG
        # -----------------------------

        result = pipeline.answer(q)

        answer = result.get(
            "answer",
            "",
        )

        contexts = result.get(
            "contexts",
            [],
        ) or []

        # -----------------------------
        # Context สำหรับ Judge
        # -----------------------------

        context_text = build_context(
            contexts
        )

        if judge_fn:
            judge = judge_fn(
                q,
                context_text,
                answer,
            )
        else:
            judge = {}

        # -----------------------------
        # Retrieval information
        # -----------------------------

        retrieved_docs = [
            c.get("doc_id")
            or c.get("file")
            for c in contexts
        ]

        retrieval_hit = (
            gold_doc in retrieved_docs
            if gold_doc
            else None
        )

        # -----------------------------
        # เก็บ row
        # -----------------------------

        rows.append(
            {
                "question": q,
                "answer": answer,
                "reference": ref,
                "adversarial": adversarial,
                "abstained": is_abstention(answer),
                "gold_doc_id": gold_doc,
                "retrieved_docs": retrieved_docs,
                "retrieval_hit": retrieval_hit,

                "contexts_preview": [
                    (c.get("text") or "")[:300]
                    for c in contexts[:3]
                ],

                "retrieval_scores": [
                    c.get("score")
                    for c in contexts[:5]
                ],

                "judge": judge,
            }
        )

    return rows


# ================================================================
# 4. Summary
# ================================================================

def summarize(
    rows: list[dict],
    bertscore_fn: Optional[Callable] = None,
    n_boot: int = 2000,
    seed: int = 42,
    label: str = "system",
) -> dict:
    """
    สรุปผลโดยแยก Answerable / Adversarial

    Answerable:
        - BERTScore
        - Faithfulness
        - Relevance
        - Answer Rate
        - False Abstention

    Adversarial:
        - Correct Abstention Rate

    Faithfulness หลัก:
        คำนวณเฉพาะข้อที่ระบบยอมตอบจริง
    """

    # ------------------------------------------------------------
    # แยกชุด
    # ------------------------------------------------------------

    answerable = [
        r
        for r in rows
        if not r["adversarial"]
    ]

    adversarial = [
        r
        for r in rows
        if r["adversarial"]
    ]

    # ------------------------------------------------------------
    # BERTScore
    # เฉพาะ Answerable ที่มี reference
    # ------------------------------------------------------------

    scored = [
        r
        for r in answerable
        if (r.get("reference") or "").strip()
    ]

    if (
        scored
        and bertscore_fn is not None
    ):
        bert = bertscore_fn(
            [r["answer"] for r in scored],
            [r["reference"] for r in scored],
        )
    else:
        bert = None

    # ------------------------------------------------------------
    # Judge score helper
    # ------------------------------------------------------------

    def judge_scores(
        key: str,
        subset: list[dict],
    ) -> list[float]:

        return [
            r["judge"][key]
            for r in subset
            if isinstance(
                r.get("judge"),
                dict,
            )
            and r["judge"].get(key)
            is not None
        ]

    # ------------------------------------------------------------
    # เฉพาะข้อที่ระบบตอบจริง
    # ------------------------------------------------------------

    answered = [
        r
        for r in answerable
        if not r["abstained"]
    ]

    faith_answered = judge_scores(
        "faithfulness",
        answered,
    )

    relevance_answered = judge_scores(
        "relevance",
        answered,
    )

    # เอาไว้ดูเทียบเท่านั้น
    faith_all = judge_scores(
        "faithfulness",
        answerable,
    )

    # ------------------------------------------------------------
    # Answer Rate
    # ------------------------------------------------------------

    answer_rate = (
        len(answered) / len(answerable)
        if answerable
        else None
    )

    # ------------------------------------------------------------
    # Abstention
    # ------------------------------------------------------------

    n_correct_abstain = sum(
        1
        for r in adversarial
        if r["abstained"]
    )

    n_false_abstain = sum(
        1
        for r in answerable
        if r["abstained"]
    )

    correct_abstention_rate = (
        n_correct_abstain / len(adversarial)
        if adversarial
        else None
    )

    false_abstention_rate = (
        n_false_abstain / len(answerable)
        if answerable
        else None
    )

    adv_flags = [
        1.0 if r["abstained"] else 0.0
        for r in adversarial
    ]

    # ------------------------------------------------------------
    # Failure Taxonomy
    # ------------------------------------------------------------

    taxonomy = {
        "retriever_miss_safe": 0,
        "retrieval_miss_answered": 0,
        "generator_over_refusal": 0,
        "generator_misread": 0,
        "unknown": 0,
    }

    for r in answerable:

        hit = r.get("retrieval_hit")

        if hit is None:
            taxonomy["unknown"] += 1
            continue

        faithfulness = (
            r.get("judge") or {}
        ).get("faithfulness")

        if not hit and r["abstained"]:

            taxonomy[
                "retriever_miss_safe"
            ] += 1

        elif not hit and not r["abstained"]:

            # ไม่ใช้คำว่า hallucination โดยอัตโนมัติ
            # เพราะ retrieval miss ไม่ได้พิสูจน์ว่า
            # คำตอบผิดเสมอไป
            taxonomy[
                "retrieval_miss_answered"
            ] += 1

        elif hit and r["abstained"]:

            taxonomy[
                "generator_over_refusal"
            ] += 1

        elif (
            hit
            and not r["abstained"]
            and faithfulness is not None
            and faithfulness <= 3
        ):

            taxonomy[
                "generator_misread"
            ] += 1

    # ------------------------------------------------------------
    # Judge failures
    # ------------------------------------------------------------

    n_judge_failed = sum(
        1
        for r in rows
        if isinstance(
            r.get("judge"),
            dict,
        )
        and not r["judge"].get(
            "ok",
            True,
        )
    )

    # ------------------------------------------------------------
    # Return summary
    # ------------------------------------------------------------

    return {
        "label": label,

        "n_total": len(rows),
        "n_answerable": len(answerable),
        "n_adversarial": len(adversarial),
        "n_judge_failed": n_judge_failed,

        # ========================================================
        # Answerable
        # ========================================================

        "answerable_evaluation": {

            "n_scored_bertscore": len(scored),

            "bertscore": bert,

            "llm_judge": {

                "faithfulness_avg_answered":
                    safe_round(
                        safe_mean(
                            faith_answered
                        )
                    ),

                "faithfulness_ci95_answered":
                    safe_ci(
                        faith_answered,
                        n_boot,
                        seed,
                    ),

                "relevance_avg_answered":
                    safe_round(
                        safe_mean(
                            relevance_answered
                        )
                    ),

                "relevance_ci95_answered":
                    safe_ci(
                        relevance_answered,
                        n_boot,
                        seed,
                    ),

                "n_answered":
                    len(answered),

                "answer_rate":
                    safe_round(
                        answer_rate
                    ),

                "faithfulness_avg_all":
                    safe_round(
                        safe_mean(
                            faith_all
                        )
                    ),

                "faithfulness_distribution_answered":
                    score_distribution(
                        faith_answered
                    ),

                "relevance_distribution_answered":
                    score_distribution(
                        relevance_answered
                    ),
            },

            "false_abstention_rate":
                safe_round(
                    false_abstention_rate
                ),

            "false_abstention_note":
                (
                    f"{n_false_abstain}/{len(answerable)}"
                    if answerable
                    else None
                ),
        },

        # ========================================================
        # Adversarial
        # ========================================================

        "adversarial_evaluation": {

            "n_questions":
                len(adversarial),

            "correct_abstention_rate":
                safe_round(
                    correct_abstention_rate
                ),

            "correct_abstention_ci95":
                safe_ci(
                    adv_flags,
                    n_boot,
                    seed,
                ),

            "correct_abstention_note":
                (
                    f"{n_correct_abstain}/{len(adversarial)}"
                    if adversarial
                    else None
                ),

            "n_failed_to_abstain":
                len(adversarial)
                - n_correct_abstain,

            "small_sample_warning":
                len(adversarial) < 15,
        },

        "failure_taxonomy": taxonomy,
    }


# ================================================================
# 5. Failure Cases
# ================================================================

def collect_failures(
    rows: list[dict],
) -> list[dict]:
    """
    ดึงเฉพาะเคสที่มีพฤติกรรมผิดปกติ

    Adversarial ที่ปฏิเสธถูกต้อง
    ไม่ถือเป็น failure
    """

    failures = []

    for r in rows:

        judge = r.get("judge") or {}

        reasons = []

        # --------------------------------------------------------
        # Adversarial ตอบออกมา
        # --------------------------------------------------------

        correct_abstain = (
            r["adversarial"]
            and r["abstained"]
        )

        if (
            r["adversarial"]
            and not r["abstained"]
        ):
            reasons.append(
                "HALLUCINATION: "
                "คำถามหลอกที่ควรปฏิเสธแต่ระบบกลับตอบ"
            )

        # --------------------------------------------------------
        # Answerable แต่ปฏิเสธ
        # --------------------------------------------------------

        if (
            not r["adversarial"]
            and r["abstained"]
        ):

            if r.get("retrieval_hit") is False:

                reasons.append(
                    "RETRIEVER MISS: "
                    "Retriever ไม่พบเอกสารที่ถูกต้อง"
                )

            else:

                reasons.append(
                    "OVER-REFUSAL: "
                    "มีเอกสารที่เกี่ยวข้องแต่ระบบปฏิเสธตอบ"
                )

        # --------------------------------------------------------
        # Judge failure
        # --------------------------------------------------------

        if not correct_abstain:

            faithfulness = judge.get(
                "faithfulness"
            )

            if (
                faithfulness is not None
                and faithfulness <= 3
            ):
                reasons.append(
                    f"LOW FAITHFULNESS ({faithfulness})"
                )

            relevance = judge.get(
                "relevance"
            )

            if (
                relevance is not None
                and relevance <= 3
            ):
                reasons.append(
                    f"LOW RELEVANCE ({relevance})"
                )

            unsupported_claims = judge.get(
                "unsupported_claims"
            )

            if unsupported_claims:

                reasons.append(
                    "UNSUPPORTED CLAIMS "
                    f"({len(unsupported_claims)} ประเด็น)"
                )

        # --------------------------------------------------------
        # เก็บเฉพาะเคสที่มี failure
        # --------------------------------------------------------

        if reasons:

            failures.append(
                {
                    **r,
                    "failure_types": reasons,
                }
            )

    return failures