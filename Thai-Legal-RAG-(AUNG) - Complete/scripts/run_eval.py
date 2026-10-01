"""Evaluation Script: ประเมิน Full RAG Pipeline

หลักการ:

Answerable
    → BERTScore
    → Faithfulness
    → Relevance
    → Answer Rate
    → False Abstention Rate

Adversarial
    → Correct Abstention Rate

Faithfulness หลักคำนวณเฉพาะข้อที่ระบบ "ยอมตอบ"
และรายงานคู่กับ Answer Rate เสมอ

Logic หลักของ Evaluation อยู่ใน src/evaluation.py
เพื่อให้ run_eval.py และเครื่องมืออื่นใช้ logic เดียวกัน
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ================================================================
# Root Directory
# ================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ================================================================
# Imports
# ================================================================

import config

from src.pipeline import RAGPipeline

from src.evaluation import (
    llm_judge,
    compute_bertscore,
    run_pipeline_on_testset,
    summarize,
    collect_failures,
)


# ================================================================
# Load Testset
# ================================================================

def load_testset() -> list[dict]:

    testset_path = getattr(
        config,
        "TESTSET_PATH",
        ROOT_DIR / "data" / "testset.json",
    )

    with open(
        testset_path,
        encoding="utf-8",
    ) as f:

        raw = json.load(f)

    if isinstance(raw, dict):

        return (
            raw.get("questions")
            or raw.get("data")
            or []
        )

    return raw


# ================================================================
# Main
# ================================================================

def main():

    mode = getattr(
        config,
        "DEFAULT_MODE",
        "hybrid_rerank",
    )

    n_boot = getattr(
        config,
        "N_BOOTSTRAP",
        2000,
    )

    seed = getattr(
        config,
        "RANDOM_SEED",
        42,
    )

    # ------------------------------------------------------------
    # Load testset
    # ------------------------------------------------------------

    testset = load_testset()

    n_adversarial = sum(
        1
        for item in testset
        if item.get("adversarial")
    )

    n_answerable = (
        len(testset)
        - n_adversarial
    )

    print(
        f"✓ Testset: {len(testset)} ข้อ "
        f"(answerable {n_answerable} / "
        f"adversarial {n_adversarial})"
    )

    print(
        f"✓ RAG mode: {mode}\n"
    )

    # ------------------------------------------------------------
    # Create Pipeline
    # ------------------------------------------------------------

    pipeline = RAGPipeline(
        mode=mode
    )

    # ------------------------------------------------------------
    # Run Pipeline
    # ------------------------------------------------------------

    rows = run_pipeline_on_testset(
        pipeline,
        testset,
        judge_fn=llm_judge,
        desc=f"Evaluating [{mode}]",
    )

    # ------------------------------------------------------------
    # Artifact directory
    # ------------------------------------------------------------

    artifacts_dir = Path(
        getattr(
            config,
            "ARTIFACTS_DIR",
            getattr(
                config,
                "INDEX_DIR",
                ROOT_DIR / "artifacts",
            ),
        )
    )

    artifacts_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    # ------------------------------------------------------------
    # Save raw results FIRST
    # ------------------------------------------------------------

    raw_path = (
        artifacts_dir
        / "eval_raw_rows.json"
    )

    raw_path.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n✓ บันทึกผลดิบไว้แล้วที่:"
        f"\n  {raw_path}"
    )

    # ------------------------------------------------------------
    # Calculate Summary
    # ------------------------------------------------------------

    print(
        "\nกำลังคำนวณ BERTScore "
        "(xlm-roberta-large) "
        "เฉพาะ Answerable..."
    )

    summary = summarize(
        rows,
        bertscore_fn=compute_bertscore,
        n_boot=n_boot,
        seed=seed,
        label=mode,
    )

    # ------------------------------------------------------------
    # Failure Analysis
    # ------------------------------------------------------------

    failures = collect_failures(
        rows
    )

    # ------------------------------------------------------------
    # Save Evaluation Results
    # ------------------------------------------------------------

    out_eval = (
        artifacts_dir
        / "eval_results.json"
    )

    out_failures = (
        artifacts_dir
        / "failure_cases.json"
    )

    out_eval.write_text(
        json.dumps(
            {
                **summary,
                "per_question": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_failures.write_text(
        json.dumps(
            failures,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Read Summary
    # ------------------------------------------------------------

    answerable_eval = (
        summary[
            "answerable_evaluation"
        ]
    )

    judge = (
        answerable_eval[
            "llm_judge"
        ]
    )

    adversarial_eval = (
        summary[
            "adversarial_evaluation"
        ]
    )

    # ------------------------------------------------------------
    # Print Summary
    # ------------------------------------------------------------

    print("\n" + "=" * 64)

    print(
        f"EVALUATION SUMMARY"
        f" — mode = {mode}"
    )

    print("=" * 64)

    # ============================================================
    # Answerable
    # ============================================================

    print(
        f"\n[ ANSWERABLE ] "
        f"n = {summary['n_answerable']}"
    )

    bert = (
        answerable_eval
        .get("bertscore")
    )

    if bert:

        print(
            f"  BERTScore Precision : "
            f"{bert.get('precision')}"
        )

        print(
            f"  BERTScore Recall    : "
            f"{bert.get('recall')}"
        )

        print(
            f"  BERTScore F1        : "
            f"{bert.get('f1')}"
        )

    print(
        f"  Answer Rate         : "
        f"{judge['answer_rate']} "
        f"({judge['n_answered']}/"
        f"{summary['n_answerable']})"
    )

    print(
        f"  Faithfulness       : "
        f"{judge['faithfulness_avg_answered']} "
        f"CI95 "
        f"{judge['faithfulness_ci95_answered']}"
    )

    print(
        f"  Relevance          : "
        f"{judge['relevance_avg_answered']} "
        f"CI95 "
        f"{judge['relevance_ci95_answered']}"
    )

    print(
        f"  Faithfulness (all) : "
        f"{judge['faithfulness_avg_all']}"
    )

    print(
        f"  False Abstention   : "
        f"{answerable_eval['false_abstention_rate']} "
        f"({answerable_eval['false_abstention_note']})"
    )

    # ============================================================
    # Adversarial
    # ============================================================

    print(
        f"\n[ ADVERSARIAL ] "
        f"n = {adversarial_eval['n_questions']}"
    )

    print(
        f"  Correct Abstention : "
        f"{adversarial_eval['correct_abstention_rate']} "
        f"({adversarial_eval['correct_abstention_note']})"
    )

    print(
        f"  CI95               : "
        f"{adversarial_eval['correct_abstention_ci95']}"
    )

    if adversarial_eval[
        "small_sample_warning"
    ]:

        print(
            "  ⚠ n < 15 — CI อาจกว้าง "
            "ควรรายงานเป็นเศษส่วนด้วย"
        )

    # ============================================================
    # Failure Taxonomy
    # ============================================================

    print(
        "\n[ FAILURE TAXONOMY ]"
    )

    for key, value in (
        summary[
            "failure_taxonomy"
        ].items()
    ):

        print(
            f"  {key:<28}: {value}"
        )

    # ============================================================
    # Judge Errors
    # ============================================================

    if summary[
        "n_judge_failed"
    ]:

        print(
            "\n⚠ LLM-as-Judge "
            f"ล้มเหลว "
            f"{summary['n_judge_failed']} ข้อ"
        )

    # ============================================================
    # Finish
    # ============================================================

    print("\n" + "=" * 64)

    print(
        f"พบ Failure Cases: "
        f"{len(failures)}/{len(rows)} ข้อ"
    )

    print(
        f"บันทึกผล Evaluation ที่:"
        f"\n  {out_eval}"
    )

    print(
        f"บันทึก Failure Cases ที่:"
        f"\n  {out_failures}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()