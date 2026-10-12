"""Ablation Study: เปรียบเทียบ 4 Retrieval Variants

Variants:
1. dense
2. bm25
3. hybrid
4. hybrid_rerank

วัด:
- Recall@k
- Hit@k
- nDCG@k
- MRR
- Latency
- Bootstrap 95% CI
- Paired Bootstrap p-value เทียบ Dense

สำคัญ:
Ablation นี้วัดเฉพาะ Retrieval
ไม่ได้วัดคุณภาพของ LLM Generator
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


# ================================================================
# Root
# ================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ================================================================
# Imports
# ================================================================

import config

from src.retrieval import Retriever

from src.metrics import (
    gold_docs_of,
    retrieved_docs_of,
    recall_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    mean,
    bootstrap_ci,
    paired_bootstrap_p,
)


# ================================================================
# Variants
# ================================================================

VARIANTS = [
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
]


# ================================================================
# Load Testset
# ================================================================

def load_testset() -> list[dict]:

    path = getattr(
        config,
        "TESTSET_PATH",
        ROOT_DIR / "data" / "testset.json",
    )

    with open(
        path,
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

    testset = load_testset()

    # ------------------------------------------------------------
    # ใช้เฉพาะ Answerable + มี Gold Document
    # ------------------------------------------------------------

    items = [
        item
        for item in testset
        if not item.get(
            "adversarial",
            False,
        )
    ]

    items = [
        item
        for item in items
        if gold_docs_of(item)
    ]

    if not items:

        raise SystemExit(
            "❌ ไม่พบคำถาม Answerable "
            "ที่มี gold document"
        )

    print(
        f"✓ พร้อมประเมิน "
        f"{len(items)} คำถาม × "
        f"{len(VARIANTS)} variants\n"
    )

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------

    eval_k_values = getattr(
        config,
        "EVAL_K_VALUES",
        [1, 3, 5],
    )

    max_k = getattr(
        config,
        "MAX_EVAL_K",
        max(eval_k_values),
    )

    rerank_pool = getattr(
        config,
        "RERANK_CANDIDATES",
        getattr(
            config,
            "TOP_K_RETRIEVE",
            30,
        ),
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
    # Artifacts
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
    # Retriever
    # ------------------------------------------------------------

    retriever = Retriever(
        index_dir=artifacts_dir
    )

    # ------------------------------------------------------------
    # Raw metrics
    # ------------------------------------------------------------

    per_question = {
        variant: {}
        for variant in VARIANTS
    }

    diagnostics = {
        variant: {
            "latency": [],
            "rank_shift": [],
            "n_candidates": [],
        }
        for variant in VARIANTS
    }

    # ------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------

    for number, item in enumerate(
        items,
        start=1,
    ):

        question = (
            item.get("question")
            or item.get("query")
            or ""
        )

        gold_docs = gold_docs_of(item)

        for variant in VARIANTS:

            start = time.perf_counter()

            hits, debug = retriever.retrieve(
                question,
                mode=variant,
                k=max_k,
                retrieve_k=rerank_pool,
                return_debug=True,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            # ----------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------

            diagnostics[
                variant
            ]["latency"].append(
                elapsed
            )

            diagnostics[
                variant
            ]["rank_shift"].append(
                debug.get(
                    "rank_shift",
                    0,
                )
            )

            diagnostics[
                variant
            ]["n_candidates"].append(
                debug.get(
                    "n_candidates",
                    debug.get(
                        "n_returned",
                        0,
                    ),
                )
            )

            # ----------------------------------------------------
            # Retrieved document IDs
            # ----------------------------------------------------

            retrieved_docs = (
                retrieved_docs_of(
                    hits
                )
            )

            metrics = (
                per_question[
                    variant
                ]
            )

            # ----------------------------------------------------
            # Recall / Hit / nDCG
            # ----------------------------------------------------

            for k in eval_k_values:

                metrics.setdefault(
                    f"recall@{k}",
                    [],
                ).append(
                    recall_at_k(
                        retrieved_docs,
                        gold_docs,
                        k,
                    )
                )

                metrics.setdefault(
                    f"hit@{k}",
                    [],
                ).append(
                    hit_at_k(
                        retrieved_docs,
                        gold_docs,
                        k,
                    )
                )

                metrics.setdefault(
                    f"ndcg@{k}",
                    [],
                ).append(
                    ndcg_at_k(
                        retrieved_docs,
                        gold_docs,
                        k,
                    )
                )

            # ----------------------------------------------------
            # MRR
            # ----------------------------------------------------

            metrics.setdefault(
                "mrr",
                [],
            ).append(
                mrr(
                    retrieved_docs,
                    gold_docs,
                    max_k,
                )
            )

        print(
            f"  ความคืบหน้า: "
            f"[{number}/{len(items)}]",
            end="\r",
        )

    print(
        "\n\nประมวลผลเสร็จสิ้น "
        "กำลังสรุปสถิติ..."
    )

    # ============================================================
    # Aggregate
    # ============================================================

    results = {}

    for variant in VARIANTS:

        aggregate = {}

        for metric_name, values in (
            per_question[
                variant
            ].items()
        ):

            if not values:
                continue

            lo, hi = bootstrap_ci(
                values,
                n_boot=n_boot,
                seed=seed,
            )

            aggregate[
                metric_name
            ] = {
                "mean": round(
                    mean(values),
                    4,
                ),
                "ci95": [
                    round(lo, 4),
                    round(hi, 4),
                ],
            }

        # --------------------------------------------------------
        # Latency
        # --------------------------------------------------------

        aggregate[
            "latency_ms_avg"
        ] = round(
            1000
            * mean(
                diagnostics[
                    variant
                ]["latency"]
            ),
            2,
        )

        # --------------------------------------------------------
        # Candidate count
        # --------------------------------------------------------

        aggregate[
            "avg_candidates"
        ] = round(
            mean(
                diagnostics[
                    variant
                ]["n_candidates"]
            ),
            1,
        )

        # --------------------------------------------------------
        # Reranker rank shift
        # --------------------------------------------------------

        if variant == "hybrid_rerank":

            aggregate[
                "avg_rank_shift"
            ] = round(
                mean(
                    diagnostics[
                        variant
                    ]["rank_shift"]
                ),
                2,
            )

        results[
            variant
        ] = aggregate

    # ============================================================
    # Statistical Significance
    # เทียบทุก variant กับ Dense
    # ============================================================

    p_values_vs_dense = {}

    target_metrics = [
        f"recall@{k}"
        for k in eval_k_values
    ]

    target_metrics.append(
        "mrr"
    )

    for variant in VARIANTS:

        if variant == "dense":
            continue

        p_values_vs_dense[
            variant
        ] = {}

        for metric_name in target_metrics:

            if (
                metric_name
                not in per_question[
                    variant
                ]
            ):
                continue

            if (
                metric_name
                not in per_question[
                    "dense"
                ]
            ):
                continue

            p_value = paired_bootstrap_p(
                per_question[
                    variant
                ][metric_name],
                per_question[
                    "dense"
                ][metric_name],
                n_boot=n_boot,
                seed=seed,
            )

            p_values_vs_dense[
                variant
            ][metric_name] = (
                round(p_value, 4)
            )

    # ============================================================
    # Save JSON
    # ============================================================

    output = {

        "n_questions": len(items),

        "variants": VARIANTS,

        "k_values": eval_k_values,

        "config": {
            "TOP_K_RETRIEVE":
                getattr(
                    config,
                    "TOP_K_RETRIEVE",
                    30,
                ),

            "RERANK_CANDIDATES":
                rerank_pool,

            "TOP_K_FINAL":
                getattr(
                    config,
                    "TOP_K_FINAL",
                    5,
                ),

            "RRF_K":
                getattr(
                    config,
                    "RRF_K",
                    60,
                ),
        },

        "results": results,

        "p_values_vs_dense":
            p_values_vs_dense,

        "per_question":
            per_question,
    }

    out_json = (
        artifacts_dir
        / "ablation_results.json"
    )

    out_json.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Markdown Table
    # ============================================================

    display_columns = (
        [
            f"recall@{k}"
            for k in eval_k_values
        ]
        + ["mrr"]
    )

    header = (
        ["| Variant |"]
        + [
            f" {column.upper()} |"
            for column in display_columns
        ]
        + [" Latency (ms) |"]
    )

    separator = (
        ["|---|"]
        + ["---|"] * (
            len(display_columns) + 1
        )
    )

    markdown_lines = [
        "".join(header),
        "".join(separator),
    ]

    for variant in VARIANTS:

        row = (
            f"| `{variant}` |"
        )

        for column in display_columns:

            value = (
                results[
                    variant
                ]
                .get(column, {})
                .get("mean", 0.0)
            )

            row += (
                f" {value:.4f} |"
            )

        row += (
            f" {results[variant]['latency_ms_avg']} |"
        )

        markdown_lines.append(
            row
        )

    markdown_table = (
        "\n".join(
            markdown_lines
        )
    )

    out_md = (
        artifacts_dir
        / "ablation_table.md"
    )

    out_md.write_text(
        markdown_table,
        encoding="utf-8",
    )

    # ============================================================
    # Print
    # ============================================================

    print("\n" + "=" * 70)

    print(
        "ABLATION STUDY — "
        "RETRIEVAL EVALUATION"
    )

    print("=" * 70)

    print(
        markdown_table
    )

    print("=" * 70)

    if "hybrid_rerank" in results:

        rank_shift = results[
            "hybrid_rerank"
        ].get(
            "avg_rank_shift",
            0,
        )

        print(
            "🔧 Cross-Encoder Reranker "
            f"สลับอันดับเฉลี่ย: "
            f"{rank_shift} ตำแหน่ง/คำถาม"
        )

    print(
        f"📁 ผลลัพธ์ JSON: "
        f"{out_json}"
    )

    print(
        f"📋 ตาราง Markdown: "
        f"{out_md}\n"
    )


if __name__ == "__main__":
    main()