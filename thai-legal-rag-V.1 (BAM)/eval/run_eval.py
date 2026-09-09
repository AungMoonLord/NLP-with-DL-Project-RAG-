"""รันการประเมินผลระบบ 1 ค่าคอนฟิก แล้วบันทึกผลลง results/

ใช้งาน:
    python eval/run_eval.py --config configs/variant_b_hybrid_rerank.yaml
    python eval/run_eval.py --config configs/baseline.yaml --no-generate   # วัดเฉพาะ retrieval
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.generator import build_generator
from src.pipeline import RAGPipeline
from eval.metrics import (LLMEvaluator, aggregate, bertscore, evaluate_retrieval,
                          rouge_l)


def load_gold(path: Path):
    """รูปแบบไฟล์ gold (JSONL) หนึ่งบรรทัดต่อหนึ่งคำถาม:
    {
      "question": "การผิดนัดชำระหนี้มีผลอย่างไร",
      "reference_answer": "...",              # เฉลยที่มนุษย์เขียน/ตรวจแล้ว
      "gold_chunk_ids": ["civil_code::u0123::p0"],   # (ไม่บังคับ) ใช้วัด retrieval
      "difficulty": "single-hop" | "multi-hop" | "unanswerable"
    }
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-generate", action="store_true",
                    help="ประเมินเฉพาะขั้นค้นคืน (ไม่เรียก LLM)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tag = args.tag or cfg.name
    gold_path = ROOT / (args.gold or cfg.evaluation.gold_path)
    gold = load_gold(gold_path)
    if args.limit:
        gold = gold[: args.limit]
    print(f"[eval] variant='{tag}'  |  ชุดทดสอบ {len(gold)} คำถาม")
    print(f"[eval] retrieval={cfg.retrieval.mode} rerank={cfg.retrieval.use_reranker} "
          f"top_k={cfg.retrieval.top_k_final} chunk={cfg.chunk.max_tokens}/{cfg.chunk.overlap_tokens}")

    pipe = RAGPipeline.from_config(cfg)
    generate = not args.no_generate

    rows = []
    for i, item in enumerate(gold, 1):
        print(f"  [{i}/{len(gold)}] {item['question'][:55]}...")
        res = pipe.answer(item["question"], generate=generate)
        row = {
            "question": item["question"],
            "reference": item.get("reference_answer", ""),
            "prediction": res.answer,
            "contexts": res.contexts,
            "citations": res.citations,
            "retrieved_ids": [r.chunk.chunk_id for r in res.retrieved],
            "gold_chunk_ids": item.get("gold_chunk_ids", []),
            "difficulty": item.get("difficulty", "unknown"),
            "latency_s": res.timings["total_s"],
            "retrieval_s": res.timings["retrieval_s"],
        }
        rows.append(row)

    # ---------- A) Retrieval metrics ----------
    metrics = {"retrieval": evaluate_retrieval(rows, cfg.evaluation.ks)}

    # ---------- B) Generation metrics ----------
    if generate:
        wanted = set(cfg.evaluation.metrics)
        preds = [r["prediction"] for r in rows]
        refs = [r["reference"] for r in rows]
        has_ref = [i for i, r in enumerate(refs) if r.strip()]

        if "rouge_l" in wanted:
            for r in rows:
                r.update(rouge_l(r["prediction"], r["reference"]) if r["reference"] else
                         {"rouge_l_f": float("nan")})

        if "bertscore" in wanted and has_ref:
            print("[eval] คำนวณ BERTScore ...")
            bs = bertscore([preds[i] for i in has_ref], [refs[i] for i in has_ref],
                           model_type=cfg.evaluation.bertscore_model)
            per_item = bs.pop("bertscore_f1_per_item")
            for pos, i in enumerate(has_ref):
                rows[i]["bertscore_f1"] = per_item[pos]

        needs_llm = wanted & {"llm_judge", "ragas_faithfulness", "ragas_answer_relevance"}
        if needs_llm:
            judge_cfg = cfg.generator
            judge_cfg.backend = cfg.evaluation.judge_backend
            judge_cfg.model = cfg.evaluation.judge_model
            evaluator = LLMEvaluator(build_generator(judge_cfg),
                                     embedder=pipe.retriever.embedder)
            for i, r in enumerate(rows, 1):
                print(f"[eval] LLM metrics {i}/{len(rows)}")
                if "llm_judge" in wanted and r["reference"]:
                    r.update(evaluator.judge(r["question"], r["prediction"], r["reference"]))
                if "ragas_faithfulness" in wanted:
                    r.update(evaluator.faithfulness(r["prediction"], r["contexts"]))
                if "ragas_answer_relevance" in wanted:
                    r.update(evaluator.answer_relevance(r["question"], r["prediction"]))

        metrics["generation"] = aggregate(
            rows, ["rouge_l_f", "bertscore_f1", "llm_judge_score",
                   "faithfulness", "answer_relevance"])

    metrics["efficiency"] = aggregate(rows, ["latency_s", "retrieval_s"])

    out_dir = ROOT / cfg.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "metrics": metrics,
        "per_item": rows,
    }
    out_path = out_dir / f"eval_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== สรุปผล ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n[eval] บันทึกผล -> {out_path}")


if __name__ == "__main__":
    main()
