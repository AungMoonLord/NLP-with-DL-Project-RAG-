"""STEP 9 — Ablation study

โจทย์บังคับให้เทียบอย่างน้อย 2 variant เราตั้งไว้ 4 เพื่อให้ "แยกอิทธิพลของแต่ละชิ้น" ได้:

  A  dense-only                    (baseline)
  B  hybrid (dense + BM25 + RRF)   -> วัดผลของ "การเติม lexical signal"
  C  hybrid + cross-encoder rerank -> วัดผลของ "การจัดอันดับใหม่ขั้นที่สอง"
  D  dense-only + chunk 1024       -> วัดผลของ "ขนาด chunk" แยกจากเรื่อง retrieval

หลักการสำคัญ: เปลี่ยนทีละตัวแปร (one factor at a time)
ถ้าเปลี่ยน chunk size พร้อมกับเปลี่ยนโหมด retrieval จะอธิบายไม่ได้ว่าอะไรทำให้ดีขึ้น

ใช้งาน:
    python eval/run_ablation.py                    # รันทุก variant
    python eval/run_ablation.py --variants A B C   # เลือกเฉพาะบางตัว
    python eval/run_ablation.py --no-generate      # เร็ว: วัดเฉพาะ retrieval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "A": ("configs/variant_a_dense.yaml", "A: Dense-only"),
    "B": ("configs/variant_b_hybrid.yaml", "B: Hybrid (dense+BM25+RRF)"),
    "C": ("configs/variant_c_hybrid_rerank.yaml", "C: Hybrid + CrossEncoder"),
    "D": ("configs/variant_d_chunk1024.yaml", "D: Dense-only, chunk=1024"),
}

TABLE_COLS = [
    ("retrieval", "recall@5", "Recall@5"),
    ("retrieval", "mrr@5", "MRR@5"),
    ("retrieval", "ndcg@5", "nDCG@5"),
    ("generation", "bertscore_f1", "BERTScore-F1"),
    ("generation", "rouge_l_f", "ROUGE-L"),
    ("generation", "llm_judge_score", "LLM-Judge (1-5)"),
    ("generation", "faithfulness", "Faithfulness"),
    ("generation", "answer_relevance", "Ans.Relevance"),
    ("efficiency", "latency_s", "Latency (s)"),
]


def run_variant(key: str, extra_args) -> dict:
    cfg_path, label = VARIANTS[key]
    print(f"\n{'='*70}\n  รัน variant {label}\n{'='*70}")
    cmd = [sys.executable, str(ROOT / "eval/run_eval.py"),
           "--config", str(ROOT / cfg_path), "--tag", key] + extra_args
    subprocess.run(cmd, check=True, cwd=ROOT)
    with open(ROOT / "results" / f"eval_{key}.json", encoding="utf-8") as f:
        return json.load(f)


def build_table(results: dict) -> str:
    keys = list(results)
    header = "| Metric | " + " | ".join(results[k]["variant_label"] for k in keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    lines = [header, sep]
    for group, metric, label in TABLE_COLS:
        row_vals = []
        any_val = False
        for k in keys:
            v = results[k]["metrics"].get(group, {}).get(metric)
            if isinstance(v, (int, float)) and v == v:
                any_val = True
                row_vals.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                row_vals.append("–")
        if any_val:
            lines.append(f"| {label} | " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def plot(results: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ablation] ไม่มี matplotlib ข้ามการวาดกราฟ")
        return
    metrics = [("retrieval", "recall@5", "Recall@5"),
               ("retrieval", "mrr@5", "MRR@5"),
               ("generation", "faithfulness", "Faithfulness"),
               ("generation", "bertscore_f1", "BERTScore F1")]
    keys = list(results)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, (g, m, label) in zip(axes, metrics):
        vals = [results[k]["metrics"].get(g, {}).get(m, float("nan")) for k in keys]
        ax.bar(keys, vals)
        ax.set_title(label)
        ax.set_ylim(0, max([v for v in vals if v == v] + [0.1]) * 1.25)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Ablation Study — Thai Legal RAG")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[ablation] บันทึกกราฟ -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    extra = []
    if args.no_generate:
        extra.append("--no-generate")
    if args.limit:
        extra += ["--limit", str(args.limit)]

    results = {}
    for k in args.variants:
        payload = run_variant(k, extra)
        payload["variant_label"] = VARIANTS[k][1]
        results[k] = payload

    table = build_table(results)
    md = ["# Ablation Results — Thai Legal RAG", "", table, "",
          "## Variants", ""]
    for k in results:
        c = results[k]["config"]
        md.append(f"- **{VARIANTS[k][1]}** — mode=`{c['retrieval']['mode']}`, "
                  f"rerank=`{c['retrieval']['use_reranker']}`, "
                  f"chunk=`{c['chunk']['max_tokens']}/{c['chunk']['overlap_tokens']}`, "
                  f"top_k=`{c['retrieval']['top_k_final']}`")
    md += ["", "## การตีความ (เขียนเองหลังเห็นตัวเลขจริง)", "",
           "> อย่าเขียนแค่ 'variant C ชนะ' — ต้องอธิบายกลไก เช่น",
           "> คำถามที่มีเลขมาตราชัดเจนได้ประโยชน์จาก BM25 มากที่สุด เพราะ ...",
           "> ส่วนคำถามเชิงนิยาม dense ทำได้ดีอยู่แล้ว เพราะ ..."]

    out_dir = ROOT / "results"
    (out_dir / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    with open(out_dir / "ablation.json", "w", encoding="utf-8") as f:
        json.dump({k: {"label": v["variant_label"], "metrics": v["metrics"]}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    plot(results, out_dir / "ablation.png")

    print("\n" + table)
    print(f"\n[ablation] บันทึกผล -> {out_dir/'ablation.md'}")


if __name__ == "__main__":
    main()
