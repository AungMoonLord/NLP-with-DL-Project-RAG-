"""STEP 10 — Failure analysis (สำหรับสไลด์ที่ 9)

จัดกลุ่มข้อที่ระบบทำได้แย่ ตาม "สาเหตุ" ไม่ใช่แค่ "คะแนนต่ำ"
กรอบการวินิจฉัยที่ใช้ (แยกความผิดของ retrieval ออกจากความผิดของ generator):

  1. RETRIEVAL_MISS    ดึง chunk เฉลยไม่เจอเลย -> ปัญหาอยู่ที่ขั้นค้นคืน
  2. RANKING_ISSUE     เจอ แต่อยู่อันดับท้าย (rank > 3) -> ปัญหาอยู่ที่การจัดอันดับ/reranker
  3. GENERATION_ISSUE  ดึงเจอเป็นอันดับต้น แต่คำตอบยังผิด -> ปัญหาอยู่ที่ LLM/prompt
  4. HALLUCINATION     faithfulness ต่ำ -> แต่งข้อมูลที่ไม่มีใน context
  5. OVER_REFUSAL      ตอบว่าไม่พบข้อมูล ทั้งที่ดึงเจอ -> prompt เข้มเกินไป

ใช้งาน:
    python eval/failure_analysis.py --results results/eval_C.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REFUSAL_MARKERS = ["ไม่พบข้อมูล", "ไม่มีข้อมูล", "ไม่สามารถตอบ"]


def diagnose(row, judge_threshold=3, faith_threshold=0.7):
    gold = set(row.get("gold_chunk_ids") or [])
    retrieved = row.get("retrieved_ids", [])
    judge = row.get("llm_judge_score")
    faith = row.get("faithfulness")
    pred = row.get("prediction", "")
    refused = any(m in pred for m in REFUSAL_MARKERS)

    if gold:
        hit_rank = next((i for i, c in enumerate(retrieved, 1) if c in gold), None)
        if hit_rank is None:
            return "RETRIEVAL_MISS"
        if refused:
            return "OVER_REFUSAL"
        if isinstance(faith, (int, float)) and faith == faith and faith < faith_threshold:
            return "HALLUCINATION"
        if isinstance(judge, (int, float)) and judge < judge_threshold:
            return "RANKING_ISSUE" if hit_rank > 3 else "GENERATION_ISSUE"
        return "OK"

    # คำถามประเภท unanswerable: "ถูก" คือการปฏิเสธ
    if row.get("difficulty") == "unanswerable":
        return "OK" if refused else "HALLUCINATION"
    return "UNSCORED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = payload["per_item"]
    for r in rows:
        r["failure_type"] = diagnose(r)

    counts = Counter(r["failure_type"] for r in rows)
    total = len(rows)

    lines = [f"# Failure Analysis — variant {payload['variant']}", "",
             f"จำนวนคำถามทั้งหมด: {total}", "", "| ประเภท | จำนวน | สัดส่วน |", "|---|---|---|"]
    for k, v in counts.most_common():
        lines.append(f"| {k} | {v} | {v/total:.1%} |")

    lines += ["", "## ตัวอย่างเคสที่ล้มเหลว (ใส่ในสไลด์ 1-2 เคส)", ""]
    for kind in ["RETRIEVAL_MISS", "RANKING_ISSUE", "HALLUCINATION",
                 "GENERATION_ISSUE", "OVER_REFUSAL"]:
        examples = [r for r in rows if r["failure_type"] == kind][:2]
        if not examples:
            continue
        lines.append(f"### {kind}")
        for e in examples:
            lines += [f"- **คำถาม:** {e['question']}",
                      f"  - เฉลย: {e.get('reference','')[:150]}",
                      f"  - ระบบตอบ: {e.get('prediction','')[:200]}",
                      f"  - chunk ที่ดึงมา: {e.get('retrieved_ids', [])[:3]}",
                      f"  - เฉลย chunk: {e.get('gold_chunk_ids', [])}",
                      f"  - judge={e.get('llm_judge_score')} faithfulness={e.get('faithfulness')}", ""]

    out = Path(args.out or ROOT / "results" / f"failure_{payload['variant']}.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"\n[failure] บันทึก -> {out}")


if __name__ == "__main__":
    main()
