# import sys; sys.path.insert(0, ".")
# import json
# from tqdm import tqdm
# from src.pipeline import RAGPipeline
# from src.evaluation import compute_bertscore, llm_judge
# import config

# with open(config.TESTSET_PATH, encoding="utf-8") as f:
#     testset = json.load(f)

# pipeline = RAGPipeline(mode="hybrid_rerank")
# records, answers, references = [], [], []

# for item in tqdm(testset):
#     result = pipeline.answer(item["question"])
#     context_text = "\n".join(c["text"] for c in result["contexts"])
#     judge = llm_judge(item["question"], context_text, result["answer"])
    
#     # รองรับทั้งคีย์ 'reference_answer' และ 'ground_truth'
#     ref = item.get("reference_answer") or item.get("ground_truth", "")
    
#     records.append({
#         "question": item["question"],
#         "answer": result["answer"],
#         "reference": ref,
#         "judge": judge,
#     })
#     answers.append(result["answer"])
#     references.append(ref)

# bert = compute_bertscore(answers, references)
# faith = [r["judge"]["faithfulness"] for r in records if r["judge"]["faithfulness"]]
# rel = [r["judge"]["relevance"] for r in records if r["judge"]["relevance"]]

# final = {
#     "bertscore": bert,
#     "llm_judge": {
#         "faithfulness_avg": round(sum(faith) / len(faith), 2),
#         "relevance_avg": round(sum(rel) / len(rel), 2),
#     },
#     "per_question": records,
# }
# with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
#     json.dump(final, f, ensure_ascii=False, indent=2)

# print(json.dumps({k: v for k, v in final.items() if k != "per_question"},
#                  ensure_ascii=False, indent=2))


"""Evaluation Script: รันการประเมิน Full RAG Pipeline บนชุดข้อมูลทดสอบ
ประเมินผลครอบคลุมทั้ง:
1. BERTScore (xlm-roberta-large)
2. LLM-as-Judge (Faithfulness & Relevance พร้อมแจกแจง Distribution)
3. Abstention Analysis (วัดความแม่นยำในการปฏิเสธคำตอบเพื่อลด Hallucination)
4. Failure Analysis Extraction (ส่งออกเคสที่มีปัญหาลง failure_cases.json อัตโนมัติ)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tqdm import tqdm

# ปรับ Root Directory ให้ค้นหาโมดูลในโปรเจกต์เจอเสมอ
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.pipeline import RAGPipeline
from src.generation import is_abstention, build_context
from src.evaluation import llm_judge, compute_bertscore
from src.metrics import mean, bootstrap_ci


def main():
    # 1. โหลด Testset อย่างปลอดภัย
    testset_path = getattr(config, "TESTSET_PATH", ROOT_DIR / "data" / "testset.json")
    with open(testset_path, encoding="utf-8") as f:
        raw_testset = json.load(f)

    if isinstance(raw_testset, dict):
        testset = raw_testset.get("questions") or raw_testset.get("data") or []
    else:
        testset = raw_testset

    print(f"✓ เริ่มรันการประเมินผล Full Pipeline บนคำถาม {len(testset)} ข้อ...\n")

    pipeline = RAGPipeline(mode="hybrid_rerank")
    rows = []

    # 2. วนลูปประเมินผลคำถามทีละข้อ
    for item in tqdm(testset, desc="Evaluating Pipeline"):
        q = item.get("question") or item.get("query")
        ref = item.get("reference_answer") or item.get("ground_truth") or ""
        adv = bool(item.get("adversarial", False))

        # รัน Pipeline (เรียก answer จาก src/pipeline.py)
        result = pipeline.answer(q)
        ans = result.get("answer", "")
        contexts = result.get("contexts", [])

        # สร้าง Context Text ส่งให้ LLM-as-Judge
        context_text = build_context(contexts)
        judge = llm_judge(q, context_text, ans)

        # ตรวจสอบการปฏิเสธคำตอบ
        abstained = is_abstention(ans)

        rows.append({
            "question": q,
            "answer": ans,
            "reference": ref,
            "adversarial": adv,
            "abstained": abstained,
            "retrieved_docs": [c.get("doc_id") or c.get("file") for c in contexts],
            "judge": judge,
        })

    normal = [r for r in rows if not r["adversarial"]]
    adv_rows = [r for r in rows if r["adversarial"]]

    # 3. คำนวณ BERTScore เฉพาะข้อที่มี Ground Truth อ้างอิง
    scored = [r for r in normal if r["reference"].strip()]
    if scored:
        print("\nกำลังคำนวณ BERTScore (xlm-roberta-large)...")
        bs = compute_bertscore(
            [r["answer"] for r in scored],
            [r["reference"] for r in scored]
        )
    else:
        bs = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 4. รวมคะแนนฝั่ง LLM-as-Judge
    def get_judge_scores(key: str, subset: list[dict]) -> list[float]:
        return [r["judge"][key] for r in subset if r["judge"].get(key) is not None]

    f_all = get_judge_scores("faithfulness", rows)
    r_all = get_judge_scores("relevance", rows)

    n_boot = getattr(config, "N_BOOTSTRAP", 2000)
    seed = getattr(config, "RANDOM_SEED", 42)
    f_ci = bootstrap_ci(f_all, n_boot=n_boot, seed=seed) if f_all else [0.0, 0.0]

    summary = {
        "n_total": len(rows),
        "n_answerable": len(normal),
        "n_adversarial": len(adv_rows),
        "bertscore": bs,
        "llm_judge": {
            "faithfulness_avg": round(mean(f_all), 3),
            "faithfulness_ci95": [round(x, 3) for x in f_ci],
            "faithfulness_min": min(f_all) if f_all else None,
            "faithfulness_distribution": {str(s): f_all.count(s) for s in (1, 2, 3, 4, 5)},
            "relevance_avg": round(mean(r_all), 3),
            "relevance_distribution": {str(s): r_all.count(s) for s in (1, 2, 3, 4, 5)},
            "n_judge_failed": sum(1 for r in rows if not r["judge"].get("ok")),
        },
        "abstention": {
            "correct_abstention_rate": round(mean([1.0 if r["abstained"] else 0.0 for r in adv_rows]), 3) if adv_rows else None,
            "false_abstention_rate": round(mean([1.0 if r["abstained"] else 0.0 for r in normal]), 3) if normal else None,
        },
    }

    # 5. วิเคราะห์และดึง Failure Cases อัตโนมัติสำหรับ Slide 9
    failures = []
    for r in rows:
        j = r["judge"]
        why = []
        if r["adversarial"] and not r["abstained"]:
            why.append("HALLUCINATION: คำถามหลอกที่ควรปฏิเสธแต่กลับตอบ")
        if not r["adversarial"] and r["abstained"]:
            why.append("OVER-REFUSAL: ตอบได้แต่ปฏิเสธ (Retriever ดึงเอกสารไม่เจอ)")
        if (j.get("faithfulness") or 5) <= 3:
            why.append(f"LOW FAITHFULNESS ({j.get('faithfulness')})")
        if (j.get("relevance") or 5) <= 3:
            why.append(f"LOW RELEVANCE ({j.get('relevance')})")
        if j.get("unsupported_claims"):
            why.append(f"UNSUPPORTED CLAIMS ({len(j['unsupported_claims'])} ประเด็น)")

        if why:
            failures.append({**r, "failure_types": why})

    # 6. บันทึกผลลัพธ์ลงไฟล์
    artifacts_dir = Path(getattr(config, "ARTIFACTS_DIR", getattr(config, "INDEX_DIR", ROOT_DIR / "artifacts")))
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    out_eval = artifacts_dir / "eval_results.json"
    out_failures = artifacts_dir / "failure_cases.json"

    out_eval.write_text(
        json.dumps({**summary, "per_question": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    out_failures.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 7. แสดงผลสรุปออกทางหน้าจอ
    print("\n" + "=" * 60)
    print("ผลการประเมินระบบ RAG ประจำโปรเจกต์ (Evaluation Summary):")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"📉 พบจุดบกพร่อง (Failures) ทั้งหมด: {len(failures)}/{len(rows)} ข้อ")
    print(f"📁 บันทึกผลการประเมินที่: {out_eval}")
    print(f"📋 นำเคสในไฟล์นี้ไปใส่ Slide 9: {out_failures}\n")


if __name__ == "__main__":
    main()
    