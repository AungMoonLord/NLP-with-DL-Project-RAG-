import sys; sys.path.insert(0, ".")
import json
from tqdm import tqdm
from src.pipeline import RAGPipeline
from src.evaluation import compute_bertscore, llm_judge
import config

with open(config.TESTSET_PATH, encoding="utf-8") as f:
    testset = json.load(f)

pipeline = RAGPipeline(mode="hybrid_rerank")
records, answers, references = [], [], []

for item in tqdm(testset):
    result = pipeline.answer(item["question"])
    context_text = "\n".join(c["text"] for c in result["contexts"])
    judge = llm_judge(item["question"], context_text, result["answer"])
    
    # รองรับทั้งคีย์ 'reference_answer' และ 'ground_truth'
    ref = item.get("reference_answer") or item.get("ground_truth", "")
    
    records.append({
        "question": item["question"],
        "answer": result["answer"],
        "reference": ref,
        "judge": judge,
    })
    answers.append(result["answer"])
    references.append(ref)

bert = compute_bertscore(answers, references)
faith = [r["judge"]["faithfulness"] for r in records if r["judge"]["faithfulness"]]
rel = [r["judge"]["relevance"] for r in records if r["judge"]["relevance"]]

final = {
    "bertscore": bert,
    "llm_judge": {
        "faithfulness_avg": round(sum(faith) / len(faith), 2),
        "relevance_avg": round(sum(rel) / len(rel), 2),
    },
    "per_question": records,
}
with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print(json.dumps({k: v for k, v in final.items() if k != "per_question"},
                 ensure_ascii=False, indent=2))
