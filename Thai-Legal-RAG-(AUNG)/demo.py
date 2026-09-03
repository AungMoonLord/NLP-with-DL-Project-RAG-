import sys; sys.path.insert(0, ".")
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(mode="hybrid_rerank")
print("Thai Legal RAG — พิมพ์คำถาม (พิมพ์ 'exit' เพื่อออก)\n")

while True:
    q = input("❓ คำถาม: ").strip()
    if q.lower() in ("exit", "quit"):
        break
    result = pipeline.answer(q)
    print(f"\n💡 คำตอบ:\n{result['answer']}\n")
    print("📄 แหล่งอ้างอิง:")
    for c in result["contexts"]:
        print(f"  - {c['doc_id']} (score: {c['score']:.4f})")
    print()
