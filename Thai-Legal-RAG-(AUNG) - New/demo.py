# import sys; sys.path.insert(0, ".")
# from src.pipeline import RAGPipeline

# pipeline = RAGPipeline(mode="hybrid_rerank")
# print("Thai Legal RAG — พิมพ์คำถาม (พิมพ์ 'exit' เพื่อออก)\n")

# while True:
#     q = input("❓ คำถาม: ").strip()
#     if q.lower() in ("exit", "quit"):
#         break
#     result = pipeline.answer(q)
#     print(f"\n💡 คำตอบ:\n{result['answer']}\n")
#     print("📄 แหล่งอ้างอิง:")
#     for c in result["contexts"]:
#         print(f"  - {c['doc_id']} (score: {c['score']:.4f})")
#     print()

import sys
sys.path.insert(0, ".")

from src.pipeline import RAGPipeline

# สำหรับ Live Demo: หากต้องการให้ Rerank ไวขึ้น สามารถใช้ mode="hybrid" หรือคง "hybrid_rerank" ไว้ได้
pipeline = RAGPipeline(mode="hybrid_rerank")

print("Thai Legal RAG — พิมพ์คำถาม (พิมพ์ 'exit' เพื่อออก)\n")

while True:
    try:
        q = input("❓ คำถาม: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        print("\n💡 คำตอบ: ")
        # ดึง Contexts และ Iterator ของ Stream
        # ส่ง retrieve_k=12 หรือ 15 เพื่อลดภาระ CPU ให้ตอบไวขึ้นตอนเดโมสด
        contexts, stream = pipeline.answer_stream(q, retrieve_k=12)

        # พิมพ์ตัวอักษรออกมาทีละ Token ทันที
        for token in stream:
            print(token, end="", flush=True)
        print("\n")

        print("📄 แหล่งอ้างอิง:")
        for c in contexts:
            score = c.get("rerank_score", c.get("score", 0.0))
            doc_name = c.get("doc_id") or c.get("file")
            print(f"  - {doc_name} (score: {score:.4f})")
        print()

    except KeyboardInterrupt:
        print("\nออกจากโปรแกรม")
        break

