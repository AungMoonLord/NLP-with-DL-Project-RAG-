from src.retrieval import Retriever
from src.generation import generate_answer, generate_answer_stream


class RAGPipeline:
    def __init__(self, mode: str = "hybrid_rerank"):
        self.mode = mode
        self.retriever = Retriever(use_reranker=(mode == "hybrid_rerank"))

    def answer(self, query: str) -> dict:
        contexts = self.retriever.retrieve(query, mode=self.mode)
        answer = generate_answer(query, contexts)
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "mode": self.mode,
        }
    # def answer_stream(self, query: str):
    #     """ค้นหาบริบทก่อน แล้วสตรีมคำตอบออกมาทีละ token"""
    #     contexts = self.retriever.retrieve(query, mode=self.mode)
    #     return contexts, generate_answer_stream(query, contexts)
    def answer_stream(self, query: str, retrieve_k: int | None = None):
        """ค้นหาบริบทก่อน แล้วสตรีมคำตอบออกมาทีละ token"""
        # ส่ง retrieve_k ต่อไปยังตัว retriever
        contexts = self.retriever.retrieve(query, mode=self.mode, retrieve_k=retrieve_k)
        return contexts, generate_answer_stream(query, contexts)
