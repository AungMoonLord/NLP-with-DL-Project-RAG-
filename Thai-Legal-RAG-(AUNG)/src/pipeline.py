from src.retrieval import Retriever
from src.generation import generate_answer


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
