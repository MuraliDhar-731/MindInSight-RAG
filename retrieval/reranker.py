from sentence_transformers import CrossEncoder
from typing import List, Dict

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N = 3

class BiomedicalReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker ready.")

    def rerank(self, query: str, results: List[Dict], top_n: int = TOP_N):
        if not results:
            return []

        # Build pairs of (query, passage) for cross-encoder
        pairs = [[query, r["context"]] for r in results]

        # Score each pair
        scores = self.model.predict(pairs)

        # Attach reranker scores
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by reranker score descending
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_n]

    def format_results(self, results):
        for i, r in enumerate(results):
            print(f"\n--- Reranked Result {i+1} ---")
            print(f"  Retrieval score : {r.get('retrieval_score', 0):.4f}")
            print(f"  Rerank score    : {r['rerank_score']:.4f}")
            print(f"  Question        : {r['question']}")
            print(f"  Answer          : {r['answer']}")
            print(f"  Context         : {r['context'][:200]}...")


if __name__ == "__main__":
    from retrieval.retriever import BiomedicalRetriever

    retriever = BiomedicalRetriever()
    reranker = BiomedicalReranker()

    query = "Does surgical treatment improve outcomes in elderly patients?"
    print(f"\nQuery: {query}")

    # Step 1 — retrieve top 10
    results = retriever.retrieve(query, top_k=10)
    print(f"\nRetrieved {len(results)} results.")

    # Step 2 — rerank to top 3
    reranked = reranker.rerank(query, results, top_n=3)
    print(f"\nTop {len(reranked)} after reranking:")
    reranker.format_results(reranked)
