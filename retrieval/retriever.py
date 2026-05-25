import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"
TOP_K = 10

class BiomedicalRetriever:
    def __init__(self, model_name=MODEL_NAME, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
        print("Loading retriever...")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Retriever ready. Index has {self.index.ntotal} vectors.")

    def retrieve(self, query, top_k=TOP_K):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["retrieval_score"] = float(score)
            results.append(chunk)
        return results

    def format_results(self, results):
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} (score: {r['retrieval_score']:.4f}) ---")
            print(f"Question : {r['question']}")
            print(f"Answer   : {r['answer']}")
            print(f"Context  : {r['context'][:200]}...")

if __name__ == "__main__":
    retriever = BiomedicalRetriever()
    test_query = "Does surgical treatment improve outcomes in elderly patients?"
    print(f"\nQuery: {test_query}")
    results = retriever.retrieve(test_query, top_k=5)
    retriever.format_results(results)
    print(f"\nRetrieved {len(results)} results.")
