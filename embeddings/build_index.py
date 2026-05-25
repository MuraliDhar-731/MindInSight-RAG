import json
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
DATA_PATH = "data/pubmedqa.json"
INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"
BATCH_SIZE = 32

def load_data(path=DATA_PATH):
    with open(path, "r") as f:
        return json.load(f)

def chunk_documents(samples):
    chunks = []
    for item in samples:
        chunks.append({
            "id": item["id"],
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
            "long_answer": item["long_answer"],
            "text": f"Question: {item['question']}\nContext: {item['context']}"
        })
    return chunks

def build_faiss_index(chunks, model_name=MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index, embeddings

def save_index(index, chunks, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {index_path}")
    print(f"Chunks saved to {chunks_path}")

def load_index(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

if __name__ == "__main__":
    samples = load_data()
    print(f"Loaded {len(samples)} samples.")
    chunks = chunk_documents(samples)
    print(f"Created {len(chunks)} chunks.")
    index, embeddings = build_faiss_index(chunks)
    save_index(index, chunks)
    print(f"Done. Total vectors: {index.ntotal}")
