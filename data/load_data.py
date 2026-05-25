# data/load_data.py
from datasets import load_dataset
import json
import os

def load_pubmedqa(split="train", max_samples=1000):
    """
    Load PubMedQA dataset from HuggingFace.
    Returns list of dicts with question, context, answer.
    """
    print(f"Loading PubMedQA ({split} split)...")
    
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        
        # Flatten context passages into one string
        context = " ".join(item["context"]["contexts"])
        
        samples.append({
            "id": item["pubid"],
            "question": item["question"],
            "context": context,
            "answer": item["final_decision"],        # yes / no / maybe
            "long_answer": item["long_answer"],       # full explanation
            "labels": item["context"]["labels"],
        })
    
    print(f"Loaded {len(samples)} samples.")
    return samples


def save_data(samples, path="data/pubmedqa.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved to {path}")


def load_saved_data(path="data/pubmedqa.json"):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    samples = load_pubmedqa(max_samples=1000)
    save_data(samples)
    
    # Quick check
    print("\nSample entry:")
    print(f"  Question : {samples[0]['question']}")
    print(f"  Answer   : {samples[0]['answer']}")
    print(f"  Context  : {samples[0]['context'][:200]}...")