from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128

class BiomedicalGenerator:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading generator: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Generator ready.")

    def extract_answer(self, query: str, contexts: List[Dict]) -> str:
        """
        Pick the best matching context by finding which study question
        is most similar to the user query, then extract from that context.
        """
        query_words = set(query.lower().split())

        best_idx = 0
        best_overlap = -1
        for i, ctx in enumerate(contexts):
            study_words = set(ctx["question"].lower().split())
            overlap = len(query_words & study_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        best = contexts[best_idx]
        context = best["context"]
        verdict = best["answer"]

        # Extract meaningful sentences
        sentences = []
        for s in context.split("."):
            s = s.strip()
            if len(s) > 40:
                sentences.append(s)

        if not sentences:
            return f"Based on the available evidence, the answer is: **{verdict.upper()}**."

        # Take the 2 most relevant sentences
        summary = ". ".join(sentences[:2]) + "."
        return (
            f"{summary}\n\n"
            f"**Evidence conclusion: {verdict.upper()}**"
        )

    def check_ambiguity(self, contexts: List[Dict]) -> Dict:
        answers = [c["answer"].lower() for c in contexts]
        yes_count = answers.count("yes")
        no_count = answers.count("no")
        maybe_count = answers.count("maybe")
        total = len(answers)
        dominant = max(yes_count, no_count, maybe_count)
        confidence = dominant / total if total > 0 else 0
        is_ambiguous = confidence < 0.6 or maybe_count > 0
        if confidence >= 0.8:
            label = "HIGH"
        elif confidence >= 0.6:
            label = "MEDIUM"
        else:
            label = "LOW"
        return {
            "is_ambiguous": is_ambiguous,
            "confidence": confidence,
            "confidence_label": label,
            "yes_count": yes_count,
            "no_count": no_count,
            "maybe_count": maybe_count,
        }

    def generate_followup(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["age", "elder", "old", "geriatric"]):
            return "Could you specify the patient's age range or any comorbidities?"
        elif any(w in q for w in ["surgery", "surgical", "operation"]):
            return "What type of surgical procedure are you asking about specifically?"
        elif any(w in q for w in ["treatment", "therapy", "chemo"]):
            return "What treatment options have already been considered?"
        elif any(w in q for w in ["pain", "symptom", "feel", "hurt"]):
            return "How long have you had these symptoms and what is the severity?"
        return "Could you provide more clinical details to improve the answer?"

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
        answer = self.extract_answer(query, contexts)
        ambiguity = self.check_ambiguity(contexts)
        followup = self.generate_followup(query) if ambiguity["is_ambiguous"] else None

        citations = []
        for i, ctx in enumerate(contexts):
            citations.append({
                "source": i + 1,
                "question": ctx["question"],
                "answer": ctx["answer"],
                "rerank_score": round(ctx.get("rerank_score", 0), 4),
            })

        return {
            "query": query,
            "answer": answer,
            "ambiguity": ambiguity,
            "followup_question": followup,
            "citations": citations,
            "num_sources": len(contexts),
        }

    def format_output(self, result: Dict):
        print("\n" + "="*60)
        print(f"QUERY: {result['query']}")
        print("="*60)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nCONFIDENCE: {result['ambiguity']['confidence_label']} ({result['ambiguity']['confidence']:.0%})")
        if result["followup_question"]:
            print(f"\nCLARIFICATION: {result['followup_question']}")
        print(f"\nSOURCES ({result['num_sources']}):")
        for c in result["citations"]:
            print(f"  [{c['source']}] {c['question']} → {c['answer'].upper()}")
        print("="*60)


if __name__ == "__main__":
    from retrieval.retriever import BiomedicalRetriever
    from retrieval.reranker import BiomedicalReranker

    retriever = BiomedicalRetriever()
    reranker = BiomedicalReranker()
    generator = BiomedicalGenerator()

    queries = [
        "Does surgical treatment improve outcomes in elderly patients?",
        "Is chemotherapy effective for elderly cancer patients?",
        "What are the effects of regional anesthesia in geriatric patients?",
        "Does laparoscopic surgery improve quality of life?",
    ]

    for query in queries:
        results = retriever.retrieve(query, top_k=10)
        reranked = reranker.rerank(query, results, top_n=3)
        output = generator.generate(query, reranked)
        generator.format_output(output)
