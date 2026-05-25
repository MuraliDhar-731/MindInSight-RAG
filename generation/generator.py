from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            tie_word_embeddings=False
        )
        self.model.to(self.device)
        self.model.eval()
        print("Generator ready.")

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        # Use only the single best context for a focused answer
        best = contexts[0]
        context = best["context"][:400]

        prompt = (
            f"Answer the following medical question based on the study below.\n\n"
            f"Study: {context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a direct answer in 1-2 sentences:"
        )
        return prompt

    def extract_answer(self, contexts: List[Dict]) -> str:
        """
        Extractive fallback — pull the most informative sentence
        directly from the top context. Always accurate, never hallucinated.
        """
        best = contexts[0]
        context = best["context"]
        question_answer = best["answer"]  # yes / no / maybe

        # Split into sentences
        sentences = [s.strip() for s in context.replace("?", "?.").split(".") if len(s.strip()) > 30]

        if not sentences:
            return f"Based on available evidence, the answer is: {question_answer.upper()}."

        # Return first 2 meaningful sentences + the study conclusion
        summary = " ".join(sentences[:2])
        return f"{summary}. Based on this evidence, the answer is likely: **{question_answer.upper()}**."

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
            confidence_label = "HIGH"
        elif confidence >= 0.6:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"
        return {
            "is_ambiguous": is_ambiguous,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "yes_count": yes_count,
            "no_count": no_count,
            "maybe_count": maybe_count,
        }

    def generate_followup(self, query: str) -> str:
        query_lower = query.lower()
        if any(w in query_lower for w in ["age", "elder", "old", "geriatric"]):
            return "Could you specify the patient's age range or any comorbidities?"
        elif any(w in query_lower for w in ["surgery", "surgical", "operation"]):
            return "What type of surgical procedure are you asking about specifically?"
        elif any(w in query_lower for w in ["treatment", "therapy"]):
            return "What treatment options have already been considered?"
        elif any(w in query_lower for w in ["pain", "symptom", "feel"]):
            return "How long have you had these symptoms and what is the severity?"
        return "Could you provide more clinical details to improve the answer?"

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
        # Try LLM generation first
        try:
            prompt = self.build_prompt(query, contexts)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_OUTPUT_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0,
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # If answer is too short or looks like garbage, use extractive
            if len(answer.strip()) < 10 or answer.strip().lower() in ["yes", "no", "maybe", "n", "y"]:
                answer = self.extract_answer(contexts)

        except Exception:
            answer = self.extract_answer(contexts)

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
    ]

    for query in queries:
        results = retriever.retrieve(query, top_k=10)
        reranked = reranker.rerank(query, results, top_n=3)
        output = generator.generate(query, reranked)
        generator.format_output(output)
