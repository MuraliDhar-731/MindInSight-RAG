from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 256

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

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_parts = []
        for i, ctx in enumerate(contexts):
            context_parts.append(f"Study {i+1}: {ctx['context'][:250]}")
        context_text = " ".join(context_parts)

        prompt = (
            f"Based on these medical studies, answer the question.\n"
            f"Studies: {context_text}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt

    def check_ambiguity(self, answer: str, contexts: List[Dict]) -> Dict:
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

    def generate_followup_question(self, query: str) -> str:
        followups = [
            "Could you provide more details about the patient's age or medical history?",
            "What type of surgical procedure are you referring to?",
            "Are there specific symptoms or conditions you are concerned about?",
            "What is the severity or duration of the condition?",
        ]
        # Simple keyword matching for follow-up
        query_lower = query.lower()
        if "age" in query_lower or "elder" in query_lower or "old" in query_lower:
            return "Could you specify the patient's age range or any comorbidities?"
        elif "surgery" in query_lower or "surgical" in query_lower:
            return "What type of surgical procedure are you asking about specifically?"
        elif "treatment" in query_lower:
            return "What treatment options have already been considered?"
        return followups[0]

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
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
                no_repeat_ngram_size=3,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ambiguity check
        ambiguity = self.check_ambiguity(answer, contexts)

        # Follow-up question if ambiguous
        followup = None
        if ambiguity["is_ambiguous"]:
            followup = self.generate_followup_question(query)

        # Citations
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
        print(f"\nCONFIDENCE: {result['ambiguity']['confidence_label']} "
              f"({result['ambiguity']['confidence']:.0%})")
        if result["followup_question"]:
            print(f"\nCLARIFICATION NEEDED:\n{result['followup_question']}")
        print(f"\nSOURCES ({result['num_sources']}):")
        for c in result["citations"]:
            print(f"  [{c['source']}] {c['question']}")
            print(f"          → {c['answer'].upper()} (rerank: {c['rerank_score']})")
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
        "What are the risks of anesthesia in elderly patients?",
    ]

    for query in queries:
        print(f"\nProcessing: {query}")
        results = retriever.retrieve(query, top_k=10)
        reranked = reranker.rerank(query, results, top_n=3)
        output = generator.generate(query, reranked)
        generator.format_output(output)
