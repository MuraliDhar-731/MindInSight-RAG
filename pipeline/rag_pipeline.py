from retrieval.retriever import BiomedicalRetriever
from retrieval.reranker import BiomedicalReranker
from generation.generator import BiomedicalGenerator

class MedInSightPipeline:
    def __init__(self):
        print("Initialising MedInSight-RAG pipeline...")
        self.retriever = BiomedicalRetriever()
        self.reranker = BiomedicalReranker()
        self.generator = BiomedicalGenerator()
        print("Pipeline ready.\n")

    def run(self, query: str, top_k: int = 10, top_n: int = 3) -> dict:
        # Step 1 — Retrieve
        results = self.retriever.retrieve(query, top_k=top_k)

        if not results:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "ambiguity": {"is_ambiguous": False, "confidence_label": "LOW", "confidence": 0},
                "followup_question": None,
                "citations": [],
                "num_sources": 0,
            }

        # Step 2 — Rerank
        reranked = self.reranker.rerank(query, results, top_n=top_n)

        # Step 3 — Generate
        output = self.generator.generate(query, reranked)

        return output

    def format(self, result: dict) -> str:
        lines = []
        lines.append(f"**Answer:** {result['answer']}")
        lines.append(f"**Confidence:** {result['ambiguity']['confidence_label']} ({result['ambiguity']['confidence']:.0%})")

        if result.get("followup_question"):
            lines.append(f"\n**Clarification needed:** {result['followup_question']}")

        lines.append(f"\n**Sources ({result['num_sources']}):**")
        for c in result["citations"]:
            lines.append(f"  [{c['source']}] {c['question']} → {c['answer'].upper()}")

        return "\n".join(lines)


if __name__ == "__main__":
    pipeline = MedInSightPipeline()

    query = "Does surgical treatment improve outcomes in elderly patients?"
    result = pipeline.run(query)
    print(pipeline.format(result))
