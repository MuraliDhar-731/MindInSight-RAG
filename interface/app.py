import gradio as gr
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import MedInSightPipeline

print("Starting MedInSight-RAG...")
pipeline = MedInSightPipeline()

def answer_question(query, history):
    if not query.strip():
        return history, "", ""

    result = pipeline.run(query)

    answer_text = result["answer"]
    confidence = result["ambiguity"]["confidence_label"]
    confidence_pct = f"{result['ambiguity']['confidence']:.0%}"

    citations_text = "### 📚 Evidence Sources\n\n"
    for c in result["citations"]:
        emoji = "✅" if c["answer"].lower() == "yes" else "❌" if c["answer"].lower() == "no" else "⚠️"
        citations_text += f"{emoji} **[{c['source']}]** {c['question']}\n"
        citations_text += f"   → **{c['answer'].upper()}** | Score: `{c['rerank_score']}`\n\n"

    conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
    confidence_md = f"### Confidence\n\n{conf_color} **{confidence}** ({confidence_pct})"

    if result.get("followup_question"):
        answer_text += f"\n\n💬 *{result['followup_question']}*"

    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer_text}
    ]
    return history, citations_text, confidence_md


def clear_all():
    return [], "### 📚 Evidence Sources\n\nSources will appear here.", "### Confidence\n\nAsk a question to see confidence."


with gr.Blocks(title="MedInSight-RAG") as demo:

    gr.HTML("""
    <div style='text-align:center; padding:20px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius:10px; margin-bottom:20px'>
        <h1 style='color:#00d4ff'>🧬 MedInSight-RAG</h1>
        <h3 style='color:#ccc'>Biomedical Question Answering with Evidence-Grounded Responses</h3>
        <p style='color:#aaa'>Powered by BioBERT · Cross-Encoder Reranking · FLAN-T5</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                layout="bubble",
            )
            with gr.Row():
                query_input = gr.Textbox(
                    label="Your biomedical question",
                    placeholder="e.g. Does surgical treatment improve outcomes in elderly patients?",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("🔍 Ask", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary")

            gr.Examples(
                examples=[
                    ["Does surgical treatment improve outcomes in elderly patients?"],
                    ["Is chemotherapy effective for elderly cancer patients?"],
                    ["What are the effects of regional anesthesia in geriatric patients?"],
                    ["Does laparoscopic surgery improve quality of life?"],
                    ["Are there cardiovascular risks in elderly surgical patients?"],
                ],
                inputs=query_input,
                label="💡 Example Questions"
            )

        with gr.Column(scale=1):
            confidence_output = gr.Markdown(
                value="### Confidence\n\nAsk a question to see confidence."
            )
            gr.HTML("<hr style='border-color:#333'>")
            citations_output = gr.Markdown(
                value="### 📚 Evidence Sources\n\nSources will appear here."
            )

    gr.HTML("""
    <div style='font-size:12px; color:#888; text-align:center; padding:10px; margin-top:10px'>
        ⚠️ MedInSight-RAG is for <strong>research and educational purposes only</strong>.
        Always consult a qualified medical professional for health decisions.
    </div>
    """)

    submit_btn.click(
        fn=answer_question,
        inputs=[query_input, chatbot],
        outputs=[chatbot, citations_output, confidence_output],
    ).then(lambda: "", outputs=query_input)

    query_input.submit(
        fn=answer_question,
        inputs=[query_input, chatbot],
        outputs=[chatbot, citations_output, confidence_output],
    ).then(lambda: "", outputs=query_input)

    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, citations_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
