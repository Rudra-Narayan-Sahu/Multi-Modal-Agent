from typing import Dict, Any


def handle_fallback(
    query: str,
    confidence: float = 0.0,
    reasoning: str = ""
) -> Dict[str, Any]:
    """
    Handles ambiguous, low-confidence, or out-of-scope requests by offering
    polite clarification and highlighting available capabilities.
    """
    answer = (
        f"I'm not entirely certain how to best handle your request (classification confidence: {confidence * 100:.1f}%).\n\n"
        f"**Could you please clarify what you would like to do?**\n\n"
        f"Here are a few ways I can help:\n"
        f"- 📄 **Document QA**: Ask specific questions about your uploaded documents or research papers.\n"
        f"- 📝 **Summarization**: Request an executive summary or key findings of any document.\n"
        f"- 📊 **Data & Metrics Analysis**: Extract or compare benchmark results and tables.\n"
        f"- 💻 **Technical & Code**: Request Python implementations, algorithms, or technical explanations.\n"
        f"- 🌐 **General Knowledge**: Ask general factual and scientific questions."
    )
    return {
        "answer": answer,
        "citations": None,
        "documents": [],
        "is_fallback": True,
        "reasoning": reasoning
    }

