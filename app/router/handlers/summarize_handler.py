import logfire
from typing import Dict, Any, Optional
from app.config import settings
from app.services.retrieval.pipeline import AdvancedRetriever, RetrievalConfig


def handle_summarization(
    query: str,
    retriever: Optional[AdvancedRetriever] = None,
    conversation_history: str = ""
) -> Dict[str, Any]:
    """Generates structured summaries of documents or topics."""
    retriever = retriever or AdvancedRetriever()

    with logfire.span("Handler: Document Summarization", query=query):
        # Retrieve broader context for summarization (top 8 chunks)
        config = RetrievalConfig(
            search_type="hybrid",
            enable_reranker=True,
            top_k=8,
            max_context_length=8000
        )
        retrieval_data = retriever.retrieve_context_with_citations(query=query, config=config)
        context = retrieval_data.get("context", "")
        citations = retrieval_data.get("citations", "")

        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = (
            "You are an executive research summarizer. "
            "Synthesize an organized, high-density summary of the provided documents. "
            "Structure your response with the following sections:\n"
            "1. **Executive Overview**\n"
            "2. **Key Findings & Contributions**\n"
            "3. **Methodology / Technical Architecture**\n"
            "4. **Evaluation & Results (if mentioned)**\n"
            "5. **Takeaway & Conclusions**\n"
            "Cite sources where applicable."
        )

        user_content = (
            f"Reference Documents:\n{context}\n\n"
            f"User Summarization Request: {query}"
        )

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "citations": citations,
            "documents": retrieval_data.get("documents", [])
        }

