import logfire
from typing import Dict, Any, Optional
from app.config import settings
from app.services.retrieval.pipeline import AdvancedRetriever, RetrievalConfig


def handle_data_analysis(
    query: str,
    retriever: Optional[AdvancedRetriever] = None,
    conversation_history: str = ""
) -> Dict[str, Any]:
    """Handles quantitative evaluation, metrics comparison, and analytical questions."""
    retriever = retriever or AdvancedRetriever()

    with logfire.span("Handler: Data Analysis", query=query):
        # Retrieve relevant data/metrics from documents
        config = RetrievalConfig(
            search_type="hybrid",
            enable_reranker=True,
            top_k=5
        )
        retrieval_data = retriever.retrieve_context_with_citations(query=query, config=config)
        context = retrieval_data.get("context", "")
        citations = retrieval_data.get("citations", "")

        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = (
            "You are a quantitative data analyst. "
            "Analyze the data, statistics, benchmark metrics, or formulas present in the context. "
            "Provide comparative tables in Markdown, highlighting numerical differences, "
            "percentage improvements, and statistical implications."
        )

        user_content = (
            f"Context Data:\n{context}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"Analytical Query: {query}"
        )

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=900
        )
        return {
            "answer": response.choices[0].message.content or "",
            "citations": citations,
            "documents": retrieval_data.get("documents", [])
        }

