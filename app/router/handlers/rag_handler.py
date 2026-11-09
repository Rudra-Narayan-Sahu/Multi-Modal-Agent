import logfire
from typing import Dict, Any, Optional
from app.config import settings
from app.services.retrieval.pipeline import AdvancedRetriever, RetrievalConfig


def handle_document_rag(
    query: str,
    retriever: Optional[AdvancedRetriever] = None,
    conversation_history: str = ""
) -> Dict[str, Any]:
    """Handles document-based question answering using Advanced RAG (Hybrid + FlashRank)."""
    retriever = retriever or AdvancedRetriever()

    with logfire.span("Handler: Document RAG", query=query):
        # 1. Retrieve top-5 hybrid & reranked context
        config = RetrievalConfig(
            search_type="hybrid",
            enable_reranker=True,
            top_k=5
        )
        retrieval_data = retriever.retrieve_context_with_citations(query=query, config=config)
        context = retrieval_data.get("context", "")
        citations = retrieval_data.get("citations", "")

        if not context.strip():
            return {
                "answer": "I searched the internal documents but could not find relevant passages matching your question.",
                "citations": "No matching documents found.",
                "documents": []
            }

        # 2. Synthesize answer with Groq LLM
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = (
            "You are a helpful, expert Enterprise RAG Assistant. "
            "Answer the user's question accurately using ONLY the provided document context. "
            "Cite document sources using bracket numbers (e.g. [1], [2]) when asserting facts. "
            "If the context does not contain the answer, state that clearly without speculating."
        )

        user_content = (
            f"Document Context:\n{context}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"User Question: {query}"
        )

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=800
        )
        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "citations": citations,
            "documents": retrieval_data.get("documents", []),
            "retrieval_count": retrieval_data.get("count", 0)
        }

