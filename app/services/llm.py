import os
import logfire
from groq import Groq
from app.config import settings
from app.services.retrieval.retriever import retrieve_documents

def _get_groq_client(api_key: str = None) -> Groq:
    key = api_key or settings.GROQ_API_KEY or settings.GROQ_FALLBACK_API_KEY
    if not key:
        raise ValueError("GROQ_API_KEY is not configured in .env")
    return Groq(api_key=key)

SYSTEM_PROMPT = """You are an intelligent, precise AI assistant specializing in Retrieval-Augmented Generation (RAG).
Your job is to answer the user's questions truthfully and informatively based on the retrieved context documents provided below.

Guidelines:
1. Prioritize facts and data given in the Context. If the context contains the answer, synthesize it clearly and cite the relevant document(s).
2. If the retrieved context is relevant, use it directly to form your answer. If the context is only partially relevant or does not contain the answer, answer helpfully while explicitly noting the limitations of the provided documents.
3. Use clean Markdown formatting with headers, bullet points, and code blocks where applicable.
4. Always be objective, accurate, and concise.
"""

def generate_rag_response(
    query: str,
    history: list[dict] = None,
    top_k: int = 5
) -> dict:
    """
    1. Embeds query and retrieves top_k similar document chunks from Qdrant.
    2. Constructs RAG prompt with context and conversation history.
    3. Calls Groq LLM to generate grounded response.
    4. Returns answer and source chunk citations.
    """
    if not query or not query.strip():
        return {
            "answer": "Please provide a query or question.",
            "sources": [],
            "model_used": "none"
        }

    # 1. Retrieve matching chunks from Qdrant
    documents = retrieve_documents(query=query, top_k=top_k)

    # 2. Build Context String
    if documents:
        context_parts = []
        for i, doc in enumerate(documents, start=1):
            source_info = f"[Doc {i}: {doc.get('file_name', 'Unknown')} (Chunk {doc.get('chunk_index', 0)}, Score: {doc.get('score', 0)})]"
            context_parts.append(f"{source_info}\n{doc.get('text', '')}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant documents found in the database."

    # 3. Construct Message Chain
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add historical messages if provided (max last 6 messages to keep context focused)
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

    # Add user turn with retrieved context
    user_prompt = f"""--- RETRIEVED CONTEXT ---
{context_text}
--- END OF CONTEXT ---

User Question: {query}"""

    messages.append({"role": "user", "content": user_prompt})

    # 4. Attempt generation with fallback models and keys
    candidates = [
        (settings.GROQ_API_KEY, settings.GROQ_MODEL),
        (settings.GROQ_API_KEY, "openai/gpt-oss-120b"),
        (settings.GROQ_API_KEY, "openai/gpt-oss-20b"),
        (settings.GROQ_API_KEY, "qwen/qwen3.6-27b"),
        (settings.GROQ_FALLBACK_API_KEY, "openai/gpt-oss-120b"),
        (settings.GROQ_FALLBACK_API_KEY, "openai/gpt-oss-20b"),
    ]

    # Deduplicate candidate pairs
    seen = set()
    unique_candidates = []
    for key, model in candidates:
        if key and model and (key, model) not in seen:
            seen.add((key, model))
            unique_candidates.append((key, model))

    last_error = None
    answer = None
    model_used = None

    for api_key, model_name in unique_candidates:
        try:
            client = _get_groq_client(api_key=api_key)
            with logfire.span("Groq RAG LLM Completion", model=model_name):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                )
            answer = response.choices[0].message.content
            model_used = model_name
            break
        except Exception as e:
            logfire.warning(f"Groq generation failed with model {model_name}: {e}")
            last_error = e
            continue

    if answer is None:
        answer = f"⚠️ Could not generate response from LLM. Error details: {last_error}"
        model_used = "error"

    # 5. Format sources for frontend presentation
    sources = [
        {
            "id": doc.get("id"),
            "file_name": doc.get("file_name", "Unknown"),
            "score": doc.get("score", 0.0),
            "chunk_index": doc.get("chunk_index", 0),
            "source_type": doc.get("source_type", "doc"),
            "text": doc.get("text", "")
        }
        for doc in documents
    ]

    return {
        "answer": answer,
        "sources": sources,
        "model_used": model_used
    }
