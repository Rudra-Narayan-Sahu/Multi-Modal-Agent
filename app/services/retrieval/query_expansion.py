import logfire
from typing import List, Optional
from app.config import settings


def _get_groq_client():
    """Lazily initialize Groq client."""
    if not settings.GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=settings.GROQ_API_KEY)
    except Exception as e:
        logfire.warning(f"Could not initialize Groq client: {e}")
        return None


def generate_sub_queries(query: str, num_queries: int = 3) -> List[str]:
    """
    Generate alternative query formulations from different angles to maximize retrieval recall.
    Falls back to heuristic query variations if LLM is unavailable.
    """
    client = _get_groq_client()
    if client:
        with logfire.span("Query Expansion (LLM)", original_query=query):
            try:
                system_prompt = (
                    f"You are an AI search query optimization assistant. "
                    f"Generate {num_queries} diverse search query variations for the user query. "
                    f"Each variation must capture different aspects, terminology, or synonyms. "
                    f"Output ONLY one query per line, without numbering, bullets, or extra text."
                )
                response = client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                content = response.choices[0].message.content or ""
                lines = [line.strip().lstrip("0123456789.-* ") for line in content.splitlines() if line.strip()]
                if lines:
                    return lines[:num_queries]
            except Exception as e:
                logfire.warning(f"LLM query expansion failed: {e}. Using heuristic fallback.")

    # Rule-based fallback if LLM is offline
    variations = [query]
    words = query.split()
    if len(words) > 3:
        variations.append(" ".join(words[:len(words)//2]))
        variations.append(" ".join(words[len(words)//2:]))
    return variations[:num_queries]


def generate_hyde_query(query: str) -> str:
    """
    Hypothetical Document Embeddings (HyDE):
    Generates a hypothetical document answering the query to better match document embeddings in vector space.
    """
    client = _get_groq_client()
    if client:
        with logfire.span("HyDE Generation", original_query=query):
            try:
                prompt = (
                    "Write a short, factual passage (2-3 sentences) answering the following question. "
                    "Use precise technical and domain vocabulary as found in research papers:\n\n"
                    f"Question: {query}"
                )
                response = client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a scientific research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                hypo_doc = response.choices[0].message.content or ""
                if hypo_doc.strip():
                    return hypo_doc.strip()
            except Exception as e:
                logfire.warning(f"HyDE generation failed: {e}")

    return query

