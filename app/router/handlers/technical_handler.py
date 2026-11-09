import logfire
from typing import Dict, Any
from app.config import settings


def handle_technical_code(query: str, conversation_history: str = "") -> Dict[str, Any]:
    """Handles technical programming, coding, debugging, and software architecture questions."""
    with logfire.span("Handler: Technical Code", query=query):
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = (
            "You are an expert Senior Software Engineer and Systems Architect. "
            "Provide production-ready, clean, well-commented code, algorithmic explanations, "
            "and best practices. Include type hints, edge-case considerations, and complexity analysis where applicable."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.append({"role": "user", "content": f"Context:\n{conversation_history}"})
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        return {
            "answer": response.choices[0].message.content or "",
            "citations": None,
            "documents": []
        }

