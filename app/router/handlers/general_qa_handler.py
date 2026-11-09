import logfire
from typing import Dict, Any
from app.config import settings


def handle_general_qa(query: str, conversation_history: str = "") -> Dict[str, Any]:
    """Handles general knowledge and conceptual questions directly via LLM."""
    with logfire.span("Handler: General QA", query=query):
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = (
            "You are a helpful and knowledgeable AI assistant. "
            "Answer the user's question clearly, thoroughly, and factually. "
            "Use clear formatting and explanations."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.append({"role": "user", "content": f"Previous conversation context:\n{conversation_history}"})
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        return {
            "answer": response.choices[0].message.content or "",
            "citations": None,
            "documents": []
        }

