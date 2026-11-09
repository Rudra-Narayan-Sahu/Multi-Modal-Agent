import logfire
from typing import Dict, Any
from app.config import settings


def handle_conversational(query: str, conversation_history: str = "") -> Dict[str, Any]:
    """Handles polite chitchat, greetings, pleasantries, and meta-assistant questions."""
    with logfire.span("Handler: Conversational", query=query):
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = (
            "You are the Multi-Agent RAG Assistant, an enterprise AI assistant specializing in "
            "document analysis, multimodal retrieval, and technical question answering. "
            "Respond politely, warmly, and concisely. If greeted, greet back and let the user know "
            "how you can assist them with document queries, summarization, data analysis, or technical questions."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.append({"role": "user", "content": f"History:\n{conversation_history}"})
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return {
            "answer": response.choices[0].message.content or "",
            "citations": None,
            "documents": []
        }

