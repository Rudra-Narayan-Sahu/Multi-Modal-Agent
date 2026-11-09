import os
import json
import logfire
from typing import List, Dict, Any, Optional
from groq import Groq
from app.config import settings


class AgentLLMClient:
    """
    Centralized LLM execution client for Multi-Agent nodes.
    Supports Groq (primary + fallback key) and Google Gemini fallback.
    """

    def __init__(self):
        self._groq_client = None
        self._init_client()

    def _init_client(self):
        api_key = settings.GROQ_API_KEY or settings.GROQ_FALLBACK_API_KEY
        if api_key:
            try:
                self._groq_client = Groq(api_key=api_key)
            except Exception as e:
                logfire.warning(f"Failed to initialize Groq client: {e}")

    def call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> str:
        """Execute chat completion with automatic fallback and observability."""
        target_model = model or settings.GROQ_MODEL

        # Attempt 1: Groq API
        if self._groq_client:
            with logfire.span("Agent LLM Call", model=target_model, json_mode=json_mode):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": target_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = self._groq_client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                    return content.strip()
                except Exception as e:
                    logfire.warning(f"Groq LLM call failed ({e}), attempting fallback...")

        # Attempt 2: Fallback Groq key
        if settings.GROQ_FALLBACK_API_KEY and settings.GROQ_FALLBACK_API_KEY != settings.GROQ_API_KEY:
            try:
                fallback_client = Groq(api_key=settings.GROQ_FALLBACK_API_KEY)
                response = fallback_client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if json_mode else None
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                logfire.warning(f"Fallback Groq key failed: {e}")

        # Attempt 3: Google Gemini fallback
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Combine messages into prompt
                prompt_parts = []
                for m in messages:
                    prompt_parts.append(f"{m['role'].upper()}: {m['content']}")
                full_prompt = "\n\n".join(prompt_parts)
                if json_mode:
                    full_prompt += "\n\nRespond strictly with valid JSON only."

                res = gemini_model.generate_content(full_prompt)
                return res.text.strip()
            except Exception as ge:
                logfire.error(f"Gemini fallback failed: {ge}")

        # If all failed or no key
        if json_mode:
            return json.dumps({"error": "LLM providers unreachable", "status": "failed"})
        return "Error: Unable to connect to LLM provider. Please check API keys."

    def call_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Calls LLM in JSON mode and parses the result safely."""
        raw = self.call(messages, json_mode=True, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt regex extraction if JSON was surrounded by markdown fences
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"raw_response": raw, "error": "Invalid JSON returned"}


agent_llm = AgentLLMClient()

