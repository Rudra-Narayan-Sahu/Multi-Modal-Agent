import re
import json
import logfire
from typing import List, Dict, Any, Optional
from app.agents.llm import agent_llm
from app.memory.manager import memory_manager
from app.memory.schemas import MemoryCategoryEnum


EXTRACTION_PROMPT = """You are a Selective Memory Extraction Agent for an AI assistant.
Analyze this user query and assistant response:

User Query: {query}
Assistant Response: {response}

Your task is to identify durable, high-value facts that should be remembered across future sessions.
Valid Categories:
- "user_preferences": User's preferred coding style, response depth, preferred frameworks, language.
- "user_context": User's profession, role, background, goals.
- "project_context": Specific project name, tech stack, architecture decisions, requirements.
- "research_memory": Key research conclusions or findings the user discovered.
- "explicit_notes": Facts the user explicitly asked to remember (e.g. "remember that I use Python 3.13").

Rules:
1. Be SELECTIVE. Do NOT extract conversational noise, greetings, transient calculations, or temporary questions.
2. Never extract passwords, API keys, secrets, or confidential credentials.
3. Formulate each memory as a clear, standalone factual statement.
4. Estimate importance (1 = low, 3 = normal, 5 = critical requirement).

Respond with valid JSON:
{{
  "should_remember": true,
  "memories": [
    {{
      "category": "project_context",
      "content": "User is building a Multi-Agent RAG Assistant using FastAPI and Qdrant.",
      "importance": 4
    }}
  ]
}}
"""


class MemoryExtractor:
    """
    Extracts durable user facts, preferences, and project context from interactions
    and persists them into long-term memory.
    """

    @staticmethod
    def _fast_explicit_check(query: str) -> Optional[Dict[str, Any]]:
        """Detects explicit user commands to remember information."""
        q_lower = query.strip().lower()
        patterns = [
            (r"(?:please\s+)?remember\s+(?:that\s+)?(.+)", MemoryCategoryEnum.EXPLICIT_NOTES),
            (r"(?:keep\s+in\s+mind|note)\s+(?:that\s+)?(.+)", MemoryCategoryEnum.EXPLICIT_NOTES),
            (r"(?:my\s+preference\s+is|i\s+prefer)\s+(.+)", MemoryCategoryEnum.USER_PREFERENCES),
            (r"(?:i\s+am\s+working\s+on|i\s+am\s+building)\s+(.+)", MemoryCategoryEnum.PROJECT_CONTEXT),
        ]
        for pat, cat in patterns:
            match = re.search(pat, q_lower)
            if match:
                raw_fact = match.group(1).strip().capitalize()
                return {
                    "category": cat.value,
                    "content": raw_fact,
                    "importance": 4
                }
        return None

    def extract_and_save(
        self,
        user_id: str,
        query: str,
        response: str,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extracts durable facts and persists them for the user."""
        q_lower = query.strip().lower()

        # Fast discard: Trivial greetings, simple math, or short generic questions
        if len(query.strip()) < 8 or q_lower in {"hello", "hi", "hey", "thanks", "thank you"}:
            return []

        # 1. Fast explicit check
        explicit = self._fast_explicit_check(query)
        if explicit:
            saved = memory_manager.save_memory(
                user_id=user_id,
                content=explicit["content"],
                category=explicit["category"],
                importance=explicit["importance"],
                source_conversation_id=conversation_id
            )
            return [saved] if saved else []

        # 2. LLM Selective Extraction
        with logfire.span("Selective Memory Extraction", query=query):
            prompt = EXTRACTION_PROMPT.format(
                query=query[:600],
                response=response[:800]
            )
            data = agent_llm.call_json([
                {"role": "system", "content": "You are a selective memory extractor."},
                {"role": "user", "content": prompt}
            ])

            if not data.get("should_remember", False):
                return []

            saved_memories = []
            for mem_item in data.get("memories", []):
                content = mem_item.get("content", "").strip()
                cat = mem_item.get("category", "user_context")
                imp = mem_item.get("importance", 3)

                if content and memory_manager.validate_memory(content):
                    saved = memory_manager.save_memory(
                        user_id=user_id,
                        content=content,
                        category=cat,
                        importance=imp,
                        source_conversation_id=conversation_id
                    )
                    if saved:
                        saved_memories.append(saved)

            return saved_memories


memory_extractor = MemoryExtractor()

