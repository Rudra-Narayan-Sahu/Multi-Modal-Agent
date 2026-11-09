import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logfire

from app.database.connection import get_db_session
from app.database.models import LongTermMemory


class MemoryRetriever:
    """
    Retrieves and ranks relevant long-term memories for an authenticated user
    to provide personalized, context-aware agent interactions.
    """

    @staticmethod
    def _tokenize(text: str) -> set:
        """Extract alphanumeric lowercase tokens."""
        return set(re.findall(r"\b\w{3,}\b", text.lower()))

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves memories matching the user query, ranked by keyword relevance
        and importance weighting.
        """
        query_tokens = self._tokenize(query)

        with get_db_session() as db:
            q = db.query(LongTermMemory).filter(LongTermMemory.user_id == user_id)
            if categories:
                q = q.filter(LongTermMemory.memory_category.in_(categories))
            all_memories = q.all()

            if not all_memories:
                return []

            scored = []
            for mem in all_memories:
                mem_tokens = self._tokenize(mem.memory_content)
                overlap = len(query_tokens.intersection(mem_tokens))
                
                # Base score from importance (1-5 scaled to 0.2 - 1.0)
                importance_score = mem.importance / 5.0
                
                # Relevance score combines keyword overlap and importance
                relevance_score = (overlap * 0.5) + (importance_score * 0.5)

                # Prioritize explicit notes or high-importance project memories even with slight overlap
                if overlap > 0 or mem.importance >= 4:
                    scored.append((relevance_score, mem))

            # Sort descending by score
            scored.sort(key=lambda x: x[0], reverse=True)
            top_memories = scored[:limit]

            results = []
            for score, m in top_memories:
                results.append({
                    "id": m.id,
                    "category": m.memory_category,
                    "content": m.memory_content,
                    "importance": m.importance,
                    "relevance_score": round(score, 3),
                    "updated_at": m.updated_at.isoformat()
                })
            return results

    def format_memories_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Formats retrieved memories for injection into the Supervisor Agent's context."""
        if not memories:
            return ""

        lines = ["### Persistent User Context & Long-Term Memories:"]
        for m in memories:
            cat = m.get("category", "Note").replace("_", " ").title()
            lines.append(f"- **[{cat}]**: {m.get('content')}")
        return "\n".join(lines)


memory_retriever = MemoryRetriever()

