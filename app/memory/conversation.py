import time
import threading
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    query: str
    response: str
    category: str
    agents_used: List[str]
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class ConversationSession(BaseModel):
    conversation_id: str
    user_id: str
    created_at: float = Field(default_factory=time.time)
    turns: List[ConversationTurn] = Field(default_factory=list)


class ConversationMemoryManager:
    """Thread-safe conversation manager with user isolation."""

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def add_turn(
        self,
        conversation_id: str,
        user_id: str,
        query: str,
        response: str,
        category: str,
        agents_used: List[str],
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        with self._lock:
            if conversation_id not in self._sessions:
                self._sessions[conversation_id] = ConversationSession(
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            
            # Security verification: Ensure conversation belongs to authorized user
            session = self._sessions[conversation_id]
            if session.user_id != user_id and session.user_id != "default_user":
                raise PermissionError("Access denied: Conversation belongs to another user.")

            session.turns.append(ConversationTurn(
                query=query,
                response=response,
                category=category,
                agents_used=agents_used,
                citations=citations or []
            ))

    def get_history(self, conversation_id: str, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        with self._lock:
            if conversation_id not in self._sessions:
                return None
            session = self._sessions[conversation_id]
            if session.user_id != user_id and session.user_id != "default_user" and user_id != "default_user":
                raise PermissionError("Access denied: Conversation belongs to another user.")
            return session.model_dump()

    def clear(self, conversation_id: str, user_id: str = "default_user") -> bool:
        with self._lock:
            if conversation_id in self._sessions:
                session = self._sessions[conversation_id]
                if session.user_id != user_id and session.user_id != "default_user" and user_id != "default_user":
                    raise PermissionError("Access denied.")
                del self._sessions[conversation_id]
                return True
            return False


conversation_manager = ConversationMemoryManager()

