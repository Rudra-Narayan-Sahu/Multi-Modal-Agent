import time
from typing import List, Dict, Any, Optional
from threading import Lock


class ConversationTurn:
    def __init__(self, role: str, content: str, intent: Optional[str] = None):
        self.role = role
        self.content = content
        self.intent = intent
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "timestamp": self.timestamp
        }


class ConversationStore:
    """Thread-safe conversation history manager for context-aware routing."""

    def __init__(self, max_history_turns: int = 10):
        self.sessions: Dict[str, List[ConversationTurn]] = {}
        self.max_history_turns = max_history_turns
        self.lock = Lock()

    def add_turn(self, conversation_id: str, role: str, content: str, intent: Optional[str] = None) -> None:
        with self.lock:
            if conversation_id not in self.sessions:
                self.sessions[conversation_id] = []
            turn = ConversationTurn(role=role, content=content, intent=intent)
            self.sessions[conversation_id].append(turn)
            if len(self.sessions[conversation_id]) > self.max_history_turns * 2:
                self.sessions[conversation_id] = self.sessions[conversation_id][-self.max_history_turns * 2:]

    def get_history(self, conversation_id: str, last_n_turns: int = 4) -> List[Dict[str, Any]]:
        with self.lock:
            turns = self.sessions.get(conversation_id, [])
            return [t.to_dict() for t in turns[-last_n_turns:]]

    def get_last_intent(self, conversation_id: str) -> Optional[str]:
        with self.lock:
            turns = self.sessions.get(conversation_id, [])
            for t in reversed(turns):
                if t.intent:
                    return t.intent
            return None

    def format_history_prompt(self, conversation_id: str) -> str:
        """Formats conversation history for LLM prompt context."""
        history = self.get_history(conversation_id, last_n_turns=4)
        if not history:
            return "No previous conversation history."
        
        formatted = []
        for t in history:
            prefix = "User" if t["role"] == "user" else "Assistant"
            intent_tag = f" [Intent: {t['intent']}]" if t.get("intent") else ""
            formatted.append(f"{prefix}{intent_tag}: {t['content']}")
        return "\n".join(formatted)

    def clear(self, conversation_id: Optional[str] = None) -> None:
        with self.lock:
            if conversation_id:
                self.sessions.pop(conversation_id, None)
            else:
                self.sessions.clear()


# Global singleton instance
conversation_store = ConversationStore()

