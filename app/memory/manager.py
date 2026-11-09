import re
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logfire

from app.database.connection import get_db_session
from app.database.models import User, LongTermMemory
from app.memory.schemas import MemoryCategoryEnum, MemoryResponse


# Privacy & Secret filters (rejects storing API keys, passwords, tokens)
SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{12,}",
    r"sk-[a-zA-Z0-9]{20,}",
    r"gsk_[a-zA-Z0-9]{20,}",
    r"tvly-[a-zA-Z0-9]{20,}",
    r"AIzaSy[a-zA-Z0-9_-]{33}",
]


class MemoryManager:
    """
    Manages the lifecycle of long-term memories with user isolation,
    deduplication, validation, and privacy filtering.
    """

    @staticmethod
    def validate_memory(content: str) -> bool:
        """Ensures content does not contain credentials, passwords, or secrets."""
        if not content or len(content.strip()) < 3:
            return False
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, content):
                logfire.warning("Memory rejected: Sensitive secret or credential pattern detected.")
                return False
        return True

    @staticmethod
    def _ensure_user_exists(user_id: str, db: Session) -> None:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            db.add(User(id=user_id))
            db.flush()

    def save_memory(
        self,
        user_id: str,
        content: str,
        category: str = "explicit_notes",
        importance: int = 3,
        source_conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Saves a new validated long-term memory for the user."""
        clean_content = content.strip()
        if not self.validate_memory(clean_content):
            return None

        with get_db_session() as db:
            self._ensure_user_exists(user_id, db)

            # Deduplication: Check if identical memory content exists for this user
            existing = (
                db.query(LongTermMemory)
                .filter(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.memory_content == clean_content
                )
                .first()
            )
            if existing:
                # Update importance if newer has higher importance
                if importance > existing.importance:
                    existing.importance = importance
                    existing.updated_at = datetime.now(timezone.utc)
                    db.commit()
                return self._to_dict(existing)

            memory = LongTermMemory(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                user_id=user_id,
                memory_category=category,
                memory_content=clean_content,
                importance=min(max(importance, 1), 5),
                source_conversation_id=source_conversation_id,
                metadata_json=json.dumps(metadata or {})
            )
            db.add(memory)
            db.commit()
            return self._to_dict(memory)

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing memory belonging to the user."""
        with get_db_session() as db:
            mem = db.query(LongTermMemory).filter(LongTermMemory.id == memory_id).first()
            if not mem:
                return None
            if mem.user_id != user_id and mem.user_id != "default_user" and user_id != "default_user":
                raise PermissionError("Access denied: Memory record belongs to another user.")

            if content is not None:
                clean_content = content.strip()
                if not self.validate_memory(clean_content):
                    raise ValueError("Memory validation failed: invalid content or contains secrets.")
                mem.memory_content = clean_content

            if category is not None:
                mem.memory_category = category

            if importance is not None:
                mem.importance = min(max(importance, 1), 5)

            if metadata is not None:
                mem.metadata_json = json.dumps(metadata)

            mem.updated_at = datetime.now(timezone.utc)
            db.commit()
            return self._to_dict(mem)

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Deletes a specific memory record with user authorization."""
        with get_db_session() as db:
            mem = db.query(LongTermMemory).filter(LongTermMemory.id == memory_id).first()
            if not mem:
                return False
            if mem.user_id != user_id and mem.user_id != "default_user" and user_id != "default_user":
                raise PermissionError("Access denied: Memory record belongs to another user.")
            db.delete(mem)
            db.commit()
            return True

    def get_user_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieves stored memories for a user, optionally filtered by category."""
        with get_db_session() as db:
            q = db.query(LongTermMemory).filter(LongTermMemory.user_id == user_id)
            if category:
                q = q.filter(LongTermMemory.memory_category == category)
            records = q.order_by(desc(LongTermMemory.importance), desc(LongTermMemory.updated_at)).limit(limit).all()
            return [self._to_dict(m) for m in records]

    def _to_dict(self, mem: LongTermMemory) -> Dict[str, Any]:
        return {
            "id": mem.id,
            "user_id": mem.user_id,
            "category": mem.memory_category,
            "content": mem.memory_content,
            "importance": mem.importance,
            "source_conversation_id": mem.source_conversation_id,
            "created_at": mem.created_at.isoformat(),
            "updated_at": mem.updated_at.isoformat(),
            "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {}
        }


memory_manager = MemoryManager()

