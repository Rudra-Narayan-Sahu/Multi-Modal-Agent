import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logfire

from app.database.connection import get_db_session, SessionLocal
from app.database.models import User, Conversation, Message
from app.memory.schemas import (
    ConversationResponse,
    ConversationListItem,
    MessageResponse
)


class ConversationHistoryManager:
    """
    Database-backed conversation history repository.
    Enforces user isolation, maintains message ordering, and provides context windows.
    """

    @staticmethod
    def _ensure_user_exists(user_id: str, db: Session) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id)
            db.add(user)
            db.flush()
        return user

    def create_conversation(
        self,
        user_id: str = "default_user",
        title: str = "New Conversation",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new conversation session for an authorized user."""
        cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        with get_db_session() as db:
            self._ensure_user_exists(user_id, db)
            conv = Conversation(id=cid, user_id=user_id, title=title)
            db.add(conv)
            db.commit()
            return {
                "id": conv.id,
                "user_id": conv.user_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat()
            }

    def list_conversations(
        self,
        user_id: str = "default_user",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all conversations belonging to the user."""
        with get_db_session() as db:
            convs = (
                db.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(desc(Conversation.updated_at))
                .offset(offset)
                .limit(limit)
                .all()
            )
            result = []
            for c in convs:
                count = db.query(Message).filter(Message.conversation_id == c.id).count()
                result.append({
                    "id": c.id,
                    "user_id": c.user_id,
                    "title": c.title,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                    "message_count": count
                })
            return result

    def get_conversation(
        self,
        conversation_id: str,
        user_id: str = "default_user"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve conversation details and recent messages with user isolation."""
        with get_db_session() as db:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return None
            if conv.user_id != user_id and conv.user_id != "default_user" and user_id != "default_user":
                raise PermissionError("Access denied: Conversation belongs to another user.")

            messages = (
                db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )

            msg_list = []
            for m in messages:
                meta = json.loads(m.metadata_json) if m.metadata_json else {}
                msg_list.append({
                    "id": m.id,
                    "conversation_id": m.conversation_id,
                    "user_id": m.user_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "metadata": meta
                })

            return {
                "id": conv.id,
                "user_id": conv.user_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "message_count": len(msg_list),
                "messages": msg_list,
                # Compatibility field
                "turns": [
                    {
                        "query": m["content"],
                        "response": msg_list[i+1]["content"] if i+1 < len(msg_list) and msg_list[i+1]["role"] == "assistant" else "",
                        "category": m.get("metadata", {}).get("category", ""),
                        "agents_used": m.get("metadata", {}).get("agents_used", []),
                        "citations": m.get("metadata", {}).get("citations", [])
                    }
                    for i, m in enumerate(msg_list) if m["role"] == "user"
                ]
            }

    def get_messages(
        self,
        conversation_id: str,
        user_id: str = "default_user",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Paginated retrieval of messages in a conversation."""
        with get_db_session() as db:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return []
            if conv.user_id != user_id and conv.user_id != "default_user" and user_id != "default_user":
                raise PermissionError("Access denied.")

            msgs = (
                db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": m.id,
                    "conversation_id": m.conversation_id,
                    "user_id": m.user_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "metadata": json.loads(m.metadata_json) if m.metadata_json else {}
                }
                for m in msgs
            ]

    def add_turn(
        self,
        conversation_id: str,
        user_id: str,
        query: str,
        response: str,
        category: str = "UNKNOWN",
        agents_used: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Persist a user query and assistant response turn into PostgreSQL/SQLite."""
        with get_db_session() as db:
            self._ensure_user_exists(user_id, db)
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                title = " ".join(query.strip().split()[:6])
                conv = Conversation(id=conversation_id, user_id=user_id, title=title)
                db.add(conv)
                db.flush()
            else:
                if conv.user_id != user_id and conv.user_id != "default_user" and user_id != "default_user":
                    raise PermissionError("Access denied: Conversation belongs to another user.")
                conv.updated_at = datetime.now(timezone.utc)

            # User message
            user_msg = Message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                content=query,
                metadata_json=json.dumps({"category": category})
            )
            db.add(user_msg)

            # Assistant message
            asst_meta = {
                "category": category,
                "agents_used": agents_used or [],
                "citations": citations or []
            }
            asst_msg = Message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=response,
                metadata_json=json.dumps(asst_meta)
            )
            db.add(asst_msg)
            db.commit()

    def delete_conversation(
        self,
        conversation_id: str,
        user_id: str = "default_user"
    ) -> bool:
        """Delete conversation and all cascaded messages."""
        with get_db_session() as db:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return False
            if conv.user_id != user_id and conv.user_id != "default_user" and user_id != "default_user":
                raise PermissionError("Access denied.")
            db.delete(conv)
            db.commit()
            return True

    def clear(self, conversation_id: str, user_id: str = "default_user") -> bool:
        """Compatibility wrapper for delete_conversation."""
        return self.delete_conversation(conversation_id, user_id)

    def get_history(self, conversation_id: str, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """Compatibility wrapper for get_conversation."""
        return self.get_conversation(conversation_id, user_id)

    def format_history_context(
        self,
        conversation_id: str,
        user_id: str = "default_user",
        max_turns: int = 5
    ) -> str:
        """
        Formats recent conversation history as a concise dialogue string
        for the Supervisor Agent's follow-up resolution and short-term memory.
        """
        data = self.get_conversation(conversation_id, user_id)
        if not data or not data.get("messages"):
            return ""

        msgs = data["messages"][- (max_turns * 2):]  # Get last N turns
        formatted = []
        for m in msgs:
            role = "User" if m["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {m['content']}")

        return "\n".join(formatted)


conversation_manager = ConversationHistoryManager()
