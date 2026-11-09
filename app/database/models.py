import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, default=utc_now)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("LongTermMemory", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")
    research_sessions = relationship("ResearchSession", back_populates="conversation", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_conversation_user_updated", "user_id", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(64), ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    metadata_json = Column(Text, default="{}")  # JSON string of agent citations, tokens, status

    conversation = relationship("Conversation", back_populates="messages")


class LongTermMemory(Base):
    __tablename__ = "long_term_memories"

    id = Column(String(64), primary_key=True, default=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    memory_category = Column(String(32), nullable=False, index=True)  # user_preferences, user_context, project_context, research_memory, explicit_notes
    memory_content = Column(Text, nullable=False)
    importance = Column(Integer, default=3)  # 1 (low) to 5 (critical)
    source_conversation_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    expires_at = Column(DateTime, nullable=True)
    metadata_json = Column(Text, default="{}")

    user = relationship("User", back_populates="memories")

    __table_args__ = (
        Index("idx_memory_user_cat", "user_id", "memory_category"),
    )


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(String(64), primary_key=True, default=lambda: f"res_{uuid.uuid4().hex[:12]}")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(String(64), ForeignKey("conversations.id"), nullable=False, index=True)
    research_topic = Column(String(255), nullable=False)
    findings = Column(Text, nullable=False)
    source_references_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=utc_now)

    conversation = relationship("Conversation", back_populates="research_sessions")


class UploadedDocument(Base):
    __tablename__ = "uploaded_documents"

    id = Column(String(64), primary_key=True, default=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_size = Column(Integer, default=0)
    chunks_count = Column(Integer, default=0)
    status = Column(String(32), default="indexed")
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User")

