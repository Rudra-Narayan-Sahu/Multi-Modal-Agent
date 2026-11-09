from enum import Enum
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MemoryCategoryEnum(str, Enum):
    USER_PREFERENCES = "user_preferences"
    USER_CONTEXT = "user_context"
    PROJECT_CONTEXT = "project_context"
    RESEARCH_MEMORY = "research_memory"
    EXPLICIT_NOTES = "explicit_notes"


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"
    user_id: str = "default_user"


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    recent_messages: List[MessageResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ConversationListItem(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class MemoryCreate(BaseModel):
    category: MemoryCategoryEnum = MemoryCategoryEnum.EXPLICIT_NOTES
    content: str = Field(..., min_length=2, description="The memory content to remember")
    importance: int = Field(default=3, ge=1, le=5)
    source_conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    category: Optional[MemoryCategoryEnum] = None
    content: Optional[str] = None
    importance: Optional[int] = Field(default=None, ge=1, le=5)
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    category: str
    content: str
    importance: int
    source_conversation_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class RelevantMemory(BaseModel):
    id: str
    category: str
    content: str
    importance: int
    relevance_score: float = 1.0
    updated_at: datetime

