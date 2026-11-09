from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from app.memory.conversation import conversation_manager
from app.memory.schemas import ConversationCreate, ConversationResponse, ConversationListItem, MessageResponse

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.post("", response_model=Dict[str, Any])
def create_conversation(payload: ConversationCreate):
    """Create a new conversation session for the authorized user."""
    return conversation_manager.create_conversation(
        user_id=payload.user_id,
        title=payload.title or "New Conversation"
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_conversations(
    user_id: str = Query("default_user"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all conversation sessions belonging to the user."""
    return conversation_manager.list_conversations(user_id=user_id, limit=limit, offset=offset)


@router.get("/{conversation_id}", response_model=Dict[str, Any])
def get_conversation_details(
    conversation_id: str,
    user_id: str = Query("default_user")
):
    """Retrieve conversation details and messages with user isolation."""
    try:
        conv = conversation_manager.get_conversation(conversation_id=conversation_id, user_id=user_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return conv
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@router.get("/{conversation_id}/messages", response_model=List[Dict[str, Any]])
def get_conversation_messages(
    conversation_id: str,
    user_id: str = Query("default_user"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Retrieve paginated message history."""
    try:
        return conversation_manager.get_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    user_id: str = Query("default_user")
):
    """Delete conversation and all its messages."""
    try:
        success = conversation_manager.delete_conversation(conversation_id=conversation_id, user_id=user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"status": "deleted", "conversation_id": conversation_id}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))

