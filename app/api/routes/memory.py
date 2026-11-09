from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from app.memory.manager import memory_manager
from app.memory.schemas import MemoryCreate, MemoryUpdate, MemoryResponse

router = APIRouter(prefix="/api/memory", tags=["Long-Term Memory"])


@router.get("", response_model=List[Dict[str, Any]])
def get_memories(
    user_id: str = Query("default_user"),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=200)
):
    """Retrieve long-term memories for the authorized user."""
    return memory_manager.get_user_memories(user_id=user_id, category=category, limit=limit)


@router.post("", response_model=Dict[str, Any])
def create_memory(
    payload: MemoryCreate,
    user_id: str = Query("default_user")
):
    """Explicitly save a user preference, project context, or fact into long-term memory."""
    mem = memory_manager.save_memory(
        user_id=user_id,
        content=payload.content,
        category=payload.category.value if hasattr(payload.category, "value") else str(payload.category),
        importance=payload.importance,
        source_conversation_id=payload.source_conversation_id,
        metadata=payload.metadata
    )
    if not mem:
        raise HTTPException(
            status_code=400,
            detail="Memory could not be saved: invalid content or contains restricted secrets."
        )
    return mem


@router.patch("/{memory_id}", response_model=Dict[str, Any])
def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    user_id: str = Query("default_user")
):
    """Update a stored memory record."""
    try:
        updated = memory_manager.update_memory(
            memory_id=memory_id,
            user_id=user_id,
            content=payload.content,
            category=payload.category.value if payload.category and hasattr(payload.category, "value") else (str(payload.category) if payload.category else None),
            importance=payload.importance,
            metadata=payload.metadata
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Memory record not found.")
        return updated
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    user_id: str = Query("default_user")
):
    """Delete a specific memory record."""
    try:
        success = memory_manager.delete_memory(memory_id=memory_id, user_id=user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory record not found.")
        return {"status": "deleted", "memory_id": memory_id}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))

