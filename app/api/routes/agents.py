from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from app.memory.conversation import conversation_manager
from app.services.tools import tool_registry
from app.config import settings

router = APIRouter(prefix="/api", tags=["Multi-Agent Metadata & Conversations"])


@router.get("/agents/status")
def get_agents_status() -> Dict[str, Any]:
    """Returns the operational status and capabilities of all 6 agents."""
    return {
        "status": "operational",
        "orchestration_framework": "LangGraph",
        "agents": [
            {
                "name": "supervisor",
                "role": "Master Orchestrator",
                "responsibilities": "Intent classification, planning, conditional routing, cycle termination",
                "status": "active"
            },
            {
                "name": "rag_agent",
                "role": "Document Knowledge Base Retrieval",
                "responsibilities": "Hybrid BM25 + Qdrant dense retrieval, FlashRank reranking, chunk provenance",
                "status": "active",
                "collection": settings.QDRANT_COLLECTION_NAME
            },
            {
                "name": "web_agent",
                "role": "External Web Search",
                "responsibilities": "Real-time search via Tavily API with DuckDuckGo fallback, URL verification",
                "status": "active",
                "providers": ["tavily", "duckduckgo"]
            },
            {
                "name": "research_agent",
                "role": "Comparative Research & Synthesis",
                "responsibilities": "Cross-source comparison, gap analysis, contradiction detection, executive synthesis",
                "status": "active"
            },
            {
                "name": "tool_agent",
                "role": "External Tool Execution",
                "responsibilities": "Safe mathematical calculations, restricted Python sandbox, datetime operations",
                "status": "active",
                "registered_tools": [t["name"] for t in tool_registry.list_tools()]
            },
            {
                "name": "response_agent",
                "role": "Grounded Response & Citation Generation",
                "responsibilities": "Final grounded answer, citation formatting, limitations reporting",
                "status": "active"
            }
        ]
    }


@router.get("/conversations/{conversation_id}")
def get_conversation_history(conversation_id: str, user_id: str = Query("default_user")) -> Dict[str, Any]:
    """Retrieves conversation history for the authorized user."""
    try:
        history = conversation_manager.get_history(conversation_id=conversation_id, user_id=user_id)
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return history
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user_id: str = Query("default_user")) -> Dict[str, Any]:
    """Deletes conversation memory."""
    try:
        deleted = conversation_manager.clear(conversation_id=conversation_id, user_id=user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"status": "deleted", "conversation_id": conversation_id}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))

