from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional

from app.router.models import RouterRequest, RouterResponse
from app.router.engine import default_router
from app.router.context_store import conversation_store
from app.config import settings
from qdrant_client import QdrantClient

# Import Multi-Agent LangGraph routes
from app.api.routes.chat import router as chat_router
from app.api.routes.agents import router as agents_router

app = FastAPI(
    title="Multi-Agent RAG Assistant API",
    description="LangGraph Multi-Agent Orchestration, Advanced RAG, and Intelligent Query Routing API.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Multi-Agent LangGraph Endpoints
app.include_router(chat_router)
app.include_router(agents_router)


@app.get("/")
def root():
    return {
        "service": "Multi-Agent RAG Assistant (LangGraph)",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "multi_agent_chat": "/api/chat",
            "multi_agent_stream": "/api/chat/stream",
            "agents_status": "/api/agents/status",
            "conversation_history": "/api/conversations/{conversation_id}",
            "legacy_query_router": "/api/query-router",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
def health_check():
    """Check connectivity to Qdrant Vector DB and LLM providers."""
    status = {
        "api": "healthy",
        "qdrant": "unknown",
        "groq": "configured" if settings.GROQ_API_KEY else "missing",
        "gemini": "configured" if settings.GEMINI_API_KEY else "missing",
        "orchestrator": "LangGraph (StateGraph)"
    }
    try:
        qc = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=3,
            prefer_grpc=False
        )
        info = qc.get_collection(settings.QDRANT_COLLECTION_NAME)
        status["qdrant"] = "connected"
        status["points_count"] = info.points_count
    except Exception as e:
        status["qdrant"] = f"unreachable: {str(e)}"

    return status


# Legacy router endpoint preserved for backward compatibility
@app.post("/api/query-router", response_model=RouterResponse)
def route_query(request: RouterRequest):
    """
    Query Intent Router endpoint (backward-compatible).
    Classifies intent, evaluates confidence, and returns the assigned route.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return default_router.process(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")


@app.post("/api/history/clear")
def clear_history(payload: Optional[Dict[str, str]] = None):
    """Clears conversation context memory for a given session or all sessions."""
    conv_id = payload.get("conversation_id") if payload else None
    conversation_store.clear(conv_id)
    return {"status": "cleared", "conversation_id": conv_id or "all"}
