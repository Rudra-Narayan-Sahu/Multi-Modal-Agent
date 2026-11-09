from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional

from app.router.models import RouterRequest, RouterResponse
from app.router.engine import default_router
from app.router.context_store import conversation_store
from app.config import settings
from app.database import init_db
from qdrant_client import QdrantClient

# Import Multi-Agent LangGraph & Memory routes
from app.api.routes.chat import router as chat_router
from app.api.routes.agents import router as agents_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.memory import router as memory_router

app = FastAPI(
    title="Multi-Agent RAG Assistant API",
    description="LangGraph Multi-Agent Orchestration, Advanced RAG, and 3-Tier Memory Management API.",
    version="2.1.0"
)

# Auto-initialize database tables on server startup
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Multi-Agent & Memory Endpoints
app.include_router(chat_router)
app.include_router(agents_router)
app.include_router(conversations_router)
app.include_router(memory_router)

# Mount React + Tailwind static frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app.mount("/app", StaticFiles(directory="app/static", html=True), name="static")


@app.get("/")
def root():
    """Serve the ChatGPT UI at the root path."""
    index_file = os.path.join("app", "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "service": "Multi-Agent RAG Assistant (LangGraph + 3-Tier Memory)",
        "version": "2.1.0",
        "status": "online",
        "endpoints": {
            "ui": "/app",
            "multi_agent_chat": "/api/chat",
            "multi_agent_stream": "/api/chat/stream",
            "conversations": "/api/conversations",
            "memory_management": "/api/memory",
            "agents_status": "/api/agents/status",
            "health": "/api/health"
        }
    }


@app.get("/api/info")
def api_info():
    """API metadata and endpoints list."""
    return {
        "service": "Multi-Agent RAG Assistant (LangGraph + 3-Tier Memory)",
        "version": "2.1.0",
        "status": "online",
        "endpoints": {
            "ui": "/",
            "multi_agent_chat": "/api/chat",
            "multi_agent_stream": "/api/chat/stream",
            "conversations": "/api/conversations",
            "memory_management": "/api/memory",
            "agents_status": "/api/agents/status",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
def health_check():
    """Check connectivity to Qdrant Vector DB, SQLite/PostgreSQL, and LLM providers."""
    status = {
        "api": "healthy",
        "database": "connected (SQLAlchemy)",
        "qdrant": "unknown",
        "groq": "configured" if settings.GROQ_API_KEY else "missing",
        "gemini": "configured" if settings.GEMINI_API_KEY else "missing",
        "orchestrator": "LangGraph (StateGraph)",
        "memory_system": "3-Tier (Short-Term, History, Long-Term)"
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
    """Query Intent Router endpoint (backward-compatible)."""
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
