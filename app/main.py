import os
import shutil
import tempfile
import logfire
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.retrieval.retriever import retrieve_documents, get_collection_stats
from app.services.retrieval.embedding import get_embedding_dim
from app.services.llm import generate_rag_response
from app.ingestion.processor import process_file, process_all_files, ensure_collection_exists

# Initialize FastAPI App
app = FastAPI(
    title="Agentic RAG Knowledge System",
    description="Production-grade RAG System with Qdrant Vector Search and Groq LLM",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or prompt")
    history: Optional[List[MessageItem]] = Field(default_factory=list, description="Conversation history")
    top_k: int = Field(default=5, description="Number of context chunks to retrieve")

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    model_used: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(default=5, description="Number of results")

class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[Dict[str, Any]]

# --- API Endpoints ---

@app.get("/api/status")
def get_system_status():
    """Returns the current vector database and system configuration status."""
    stats = get_collection_stats()
    return {
        "status": "healthy",
        "collection": stats,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dim": get_embedding_dim(),
        "llm_model": settings.GROQ_MODEL,
        "has_qdrant_configured": bool(settings.QDRANT_URL and settings.QDRANT_API_KEY),
        "has_groq_configured": bool(settings.GROQ_API_KEY),
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    RAG Chat endpoint: Embeds query -> Retrieves relevant chunks -> Generates answer via LLM.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    history_dicts = [{"role": h.role, "content": h.content} for h in (req.history or [])]
    
    result = generate_rag_response(
        query=req.message,
        history=history_dicts,
        top_k=req.top_k
    )
    
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        model_used=result["model_used"]
    )

@app.post("/api/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    """
    Direct similarity search: Embeds query -> Performs cosine similarity search in Qdrant.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    docs = retrieve_documents(query=req.query, top_k=req.top_k)
    return SearchResponse(
        query=req.query,
        total=len(docs),
        results=docs
    )

@app.post("/api/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Triggers batch ingestion of all documents in the DATA directory.
    """
    result = process_all_files("DATA")
    return {
        "success": True,
        "message": f"Processed {result.get('total_files', 0)} files, generated {result.get('total_chunks', 0)} chunks.",
        "details": result
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Direct document upload: parses, chunks, embeds, and stores in Qdrant.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Save uploaded file to temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file into Qdrant
        chunks_count = process_file(
            file_path=temp_path,
            file_name=file.filename,
            source_name=f"Upload: {file.filename}"
        )

        return {
            "success": True,
            "file_name": file.filename,
            "chunks_stored": chunks_count,
            "message": f"Successfully ingested {file.filename} into Qdrant ({chunks_count} chunks created)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")
    finally:
        # Clean up temp folder
        shutil.rmtree(temp_dir, ignore_errors=True)

# Mount Static Files & Serve UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    """Serves the web dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Agentic RAG API is running. UI files not found in app/static/."}
