import os
import shutil
import tempfile
import logfire
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.database.models import User, UploadedDocument
from app.ingestion.processor import process_file, detect_source_type
from app.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models

router = APIRouter(prefix="/api", tags=["Document Upload & Ingestion"])

# Allowed extensions
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".html", ".htm",
    ".txt", ".md", ".json", ".csv", ".py", ".log"
}


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    file_type: str
    file_size: int
    chunks_count: int
    status: str
    created_at: str


class UploadResultResponse(BaseModel):
    success: bool
    file_name: str
    file_type: str
    file_size: int
    chunks_stored: int
    message: str


@router.post("/upload", response_model=UploadResultResponse)
def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db)
):
    """
    Direct user document upload:
    Accepts PDF, Word, PowerPoint, HTML, Markdown, CSV, JSON, and Text files.
    Parses, chunks, embeds, and stores points in Qdrant Vector DB for multi-agent RAG.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Ensure user exists in database
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.commit()

    # Save to temp directory safely
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        content = file.file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        source_type = detect_source_type(file.filename)

        # Ingest into Qdrant Cloud knowledge base
        chunks_count = process_file(
            file_path=temp_path,
            file_name=file.filename,
            source_type=source_type,
            source_name=f"User Upload ({user_id}): {file.filename}"
        )

        # Save document record to DB
        doc_record = UploadedDocument(
            user_id=user_id,
            file_name=file.filename,
            file_type=source_type,
            file_size=file_size,
            chunks_count=chunks_count,
            status="indexed"
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        logfire.info(f"User {user_id} uploaded {file.filename} -> {chunks_count} chunks indexed.")

        return UploadResultResponse(
            success=True,
            file_name=file.filename,
            file_type=source_type,
            file_size=file_size,
            chunks_stored=chunks_count,
            message=f"Successfully indexed '{file.filename}' into knowledge base ({chunks_count} chunks created)."
        )

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Failed to ingest uploaded document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/documents", response_model=List[DocumentResponse])
def list_uploaded_documents(
    user_id: str = "default_user",
    db: Session = Depends(get_db)
):
    """List all documents uploaded by the user with indexing status and chunk counts."""
    docs = db.query(UploadedDocument).filter(
        UploadedDocument.user_id == user_id
    ).order_by(UploadedDocument.created_at.desc()).all()

    return [
        DocumentResponse(
            id=d.id,
            file_name=d.file_name,
            file_type=d.file_type,
            file_size=d.file_size or 0,
            chunks_count=d.chunks_count or 0,
            status=d.status or "indexed",
            created_at=d.created_at.isoformat()
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: str,
    user_id: str = "default_user",
    db: Session = Depends(get_db)
):
    """Delete an uploaded document metadata from DB."""
    doc = db.query(UploadedDocument).filter(
        UploadedDocument.id == doc_id,
        UploadedDocument.user_id == user_id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_name = doc.file_name
    db.delete(doc)
    db.commit()

    return {"success": True, "message": f"Document '{file_name}' removed from your collection."}
