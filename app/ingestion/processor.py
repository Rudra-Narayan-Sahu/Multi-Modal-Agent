import os
import sys
import uuid
import logfire
from app.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.pdf import parse_pdf 
from app.ingestion.loaders.office import parse_office
from app.ingestion.chunking.splitter import chunk_text

try:
    logfire.configure(service_name="Ingestion Processor", send_to_logfire="if-token-present")
except Exception:
    pass
PROCESSOR_DIR = "processed_data"
_client = None

def get_qdrant_client() -> QdrantClient:
    """Lazily instantiate QdrantClient and ensure target collection exists."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False
        )
        try:
            if not _client.collection_exists(settings.QDRANT_COLLECTION_NAME):
                logfire.info(f"Creating Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
                _client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=get_embedding_dim(), 
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logfire.warning(f"Qdrant collection verification skipped: {e}")
    return _client

# Backwards compatibility
client = None

def save_data_locally(data:dict,source_type:str,source_name:str)->str:
    """
    Saves the chunk data in locally.
    Returns the path to the saved file.
    """
    if not os.path.exists(PROCESSOR_DIR):
        os.makedirs(PROCESSOR_DIR)
    file_name=f"{source_type}_{source_name}_{uuid.uuid4().hex}.txt"
    file_path=os.path.join(PROCESSOR_DIR,file_name)
    with open(file_path,'w',encoding='utf-8', errors='replace') as f:
        f.write(data['text'])
    return file_path

def detect_source_type(file_name: str) -> str:
    """Detect document parser type from file extension."""
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".docx", ".doc", ".pptx"]:
        return "docx"
    elif ext in [".html", ".htm"]:
        return "html"
    elif ext in [".txt", ".md", ".json", ".csv", ".py", ".rst", ".log"]:
        return "text"
    return "text"


def process_file(file_path: str, file_name: str, source_type: str = None, source_name: str = None) -> int:
    """
    Processes a file based on its source type and saves the processed data to Qdrant.
    Returns the number of chunks saved.
    """
    if source_type is None:
        source_type = detect_source_type(file_name)
    if source_name is None:
        source_name = f"File: {file_name}"

    logfire.info(f"Processing file: {file_name} of type: {source_type} from source: {source_name}")
    try:
        if source_type == "pdf":
            text = parse_pdf(file_path)
        elif source_type == "docx":
            text = parse_office(file_path)
        elif source_type == "html":
            text = parse_html(file_path)
        elif source_type == "text":
            text = parse_text(file_path)
        else:
            text = parse_text(file_path)
        
        if not text or not text.strip():
            logfire.warning(f"No text extracted from {file_name}. Skipping.")
            return 0
            
        # Chunk the text to avoid hitting LLM token limits and create meaningful search blocks
        chunks = chunk_text(text)
        if not chunks:
            logfire.warning(f"No valid chunks from {file_name}. Skipping.")
            return 0
        
        # Clean any surrogate characters from extraction
        chunks = [c.encode("utf-8", "ignore").decode("utf-8") for c in chunks]
        
        data = {"text": text}
        saved_file_path = save_data_locally(data, source_type, source_name)
        
        # Embedding the chunks
        embeddings = embed_texts(chunks)
        
        # Saving to Qdrant
        points = []
        for idx, (chunk_text_str, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "file_name": file_name,
                        "source_type": source_type,
                        "source_name": source_name,
                        "local_path": saved_file_path,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": chunk_text_str
                    }
                )
            )
            
        qc = get_qdrant_client()
        qc.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )
        
        logfire.info(f"Successfully processed and saved {len(points)} chunks for {file_name} to Qdrant.")
        return len(points)
    except Exception as e:
        logfire.error(f"Failed to process {file_name}: {e}")
        raise