import os
import uuid
import glob
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

PROCESSOR_DIR = "processed_data"

def get_qdrant_client() -> QdrantClient:
    """Instantiate a Qdrant client."""
    if not settings.QDRANT_URL:
        raise ValueError("QDRANT_URL is not set in environment or configuration.")
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30.0,
    )

def ensure_collection_exists(client: QdrantClient = None, collection_name: str = None) -> None:
    """Ensure the target collection exists with the exact vector dimension."""
    if client is None:
        client = get_qdrant_client()
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION_NAME

    dim = get_embedding_dim()

    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        current_dim = info.config.params.vectors.size
        if current_dim != dim:
            logfire.warning(f"Qdrant collection {collection_name} has dimension {current_dim}, expected {dim}. Re-creating collection...")
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )
            logfire.info(f"Re-created Qdrant collection {collection_name} with dimension {dim}.")
    else:
        logfire.info(f"Creating Qdrant collection {collection_name} with dimension {dim}.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            )
        )
        logfire.info(f"Created Qdrant collection {collection_name}.")

def save_data_locally(data: dict, source_type: str, source_name: str) -> str:
    """Saves raw extracted text locally for provenance."""
    if not os.path.exists(PROCESSOR_DIR):
        os.makedirs(PROCESSOR_DIR, exist_ok=True)
    safe_name = "".join(c for c in source_name if c.isalnum() or c in "._- ")
    file_name = f"{source_type}_{safe_name}_{uuid.uuid4().hex[:8]}.txt"
    file_path = os.path.join(PROCESSOR_DIR, file_name)
    with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(data['text'])
    return file_path

def detect_source_type(file_path: str) -> str:
    """Detect source type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".docx", ".pptx", ".doc"]:
        return "docx"
    elif ext in [".html", ".htm"]:
        return "html"
    elif ext in [".txt", ".md", ".json", ".csv"]:
        return "text"
    return "text"

def process_file(
    file_path: str,
    file_name: str = None,
    source_type: str = None,
    source_name: str = "Local"
) -> int:
    """
    Processes a file: parses text, generates chunks, computes embeddings, and stores in Qdrant.
    Returns the number of chunks saved.
    """
    if file_name is None:
        file_name = os.path.basename(file_path)
    if source_type is None:
        source_type = detect_source_type(file_path)

    logfire.info(f"Processing file: {file_name} (type: {source_type})")
    
    # 1. Parse text from file
    if source_type == "pdf":
        text = parse_pdf(file_path)
    elif source_type in ["docx", "office"]:
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

    # 2. Chunk text
    chunks = chunk_text(text)
    if not chunks:
        logfire.warning(f"No valid chunks generated for {file_name}. Skipping.")
        return 0

    # 3. Save raw extracted text locally
    saved_file_path = save_data_locally({"text": text}, source_type, file_name)

    # 4. Generate embeddings
    embeddings = embed_texts(chunks)

    # 5. Connect to Qdrant & ensure collection exists
    client = get_qdrant_client()
    ensure_collection_exists(client, settings.QDRANT_COLLECTION_NAME)

    # 6. Upsert points to Qdrant
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

    # Batch upsert in slices of 100 points
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=batch
        )

    logfire.info(f"Successfully stored {len(points)} chunks for {file_name} in Qdrant.")
    return len(points)

def process_all_files(data_dir: str = "DATA") -> dict:
    """
    Ingests all supported documents found in the specified directory.
    """
    if not os.path.isabs(data_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, data_dir)

    logfire.info(f"Scanning for documents in: {data_dir}")
    if not os.path.exists(data_dir):
        return {"error": f"Directory not found: {data_dir}", "processed_files": [], "total_chunks": 0}

    supported_extensions = ["*.pdf", "*.txt", "*.md", "*.docx", "*.pptx", "*.html"]
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))

    results = []
    total_chunks = 0
    errors = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            chunks_count = process_file(
                file_path=file_path,
                file_name=file_name,
                source_name="Data Directory"
            )
            results.append({"file_name": file_name, "chunks": chunks_count, "status": "success"})
            total_chunks += chunks_count
        except Exception as e:
            logfire.error(f"Error processing {file_name}: {e}")
            errors.append({"file_name": file_name, "error": str(e)})

    return {
        "total_files": len(files),
        "processed_files": results,
        "total_chunks": total_chunks,
        "errors": errors
    }