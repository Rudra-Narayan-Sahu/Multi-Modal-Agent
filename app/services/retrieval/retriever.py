import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embedding import embed_query, get_embedding_dim

_client = None

def get_retriever_client() -> QdrantClient:
    global _client
    if _client is None:
        if not settings.QDRANT_URL:
            raise ValueError("QDRANT_URL is not set.")
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30.0,
        )
    return _client

def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the user query and perform cosine similarity search against Qdrant.
    Returns ranked list of matching chunks with similarity score and metadata.
    """
    if not query or not query.strip():
        return []

    try:
        client = get_retriever_client()
        collection_name = settings.QDRANT_COLLECTION_NAME

        if not client.collection_exists(collection_name):
            logfire.warning(f"Collection {collection_name} does not exist in Qdrant.")
            return []

        query_vector = embed_query(query)
        
        # Use query_points (or search fallback)
        try:
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(results, "points", [])
        except Exception:
            # Fallback for older client versions
            points = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )

        documents = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            score = getattr(point, "score", 0.0)
            documents.append(
                {
                    "id": str(point.id),
                    "score": round(float(score), 4),
                    "text": payload.get("text", ""),
                    "file_name": payload.get("file_name", "Unknown"),
                    "source_type": payload.get("source_type", "doc"),
                    "source_name": payload.get("source_name", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 1),
                }
            )

        logfire.info(f"Retrieved {len(documents)} matching chunks for query: '{query[:40]}...'")
        return documents

    except Exception as e:
        logfire.error(f"Failed to retrieve documents: {e}")
        return []

def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieve and join the most relevant document chunks into a single context string.
    """
    documents = retrieve_documents(query, top_k=top_k)
    if not documents:
        return ""

    context_parts = []
    for i, doc in enumerate(documents, start=1):
        file_name = doc.get("file_name", "Unknown")
        chunk_idx = doc.get("chunk_index", 0)
        text = doc.get("text", "").strip()
        score = doc.get("score", 0.0)
        context_parts.append(
            f"[Document {i}: {file_name} (Chunk #{chunk_idx}, Relevance Score: {score})]\n{text}"
        )
    return "\n\n---\n\n".join(context_parts)

def get_collection_stats() -> dict:
    """Retrieve collection status and point count."""
    try:
        client = get_retriever_client()
        collection_name = settings.QDRANT_COLLECTION_NAME
        if not client.collection_exists(collection_name):
            return {
                "exists": False,
                "collection_name": collection_name,
                "points_count": 0,
                "vector_size": get_embedding_dim(),
                "status": "not_found"
            }
        info = client.get_collection(collection_name)
        return {
            "exists": True,
            "collection_name": collection_name,
            "points_count": getattr(info, "points_count", 0),
            "vector_size": info.config.params.vectors.size,
            "status": str(info.status.value) if hasattr(info.status, "value") else str(info.status)
        }
    except Exception as e:
        logfire.error(f"Error getting collection stats: {e}")
        return {
            "exists": False,
            "collection_name": settings.QDRANT_COLLECTION_NAME,
            "points_count": 0,
            "vector_size": get_embedding_dim(),
            "error": str(e)
        }

