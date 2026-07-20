import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embedding import embed_query

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=True,
)


def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the top relevant documents from Qdrant for a user query.
    """
    try:
        query_vector = embed_query(query)
        results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        documents = []
        for point in getattr(results, "points", []):
            payload = point.payload or {}
            documents.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "text": payload.get("text"),
                    "file_name": payload.get("file_name"),
                    "source_type": payload.get("source_type"),
                    "source_name": payload.get("source_name"),
                    "local_path": payload.get("local_path"),
                }
            )
        return documents
    except Exception as e:
        logfire.error(f"Failed to retrieve documents: {e}")
        return []


def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieve and join the most relevant document chunks into a single context string.
    """
    documents = retrieve_documents(query, top_k=top_k)
    context_parts = []
    for doc in documents:
        if doc.get("text"):
            context_parts.append(doc["text"])
    return "\n\n".join(context_parts)
