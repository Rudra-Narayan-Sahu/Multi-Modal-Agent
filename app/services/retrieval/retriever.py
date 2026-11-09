import logfire
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.pipeline import AdvancedRetriever, RetrievalConfig

# Singleton default retriever instance
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=True,
    timeout=10,
)
_default_pipeline = AdvancedRetriever(client=client)


def retrieve_documents(
    query: str,
    top_k: int = 5,
    search_type: str = "hybrid",
    enable_reranker: bool = True,
    enable_mmr: bool = False,
    score_threshold: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top relevant documents using the Advanced RAG Pipeline
    (Dense + BM25 Hybrid, Cross-Encoder Reranking, Metadata Filters).
    """
    config = RetrievalConfig(
        search_type=search_type,  # "dense", "sparse", or "hybrid"
        top_k=top_k,
        enable_reranker=enable_reranker,
        enable_mmr=enable_mmr,
        score_threshold=score_threshold,
        filters=filters,
    )
    return _default_pipeline.retrieve(query=query, config=config)


def retrieve_context(
    query: str,
    top_k: int = 5,
    include_citations: bool = False,
    search_type: str = "hybrid",
    enable_reranker: bool = True,
    filters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Retrieve and join the most relevant document chunks into a single context string.
    Optionally appends document citations and provenance metadata.
    """
    config = RetrievalConfig(
        search_type=search_type,
        top_k=top_k,
        enable_reranker=enable_reranker,
        filters=filters,
    )
    result = _default_pipeline.retrieve_context_with_citations(query=query, config=config)
    
    if include_citations:
        return f"{result['context']}\n\n{result['citations']}"
    return result["context"]
