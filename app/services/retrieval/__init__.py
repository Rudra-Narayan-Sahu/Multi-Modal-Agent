from app.services.retrieval.embedding import embed_query, embed_texts, get_embedding_dim
from app.services.retrieval.retriever import retrieve_context, retrieve_documents
from app.services.retrieval.pipeline import AdvancedRetriever, RetrievalConfig
from app.services.retrieval.hybrid import get_bm25_index, reciprocal_rank_fusion
from app.services.retrieval.reranker import rerank_documents
from app.services.retrieval.diversity import apply_mmr_to_documents, maximal_marginal_relevance
from app.services.retrieval.query_expansion import generate_sub_queries, generate_hyde_query
from app.services.retrieval.compressor import deduplicate_documents, compress_context, format_citations

__all__ = [
    "embed_query",
    "embed_texts",
    "get_embedding_dim",
    "retrieve_context",
    "retrieve_documents",
    "AdvancedRetriever",
    "RetrievalConfig",
    "get_bm25_index",
    "reciprocal_rank_fusion",
    "rerank_documents",
    "apply_mmr_to_documents",
    "maximal_marginal_relevance",
    "generate_sub_queries",
    "generate_hyde_query",
    "deduplicate_documents",
    "compress_context",
    "format_citations",
]
