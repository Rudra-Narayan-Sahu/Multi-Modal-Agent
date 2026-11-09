import logfire
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.hybrid import get_bm25_index, reciprocal_rank_fusion
from app.services.retrieval.reranker import rerank_documents
from app.services.retrieval.diversity import apply_mmr_to_documents
from app.services.retrieval.query_expansion import generate_sub_queries, generate_hyde_query
from app.services.retrieval.compressor import deduplicate_documents, compress_context, format_citations


@dataclass
class RetrievalConfig:
    """Configuration settings for the Advanced Retrieval Pipeline."""
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 5
    dense_top_k: int = 15
    sparse_top_k: int = 15
    enable_reranker: bool = True
    rerank_top_k: int = 5
    enable_mmr: bool = False
    mmr_lambda: float = 0.7
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    enable_query_expansion: bool = False
    use_hyde: bool = False
    max_context_length: int = 6000
    rrf_k: int = 60


class AdvancedRetriever:
    """
    Production-oriented RAG Retrieval Pipeline orchestrating:
    - Hybrid Search (Dense + BM25 with RRF)
    - Metadata-based filtering
    - Cross-Encoder Reranking (FlashRank)
    - Maximum Marginal Relevance (MMR diversity)
    - Context compression, deduplication, and citations
    """

    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=10,
        )
        self.bm25 = get_bm25_index(self.client)

    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        """Convert a dictionary of filters into Qdrant models.Filter."""
        if not filters:
            return None

        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
        return models.Filter(must=must_conditions) if must_conditions else None

    def _dense_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Dense cosine vector search in Qdrant."""
        try:
            query_vector = embed_query(query)
            qdrant_filter = self._build_qdrant_filter(filters)

            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            documents = []
            for point in getattr(results, "points", []):
                payload = point.payload or {}
                documents.append({
                    "id": str(point.id),
                    "score": float(point.score),
                    "dense_score": float(point.score),
                    "text": payload.get("text", ""),
                    "file_name": payload.get("file_name"),
                    "source_type": payload.get("source_type"),
                    "source_name": payload.get("source_name"),
                    "chunk_index": payload.get("chunk_index"),
                    "local_path": payload.get("local_path"),
                    "retrieval_source": "dense",
                })
            return documents
        except Exception as e:
            logfire.error(f"Dense vector retrieval failed: {e}")
            return []

    def _sparse_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Sparse BM25 keyword search."""
        return self.bm25.search(query=query, top_k=top_k, filters=filters)

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the full end-to-end multi-stage retrieval pipeline:
        1. Query Expansion / HyDE
        2. Candidate Retrieval (Dense / Sparse / Hybrid RRF)
        3. Lexical Deduplication
        4. Cross-Encoder Reranking
        5. MMR Diversification
        6. Score Thresholding
        """
        cfg = config or RetrievalConfig()
        search_query = query

        # 1. Query Expansion / HyDE
        if cfg.use_hyde:
            search_query = generate_hyde_query(query)
            logfire.info(f"Using HyDE query representation: {search_query[:80]}...")

        queries_to_search = [search_query]
        if cfg.enable_query_expansion:
            expanded = generate_sub_queries(query, num_queries=2)
            for eq in expanded:
                if eq not in queries_to_search:
                    queries_to_search.append(eq)

        # 2. Candidate Retrieval across query formulations
        all_candidates: List[Dict[str, Any]] = []

        for q in queries_to_search:
            if cfg.search_type == "dense":
                candidates = self._dense_search(q, top_k=cfg.dense_top_k, filters=cfg.filters)
            elif cfg.search_type == "sparse":
                candidates = self._sparse_search(q, top_k=cfg.sparse_top_k, filters=cfg.filters)
            else:  # hybrid
                dense_docs = self._dense_search(q, top_k=cfg.dense_top_k, filters=cfg.filters)
                sparse_docs = self._sparse_search(q, top_k=cfg.sparse_top_k, filters=cfg.filters)
                candidates = reciprocal_rank_fusion(
                    dense_results=dense_docs,
                    sparse_results=sparse_docs,
                    rrf_k=cfg.rrf_k,
                    top_k=max(cfg.dense_top_k, cfg.sparse_top_k)
                )
            all_candidates.extend(candidates)

        # 3. Deduplication of candidate chunks
        unique_candidates = deduplicate_documents(all_candidates, similarity_threshold=0.85)

        # 4. Cross-Encoder Reranking (FlashRank)
        if cfg.enable_reranker and unique_candidates:
            rerank_k = cfg.rerank_top_k if not cfg.enable_mmr else max(cfg.rerank_top_k, cfg.top_k * 2)
            ranked_docs = rerank_documents(
                query=query,
                documents=unique_candidates,
                top_k=rerank_k,
                score_threshold=cfg.score_threshold
            )
        else:
            ranked_docs = unique_candidates

        # 5. Maximum Marginal Relevance (MMR)
        if cfg.enable_mmr and len(ranked_docs) > cfg.top_k:
            final_docs = apply_mmr_to_documents(
                query=query,
                documents=ranked_docs,
                top_k=cfg.top_k,
                lambda_mult=cfg.mmr_lambda
            )
        else:
            final_docs = ranked_docs[:cfg.top_k]

        # 6. Final Score Threshold Filter
        if cfg.score_threshold is not None:
            final_docs = [d for d in final_docs if d.get("score", 0.0) >= cfg.score_threshold]

        return final_docs

    def retrieve_context_with_citations(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> Dict[str, Any]:
        """
        Retrieves top documents and packages them with:
        - Structured context block ready for LLM consumption
        - Verifiable source citations and provenance metadata
        """
        cfg = config or RetrievalConfig()
        docs = self.retrieve(query=query, config=cfg)
        context = compress_context(docs, max_characters=cfg.max_context_length)
        citations = format_citations(docs)

        return {
            "query": query,
            "context": context,
            "citations": citations,
            "documents": docs,
            "count": len(docs)
        }

