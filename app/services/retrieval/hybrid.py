import os
import re
import json
import logfire
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from app.config import settings

BM25_CACHE_PATH = os.path.join("processed_data", "bm25_corpus.json")


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """
    In-memory BM25 index over ingested knowledge base chunks.
    Can sync from Qdrant Cloud or load from a local cache.
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_initialized: bool = False

    def load_or_build(self, client: Optional[QdrantClient] = None, force_rebuild: bool = False) -> None:
        """Load from local cache if valid, otherwise fetch from Qdrant and build index."""
        if self._is_initialized and not force_rebuild:
            return

        if not force_rebuild and os.path.exists(BM25_CACHE_PATH):
            try:
                with open(BM25_CACHE_PATH, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                if self.documents:
                    self.corpus_tokens = [tokenize(doc.get("text", "")) for doc in self.documents]
                    self.bm25 = BM25Okapi(self.corpus_tokens)
                    self._is_initialized = True
                    logfire.info(f"Loaded BM25 index with {len(self.documents)} chunks from cache.")
                    return
            except Exception as e:
                logfire.warning(f"Failed to load cached BM25 index: {e}. Rebuilding...")

        self.rebuild_from_qdrant(client)

    def rebuild_from_qdrant(self, client: Optional[QdrantClient] = None) -> None:
        """Fetch all points from Qdrant and construct the BM25 model."""
        if client is None:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=True
            )

        logfire.info("Fetching corpus from Qdrant for BM25 indexing...")
        docs: List[Dict[str, Any]] = []
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                payload = r.payload or {}
                docs.append({
                    "id": str(r.id),
                    "text": payload.get("text", ""),
                    "file_name": payload.get("file_name"),
                    "source_type": payload.get("source_type"),
                    "source_name": payload.get("source_name"),
                    "chunk_index": payload.get("chunk_index"),
                    "local_path": payload.get("local_path"),
                })
            if next_offset is None or not records:
                break
            offset = next_offset

        self.documents = docs
        if self.documents:
            self.corpus_tokens = [tokenize(d.get("text", "")) for d in self.documents]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self._is_initialized = True

            os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
            try:
                with open(BM25_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.documents, f, ensure_ascii=False)
            except Exception as e:
                logfire.warning(f"Could not cache BM25 corpus to disk: {e}")

            logfire.info(f"BM25 index built with {len(self.documents)} documents.")
        else:
            logfire.warning("No documents retrieved from Qdrant for BM25 indexing.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search over the indexed chunks with optional metadata filtering."""
        if not self._is_initialized or self.bm25 is None or not self.documents:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Filter and rank
        scored_docs = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            doc = self.documents[idx]

            # Apply metadata filters if specified
            if filters:
                matches = True
                for k, v in filters.items():
                    if doc.get(k) != v:
                        matches = False
                        break
                if not matches:
                    continue

            doc_copy = dict(doc)
            doc_copy["score"] = float(score)
            doc_copy["retrieval_source"] = "bm25"
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


# Global singleton instance
_bm25_index = BM25Index()


def get_bm25_index(client: Optional[QdrantClient] = None) -> BM25Index:
    """Retrieve or lazily initialize the singleton BM25 index."""
    if not _bm25_index._is_initialized:
        _bm25_index.load_or_build(client=client)
    return _bm25_index


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Combine dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).
    RRF Score = SUM( 1 / (rrf_k + rank_i) )
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Helper to build unique key for deduplication
    def doc_key(doc: Dict[str, Any]) -> str:
        return doc.get("id") or f"{doc.get('file_name')}_{doc.get('chunk_index')}"

    # Process dense results
    for rank, doc in enumerate(dense_results, start=1):
        key = doc_key(doc)
        if key not in doc_map:
            doc_map[key] = dict(doc)
            doc_map[key]["dense_score"] = doc.get("score", 0.0)
            doc_map[key]["sparse_score"] = 0.0
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    # Process sparse (BM25) results
    for rank, doc in enumerate(sparse_results, start=1):
        key = doc_key(doc)
        if key not in doc_map:
            doc_map[key] = dict(doc)
            doc_map[key]["dense_score"] = 0.0
            doc_map[key]["sparse_score"] = doc.get("score", 0.0)
        else:
            doc_map[key]["sparse_score"] = doc.get("score", 0.0)
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    # Sort documents by RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_docs = []
    for key in sorted_keys[:top_k]:
        doc = doc_map[key]
        doc["rrf_score"] = scores[key]
        doc["score"] = scores[key]  # Primary ranking score
        doc["retrieval_source"] = "hybrid_rrf"
        fused_docs.append(doc)

    return fused_docs

