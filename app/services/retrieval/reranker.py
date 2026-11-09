import os
import tempfile
import logfire
from typing import List, Dict, Any, Optional

_ranker_instance = None


def _get_ranker(model_name: str = "ms-marco-TinyBERT-L-2-v2"):
    """Initialize or return the singleton FlashRank Ranker instance."""
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker
            # Use cached directory where model is downloaded
            cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
            os.makedirs(cache_dir, exist_ok=True)
            _ranker_instance = Ranker(model_name=model_name, cache_dir=cache_dir)
            logfire.info(f"FlashRank Ranker initialized with model: {model_name}")
        except Exception as e:
            logfire.error(f"Failed to initialize FlashRank Ranker: {e}")
            _ranker_instance = None
    return _ranker_instance


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Reranks a list of candidate documents using FlashRank cross-encoder.
    Preserves original metadata and adds `rerank_score`.
    """
    if not documents:
        return []

    ranker = _get_ranker()
    if ranker is None:
        logfire.warning("Reranker unavailable — returning original document ranking.")
        for d in documents:
            if "score" not in d:
                d["score"] = 0.0
        return documents[:top_k]

    with logfire.span("Cross-Encoder Reranking", query=query, candidates=len(documents)):
        try:
            from flashrank import RerankRequest

            # Build passages preserving reference to original document index
            passages = []
            for idx, doc in enumerate(documents):
                passages.append({
                    "id": idx,
                    "text": doc.get("text", "")
                })

            rerank_request = RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rerank(rerank_request)

            reranked_docs: List[Dict[str, Any]] = []
            for item in ranked_results:
                idx = item["id"]
                score = float(item["score"])

                if score_threshold is not None and score < score_threshold:
                    continue

                doc = dict(documents[idx])
                doc["pre_rerank_score"] = doc.get("score")
                doc["rerank_score"] = score
                doc["score"] = score  # Primary ranking score
                reranked_docs.append(doc)

            logfire.info(f"Reranked {len(documents)} candidates into {len(reranked_docs[:top_k])} results.")
            return reranked_docs[:top_k]

        except Exception as e:
            logfire.error(f"Reranking execution failed: {e}")
            return documents[:top_k]
