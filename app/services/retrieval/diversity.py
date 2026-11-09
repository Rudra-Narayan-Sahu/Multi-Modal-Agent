import numpy as np
import logfire
from typing import List, Dict, Any, Optional
from app.services.retrieval.embedding import embed_texts, embed_query


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between 2D array a and 2D array b."""
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    a_norm[a_norm == 0] = 1e-10
    b_norm[b_norm == 0] = 1e-10
    return np.dot(a / a_norm, (b / b_norm).T)


def maximal_marginal_relevance(
    query_vector: List[float],
    doc_vectors: List[List[float]],
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    lambda_mult: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Selects top_k documents using Maximum Marginal Relevance (MMR).
    MMR = argmax [ lambda * Sim(doc, query) - (1 - lambda) * max_{s in S} Sim(doc, s) ]
    
    Args:
        query_vector: Embedded representation of the query.
        doc_vectors: Embedded representations of candidate documents.
        documents: Candidate document dictionaries.
        top_k: Number of diverse documents to return.
        lambda_mult: Balance between relevance (1.0) and diversity (0.0). Default 0.7.
    """
    if not documents or not doc_vectors:
        return []

    top_k = min(top_k, len(documents))
    q_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    d_vecs = np.array(doc_vectors, dtype=np.float32)

    # Compute similarity to query
    query_sims = cosine_similarity(d_vecs, q_vec).flatten()

    # Pre-compute document-to-document similarity matrix
    doc_sims = cosine_similarity(d_vecs, d_vecs)

    selected_indices: List[int] = []
    unselected_indices = list(range(len(documents)))

    # Select the first document with highest query similarity
    first_idx = int(np.argmax(query_sims))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    # Greedily select remaining documents
    while len(selected_indices) < top_k and unselected_indices:
        best_mmr = -float("inf")
        best_idx = None

        for idx in unselected_indices:
            relevance = query_sims[idx]
            max_redundancy = max(doc_sims[idx, s] for s in selected_indices)
            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * max_redundancy)

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        else:
            break

    mmr_docs: List[Dict[str, Any]] = []
    for rank, idx in enumerate(selected_indices, start=1):
        doc = dict(documents[idx])
        doc["mmr_rank"] = rank
        doc["original_score"] = doc.get("score")
        mmr_docs.append(doc)

    return mmr_docs


def apply_mmr_to_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    lambda_mult: float = 0.7
) -> List[Dict[str, Any]]:
    """Convenience helper: computes vectors on the fly if not already provided."""
    if not documents:
        return []
    if len(documents) <= top_k:
        return documents

    with logfire.span("MMR Diversification", candidates=len(documents), top_k=top_k, lambda_mult=lambda_mult):
        try:
            q_vec = embed_query(query)
            texts = [d.get("text", "") for d in documents]
            d_vecs = embed_texts(texts)
            return maximal_marginal_relevance(
                query_vector=q_vec,
                doc_vectors=d_vecs,
                documents=documents,
                top_k=top_k,
                lambda_mult=lambda_mult
            )
        except Exception as e:
            logfire.error(f"MMR failed: {e}. Returning original ranking.")
            return documents[:top_k]

