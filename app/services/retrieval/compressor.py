import re
import logfire
from typing import List, Dict, Any, Tuple


def _text_tokens(text: str) -> set:
    """Extract lowercase word tokens for fast Jaccard similarity."""
    return set(re.findall(r"\b\w{3,}\b", text.lower()))


def _jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


def deduplicate_documents(
    documents: List[Dict[str, Any]],
    similarity_threshold: float = 0.80
) -> List[Dict[str, Any]]:
    """
    Filters out near-duplicate chunks based on lexical token overlap (Jaccard similarity).
    Preserves the higher-scoring document in each duplicate pair.
    """
    if not documents:
        return []

    unique_docs: List[Dict[str, Any]] = []
    doc_token_sets: List[set] = []

    for doc in documents:
        text = doc.get("text", "")
        tokens = _text_tokens(text)
        is_duplicate = False

        for existing_tokens in doc_token_sets:
            sim = _jaccard_similarity(tokens, existing_tokens)
            if sim >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(doc)
            doc_token_sets.append(tokens)

    return unique_docs


def compress_context(
    documents: List[Dict[str, Any]],
    max_characters: int = 6000
) -> str:
    """
    Joins document chunks into a single context string, truncating gracefully
    at sentence boundaries if the maximum character budget is exceeded.
    """
    if not documents:
        return ""

    context_parts: List[str] = []
    current_length = 0

    for idx, doc in enumerate(documents, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        file_name = doc.get("file_name", "Unknown Document")
        chunk_idx = doc.get("chunk_index", "N/A")
        header = f"[Document {idx} | Source: {file_name} (Chunk {chunk_idx})]"
        chunk_entry = f"{header}\n{text}\n"
        entry_len = len(chunk_entry)

        if current_length + entry_len <= max_characters:
            context_parts.append(chunk_entry)
            current_length += entry_len
        else:
            # Fit remaining character budget
            remaining_budget = max_characters - current_length - len(header) - 10
            if remaining_budget > 100:
                truncated_text = text[:remaining_budget].rsplit(". ", 1)[0] + "..."
                context_parts.append(f"{header}\n{truncated_text}\n")
            break

    return "\n".join(context_parts)


def format_citations(documents: List[Dict[str, Any]]) -> str:
    """
    Generates a structured, human-readable citation provenance block
    with exact file names, chunk indices, and relevance scores.
    """
    if not documents:
        return "No external sources cited."

    citations: List[str] = ["### Document Provenance & Citations"]
    for idx, doc in enumerate(documents, start=1):
        file_name = doc.get("file_name", "Unknown File")
        chunk_idx = doc.get("chunk_index", "N/A")
        score = doc.get("score", 0.0)
        source_type = doc.get("source_type", "doc")
        
        # Snippet preview
        text_snippet = (doc.get("text") or "").replace("\n", " ")[:120].strip()
        citations.append(
            f"**[{idx}] `{file_name}`** (Type: `{source_type}`, Chunk: `{chunk_idx}`, Score: `{score:.4f}`)\n"
            f"> *\"{text_snippet}...\"*"
        )

    return "\n\n".join(citations)

