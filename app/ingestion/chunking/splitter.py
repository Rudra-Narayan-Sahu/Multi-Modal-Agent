from typing import List
import logfire

def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    docs = []
    current_doc = []
    total = 0
    for d in splits:
        _len = len(d) + (len(separator) if current_doc else 0)
        if total + _len > chunk_size and current_doc:
            docs.append(separator.join(current_doc))
            while total > chunk_overlap or (total + _len > chunk_size and total > 0):
                total -= len(current_doc[0]) + (len(separator) if len(current_doc) > 1 else 0)
                current_doc = current_doc[1:]
        current_doc.append(d)
        total += _len
    if current_doc:
        docs.append(separator.join(current_doc))
    return docs

def _split_recursive(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    final_chunks = []
    separator = separators[-1]
    new_separators = []
    for i, s in enumerate(separators):
        if s == "":
            separator = s
            break
        if s in text:
            separator = s
            new_separators = separators[i + 1:]
            break

    splits = text.split(separator) if separator else list(text)
    good_splits = []
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))
                good_splits = []
            if not new_separators:
                final_chunks.append(s[:chunk_size])
            else:
                final_chunks.extend(_split_recursive(s, new_separators, chunk_size, chunk_overlap))
    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))
    return final_chunks

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    """
    Split text into chunks of specified size and overlap preserving semantic boundaries.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text or not text.strip(): 
            return []
            
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = _split_recursive(text, separators, chunk_size, chunk_overlap)
        valid_chunks = [c.strip() for c in chunks if c.strip()]
        logfire.info(f"Generated {len(valid_chunks)} chunks")
        return valid_chunks
