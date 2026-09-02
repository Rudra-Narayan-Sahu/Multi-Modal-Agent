import logfire
from sentence_transformers import SentenceTransformer
from app.config import settings

BATCH_SIZE = 32
_model = None
_embedding_dim = None

def _get_model() -> SentenceTransformer:
    global _model, _embedding_dim
    if _model is None:
        model_name = settings.EMBEDDING_MODEL_NAME or "all-MiniLM-L6-v2"
        logfire.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        if hasattr(_model, "get_embedding_dimension"):
            _embedding_dim = _model.get_embedding_dimension()
        else:
            _embedding_dim = _model.get_sentence_embedding_dimension()
        logfire.info(f"Embedding model loaded successfully. Dimension: {_embedding_dim}")
    return _model

def get_embedding_dim() -> int:
    """Return the vector dimension for the active embedding model."""
    global _embedding_dim
    if _embedding_dim is None:
        _get_model()
    return _embedding_dim

def embed_query(query: str) -> list[float]:
    """Generate vector embedding for a single search query."""
    if not query or not query.strip():
        dim = get_embedding_dim()
        return [0.0] * dim
    model = _get_model()
    embedding = model.encode(query.strip(), convert_to_numpy=True)
    return embedding.tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for a list of text chunks in batches."""
    if not texts:
        return []
    
    model = _get_model()
    cleaned_texts = [t if t.strip() else " " for t in texts]
    
    with logfire.span("Embed texts", total_chunks=len(cleaned_texts), batch_size=BATCH_SIZE):
        embeddings = model.encode(
            cleaned_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    return embeddings.tolist()