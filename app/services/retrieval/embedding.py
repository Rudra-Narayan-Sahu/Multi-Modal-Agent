import time
import logfire
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_hf():
    """Try one embed call to verify HuggingFace API is reachable. Returns model or None."""
    try:
        model = HuggingFaceInferenceAPIEmbeddings(
            api_key=settings.HF_TOKEN,
            model_name="BAAI/bge-large-en-v1.5" # Excellent open-source embedding model
        )
        model.embed_query("probe")
        logfire.info("HuggingFace embeddings ready (bge-large-en-v1.5, 1024-dim).")
        return model
    except Exception as e:
        logfire.warning(f"HuggingFace probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process."""
    global _active_model, _model_type, _GEMINI_DIM
    if _active_model is not None:
        return

    hf = _probe_hf()
    if hf:
        _active_model = hf
        _model_type = "huggingface"
        _GEMINI_DIM = 1024 # BAAI/bge-large-en-v1.5 dimension
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "huggingface" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "huggingface":
        for attempt in range(8):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "too many requests"))
                if is_rate_limit and attempt < 7:
                    wait = 15 * (attempt + 1)
                    logfire.warning(
                        f"HF rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/8)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"HF embedding failed: {e}")
                    raise
        raise RuntimeError("HF rate limit persisted after 8 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "huggingface":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
            
        # Proactively sleep for 2 seconds after each batch
        if _model_type == "huggingface":
            time.sleep(2)
            
    return all_embeddings