import time
import logfire
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_HF_DIM = 1024
_FASTEMBED_DIM = 384

_active_model = None
_model_type: str | None = None  


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Initialise Gemini embeddings if GEMINI_API_KEY is available."""
    if not settings.GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Test probe
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content="probe"
        )
        if "embedding" in res and res["embedding"]:
            logfire.info("Gemini embeddings ready (gemini-embedding-001, 3072-dim).")
            return genai
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}")
    return None


def _probe_hf():
    """Try one embed call to verify HuggingFace API is reachable. Returns model or None."""
    if not settings.HF_TOKEN:
        return None
    try:
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        model = HuggingFaceInferenceAPIEmbeddings(
            api_key=settings.HF_TOKEN,
            model_name="BAAI/bge-large-en-v1.5"
        )
        model.embed_query("probe")
        logfire.info("HuggingFace embeddings ready (bge-large-en-v1.5, 1024-dim).")
        return model
    except Exception as e:
        logfire.warning(f"HuggingFace probe failed: {e}. Will use FastEmbed fallback.")
        return None


def _load_fallback():
    """Fallback to FastEmbed (ONNX, fast, lightweight)."""
    try:
        from fastembed import TextEmbedding
        logfire.info("Loading FastEmbed fallback (bge-small-en-v1.5, 384-dim).")
        return TextEmbedding("BAAI/bge-small-en-v1.5")
    except Exception:
        from sentence_transformers import SentenceTransformer
        logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
        return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
        return

    hf = _probe_hf()
    if hf:
        _active_model = hf
        _model_type = "huggingface"
        return

    _active_model = _load_fallback()
    _model_type = "fastembed" if hasattr(_active_model, "embed") else "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    if _model_type == "gemini":
        return _GEMINI_DIM
    elif _model_type == "huggingface":
        return _HF_DIM
    elif _model_type == "fastembed":
        return _FASTEMBED_DIM
    return 768


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    # Sanitize surrogate characters from raw PDF text
    batch = [t.encode("utf-8", "ignore").decode("utf-8") for t in batch]
    if _model_type == "gemini":
        for attempt in range(8):
            try:
                res = _active_model.embed_content(
                    model="models/gemini-embedding-001",
                    content=batch
                )
                return res["embedding"]
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 7:
                    wait = 25 * (attempt + 1)
                    logfire.warning(f"Gemini rate limit hit — retrying in {wait}s (attempt {attempt + 1}/8).")
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 8 attempts.")

    elif _model_type == "huggingface":
        for attempt in range(8):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "too many requests"))
                if is_rate_limit and attempt < 7:
                    wait = 15 * (attempt + 1)
                    logfire.warning(f"HF rate limit hit — retrying in {wait}s (attempt {attempt + 1}/8).")
                    time.sleep(wait)
                else:
                    logfire.error(f"HF embedding failed: {e}")
                    raise
        raise RuntimeError("HF rate limit persisted after 8 attempts.")

    elif _model_type == "fastembed":
        return [v.tolist() for v in _active_model.embed(batch)]
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    query = query.encode("utf-8", "ignore").decode("utf-8")
    if _model_type == "gemini":
        res = _active_model.embed_content(
            model="models/gemini-embedding-001",
            content=query
        )
        return res["embedding"]
    elif _model_type == "huggingface":
        return _active_model.embed_query(query)
    elif _model_type == "fastembed":
        return list(_active_model.embed([query]))[0].tolist()
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
            
        if _model_type in ("gemini", "huggingface"):
            time.sleep(1)
            
    return all_embeddings
