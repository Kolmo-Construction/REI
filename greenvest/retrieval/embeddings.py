"""
Shared embedding model instances for all retrieval branches.

A single module-level cache prevents both branches loading BAAI/bge-large-en-v1.5
independently (the previous double-load cost ~6s on cold start).

Call warmup_models() once at process startup to pay the load cost upfront,
before the first user request arrives.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

DENSE_MODEL  = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL)


@lru_cache(maxsize=1)
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL)


def warmup_models() -> None:
    """
    Load both models into the LRU cache synchronously.
    Call once at startup (e.g. in scripts/run_agent.py or app lifespan).
    After this returns, all retrieval calls skip the load cost entirely.
    """
    import time
    t = time.monotonic()
    dense_model()
    sparse_model()
    elapsed = round((time.monotonic() - t) * 1000)
    # Use print here — logging may not be configured yet at startup
    print(f"[embeddings] models warm ({elapsed}ms)")
