"""
Branch A: Expert Advice vector search against Qdrant rei_expert_advice collection.

Dense-only search (BAAI/bge-large-en-v1.5) — expert advice is narrative text,
so semantic similarity dominates. No sparse/RRF needed here.

Returns top 3 chunks as dicts with keys: title, section, chunk_text, url.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

import structlog
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient

from greenvest.config import settings
from greenvest.state import GreenvestState

log = structlog.get_logger(__name__)

COLLECTION_NAME = "rei_expert_advice"
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K = 3


@lru_cache(maxsize=1)
def _dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL)


def _qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.QDRANT_URL)


def _build_query(state: GreenvestState) -> str:
    """Enrich the raw query with activity and environment for better recall."""
    parts = [state.get("query", "")]
    if state.get("activity"):
        parts.append(state["activity"].replace("_", " "))
    if state.get("user_environment"):
        parts.append(state["user_environment"].replace("_", " "))
    return ". ".join(p for p in parts if p)


async def search_expert_advice(state: GreenvestState) -> list[dict]:
    """
    Dense ANN search against rei_expert_advice.
    Returns top 3 chunks as dicts ready for context assembly.
    """
    query = _build_query(state)

    loop = asyncio.get_event_loop()
    dense_vec = await loop.run_in_executor(
        None,
        lambda: list(_dense_model().embed([query]))[0].tolist(),
    )

    results = await _qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using="dense_text",
        limit=TOP_K,
        with_payload=True,
    )

    chunks = [
        {
            "title":      hit.payload.get("title", ""),
            "section":    hit.payload.get("section", ""),
            "chunk_text": hit.payload.get("chunk_text", ""),
            "url":        hit.payload.get("url", ""),
            "score":      hit.score,
        }
        for hit in results.points
    ]

    log.info(
        "branch_a_expert",
        session_id=state["session_id"],
        matched=len(chunks),
        top_score=chunks[0]["score"] if chunks else None,
    )

    return chunks
