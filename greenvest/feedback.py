"""
Interaction recorder — captures customer Q&A pairs and behavioral signals.

Two purposes:
  1. Short-term: LLM-as-a-Judge evaluation source (Phase 18)
  2. Long-term: curated interactions get promoted into rei_expert_advice
     (a human or automated review pipeline promotes high-quality Q&A into
     the expert advice index, closing the feedback loop)

Storage: Qdrant collection "customer_interactions" (same instance as products).
Each point = one completed interaction (query → recommendation).

Behavioral signals recorded:
  - thumbs_up / thumbs_down (explicit)
  - follow_up_asked (implicit: customer sent another message after recommendation)
  - session_ended_after_recommendation (implicit: session closed within 2 min)
  - clarification_turns (how many rounds before READY_TO_SEARCH)
"""
from __future__ import annotations

import hashlib
import time
from typing import Literal, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct

from greenvest.config import settings
from greenvest.state import GreenvestState

COLLECTION_NAME = "customer_interactions"
PROMOTED_TAG = "promoted_to_expert_advice"


def _qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.QDRANT_URL)


def _interaction_id(session_id: str, ts: float) -> int:
    raw = f"{session_id}|{ts}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


async def ensure_collection() -> None:
    """Idempotent — call once at server startup."""
    client = _qdrant()
    existing = {c.name for c in (await client.get_collections()).collections}
    if COLLECTION_NAME in existing:
        return

    # No vectors — this collection is payload-only (used for filtering/lookup,
    # not ANN search). Qdrant supports payload-only collections via a dummy
    # sparse config; we simply store and filter by payload fields.
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},          # no vectors — payload-only
        sparse_vectors_config={},
    )

    for field, schema_type in [
        ("session_id",   models.PayloadSchemaType.KEYWORD),
        ("intent",       models.PayloadSchemaType.KEYWORD),
        ("activity",     models.PayloadSchemaType.KEYWORD),
        ("rating",       models.PayloadSchemaType.KEYWORD),
        ("promoted",     models.PayloadSchemaType.BOOL),
        ("timestamp",    models.PayloadSchemaType.FLOAT),
    ]:
        await client.create_payload_index(COLLECTION_NAME, field, schema_type)


async def record_interaction(
    state: GreenvestState,
    rating: Optional[Literal["thumbs_up", "thumbs_down"]] = None,
    follow_up_asked: bool = False,
    session_ended_after_recommendation: bool = False,
) -> None:
    """
    Call this after the synthesizer node completes.
    `rating` is None until the customer explicitly rates (thumbs up/down).
    Call `update_rating()` later when the signal arrives.
    """
    ts = time.time()
    point_id = _interaction_id(state["session_id"], ts)

    payload = {
        # Core Q&A pair — the raw material for expert advice promotion
        "query":              state.get("query", ""),
        "recommendation":     state.get("recommendation", ""),

        # Intent classification
        "intent":             state.get("intent"),
        "activity":           state.get("activity"),
        "user_environment":   state.get("user_environment"),
        "experience_level":   state.get("experience_level"),
        "derived_specs":      state.get("derived_specs", []),

        # Retrieval metadata
        "catalog_skus":       [p.get("sku") for p in state.get("catalog_results", [])],
        "expert_chunks_used": len(state.get("expert_context", [])),

        # Behavioral signals
        "clarification_turns":                    state.get("clarification_count", 0),
        "follow_up_asked":                        follow_up_asked,
        "session_ended_after_recommendation":     session_ended_after_recommendation,
        "rating":                                 rating,  # None until explicit signal

        # Lifecycle
        "timestamp":  ts,
        "promoted":   False,    # True once reviewed + added to rei_expert_advice
        "session_id": state["session_id"],
        "store_id":   state.get("store_id", ""),
    }

    await _qdrant().upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector={}, payload=payload)],
    )


async def update_rating(
    session_id: str,
    rating: Literal["thumbs_up", "thumbs_down"],
) -> None:
    """
    Called when the customer taps thumbs up / thumbs down in the UI.
    Patches the rating field on the existing interaction point.
    """
    client = _qdrant()

    # Find the most recent interaction for this session
    results = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )]
        ),
        limit=1,
        order_by=models.OrderBy(key="timestamp", direction=models.Direction.DESC),
        with_payload=True,
    )

    points, _ = results
    if not points:
        return

    await client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"rating": rating},
        points=[points[0].id],
    )


async def get_promotable_interactions(
    min_rating: Literal["thumbs_up"] = "thumbs_up",
    limit: int = 50,
) -> list[dict]:
    """
    Returns interactions eligible to be promoted into rei_expert_advice.
    Criteria: thumbs_up + not yet promoted + recommendation is non-empty.

    A human reviewer (or automated script) calls this, reviews the Q&A pairs,
    and calls promote_to_expert_advice() on approved ones.
    """
    client = _qdrant()
    results, _ = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="rating",    match=models.MatchValue(value="thumbs_up")),
                models.FieldCondition(key="promoted",  match=models.MatchValue(value=False)),
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    return [p.payload for p in results]


async def promote_to_expert_advice(
    session_id: str,
    point_id: int,
) -> dict:
    """
    Marks the interaction as promoted and returns the chunk dict ready to be
    passed to index_expert_advice_chunk() in scripts/index_expert_advice.py.

    The actual embedding + Qdrant upsert into rei_expert_advice happens in
    that script (run offline / as a cron job).
    """
    await _qdrant().set_payload(
        collection_name=COLLECTION_NAME,
        payload={"promoted": True},
        points=[point_id],
    )

    # Fetch the payload to build the chunk
    results, _ = await _qdrant().scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="session_id", match=models.MatchValue(value=session_id)
            )]
        ),
        limit=1,
        with_payload=True,
    )
    p = results[0].payload if results else {}

    # Shape matches what the scraper produces — same indexing pipeline
    return {
        "url":        f"internal://interactions/{session_id}",
        "title":      f"Customer Q&A: {p.get('activity', 'general')}",
        "section":    p.get("query", ""),
        "category":   p.get("activity", "general"),
        "chunk_text": (
            f"Customer question: {p.get('query', '')}\n"
            f"REI recommendation: {p.get('recommendation', '')}"
        ),
        "source":     "customer_interaction",
    }
