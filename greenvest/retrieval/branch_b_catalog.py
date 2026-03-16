"""
Branch B — Product Catalog Retrieval.

Phase 11: Qdrant hybrid search (SPLADE sparse + BAAI/bge-large-en-v1.5 dense, server-side RRF).
Interface is identical to the Phase 6 flat-JSON version — no callers changed.
"""
from __future__ import annotations
import asyncio
import re
from typing import Optional

import structlog
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import SparseVector

from greenvest.config import settings
from greenvest.retrieval.embeddings import dense_model as _dense_model, sparse_model as _sparse_model
from greenvest.state import GreenvestState

log = structlog.get_logger(__name__)

COLLECTION_NAME = "rei_products"

# Common outdoor brand tokens — triggers sparse-heavy RRF weighting
_BRAND_TOKENS = {
    "rei", "patagonia", "north face", "tnf", "black diamond",
    "petzl", "arcteryx", "arc'teryx", "marmot", "osprey", "deuter",
    "gregory", "msr", "big agnes", "nemo", "sea to summit",
    "jetboil", "garmin", "suunto", "salomon", "scarpa", "la sportiva",
    "gore-tex", "goretex", "primaloft", "polartec",
    "therm-a-rest", "thermarest", "western mountaineering", "kelty", "nemo",
    "merrell",
}



def _qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.QDRANT_URL)


def _detect_brand_token(query: str) -> bool:
    q = query.lower()
    return any(brand in q for brand in _BRAND_TOKENS)


def _build_filter(derived_specs: dict[str, str]) -> Optional[models.Filter]:
    """Convert derived_specs into a Qdrant Filter applied during ANN traversal (pre-filtering)."""
    conditions = []
    numeric_fields = {"temp_rating_f", "r_value", "weight_oz"}
    keyword_fields = {"fill_type", "water_resistance"}

    for key, value in derived_specs.items():
            if key not in numeric_fields | keyword_fields:
                continue

            if key in numeric_fields and isinstance(value, str):
                m = re.match(r"^([<>]=?)(-?\d+\.?\d*)$", value.strip())
                if m:
                    op, num = m.group(1), float(m.group(2))
                    range_kwargs: dict = {}
                    if op == "<=": range_kwargs["lte"] = num
                    elif op == ">=": range_kwargs["gte"] = num
                    elif op == "<":  range_kwargs["lt"] = num
                    elif op == ">":  range_kwargs["gt"] = num
                    conditions.append(
                        models.FieldCondition(key=key, range=models.Range(**range_kwargs))
                    )

            elif key in keyword_fields and isinstance(value, str):
                if " OR " in value:
                    options = [v.strip() for v in value.split(" OR ")]
                    conditions.append(
                        models.FieldCondition(key=key, match=models.MatchAny(any=options))
                    )
                else:
                    conditions.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    )

    return models.Filter(must=conditions) if conditions else None


def _build_query_document(state: GreenvestState) -> str:
    """Enrich the raw query with extracted state for better ANN precision."""
    parts = [state.get("query", "")]
    if state.get("activity"):
        parts.append(state["activity"].replace("_", " "))
    if state.get("user_environment"):
        parts.append(state["user_environment"].replace("_", " "))
    for k, v in state.get("derived_specs", {}).items():
        parts.append(f"{k}: {v}")
    return ". ".join(p for p in parts if p)


async def search_catalog(state: GreenvestState) -> list[dict]:
    """
    Hybrid search: SPLADE sparse + BAAI/bge-large-en-v1.5 dense, fused server-side via RRF.
    Returns top 5 products as dicts (same shape as sample_products.json).
    """
    query_doc = _build_query_document(state)
    spec_filter = _build_filter(state.get("derived_specs", {}))

    # Dense and sparse embeddings generated concurrently (both run in executor — CPU-bound)
    loop = asyncio.get_event_loop()
    dense_task = loop.run_in_executor(
        None,
        lambda: list(_dense_model().embed([query_doc]))[0],
    )
    sparse_task = loop.run_in_executor(
        None,
        lambda: list(_sparse_model().embed([query_doc]))[0],
    )

    dense_result, sparse_result = await asyncio.gather(dense_task, sparse_task)
    dense_vec = dense_result.tolist()
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    # Brand queries favor sparse (exact token match dominates)
    has_brand = _detect_brand_token(state.get("query", ""))
    sparse_weight, dense_weight = (2.0, 1.0) if has_brand else (1.0, 2.0)

    results = await _qdrant().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=sparse_vec,
                using="sparse_text",
                filter=spec_filter,
                limit=50,
            ),
            models.Prefetch(
                query=dense_vec,
                using="dense_text",
                filter=spec_filter,
                limit=50,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[sparse_weight, dense_weight])),
        limit=5,
        with_payload=True,
    )

    hits = [point.payload for point in results.points]

    log.info(
        "branch_b_catalog",
        session_id=state["session_id"],
        matched=len(hits),
        brand_query=has_brand,
        filter_active=spec_filter is not None,
    )

    return hits
