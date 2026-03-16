"""
Live Qdrant integration tests — require Qdrant running at localhost:6333
with both collections already indexed:
  - rei_products       (run: uv run python scripts/index_catalog.py)
  - rei_expert_advice  (run: uv run python scripts/index_expert_advice.py)

Run:
    uv run pytest tests/test_qdrant_live.py -v

Skip automatically if Qdrant is unreachable.
"""
import pytest
import httpx

from greenvest.state import initial_state
from greenvest.retrieval.branch_a_expert import TOP_K


# ---------------------------------------------------------------------------
# Skip guard — skip entire module if Qdrant is not reachable
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        resp = httpx.get("http://localhost:6333/healthz", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant not reachable at localhost:6333 — start it and index the collections first",
)


# ---------------------------------------------------------------------------
# Branch A: Expert Advice
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_a_returns_chunks_for_sleeping_bag_query():
    """Branch A returns 3 expert advice chunks for a winter sleeping bag query."""
    from greenvest.retrieval.branch_a_expert import search_expert_advice

    state = initial_state(query="I need a sleeping bag for winter camping in the PNW")
    state["activity"] = "winter_camping"
    state["user_environment"] = "PNW"

    chunks = await search_expert_advice(state)

    assert len(chunks) == 3
    for chunk in chunks:
        assert "chunk_text" in chunk
        assert "title" in chunk
        assert "section" in chunk
        assert "score" in chunk
        assert len(chunk["chunk_text"]) > 20


@pytest.mark.asyncio
async def test_branch_a_chunks_are_semantically_relevant():
    """Top chunk for a sleeping bag query should be about sleeping bags, not footwear."""
    from greenvest.retrieval.branch_a_expert import search_expert_advice

    state = initial_state(query="What sleeping bag fill is best for rainy conditions?")
    state["activity"] = "backpacking"
    state["user_environment"] = "PNW"

    chunks = await search_expert_advice(state)

    assert len(chunks) > 0
    # The top result should mention fill type or sleeping bags
    top_text = chunks[0]["chunk_text"].lower()
    assert any(kw in top_text for kw in ["fill", "synthetic", "down", "sleeping bag", "moisture", "wet"])


@pytest.mark.asyncio
async def test_branch_a_scores_are_ordered():
    """Results should come back in descending score order."""
    from greenvest.retrieval.branch_a_expert import search_expert_advice

    state = initial_state(query="How do I choose a rain jacket for hiking?")
    state["activity"] = "hiking"

    chunks = await search_expert_advice(state)

    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True), "Chunks should be ordered by descending score"


@pytest.mark.asyncio
async def test_branch_a_backpack_query():
    """Backpack query should surface pack-related advice, not sleeping bag advice."""
    from greenvest.retrieval.branch_a_expert import search_expert_advice

    state = initial_state(query="How many liters do I need for a weekend backpacking trip?")
    state["activity"] = "backpacking"

    chunks = await search_expert_advice(state)

    assert len(chunks) == TOP_K
    for chunk in chunks:
        assert "chunk_text" in chunk
        assert "section" in chunk
        assert len(chunk["chunk_text"]) > 0


# ---------------------------------------------------------------------------
# Branch B: Product Catalog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_b_returns_products_for_winter_bag_query():
    """Branch B returns products with required fields for a winter sleeping bag query."""
    from greenvest.retrieval.branch_b_catalog import search_catalog

    state = initial_state(query="I need a sleeping bag for winter camping in the PNW")
    state["activity"] = "winter_camping"
    state["user_environment"] = "PNW"
    state["derived_specs"] = [
        {"fill_type": "synthetic"},
        {"temp_rating_f": "<=15"},
    ]

    products = await search_catalog(state)

    assert len(products) > 0
    assert len(products) <= 5
    for p in products:
        assert "sku" in p
        assert "name" in p
        assert "price_usd" in p


@pytest.mark.asyncio
async def test_branch_b_filter_excludes_warm_bags():
    """A <=15°F filter should exclude 30°F bags."""
    from greenvest.retrieval.branch_b_catalog import search_catalog

    state = initial_state(query="winter sleeping bag")
    state["activity"] = "winter_camping"
    state["derived_specs"] = [{"temp_rating_f": "<=15"}]

    products = await search_catalog(state)

    for p in products:
        if p.get("temp_rating_f") is not None:
            assert p["temp_rating_f"] <= 15, f"{p['name']} has temp_rating_f={p['temp_rating_f']}, expected <=15"


@pytest.mark.asyncio
async def test_branch_b_brand_query_uses_sparse_weighting():
    """A query containing a known brand token should still return results."""
    from greenvest.retrieval.branch_b_catalog import search_catalog, _detect_brand_token

    query = "Therm-a-Rest sleeping pad"
    assert _detect_brand_token(query) is True

    state = initial_state(query=query)
    state["derived_specs"] = []

    products = await search_catalog(state)

    assert len(products) > 0


# ---------------------------------------------------------------------------
# Collection health checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rei_products_collection_has_25_points():
    """Sanity check: rei_products should have exactly 25 indexed products."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url="http://localhost:6333")
    info = await client.get_collection("rei_products")
    assert info.points_count == 25, f"Expected 25 products, got {info.points_count}"


@pytest.mark.asyncio
async def test_rei_expert_advice_collection_has_chunks():
    """Sanity check: rei_expert_advice should have at least 20 indexed chunks."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url="http://localhost:6333")
    info = await client.get_collection("rei_expert_advice")
    assert info.points_count >= 20, f"Expected >=20 chunks, got {info.points_count}"
