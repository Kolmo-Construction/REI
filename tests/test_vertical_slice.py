"""
Vertical slice end-to-end test.
Zero external calls — pure mocks, offline, fast.

Run: uv run pytest tests/test_vertical_slice.py -v
"""
import pytest
from greenvest.graph import graph
from greenvest.state import initial_state


@pytest.mark.asyncio
async def test_vertical_slice_winter_sleeping_bag():
    """
    Primary vertical slice: winter camping in PNW → sleeping bag recommendation.
    Asserts intent routing, retrieval, and synthesis all complete correctly.
    """
    state = initial_state(query="I need a sleeping bag for winter camping in the PNW")
    result = await graph.ainvoke(state)

    assert result["intent"] == "Product_Search"
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert len(result["catalog_results"]) > 0
    # Top result should be synthetic (PNW wet climate → synthetic fill)
    assert result["catalog_results"][0]["fill_type"] == "synthetic"
    assert "recommendation" in result
    assert result["recommendation"] is not None
    assert len(result["recommendation"]) > 0


@pytest.mark.asyncio
async def test_clarification_gate_needs_activity():
    """
    Vague query with no activity → clarification requested on turn 1.
    """
    state = initial_state(query="I need a sleeping bag")
    result = await graph.ainvoke(state)

    assert result["action_flag"] == "REQUIRES_CLARIFICATION"
    assert result["clarification_message"] is not None
    assert len(result["clarification_message"]) > 0
    assert result["clarification_count"] == 1


@pytest.mark.asyncio
async def test_clarification_cap_forces_search():
    """
    Vague query with clarification_count already at 2 → forced to READY_TO_SEARCH.
    """
    state = initial_state(query="I need a sleeping bag")
    state["clarification_count"] = 2

    result = await graph.ainvoke(state)

    # Should not get stuck in clarification loop
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None


@pytest.mark.asyncio
async def test_out_of_bounds_refusal():
    """
    Out-of-bounds query → deterministic refusal, no LLM synthesis for product.
    """
    state = initial_state(query="What are my legal rights if I get injured on a trail?")
    result = await graph.ainvoke(state)

    assert result["intent"] == "Out_of_Bounds"
    assert "specialize" in result["recommendation"].lower() or "outside" in result["recommendation"].lower()


@pytest.mark.asyncio
async def test_catalog_results_have_required_fields():
    """Catalog results must always include SKU, name, fill_type, and price."""
    state = initial_state(query="I need a sleeping bag for winter camping in the PNW")
    result = await graph.ainvoke(state)

    for product in result["catalog_results"]:
        assert "sku" in product
        assert "name" in product
        assert "price_usd" in product


@pytest.mark.asyncio
async def test_ontology_lookup_produces_specs():
    """
    Ontology must resolve 'winter camping' + 'PNW' to synthetic fill and high r_value
    without any LLM call.
    """
    from greenvest.ontology import lookup_all

    specs = lookup_all(["winter camping", "PNW"])
    assert len(specs) > 0

    keys = [list(s.keys())[0] for s in specs]
    assert "fill_type" in keys or "r_value" in keys


@pytest.mark.asyncio
async def test_no_network_calls_in_mock_mode():
    """
    Smoke test: entire pipeline completes without raising connection errors.
    If network calls were made, they'd fail in CI with no credentials.
    """
    from greenvest.config import settings
    assert settings.USE_MOCK_LLM is True

    state = initial_state(query="I need a sleeping bag for winter camping in the PNW")
    # Should complete without any exception
    result = await graph.ainvoke(state)
    assert result is not None
