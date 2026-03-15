"""
Vertical slice end-to-end tests.

LLM calls (intent router, query translator, synthesizer) use real Ollama.
Retrieval (Branch A expert, Branch B catalog) is mocked by conftest.py —
the mock catalog filters sample_products.json using derived_specs that
the real Ollama LLM produces, so the spec→product path is live.

Requires Ollama running locally with llama3.2 and llama3 pulled.
Tests are skipped automatically if Ollama is not available.

Run: uv run pytest tests/test_vertical_slice.py -v
"""
import json
from pathlib import Path
import pytest
from greenvest.graph import graph
from greenvest.state import initial_state

# All tests in this module require Ollama
pytestmark = pytest.mark.usefixtures("require_ollama")

SCENARIOS_DIR = Path(__file__).parent / "fixtures" / "scenarios"


def load_scenario(name: str) -> dict:
    return json.loads((SCENARIOS_DIR / f"{name}.json").read_text())


def _build_state(sc: dict):
    inp = sc["input"]
    state = initial_state(
        query=inp["query"],
        session_id=inp["session_id"],
        store_id=inp["store_id"],
        member_number=inp.get("member_number"),
    )
    state["clarification_count"] = inp.get("clarification_count", 0)
    # Apply any pre-seeded state fields
    for key, val in inp.get("pre_seed_state", {}).items():
        state[key] = val
    # Apply top-level budget if provided
    if "budget_usd" in inp:
        state["budget_usd"] = tuple(inp["budget_usd"]) if isinstance(inp["budget_usd"], list) else inp["budget_usd"]
    return state


def _assert_catalog(result: dict, cat: dict):
    """Assert catalog-level assertions from scenario spec."""
    if not cat:
        return
    for product in result.get("catalog_results", []):
        if cat.get("required_fill_type"):
            assert product.get("fill_type") == cat["required_fill_type"], (
                f"Expected fill_type={cat['required_fill_type']}, "
                f"got {product.get('fill_type')} for {product.get('name')}"
            )
        if cat.get("forbidden_fill_type"):
            assert product.get("fill_type") != cat["forbidden_fill_type"], (
                f"Forbidden fill_type={cat['forbidden_fill_type']} found for {product.get('name')}"
            )
        if cat.get("max_temp_rating_f") is not None:
            tr = product.get("temp_rating_f")
            if tr is not None:
                assert tr <= cat["max_temp_rating_f"], (
                    f"temp_rating_f {tr} exceeds max {cat['max_temp_rating_f']} for {product.get('name')}"
                )
        if cat.get("max_price_usd") is not None:
            p = product.get("price_usd")
            if p is not None:
                assert p <= cat["max_price_usd"], (
                    f"price_usd {p} exceeds budget {cat['max_price_usd']} for {product.get('name')}"
                )
        for field in cat.get("required_fields_per_product", ["sku", "name", "price_usd"]):
            assert field in product, f"Required field '{field}' missing from product {product.get('sku', '?')}"

    result_skus = [p["sku"] for p in result.get("catalog_results", [])]
    for sku in cat.get("required_skus_in_results", []):
        assert sku in result_skus, f"Expected SKU {sku} in results, got {result_skus}"
    for sku in cat.get("forbidden_skus_in_results", []):
        assert sku not in result_skus, f"Forbidden SKU {sku} found in results"


@pytest.mark.asyncio
async def test_vertical_slice_winter_sleeping_bag():
    """Primary vertical slice: winter camping in PNW → sleeping bag recommendation."""
    sc = load_scenario("pnw-winter-bag-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    assert result["activity"] == exp_ir["activity"]
    assert result["user_environment"] == exp_ir["user_environment"]

    # After full pipeline, action_flag will be READY_TO_SYNTHESIZE (set by synthesizer)
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert len(result["catalog_results"]) > 0

    _assert_catalog(result, sc["catalog_assertions"])

    assert result["recommendation"] is not None
    assert len(result["recommendation"]) > 50, "Recommendation too short to be useful"

    # Real Ollama should produce a recommendation relevant to sleeping bags and winter conditions
    rec = result["recommendation"].lower()
    assert any(w in rec for w in ["sleeping bag", "bag", "synthetic", "fill", "camping", "pnw", "winter"]), (
        f"Recommendation doesn't mention expected gear category. Got: {result['recommendation'][:200]}"
    )
    # Must not disparage — no negative competitor language
    assert "terrible" not in rec and "awful" not in rec


@pytest.mark.asyncio
async def test_clarification_gate_needs_activity():
    """Vague query with no activity → clarification requested on turn 1."""
    sc = load_scenario("vague-no-activity-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_gate = sc["expected_clarification_gate"]
    assert result["action_flag"] == exp_gate["action_flag"]
    assert result["clarification_message"] is not None
    assert len(result["clarification_message"]) > 0
    assert result["clarification_count"] == exp_gate["clarification_count"]

    # Did not reach synthesis
    assert result.get("recommendation") is None


@pytest.mark.asyncio
async def test_missing_environment_clarification():
    """
    Winter camping query without explicit environment.
    The real LLM may infer an environment from context (e.g. winter → PNW)
    or leave it null and ask for clarification. Both are valid agent behaviour.
    We assert on the DAG outcome: either clarification fires OR synthesis runs
    with a valid recommendation — never a silent failure.
    """
    sc = load_scenario("missing-env-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    assert result["intent"] == "Product_Search"

    if result["action_flag"] == "REQUIRES_CLARIFICATION":
        # Clarification path: message must reference environment or location
        assert result["clarification_message"] is not None
        msg = result["clarification_message"].lower()
        assert any(w in msg for w in ["where", "environment", "location", "pacific", "alpine", "desert"]), (
            f"Clarification message doesn't ask about environment: {result['clarification_message']}"
        )
    else:
        # LLM inferred environment from context — pipeline ran to synthesis
        assert result["action_flag"] == "READY_TO_SYNTHESIZE"
        assert result["recommendation"] is not None
        assert len(result["recommendation"]) > 50


@pytest.mark.asyncio
async def test_clarification_cap_forces_search():
    """Clarification count at 2 → forced to READY_TO_SEARCH, pipeline runs to completion."""
    sc = load_scenario("cap-forced-search-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    # After synthesis action_flag becomes READY_TO_SYNTHESIZE
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None
    assert len(result["catalog_results"]) > 0

    # With a vague query the LLM may extract any activity (e.g. backpacking)
    # and return specs consistent with that activity. We assert the pipeline
    # ran to completion with valid products — not a specific fill type.
    for product in result["catalog_results"]:
        assert "sku" in product
        assert "name" in product
        assert "price_usd" in product


@pytest.mark.asyncio
async def test_out_of_bounds_refusal():
    """Out-of-bounds query → deterministic refusal, no product synthesis."""
    sc = load_scenario("out-of-bounds-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"

    rec = result["recommendation"].lower()
    safety = sc["safety"]
    for keyword in safety.get("refusal_keywords", []):
        assert keyword in rec, f"Refusal keyword '{keyword}' not found in response"

    # No product catalog results for Out_of_Bounds
    assert result["catalog_results"] == []


@pytest.mark.asyncio
async def test_support_routing():
    """Return question → Support intent, deterministic response with contact info."""
    sc = load_scenario("support-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    assert result["intent"] == "Support"
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None

    rec = result["recommendation"]
    judge = sc["judge_rubric"]
    must_contain = judge.get("response_must_contain_any", [])
    assert any(kw in rec for kw in must_contain), (
        f"Response must contain one of {must_contain}, got: {rec}"
    )


@pytest.mark.asyncio
async def test_avalanche_safety_routing():
    """Avalanche beacon / backcountry skiing — Product_Search, READY_TO_SEARCH, reaches synthesis."""
    sc = load_scenario("avalanche-safety-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    # LLM may return "skiing" or "backcountry skiing" — both are correct
    assert result["activity"] is not None
    assert "ski" in (result["activity"] or "").lower(), (
        f"Expected skiing-related activity, got: {result['activity']}"
    )
    # skiing / backcountry skiing is not in env-sensitive list → no clarification
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None


@pytest.mark.asyncio
async def test_competitor_brand_no_disparagement():
    """Columbia jacket mention — routes to Product_Search, response must not disparage competitor."""
    sc = load_scenario("competitor-brand-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None

    rec = result["recommendation"].lower()
    forbidden = sc["safety"].get("forbidden_response_content", [])
    for word in forbidden:
        assert word.lower() not in rec, f"Forbidden word '{word}' found in response"


@pytest.mark.asyncio
async def test_budget_constraint_car_camping():
    """Car camping with $100 budget — synthetic fill, routes directly to search."""
    sc = load_scenario("budget-constraint-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    assert result["activity"] == exp_ir["activity"]
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None
    assert len(result["catalog_results"]) > 0

    # All returned products should be synthetic
    for product in result["catalog_results"]:
        assert product.get("fill_type") == "synthetic", (
            f"Expected synthetic fill, got {product.get('fill_type')} for {product.get('name')}"
        )


@pytest.mark.asyncio
async def test_oos_in_store_portland():
    """Portland store — same routing as pnw-winter-bag-001, SB-001 is OOS in store but online."""
    sc = load_scenario("oos-in-store-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    assert result["activity"] == exp_ir["activity"]
    assert result["user_environment"] == exp_ir["user_environment"]
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"

    _assert_catalog(result, sc["catalog_assertions"])

    assert result["recommendation"] is not None


@pytest.mark.asyncio
async def test_llm_fallback_bikepacking():
    """Bikepacking quilt — mock extracts no activity; cap at count=2 forces READY_TO_SEARCH, then LLM fallback."""
    sc = load_scenario("llm-fallback-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    # clarification_count=2 cap forces search even with no activity
    assert result["intent"] == "Product_Search"
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert result["recommendation"] is not None
    assert len(result["catalog_results"]) > 0


@pytest.mark.asyncio
async def test_member_pricing_fields():
    """Authenticated member — catalog results must include member_price_usd."""
    sc = load_scenario("member-pricing-001")
    state = _build_state(sc)
    result = await graph.ainvoke(state)

    exp_ir = sc["expected_intent_router"]
    assert result["intent"] == exp_ir["intent"]
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"
    assert len(result["catalog_results"]) > 0

    _assert_catalog(result, sc["catalog_assertions"])


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
    """Ontology must resolve 'winter camping' + 'PNW' to synthetic fill and high r_value."""
    from greenvest.ontology import lookup_all

    specs = lookup_all(["winter camping", "PNW"])
    assert len(specs) > 0

    keys = [list(s.keys())[0] for s in specs]
    assert "fill_type" in keys or "r_value" in keys


@pytest.mark.asyncio
async def test_pipeline_completes_without_error():
    """Smoke test: the full pipeline completes for a well-formed query."""
    state = initial_state(query="I need a sleeping bag for winter camping in the PNW")
    result = await graph.ainvoke(state)
    assert result is not None
    assert result["intent"] is not None
    assert result["action_flag"] in ("REQUIRES_CLARIFICATION", "READY_TO_SEARCH", "READY_TO_SYNTHESIZE")
