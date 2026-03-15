"""
Pytest fixtures for Greenvest integration tests.

LLM calls (intent_router, query_translator, synthesizer) use REAL Ollama.
  - Set USE_MOCK_LLM=false so the graph routes through ollama_llm.py
  - Tests that need Ollama request the require_ollama fixture; they are
    skipped automatically if Ollama is not running locally.

Retrieval (Branch A expert, Branch B catalog) stays mocked — Qdrant is
not available in the test environment. The mock catalog filters
sample_products.json by derived_specs produced by the real Ollama LLM,
so the end-to-end spec→product path is still exercised.

Branch C (inventory SQL) is already stubbed in the source code.
"""
import json
import os
import re
from pathlib import Path

import httpx
import pytest

# Use real Ollama LLMs — not the text-matching mock
os.environ["USE_MOCK_LLM"] = "false"

PRODUCTS = json.loads((Path(__file__).parent.parent / "data" / "sample_products.json").read_text())
INVENTORY = json.loads((Path(__file__).parent / "fixtures" / "inventory.json").read_text())


# ---------------------------------------------------------------------------
# Ollama availability
# ---------------------------------------------------------------------------

def _ollama_is_available() -> bool:
    """Return True if the local Ollama server is reachable."""
    try:
        from greenvest.config import settings
        r = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_ollama():
    """
    Session-scoped fixture. Tests that request it are skipped when
    Ollama is not running. Start Ollama with: ollama serve
    """
    if not _ollama_is_available():
        pytest.skip(
            "Ollama is not running. Start it with: ollama serve\n"
            "Required models: ollama pull llama3.2 && ollama pull llama3"
        )

MOCK_EXPERT_CHUNKS = [
    {
        "title": "Sleeping Bag Guide",
        "section": "Fill Types",
        "chunk_text": "Synthetic fill retains warmth when wet, making it ideal for PNW conditions.",
        "url": "https://www.rei.com/learn/expert-advice/sleeping-bags.html",
        "score": 0.92,
    },
    {
        "title": "Winter Camping Tips",
        "section": "Temperature Ratings",
        "chunk_text": "Choose a bag rated 10-15F lower than the coldest temperature you expect.",
        "url": "https://www.rei.com/learn/expert-advice/winter-camping.html",
        "score": 0.88,
    },
    {
        "title": "REI Gear Guide",
        "section": "PNW Specific",
        "chunk_text": "In the Pacific Northwest, moisture management is critical. Hydrophobic down or synthetic insulation recommended.",
        "url": "https://www.rei.com/learn/expert-advice/pnw-camping.html",
        "score": 0.85,
    },
]


def _filter_products(state) -> list[dict]:
    """Filter sample_products.json by derived_specs from state."""
    specs = state.get("derived_specs", [])
    results = list(PRODUCTS)

    for spec in specs:
        for key, value in spec.items():
            if key == "fill_type" and isinstance(value, str):
                # Handle "down OR synthetic" style values by skipping (no filter)
                if " OR " not in value:
                    results = [p for p in results if p.get("fill_type") == value]
            elif key == "temp_rating_f" and isinstance(value, str):
                m = re.match(r"^([<>]=?)(-?\d+\.?\d*)$", value.strip())
                if m:
                    op, num = m.group(1), float(m.group(2))
                    if op == "<=":
                        results = [p for p in results if p.get("temp_rating_f") is not None and p["temp_rating_f"] <= num]
                    elif op == ">=":
                        results = [p for p in results if p.get("temp_rating_f") is not None and p["temp_rating_f"] >= num]
                    elif op == "<":
                        results = [p for p in results if p.get("temp_rating_f") is not None and p["temp_rating_f"] < num]
                    elif op == ">":
                        results = [p for p in results if p.get("temp_rating_f") is not None and p["temp_rating_f"] > num]
            elif key == "r_value" and isinstance(value, str):
                m = re.match(r"^([<>]=?)(-?\d+\.?\d*)$", value.strip())
                if m:
                    op, num = m.group(1), float(m.group(2))
                    if op == ">=":
                        results = [p for p in results if p.get("r_value") is not None and p["r_value"] >= num]
                    elif op == ">":
                        results = [p for p in results if p.get("r_value") is not None and p["r_value"] > num]
                    elif op == "<=":
                        results = [p for p in results if p.get("r_value") is not None and p["r_value"] <= num]
                    elif op == "<":
                        results = [p for p in results if p.get("r_value") is not None and p["r_value"] < num]
            elif key == "weight_oz" and isinstance(value, str):
                m = re.match(r"^([<>]=?)(-?\d+\.?\d*)$", value.strip())
                if m:
                    op, num = m.group(1), float(m.group(2))
                    if op == "<":
                        results = [p for p in results if p.get("weight_oz") is not None and p["weight_oz"] < num]
                    elif op == "<=":
                        results = [p for p in results if p.get("weight_oz") is not None and p["weight_oz"] <= num]
                    elif op == ">":
                        results = [p for p in results if p.get("weight_oz") is not None and p["weight_oz"] > num]
                    elif op == ">=":
                        results = [p for p in results if p.get("weight_oz") is not None and p["weight_oz"] >= num]

    return results[:5]


def _mock_inventory(state) -> list[dict]:
    """Return inventory from fixture for the session's store_id."""
    store_id = state.get("store_id", "REI-Seattle")
    store_inv = INVENTORY.get(store_id, {})
    catalog_skus = [p.get("sku") for p in state.get("catalog_results", [])]
    result = []
    for sku in catalog_skus:
        if sku in store_inv:
            product = next((p for p in PRODUCTS if p["sku"] == sku), {})
            entry = {
                **store_inv[sku],
                "sku": sku,
                "product_name": product.get("name", sku),
                "price_usd": product.get("price_usd"),
                "member_price_usd": product.get("member_price_usd"),
            }
            if entry["store_stock_qty"] > 0 or entry["online_stock_qty"] > 0:
                result.append(entry)
    return result


@pytest.fixture(autouse=True)
def mock_retrieval(monkeypatch):
    """Patch Branch A and B for all tests automatically."""

    async def fake_expert(state):
        return MOCK_EXPERT_CHUNKS

    async def fake_catalog(state):
        return _filter_products(state)

    monkeypatch.setattr("greenvest.retrieval.branch_a_expert.search_expert_advice", fake_expert)
    monkeypatch.setattr("greenvest.retrieval.branch_b_catalog.search_catalog", fake_catalog)
    monkeypatch.setattr("greenvest.graph.search_expert_advice", fake_expert)
    monkeypatch.setattr("greenvest.graph.search_catalog", fake_catalog)
