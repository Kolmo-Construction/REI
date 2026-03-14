"""
Deterministic mock LLM responses — no network calls, no API keys required.
Used when config.USE_MOCK_LLM is True.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


def mock_intent_router(query: str) -> dict:
    """
    Returns a deterministic intent parse.
    Handles a handful of test cases; defaults to Product_Search.
    """
    q = query.lower()

    if any(w in q for w in ["medical", "legal", "financial", "tax"]):
        return {
            "intent": "Out_of_Bounds",
            "activity": None,
            "user_environment": None,
            "experience_level": None,
        }

    if any(w in q for w in ["return", "order", "warranty", "repair"]):
        return {
            "intent": "Support",
            "activity": None,
            "user_environment": None,
            "experience_level": None,
        }

    if any(w in q for w in ["how to", "what is", "explain", "teach"]):
        return {
            "intent": "Education",
            "activity": None,
            "user_environment": None,
            "experience_level": None,
        }

    # Default: Product_Search with PNW winter camping signal
    activity = None
    environment = None

    if "winter camping" in q or "winter camp" in q:
        activity = "winter_camping"
    elif "backpacking" in q or "backpack" in q:
        activity = "backpacking"
    elif "car camping" in q or "car camp" in q:
        activity = "car_camping"
    elif "thru" in q or "thru-hik" in q:
        activity = "thru_hiking"

    if "pnw" in q or "pacific northwest" in q or "washington" in q or "oregon" in q:
        environment = "PNW_winter"
    elif "desert" in q:
        environment = "desert_summer"
    elif "alpine" in q:
        environment = "alpine"

    return {
        "intent": "Product_Search",
        "activity": activity,
        "user_environment": environment,
        "experience_level": None,
    }


def mock_query_translator(state: "GreenvestState") -> dict:
    """
    Returns deterministic derived specs for winter camping / PNW queries.
    """
    activity = state.get("activity", "")
    environment = state.get("user_environment", "")

    specs = []
    confidence = 0.85

    if activity == "winter_camping" or environment in ("PNW_winter", "alpine"):
        specs.append({"fill_type": "synthetic"})
        specs.append({"r_value": ">=4.5"})
        specs.append({"temp_rating_f": "<=15"})
        confidence = 0.92
    elif activity == "backpacking":
        specs.append({"fill_type": "down"})
        specs.append({"weight_oz": "<32"})
        confidence = 0.90
    elif activity == "car_camping":
        specs.append({"fill_type": "synthetic"})
        confidence = 0.88

    if not specs:
        # Minimal fallback
        specs.append({"fill_type": "synthetic"})
        confidence = 0.75

    return {
        "derived_specs": specs,
        "spec_confidence": confidence,
    }


def mock_synthesizer(prompt: str) -> str:
    return (
        "Based on your winter camping trip in the PNW, I'd recommend the REI Co-op Magma "
        "15 Sleeping Bag. Its synthetic fill handles the wet Pacific Northwest conditions "
        "better than down — moisture won't kill its loft mid-trip. The EN-tested 15°F rating "
        "gives you real warmth headroom for shoulder-season cold snaps, and at 2 lbs 14 oz "
        "it won't punish your pack weight. Pair it with the REI Co-op Trailbreak Sleeping Pad "
        "(R-value 4.5) so ground cold doesn't undercut the bag. If you run cold, size down to "
        "the 0°F version — it's worth it for PNW winter. A Greenvest at your local REI "
        "Seattle store can fit-check the bag for your sleep style."
    )
