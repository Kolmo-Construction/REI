"""
Deterministic mock LLM responses — no network calls, no API keys required.
Used when config.USE_MOCK_LLM is True.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


def _extract_activity(q: str) -> str | None:
    """Extract and normalize activity from a lowercased query string."""
    if "winter camping" in q or "winter camp" in q:
        return "winter_camping"
    if "thru-hik" in q or "thru hik" in q or "thru hiking" in q or "thru-hiking" in q:
        return "thru_hiking"
    if "backpacking" in q or "backpack" in q:
        return "backpacking"
    if "car camping" in q or "car camp" in q:
        return "car_camping"
    if "rock climbing" in q or "rock climb" in q or "bouldering" in q:
        return "rock_climbing"
    if "ski" in q or "skiing" in q or "snowboard" in q:
        return "skiing"
    if "mountaineer" in q:
        return "mountaineering"
    if "alpine climbing" in q or "alpine climb" in q:
        return "alpine_climbing"
    # hiking synonyms and common misspellings
    if any(w in q for w in ["hiking", "hikking", "hikin", "hike", "trekking", "trek", "trail walk", "trail run"]):
        return "day_hiking"
    if "camping" in q or "camp" in q:
        return "car_camping"
    return None


def _extract_environment(q: str) -> str | None:
    if "pnw" in q or "pacific northwest" in q or "washington" in q or "oregon" in q:
        return "PNW_winter"
    if "desert" in q:
        return "desert_summer"
    if "alpine" in q:
        return "alpine"
    if "coastal" in q or "coast" in q:
        return "coastal"
    return None


def mock_intent_router(query: str) -> dict:
    """
    Returns a deterministic intent parse.
    Intent and entity extraction are independent — activity/environment are
    always extracted regardless of which intent is detected.
    """
    q = query.lower()

    # Classify intent
    if any(w in q for w in ["medical", "legal", "financial", "tax"]):
        intent = "Out_of_Bounds"
    elif any(w in q for w in ["return", "order", "warranty", "repair"]):
        intent = "Support"
    elif any(w in q for w in ["how to", "what is", "explain", "teach", "best", "recommend", "should i"]):
        intent = "Education"
    else:
        intent = "Product_Search"

    # Entity extraction is always performed, regardless of intent
    activity = _extract_activity(q)
    environment = _extract_environment(q)

    return {
        "intent": intent,
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
