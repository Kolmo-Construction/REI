"""
Unit tests for deterministic nodes: clarification_gate and ontology lookup.
Calls node functions directly — no graph invocation.

Run: uv run pytest tests/test_deterministic_nodes.py -v
"""
import pytest
from greenvest.nodes.clarification_gate import clarification_gate
from greenvest.ontology import lookup, lookup_all
from greenvest.state import initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> dict:
    """Build a minimal state dict with defaults, applying overrides."""
    base = initial_state(
        query="test query",
        session_id="unit-test-session",
        store_id="REI-Seattle",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# clarification_gate — Out_of_Bounds branch
# ---------------------------------------------------------------------------

def test_gate_out_of_bounds_returns_ready_to_synthesize():
    state = _state(intent="Out_of_Bounds")
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"


def test_gate_out_of_bounds_does_not_set_clarification_message():
    state = _state(intent="Out_of_Bounds")
    result = clarification_gate(state)
    assert "clarification_message" not in result or result.get("clarification_message") is None


# ---------------------------------------------------------------------------
# clarification_gate — Support branch
# ---------------------------------------------------------------------------

def test_gate_support_returns_ready_to_synthesize():
    state = _state(intent="Support")
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SYNTHESIZE"


def test_gate_support_does_not_increment_clarification_count():
    state = _state(intent="Support", clarification_count=0)
    result = clarification_gate(state)
    assert result.get("clarification_count", 0) == 0


# ---------------------------------------------------------------------------
# clarification_gate — no activity, count=0
# ---------------------------------------------------------------------------

def test_gate_no_activity_count_zero_requires_clarification():
    state = _state(intent="Product_Search", activity=None, clarification_count=0)
    result = clarification_gate(state)
    assert result["action_flag"] == "REQUIRES_CLARIFICATION"
    assert result["clarification_count"] == 1
    assert result["clarification_message"] is not None
    assert len(result["clarification_message"]) > 0


def test_gate_no_activity_question_mentions_activity():
    state = _state(intent="Product_Search", activity=None, clarification_count=0)
    result = clarification_gate(state)
    msg = result["clarification_message"].lower()
    assert any(w in msg for w in ["activity", "backpacking", "camping", "hiking"])


# ---------------------------------------------------------------------------
# clarification_gate — env-sensitive activity, no env, count=0
# ---------------------------------------------------------------------------

def test_gate_winter_camping_no_env_count_zero_requires_clarification():
    state = _state(
        intent="Product_Search",
        activity="winter_camping",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "REQUIRES_CLARIFICATION"
    assert result["clarification_count"] == 1
    assert result["clarification_message"] is not None


def test_gate_backpacking_no_env_count_zero_requires_clarification():
    state = _state(
        intent="Product_Search",
        activity="backpacking",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "REQUIRES_CLARIFICATION"


def test_gate_alpine_climbing_no_env_requires_clarification():
    state = _state(
        intent="Product_Search",
        activity="alpine_climbing",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "REQUIRES_CLARIFICATION"


def test_gate_mountaineering_no_env_requires_clarification():
    state = _state(
        intent="Product_Search",
        activity="mountaineering",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "REQUIRES_CLARIFICATION"


def test_gate_thru_hiking_no_env_requires_clarification():
    state = _state(
        intent="Product_Search",
        activity="thru_hiking",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "REQUIRES_CLARIFICATION"


# ---------------------------------------------------------------------------
# clarification_gate — env-sensitive activity, no env, count=2 (cap)
# ---------------------------------------------------------------------------

def test_gate_env_sensitive_count_two_forces_ready_to_search():
    state = _state(
        intent="Product_Search",
        activity="winter_camping",
        user_environment=None,
        clarification_count=2,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


def test_gate_backpacking_no_env_count_two_forces_search():
    state = _state(
        intent="Product_Search",
        activity="backpacking",
        user_environment=None,
        clarification_count=2,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


# ---------------------------------------------------------------------------
# clarification_gate — non-env-sensitive activity, no env → READY_TO_SEARCH
# ---------------------------------------------------------------------------

def test_gate_car_camping_no_env_ready_to_search():
    """car_camping is NOT in the env-sensitive set — should proceed directly."""
    state = _state(
        intent="Product_Search",
        activity="car_camping",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


def test_gate_skiing_no_env_ready_to_search():
    """skiing is NOT in the env-sensitive set — should proceed directly."""
    state = _state(
        intent="Product_Search",
        activity="skiing",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


def test_gate_day_hiking_no_env_ready_to_search():
    """day_hiking is NOT in the env-sensitive set."""
    state = _state(
        intent="Product_Search",
        activity="day_hiking",
        user_environment=None,
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


# ---------------------------------------------------------------------------
# clarification_gate — env-sensitive activity WITH env → READY_TO_SEARCH
# ---------------------------------------------------------------------------

def test_gate_winter_camping_with_pnw_env_ready_to_search():
    state = _state(
        intent="Product_Search",
        activity="winter_camping",
        user_environment="PNW_winter",
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


def test_gate_backpacking_with_alpine_env_ready_to_search():
    state = _state(
        intent="Product_Search",
        activity="backpacking",
        user_environment="alpine",
        clarification_count=0,
    )
    result = clarification_gate(state)
    assert result["action_flag"] == "READY_TO_SEARCH"


# ---------------------------------------------------------------------------
# Ontology lookup — individual term lookups
# ---------------------------------------------------------------------------

def test_ontology_winter_camping_returns_specs():
    result = lookup("winter camping")
    assert result is not None
    assert len(result) > 0
    keys = [list(s.keys())[0] for s in result]
    assert "fill_type" in keys or "temp_rating_f" in keys


def test_ontology_winter_camping_fill_type_is_synthetic():
    result = lookup("winter camping")
    assert result is not None
    fill_specs = [s for s in result if "fill_type" in s]
    assert len(fill_specs) > 0
    assert fill_specs[0]["fill_type"] == "synthetic"


def test_ontology_pnw_returns_specs():
    result = lookup("PNW")
    assert result is not None
    assert len(result) > 0


def test_ontology_backpacking_returns_specs():
    result = lookup("backpacking")
    assert result is not None
    assert len(result) > 0
    keys = [list(s.keys())[0] for s in result]
    assert "weight_oz" in keys or "fill_type" in keys


def test_ontology_unknown_term_returns_none():
    result = lookup("unknown_term_xyz_12345")
    assert result is None


def test_ontology_lookup_all_winter_camping_pnw():
    specs = lookup_all(["winter camping", "PNW"])
    assert len(specs) > 0
    keys = [list(s.keys())[0] for s in specs]
    assert "fill_type" in keys


def test_ontology_lookup_all_deduplicates_keys():
    """lookup_all must not return duplicate spec keys."""
    specs = lookup_all(["winter camping", "winter camping", "PNW"])
    keys = [list(s.keys())[0] for s in specs]
    assert len(keys) == len(set(keys)), "Duplicate spec keys returned by lookup_all"


def test_ontology_lookup_all_empty_list_returns_empty():
    specs = lookup_all([])
    assert specs == []


def test_ontology_alpine_returns_r_value():
    result = lookup("alpine")
    assert result is not None
    keys = [list(s.keys())[0] for s in result]
    assert "r_value" in keys


def test_ontology_car_camping_sleeping_bags():
    result = lookup("car camping")
    assert result is not None
    fill_specs = [s for s in result if "fill_type" in s]
    assert len(fill_specs) > 0
    assert fill_specs[0]["fill_type"] == "synthetic"
