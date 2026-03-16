"""
Tests for P0 and P1 fixes in the autonomous optimization pipeline.

Pure-logic tests (no LLM):
  - _evaluate_candidate  (P0.2 / P0.3)
  - _gradient_target_key (P1.6)

Ollama-backed integration tests (use gemma2:9b as critic fallback):
  - analyze_failures parallel dispatch  (P1.5)
  - analyze_failures tried_fixes filter (P1.7)
"""
from __future__ import annotations

import pytest

from eval.autonomous_optimize import (
    _evaluate_candidate,
    IMPROVEMENT_THRESHOLD,
    SAFETY_FLOOR,
)
from eval.critic import TextualGradient, analyze_failures
from eval.optimizer_agent import _gradient_target_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gradient(
    fix_type: str,
    node_name: str | None = None,
    alias_key: str | None = None,
    scenario_id: str = "test_sc",
    target_node: str = "synthesizer",
    target_file: str = "greenvest/nodes/synthesizer.py",
    confidence: float = 0.8,
) -> TextualGradient:
    fix: dict = {}
    if node_name:
        fix["node_name"] = node_name
    if alias_key:
        fix["alias_key"] = alias_key
    return TextualGradient(
        scenario_id=scenario_id,
        query="test query",
        scores={},
        failure_mode="low_accuracy",
        target_node=target_node,
        target_file=target_file,
        diagnosis="test diagnosis",
        suggested_fix_type=fix_type,
        suggested_fix=fix,
        confidence=confidence,
    )


# Minimal eval results with two failing scenarios (composite < 0.80)
_FAKE_EVAL_RESULTS: dict = {
    "composite": 0.55,
    "persona": 0.60,
    "accuracy": 0.50,
    "safety": 0.65,
    "relevance": 0.55,
    "scenario_count": 2,
    "judged_count": 2,
    "per_scenario": [
        {
            "scenario_id": "sc_001",
            "persona": 0.60,
            "accuracy": 0.50,
            "safety": 0.65,
            "relevance": 0.55,
            "composite": 0.56,
            "reasoning": (
                "The response lacked specific product recommendations "
                "and did not address the safety concerns for winter camping."
            ),
        },
        {
            "scenario_id": "sc_002",
            "persona": 0.65,
            "accuracy": 0.55,
            "safety": 0.68,
            "relevance": 0.50,
            "composite": 0.60,
            "reasoning": (
                "The agent failed to translate the alpine climbing query "
                "into the correct temperature and insulation specs."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# _evaluate_candidate — pure logic, no LLM
# ---------------------------------------------------------------------------

class TestEvaluateCandidate:

    def test_safety_none_reverts(self):
        """safety=None must revert — judge failed to score, cannot confirm safe."""
        code, reason = _evaluate_candidate(0.80, {"composite": 0.90, "safety": None})
        assert code == "REVERT_SAFETY"
        assert "None" in reason

    def test_safety_none_beats_good_composite(self):
        """safety=None must still revert even when composite looks great."""
        code, _ = _evaluate_candidate(0.50, {"composite": 0.99, "safety": None})
        assert code == "REVERT_SAFETY"

    def test_safety_below_floor_reverts(self):
        code, reason = _evaluate_candidate(
            0.80, {"composite": 0.90, "safety": SAFETY_FLOOR - 0.01}
        )
        assert code == "REVERT_SAFETY"
        assert "floor" in reason

    def test_safety_floor_beats_composite_improvement(self):
        """Safety floor violation must revert even if composite improves."""
        code, _ = _evaluate_candidate(0.50, {"composite": 0.99, "safety": 0.60})
        assert code == "REVERT_SAFETY"

    def test_composite_none_reverts(self):
        """composite=None → judge down; never silently accept."""
        code, reason = _evaluate_candidate(0.80, {"composite": None, "safety": 0.90})
        assert code == "REVERT_NO_SCORES"
        assert "None" in reason

    def test_baseline_none_reverts(self):
        """No baseline to compare against → revert (no silent accept)."""
        code, _ = _evaluate_candidate(None, {"composite": 0.90, "safety": 0.90})
        assert code == "REVERT_NO_SCORES"

    def test_improvement_at_exact_threshold_keeps(self):
        baseline = 0.75
        candidate = baseline + IMPROVEMENT_THRESHOLD
        code, reason = _evaluate_candidate(
            baseline, {"composite": candidate, "safety": 0.85}
        )
        assert code == "KEEP"
        assert "≥" in reason

    def test_improvement_above_threshold_keeps(self):
        code, _ = _evaluate_candidate(0.70, {"composite": 0.85, "safety": 0.80})
        assert code == "KEEP"

    def test_improvement_just_below_threshold_reverts(self):
        baseline = 0.75
        candidate = baseline + IMPROVEMENT_THRESHOLD - 0.001
        code, _ = _evaluate_candidate(
            baseline, {"composite": candidate, "safety": 0.85}
        )
        assert code == "REVERT_THRESHOLD"

    def test_regression_reverts(self):
        code, _ = _evaluate_candidate(0.85, {"composite": 0.70, "safety": 0.90})
        assert code == "REVERT_THRESHOLD"

    def test_safety_exactly_at_floor_keeps(self):
        """safety == floor is acceptable (strict less-than check)."""
        baseline = 0.70
        candidate = baseline + IMPROVEMENT_THRESHOLD
        code, _ = _evaluate_candidate(
            baseline, {"composite": candidate, "safety": SAFETY_FLOOR}
        )
        assert code == "KEEP"


# ---------------------------------------------------------------------------
# _gradient_target_key — pure logic, no LLM
# ---------------------------------------------------------------------------

class TestGradientTargetKey:

    def test_patch_prompt_same_node_same_key(self):
        g1 = _make_gradient("patch_prompt", node_name="_REI_PERSONA")
        g2 = _make_gradient("patch_prompt", node_name="_REI_PERSONA")
        assert _gradient_target_key(g1) == _gradient_target_key(g2)

    def test_patch_prompt_different_nodes_different_keys(self):
        g1 = _make_gradient("patch_prompt", node_name="_REI_PERSONA")
        g2 = _make_gradient("patch_prompt", node_name="_OUT_OF_BOUNDS_RESPONSE")
        assert _gradient_target_key(g1) != _gradient_target_key(g2)

    def test_update_ontology_same_alias_same_key(self):
        g1 = _make_gradient("update_ontology", alias_key="winter camping / winter camp")
        g2 = _make_gradient("update_ontology", alias_key="winter camping / winter camp")
        assert _gradient_target_key(g1) == _gradient_target_key(g2)

    def test_update_ontology_different_alias_different_key(self):
        g1 = _make_gradient("update_ontology", alias_key="winter camping / winter camp")
        g2 = _make_gradient("update_ontology", alias_key="alpine")
        assert _gradient_target_key(g1) != _gradient_target_key(g2)

    def test_patch_phrase_list_always_same_key(self):
        g1 = _make_gradient("patch_phrase_list")
        g2 = _make_gradient("patch_phrase_list")
        assert _gradient_target_key(g1) == _gradient_target_key(g2)

    def test_different_fix_types_different_keys(self):
        g_prompt = _make_gradient("patch_prompt", node_name="_REI_PERSONA")
        g_phrase = _make_gradient("patch_phrase_list")
        g_onto = _make_gradient("update_ontology", alias_key="alpine")
        keys = {_gradient_target_key(g_prompt), _gradient_target_key(g_phrase), _gradient_target_key(g_onto)}
        assert len(keys) == 3

    def test_deduplication_keeps_highest_confidence(self):
        """Simulate the dedup dict logic: higher confidence wins."""
        g_low = _make_gradient("patch_prompt", node_name="_REI_PERSONA", confidence=0.5)
        g_high = _make_gradient("patch_prompt", node_name="_REI_PERSONA", confidence=0.9)
        gradients = [g_low, g_high]
        seen: dict = {}
        for g in gradients:
            key = _gradient_target_key(g)
            if key not in seen or g.confidence > seen[key].confidence:
                seen[key] = g
        assert len(seen) == 1
        assert list(seen.values())[0].confidence == 0.9


# ---------------------------------------------------------------------------
# analyze_failures — requires Ollama (gemma2:9b as critic LLM fallback)
# ---------------------------------------------------------------------------

async def test_analyze_failures_returns_gradients(require_ollama):
    """Parallel critic calls return at least one gradient for clearly failing scenarios."""
    gradients = await analyze_failures(_FAKE_EVAL_RESULTS)
    assert len(gradients) >= 1
    for g in gradients:
        assert g.scenario_id in ("sc_001", "sc_002")
        assert g.target_node in (
            "intent_router", "clarification_gate",
            "query_translator", "ontology", "synthesizer",
        )
        assert g.suggested_fix_type in (
            "patch_prompt", "update_ontology", "patch_phrase_list",
        )
        assert 0.0 <= g.confidence <= 1.0
        assert g.diagnosis  # non-empty


async def test_analyze_failures_tried_fixes_filters_all(require_ollama):
    """Gradients matching tried_fixes entries are not returned in subsequent calls."""
    # First pass — obtain real gradients from Ollama
    gradients = await analyze_failures(_FAKE_EVAL_RESULTS)
    assert gradients, "Need at least one gradient to test filtering"

    # Build tried_fixes from all returned gradients
    tried: set = {
        (g.scenario_id, g.suggested_fix_type, g.target_node)
        for g in gradients
    }

    # Second pass — everything in tried_fixes should be filtered
    filtered = await analyze_failures(_FAKE_EVAL_RESULTS, tried_fixes=tried)
    for g in filtered:
        assert (g.scenario_id, g.suggested_fix_type, g.target_node) not in tried


async def test_analyze_failures_partial_tried_fixes(require_ollama):
    """Blocking only one entry leaves the others still diagnosable."""
    gradients = await analyze_failures(_FAKE_EVAL_RESULTS)
    if len(gradients) < 2:
        pytest.skip("Need ≥2 distinct gradients to test partial filtering")

    first = gradients[0]
    tried: set = {(first.scenario_id, first.suggested_fix_type, first.target_node)}

    filtered = await analyze_failures(_FAKE_EVAL_RESULTS, tried_fixes=tried)

    # The blocked entry must not appear
    for g in filtered:
        assert not (
            g.scenario_id == first.scenario_id
            and g.suggested_fix_type == first.suggested_fix_type
            and g.target_node == first.target_node
        )
