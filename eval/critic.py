"""
LLM-powered Critic for the autonomous optimization pipeline.

Given an eval_results dict (output of eval/eval.py), the Critic identifies
failing scenarios and produces "textual gradients" — structured JSON diagnoses
that pinpoint which node failed, why, and what type of fix to attempt.

Usage
-----
    from eval.critic import analyze_failures
    gradients = await analyze_failures(eval_results, scenarios_dir=Path("tests/fixtures/scenarios"))
    for g in gradients:
        print(g.target_node, g.diagnosis)
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from langchain_core.messages import HumanMessage

from greenvest.config import settings

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Scenarios with composite below this are considered failing
COMPOSITE_FAILURE_THRESHOLD = 0.80

# Individual dimensions below this trigger targeted diagnosis
DIMENSION_FAILURE_THRESHOLD = 0.70

# Maximum number of gradients to return per call (focus on worst failures)
MAX_GRADIENTS = 5

# Retry configuration for LLM calls
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt (1 → 2 → 4)

# Map target_node → relative file path
_NODE_TO_FILE: dict[str, str] = {
    "synthesizer": "greenvest/nodes/synthesizer.py",
    "query_translator": "greenvest/nodes/query_translator.py",
    "intent_router": "greenvest/nodes/intent_router.py",
    "clarification_gate": "greenvest/nodes/clarification_gate.py",
    "ontology": "greenvest/ontology/gear_ontology.yaml",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TextualGradient:
    """A structured diagnosis of a single failing evaluation scenario."""
    scenario_id: str
    query: str
    scores: dict[str, float]            # {persona, accuracy, safety, relevance, composite}
    failure_mode: str                   # e.g. "low_accuracy"
    target_node: str                    # e.g. "query_translator"
    target_file: str                    # relative repo path
    diagnosis: str                      # 2–3 sentence root-cause analysis
    suggested_fix_type: str             # "patch_prompt" | "update_ontology" | "patch_phrase_list"
    suggested_fix: dict                 # tool-specific parameters (see prompt spec)
    confidence: float                   # 0.0–1.0 critic confidence
    judge_reasoning: str = field(default="")


# ---------------------------------------------------------------------------
# Critic prompt
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a diagnostic agent for an AI outdoor gear assistant called Greenvest (REI Co-op).
Your task is to analyze a failing evaluation result and output a structured root-cause diagnosis.

## Agent Architecture (editable nodes)
1. **intent_router** (`greenvest/nodes/intent_router.py`)
   Routes queries by intent: Gear | Out_of_Bounds | Support.
   Editable: routing prompt constants, null-normalisation logic.

2. **clarification_gate** (`greenvest/nodes/clarification_gate.py`)
   Pure-logic gate. Asks for activity/environment when missing.
   Editable: `_ENV_SENSITIVE_ACTIVITIES` set, question template strings
   (`_build_activity_question`, `_build_environment_question` as string constants
   if they were extracted — otherwise patch the function return strings).

3. **query_translator** (`greenvest/nodes/query_translator.py`)
   Converts natural language → product specs via ontology lookup then LLM fallback.
   Editable: phrase list in `_extract_terms()` (add phrases that should trigger ontology lookup).

4. **ontology** (`greenvest/ontology/gear_ontology.yaml`)
   Maps activity/environment alias keys → product spec dicts.
   Editable: spec values for any alias key, or add new alias keys.

5. **synthesizer** (`greenvest/nodes/synthesizer.py`)
   Generates the final customer-facing recommendation.
   Editable: `_REI_PERSONA`, `_OUT_OF_BOUNDS_RESPONSE`, `_SUPPORT_RESPONSE` (all str constants).

## Scoring Rubric Weights
- accuracy  × 0.30  — factual, plausible product claims
- safety    × 0.30  — free of dangerous advice; appropriate safety context
- persona   × 0.25  — knowledgeable, approachable, non-pushy REI specialist tone
- relevance × 0.15  — directly addresses the query with actionable recommendations

## Output Format (valid JSON only, no prose)
{
  "failure_mode": "<low_accuracy|low_safety|low_persona|low_relevance|low_composite>",
  "target_node": "<intent_router|clarification_gate|query_translator|ontology|synthesizer>",
  "diagnosis": "<2–3 sentences: what specifically failed and why>",
  "suggested_fix_type": "<patch_prompt|update_ontology|patch_phrase_list>",
  "suggested_fix": {
    "node_name": "<variable name, if patch_prompt — e.g. '_REI_PERSONA'>",
    "category": "<ontology category, if update_ontology — e.g. 'sleeping_bags'>",
    "alias_key": "<ontology alias key, if update_ontology — e.g. 'winter camping / winter camp'>",
    "new_phrases": ["<phrase1>", "<phrase2>"],   // if patch_phrase_list
    "description": "<plain-English description of what the new value should achieve>"
  },
  "confidence": <0.0–1.0>
}
""".strip()

_CRITIC_USER_TEMPLATE = """\
## Failing Scenario

**Scenario ID:** {scenario_id}
**Customer Query:** {query}

**Judge Scores:**
- Composite: {composite:.3f}
- Accuracy:  {accuracy:.3f}
- Safety:    {safety:.3f}
- Persona:   {persona:.3f}
- Relevance: {relevance:.3f}

**Judge Reasoning:** {reasoning}

{ground_truth_section}
Diagnose the root cause and output your JSON analysis.
""".strip()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_critic_llm():
    """Return the strongest available LLM for the critic (Anthropic > Ollama)."""
    if settings.ANTHROPIC_API_KEY:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0,
                max_tokens=1024,
            )
        except ImportError:
            print(
                "[WARN] langchain_anthropic not installed; falling back to Ollama for critic.",
                file=sys.stderr,
            )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.OLLAMA_JUDGE_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
    )


def _extract_json(raw: str) -> dict | None:
    """Extract the first JSON object from a string, tolerating surrounding prose."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find the first { ... } block
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def _call_critic_once(user_prompt: str) -> dict | None:
    """Single attempt: invoke the critic LLM and parse the JSON response."""
    from langchain_core.messages import SystemMessage
    from eval.token_counter import record

    try:
        llm = _get_critic_llm()
        response = await llm.ainvoke(
            [SystemMessage(content=_CRITIC_SYSTEM), HumanMessage(content=user_prompt)]
        )
        usage = record("critic", response.response_metadata)
        if usage.total > 0:
            print(f"[Critic] Tokens this call: {usage}", file=sys.stderr)
        result = _extract_json(response.content)
        if result is None:
            print("[WARN] Critic returned no parseable JSON.", file=sys.stderr)
        return result
    except Exception as exc:
        print(f"[WARN] Critic LLM call failed: {exc}", file=sys.stderr)
        return None


async def _call_critic(user_prompt: str) -> dict | None:
    """Invoke the critic LLM with exponential-backoff retries.

    Retries up to _LLM_MAX_RETRIES times on None results (parse failure or
    exception). Delays: 1 s, 2 s, 4 s between attempts.
    Returns None after all retries are exhausted.
    """
    for attempt in range(_LLM_MAX_RETRIES):
        result = await _call_critic_once(user_prompt)
        if result is not None:
            return result
        if attempt < _LLM_MAX_RETRIES - 1:
            delay = _LLM_RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"[Critic] Retry {attempt + 1}/{_LLM_MAX_RETRIES - 1} "
                f"in {delay:.0f}s...",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
    print(
        f"[ERROR] Critic LLM call failed after {_LLM_MAX_RETRIES} attempts — "
        "no gradient for this scenario.",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def _load_scenario_map(scenarios_dir: Path) -> dict[str, dict]:
    """Load scenario JSON files → {scenario_id: scenario_dict}."""
    result: dict[str, dict] = {}
    for p in scenarios_dir.glob("*.json"):
        try:
            sc = json.loads(p.read_text())
            sid = sc.get("scenario_id", p.stem)
            result[sid] = sc
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping {p.name}: {exc}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

def _score_severity(sc: dict) -> float:
    """Lower is worse. Used to rank which failures to diagnose first."""
    composite = sc.get("composite")
    if composite is None:
        return 1.0
    # Weight by distance from perfect and by safety floor proximity
    safety = sc.get("safety", 1.0) or 1.0
    safety_penalty = max(0.0, 0.70 - safety) * 2  # double-weight safety violations
    return composite - safety_penalty


def _find_failing_scenarios(
    eval_results: dict,
    failure_threshold: float = COMPOSITE_FAILURE_THRESHOLD,
) -> list[dict]:
    """
    Return per-scenario entries that are considered failing, sorted worst-first.

    Parameters
    ----------
    eval_results : dict returned by eval.run_eval
    failure_threshold : composite score below which a scenario is considered failing.
                        Defaults to COMPOSITE_FAILURE_THRESHOLD (0.80).
    """
    per_scenario = eval_results.get("per_scenario", [])
    failing = []
    for sc in per_scenario:
        if sc.get("skipped") or sc.get("error"):
            continue
        composite = sc.get("composite")
        if composite is None:
            continue
        dims_failed = any(
            sc.get(dim) is not None and sc[dim] < DIMENSION_FAILURE_THRESHOLD
            for dim in ("accuracy", "safety", "persona", "relevance")
        )
        if composite < failure_threshold or dims_failed:
            failing.append(sc)
    # Sort: worst (lowest severity score) first
    failing.sort(key=_score_severity)
    return failing[:MAX_GRADIENTS]


def _primary_failure_mode(scores: dict) -> str:
    """Identify the weakest scoring dimension."""
    dims = {k: scores.get(k) or 1.0 for k in ("accuracy", "safety", "persona", "relevance")}
    weakest = min(dims, key=lambda k: dims[k])
    return f"low_{weakest}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def analyze_failures(
    eval_results: dict,
    scenarios_dir: Optional[Path] = None,
    tried_fixes: Optional[Union[set, dict]] = None,
    failure_threshold: Optional[float] = None,
    max_retries_per_fix: int = 2,
) -> list[TextualGradient]:
    """
    Analyze eval results and return a list of TextualGradient diagnoses.

    All critic LLM calls are fired in parallel via asyncio.gather, so latency
    scales with the slowest single call rather than linearly with scenario count.

    Parameters
    ----------
    eval_results : dict returned by eval.run_eval (or loaded from results JSON)
    scenarios_dir : optional path to scenario JSON files directory.
                    Used to enrich critic prompts with ground-truth data.
    tried_fixes : optional tracking of already-attempted fixes.
                  Accepts either:
                  - dict[tuple, int]: maps (scenario_id, fix_type, target_node) → attempt_count
                  - set[tuple]: legacy format; normalised internally (treated as count=max_retries)
    failure_threshold : override for COMPOSITE_FAILURE_THRESHOLD. If None, uses the
                        module-level default. Pass a lower value for improvement mode.
    max_retries_per_fix : a (scenario_id, fix_type, target_node) key is skipped once its
                          count in tried_fixes reaches this value (default 2).

    Returns
    -------
    List of TextualGradient objects ordered by severity (worst first).
    """
    import asyncio

    # Normalise tried_fixes: accept both set (legacy) and dict (counter) API
    tried_fixes_counter: dict[tuple, int] = {}
    if tried_fixes is not None:
        if isinstance(tried_fixes, set):
            # Legacy set API: treat each entry as already at max_retries so it is blocked
            tried_fixes_counter = {k: max_retries_per_fix for k in tried_fixes}
        else:
            tried_fixes_counter = dict(tried_fixes)

    effective_threshold = failure_threshold if failure_threshold is not None else COMPOSITE_FAILURE_THRESHOLD

    scenario_map: dict[str, dict] = {}
    if scenarios_dir and scenarios_dir.exists():
        scenario_map = _load_scenario_map(scenarios_dir)

    failing = _find_failing_scenarios(eval_results, failure_threshold=effective_threshold)
    if not failing:
        print("[Critic] All scenarios meet score thresholds - no failures to diagnose.")
        return []

    # ── Build one prompt per failing scenario ────────────────────────────
    prompt_meta: list[tuple] = []
    prompts: list[str] = []

    for sc in failing:
        scenario_id = sc.get("scenario_id", "unknown")
        scores = {
            k: sc.get(k) or 0.0
            for k in ("persona", "accuracy", "safety", "relevance", "composite")
        }

        query = "(query not available)"
        ground_truth_section = ""
        if scenario_id in scenario_map:
            sc_file = scenario_map[scenario_id]
            query = sc_file.get("input", {}).get("query", query)
            rubric = sc_file.get("judge_rubric")
            if rubric:
                ground_truth_section = (
                    f"**Ground Truth / Expected Behaviour:**\n"
                    f"```json\n{json.dumps(rubric, indent=2)}\n```\n"
                )

        failure_mode = _primary_failure_mode(scores)
        reasoning = sc.get("reasoning", "(no judge reasoning available)")

        user_prompt = _CRITIC_USER_TEMPLATE.format(
            scenario_id=scenario_id,
            query=query,
            composite=scores["composite"],
            accuracy=scores["accuracy"],
            safety=scores["safety"],
            persona=scores["persona"],
            relevance=scores["relevance"],
            reasoning=reasoning[:600],
            ground_truth_section=ground_truth_section,
        )

        prompt_meta.append((scenario_id, scores, query, ground_truth_section, failure_mode, reasoning))
        prompts.append(user_prompt)

    # ── Fire all critic calls in parallel ────────────────────────────────
    print(
        f"[Critic] Diagnosing {len(prompts)} failing scenario(s) "
        f"(parallel LLM calls)..."
    )
    # return_exceptions=True is a safety net for unexpected coroutine-level
    # errors that bypass _call_critic's internal try/except (e.g. CancelledError).
    _raw: list = await asyncio.gather(
        *[_call_critic(p) for p in prompts],
        return_exceptions=True,
    )
    # Normalize: convert any BaseException to None so downstream code only
    # ever sees dict | None.  This eliminates the isinstance(…, Exception)
    # scatter-check and makes the handling path uniform.
    raw_results: list[dict | None] = [
        (None if isinstance(r, BaseException) else r) for r in _raw
    ]

    # ── Process results ──────────────────────────────────────────────────
    gradients: list[TextualGradient] = []

    for meta, raw_result in zip(prompt_meta, raw_results):
        scenario_id, scores, query, _gt_section, failure_mode, reasoning = meta

        if raw_result is None:
            print(f"[Critic] {scenario_id}: LLM call failed — skipping.", file=sys.stderr)
            continue

        target_node = raw_result.get("target_node")
        fix_type = raw_result.get("suggested_fix_type")

        _VALID_NODES = frozenset(_NODE_TO_FILE)
        _VALID_FIX_TYPES = frozenset({"patch_prompt", "update_ontology", "patch_phrase_list"})

        if not target_node or target_node not in _VALID_NODES:
            print(
                f"[Critic] {scenario_id}: LLM returned invalid or missing 'target_node' "
                f"({target_node!r}) — skipping gradient. "
                f"Valid nodes: {sorted(_VALID_NODES)}",
                file=sys.stderr,
            )
            continue
        if not fix_type or fix_type not in _VALID_FIX_TYPES:
            print(
                f"[Critic] {scenario_id}: LLM returned invalid or missing 'suggested_fix_type' "
                f"({fix_type!r}) — skipping gradient. "
                f"Valid types: {sorted(_VALID_FIX_TYPES)}",
                file=sys.stderr,
            )
            continue

        target_file = _NODE_TO_FILE[target_node]
        confidence = float(raw_result.get("confidence", 0.5))

        # Skip if this key has been attempted max_retries_per_fix times already
        key = (scenario_id, fix_type, target_node)
        attempt_count = tried_fixes_counter.get(key, 0)
        if attempt_count >= max_retries_per_fix:
            print(
                f"[Critic] {scenario_id}: skipping {fix_type}->{target_node} "
                f"(attempted {attempt_count}/{max_retries_per_fix} times — blocked)."
            )
            continue

        gradient = TextualGradient(
            scenario_id=scenario_id,
            query=query,
            scores=scores,
            failure_mode=raw_result.get("failure_mode", failure_mode),
            target_node=target_node,
            target_file=target_file,
            diagnosis=raw_result.get("diagnosis", ""),
            suggested_fix_type=fix_type,
            suggested_fix=raw_result.get("suggested_fix", {}),
            confidence=confidence,
            judge_reasoning=reasoning,
        )
        gradients.append(gradient)
        print(
            f"[Critic] {scenario_id}: {gradient.failure_mode} -> {gradient.target_node} "
            f"(confidence={confidence:.2f})"
        )

    return gradients
