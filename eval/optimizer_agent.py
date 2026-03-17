"""
Optimizer Agent for the autonomous optimization pipeline.

Takes a TextualGradient from the Critic, reads the current value of the
targeted code element, calls an LLM to generate a specific replacement,
and applies it via the deterministic edit_tools.

Usage
-----
    from eval.critic import TextualGradient
    from eval.optimizer_agent import generate_and_apply

    result = await generate_and_apply(gradient)
    if result.success:
        print(f"Applied: {result.summary}")
    else:
        print(f"Failed:  {result.error}")
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from greenvest.config import settings
from eval.critic import TextualGradient
from eval.edit_tools import (
    REPO_ROOT,
    EditError,
    patch_prompt,
    patch_list_literal,
    patch_phrase_list,
    update_ontology,
    read_node_value,
    ONTOLOGY_PATH,
    _QUERY_TRANSLATOR,
)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ApplyResult:
    """Outcome of a single optimizer edit attempt."""
    success: bool
    summary: str
    edit_type: str
    filepath: str
    old_value: object = field(default=None)
    new_value: object = field(default=None)
    error: str = field(default="")


# ---------------------------------------------------------------------------
# Optimizer prompts
# ---------------------------------------------------------------------------

_OPTIMIZER_SYSTEM = """\
You are a code-optimization agent for an AI outdoor gear assistant called Greenvest (REI Co-op).

You will be given:
1. A structured failure diagnosis (textual gradient) from a Critic agent.
2. The CURRENT VALUE of the code element that needs to change.
3. The edit type you must produce.

Your job is to output a single, targeted replacement that addresses the diagnosed failure
while preserving all other behaviour.

## Rules
- Output ONLY valid JSON — no prose, no markdown fences.
- For `patch_prompt`: the new string must be non-empty, professional, and directly fix
  the diagnosed failure. Keep the same approximate length as the original.
- For `update_ontology`: new_specs must be a flat dict of spec_key → string value.
  Only include specs that directly fix the diagnosis. Do not remove existing specs.
- For `patch_phrase_list`: new_phrases must include ALL old phrases PLUS any new ones.
  Never remove existing phrases.
- Do NOT hallucinate product names, spec values, or technical standards.

## Output Format

For patch_prompt:
{
  "edit_type": "patch_prompt",
  "node_name": "<variable name>",
  "new_value": "<the full replacement string>",
  "rationale": "<one sentence explaining the change>"
}

For update_ontology:
{
  "edit_type": "update_ontology",
  "category": "<sleeping_bags|sleeping_pads|footwear|jackets|backpacks>",
  "alias_key": "<exact alias key from ontology>",
  "new_specs": { "<spec_key>": "<value>", ... },
  "rationale": "<one sentence explaining the change>"
}

For patch_phrase_list:
{
  "edit_type": "patch_phrase_list",
  "new_phrases": ["<phrase1>", "<phrase2>", ...],
  "rationale": "<one sentence explaining the new phrases>"
}

For patch_set_literal:
{
  "edit_type": "patch_set_literal",
  "node_name": "<variable name, e.g. '_ENV_SENSITIVE_ACTIVITIES'>",
  "target_file": "<relative path, e.g. 'greenvest/nodes/clarification_gate.py'>",
  "new_items": ["<item1>", "<item2>", ...],
  "rationale": "<one sentence explaining the change>"
}
""".strip()

_OPTIMIZER_USER_TEMPLATE = """\
## Failure Diagnosis

**Scenario:** {scenario_id}
**Query:** {query}
**Failure Mode:** {failure_mode}
**Target Node:** {target_node}
**Diagnosis:** {diagnosis}
**Suggested Fix Type:** {suggested_fix_type}
**Fix Hint:** {fix_description}

## Current Value of Target Element

**Edit Type:** {suggested_fix_type}
**Target:** {target_description}

```
{current_value}
```

Generate a replacement that fixes the diagnosed failure. Output JSON only.
""".strip()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_optimizer_llm():
    """Return the strongest available LLM for the optimizer (Anthropic > Ollama)."""
    if settings.ANTHROPIC_API_KEY:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.1,   # Slight creativity for rephrasing; low for spec changes
                max_tokens=2048,
            )
        except ImportError:
            print(
                "[WARN] langchain_anthropic not installed; falling back to Ollama for optimizer.",
                file=sys.stderr,
            )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.OLLAMA_JUDGE_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
    )


def _extract_json(raw: str) -> dict | None:
    """Extract the first JSON object from a string, tolerating surrounding prose."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
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


# Retry configuration for LLM calls
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt (1 → 2 → 4)


async def _call_optimizer_once(user_prompt: str) -> dict | None:
    """Single attempt: invoke the optimizer LLM and parse the JSON response."""
    from eval.token_counter import record
    try:
        llm = _get_optimizer_llm()
        response = await llm.ainvoke(
            [SystemMessage(content=_OPTIMIZER_SYSTEM), HumanMessage(content=user_prompt)]
        )
        usage = record("optimizer", response.response_metadata)
        if usage.total > 0:
            print(f"[Optimizer] Tokens this call: {usage}", file=sys.stderr)
        result = _extract_json(response.content)
        if result is None:
            print("[WARN] Optimizer returned no parseable JSON.", file=sys.stderr)
        return result
    except Exception as exc:
        print(f"[WARN] Optimizer LLM call failed: {exc}", file=sys.stderr)
        return None


async def _call_optimizer(user_prompt: str) -> dict | None:
    """Invoke the optimizer LLM with exponential-backoff retries.

    Retries up to _LLM_MAX_RETRIES times on None results (parse failure or
    exception). Delays: 1 s, 2 s, 4 s between attempts.
    Returns None after all retries are exhausted.
    """
    for attempt in range(_LLM_MAX_RETRIES):
        result = await _call_optimizer_once(user_prompt)
        if result is not None:
            return result
        if attempt < _LLM_MAX_RETRIES - 1:
            delay = _LLM_RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"[Optimizer] Retry {attempt + 1}/{_LLM_MAX_RETRIES - 1} "
                f"in {delay:.0f}s...",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
    print(
        f"[ERROR] Optimizer LLM call failed after {_LLM_MAX_RETRIES} attempts — "
        "no edit plan generated.",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _read_current_ontology_section(alias_key: str, category: Optional[str] = None) -> str:
    """Read the relevant part of the ontology for the optimizer's context."""
    try:
        raw = ONTOLOGY_PATH.read_text(encoding="utf-8")
        lines = raw.splitlines()
        # Find the line containing the alias_key
        for i, line in enumerate(lines):
            if alias_key in line:
                # Return the alias key line + next few spec lines
                end = min(i + 8, len(lines))
                return "\n".join(lines[max(0, i - 1) : end])
        # Fallback: return full ontology (truncated)
        return raw[:1500]
    except Exception:
        return "(ontology not readable)"


def _read_current_phrase_list() -> str:
    """Read the current phrase list from query_translator._extract_terms."""
    try:
        import ast
        source = _QUERY_TRANSLATOR.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_terms":
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and isinstance(child.iter, ast.List):
                        try:
                            return repr(ast.literal_eval(child.iter))
                        except Exception:
                            return "(phrase list not evaluable)"
        return "(phrase list not found)"
    except Exception as exc:
        return f"(error reading phrase list: {exc})"


def _build_context(gradient: TextualGradient) -> tuple[str, str]:
    """
    Return (target_description, current_value_repr) for the optimizer prompt.
    """
    fix_type = gradient.suggested_fix_type
    fix = gradient.suggested_fix

    if fix_type == "patch_prompt":
        node_name = fix.get("node_name", "_REI_PERSONA")
        target_file = REPO_ROOT / gradient.target_file
        current = read_node_value(str(target_file), node_name)
        current_repr = repr(current) if current is not None else "(variable not found)"
        return (f"`{node_name}` in `{gradient.target_file}`", current_repr)

    elif fix_type == "update_ontology":
        alias_key = fix.get("alias_key", "")
        category = fix.get("category")
        current_repr = _read_current_ontology_section(alias_key, category)
        return (f"Ontology section for '{alias_key}'", current_repr)

    elif fix_type == "patch_phrase_list":
        current_repr = _read_current_phrase_list()
        return ("Phrase list in `_extract_terms()` of `query_translator.py`", current_repr)

    elif fix_type == "patch_set_literal":
        node_name = fix.get("node_name", "_ENV_SENSITIVE_ACTIVITIES")
        target_file = REPO_ROOT / gradient.target_file
        current = read_node_value(str(target_file), node_name)
        current_repr = repr(current) if current is not None else "(variable not found)"
        return (f"`{node_name}` (set) in `{gradient.target_file}`", current_repr)

    else:
        return (gradient.target_file, "(unknown fix type)")


# ---------------------------------------------------------------------------
# Apply logic
# ---------------------------------------------------------------------------

def _apply_patch_prompt(plan: dict, gradient: TextualGradient) -> ApplyResult:
    """Apply a patch_prompt edit plan."""
    node_name = plan.get("node_name") or gradient.suggested_fix.get("node_name")
    new_value = plan.get("new_value")
    if not node_name:
        return ApplyResult(False, "", "patch_prompt", gradient.target_file,
                           error="'node_name' missing from optimizer plan.")
    if not new_value or not isinstance(new_value, str):
        return ApplyResult(False, "", "patch_prompt", gradient.target_file,
                           error="'new_value' missing or not a string in optimizer plan.")

    target_path = REPO_ROOT / gradient.target_file

    try:
        existing = read_node_value(str(target_path), node_name)
    except Exception as exc:
        return ApplyResult(False, "", "patch_prompt", gradient.target_file,
                           error=f"Could not read '{node_name}' from {gradient.target_file}: {exc}")
    if existing is None:
        return ApplyResult(False, "", "patch_prompt", gradient.target_file,
                           error=(
                               f"Variable '{node_name}' not found in {gradient.target_file}. "
                               "The LLM may have hallucinated the variable name."
                           ))

    try:
        old_value = patch_prompt(str(target_path), node_name, new_value)
        return ApplyResult(
            success=True,
            summary=(
                f"Patched `{node_name}` in `{gradient.target_file}`: "
                f"{len(new_value)} chars (was {len(old_value)} chars). "
                f"Rationale: {plan.get('rationale', '')}"
            ),
            edit_type="patch_prompt",
            filepath=gradient.target_file,
            old_value=old_value,
            new_value=new_value,
        )
    except EditError as exc:
        return ApplyResult(False, "", "patch_prompt", gradient.target_file, error=str(exc))


def _apply_update_ontology(plan: dict, gradient: TextualGradient) -> ApplyResult:
    """Apply an update_ontology edit plan."""
    category = plan.get("category") or gradient.suggested_fix.get("category")
    alias_key = plan.get("alias_key") or gradient.suggested_fix.get("alias_key")
    new_specs = plan.get("new_specs")

    if not alias_key:
        return ApplyResult(False, "", "update_ontology", gradient.target_file,
                           error="'alias_key' missing from optimizer plan.")
    if not new_specs or not isinstance(new_specs, dict):
        return ApplyResult(False, "", "update_ontology", gradient.target_file,
                           error="'new_specs' missing or not a dict in optimizer plan.")

    try:
        old_specs = update_ontology(str(ONTOLOGY_PATH), alias_key, new_specs, category=category)
        return ApplyResult(
            success=True,
            summary=(
                f"Updated ontology '{alias_key}' in '{category}': "
                f"{new_specs}. "
                f"Rationale: {plan.get('rationale', '')}"
            ),
            edit_type="update_ontology",
            filepath=gradient.target_file,
            old_value=old_specs,
            new_value=new_specs,
        )
    except EditError as exc:
        return ApplyResult(False, "", "update_ontology", gradient.target_file, error=str(exc))


def _apply_patch_phrase_list(plan: dict, gradient: TextualGradient) -> ApplyResult:
    """Apply a patch_phrase_list edit plan."""
    new_phrases = plan.get("new_phrases")
    if not new_phrases or not isinstance(new_phrases, list):
        return ApplyResult(False, "", "patch_phrase_list", gradient.target_file,
                           error="'new_phrases' missing or not a list in optimizer plan.")

    try:
        old_phrases = patch_phrase_list(str(_QUERY_TRANSLATOR), new_phrases)
        added = [p for p in new_phrases if p not in old_phrases]
        return ApplyResult(
            success=True,
            summary=(
                f"Updated phrase list: added {added}. "
                f"Rationale: {plan.get('rationale', '')}"
            ),
            edit_type="patch_phrase_list",
            filepath="greenvest/nodes/query_translator.py",
            old_value=old_phrases,
            new_value=new_phrases,
        )
    except EditError as exc:
        return ApplyResult(False, "", "patch_phrase_list", gradient.target_file, error=str(exc))


def _apply_patch_set_literal(plan: dict, gradient: TextualGradient) -> ApplyResult:
    """Apply a patch_set_literal edit plan."""
    node_name = plan.get("node_name") or gradient.suggested_fix.get("node_name")
    new_items = plan.get("new_items")
    if not node_name:
        return ApplyResult(False, "", "patch_set_literal", gradient.target_file,
                           error="'node_name' missing from optimizer plan.")
    if not new_items or not isinstance(new_items, list):
        return ApplyResult(False, "", "patch_set_literal", gradient.target_file,
                           error="'new_items' missing or not a list in optimizer plan.")

    target_path = REPO_ROOT / gradient.target_file
    try:
        old_items = patch_list_literal(str(target_path), node_name, set(new_items))
        added = [i for i in new_items if i not in old_items]
        return ApplyResult(
            success=True,
            summary=(
                f"Updated set `{node_name}` in `{gradient.target_file}`: "
                f"added {added}. "
                f"Rationale: {plan.get('rationale', '')}"
            ),
            edit_type="patch_set_literal",
            filepath=gradient.target_file,
            old_value=old_items,
            new_value=new_items,
        )
    except EditError as exc:
        return ApplyResult(False, "", "patch_set_literal", gradient.target_file, error=str(exc))


def _apply_plan(plan: dict, gradient: TextualGradient) -> ApplyResult:
    """Dispatch to the correct apply function based on edit_type."""
    edit_type = plan.get("edit_type", gradient.suggested_fix_type)
    if edit_type == "patch_prompt":
        return _apply_patch_prompt(plan, gradient)
    elif edit_type == "update_ontology":
        return _apply_update_ontology(plan, gradient)
    elif edit_type == "patch_phrase_list":
        return _apply_patch_phrase_list(plan, gradient)
    elif edit_type == "patch_set_literal":
        return _apply_patch_set_literal(plan, gradient)
    else:
        return ApplyResult(
            False, "", edit_type, gradient.target_file,
            error=f"Unknown edit_type '{edit_type}' in optimizer plan."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_edit_plan(gradient: TextualGradient) -> dict | None:
    """
    Call the optimizer LLM and return the raw edit plan dict — no file writes.

    Use this when you want to inspect or diff the planned change before applying
    it (e.g. in grid-search mode). The returned dict can be passed directly to
    apply_plan().

    Parameters
    ----------
    gradient : TextualGradient from the Critic

    Returns
    -------
    Raw plan dict from the optimizer LLM, or None if the LLM call failed.
    """
    target_description, current_value_repr = _build_context(gradient)
    fix = gradient.suggested_fix

    user_prompt = _OPTIMIZER_USER_TEMPLATE.format(
        scenario_id=gradient.scenario_id,
        query=gradient.query,
        failure_mode=gradient.failure_mode,
        target_node=gradient.target_node,
        diagnosis=gradient.diagnosis,
        suggested_fix_type=gradient.suggested_fix_type,
        fix_description=fix.get("description", "(see diagnosis)"),
        target_description=target_description,
        current_value=current_value_repr[:2000],
    )

    print(
        f"[Optimizer] Generating edit plan for {gradient.target_node} "
        f"({gradient.suggested_fix_type})..."
    )
    plan = await _call_optimizer(user_prompt)
    if plan is None:
        print(
            f"[Optimizer] LLM returned no parseable plan for "
            f"{gradient.scenario_id}.",
            file=sys.stderr,
        )
    return plan


# Expose the internal apply dispatcher as a public function so grid_search
# can apply a plan generated by generate_edit_plan() without re-running the LLM.
apply_plan = _apply_plan


async def generate_and_apply(gradient: TextualGradient) -> ApplyResult:
    """
    Generate a specific code edit for a TextualGradient and apply it.

    Convenience wrapper around generate_edit_plan() + apply_plan().

    Parameters
    ----------
    gradient : TextualGradient from the Critic

    Returns
    -------
    ApplyResult — check `.success` and `.error` fields.
    """
    plan = await generate_edit_plan(gradient)
    if plan is None:
        return ApplyResult(
            False, "", gradient.suggested_fix_type, gradient.target_file,
            error="Optimizer LLM returned no parseable plan."
        )

    result = apply_plan(plan, gradient)
    if result.success:
        print(f"[Optimizer] Applied: {result.summary}")
    else:
        print(f"[Optimizer] Edit failed: {result.error}", file=sys.stderr)

    return result


def _gradient_target_key(gradient: TextualGradient) -> tuple:
    """
    Return a hashable key that uniquely identifies the code element this gradient
    wants to edit. Used to deduplicate before applying: if two gradients target the
    same variable/key, only the higher-confidence one is attempted so they cannot
    stomp each other.
    """
    fix = gradient.suggested_fix
    fix_type = gradient.suggested_fix_type
    if fix_type == "patch_prompt":
        return (gradient.target_file, fix_type, fix.get("node_name", ""))
    if fix_type == "update_ontology":
        return (gradient.target_file, fix_type, fix.get("alias_key", ""))
    # patch_phrase_list always targets the same list, so key on fix_type alone
    return (gradient.target_file, fix_type, "")


async def generate_and_apply_all(
    gradients: list[TextualGradient],
    max_edits: int = 1,
) -> tuple[list[ApplyResult], list[TextualGradient]]:
    """
    Apply up to `max_edits` edit plans from the gradient list (highest-confidence first).

    Deduplicates gradients by their edit target before applying so two gradients
    that both target the same variable cannot stomp each other. When duplicates
    exist the highest-confidence gradient wins.

    Stops after the first successful edit per iteration to keep changes atomic
    and reversible. Use max_edits > 1 only for batch pre-commits.

    Parameters
    ----------
    gradients : list from analyze_failures()
    max_edits : maximum number of edits to attempt

    Returns
    -------
    (results, attempted) where `attempted` is the list of TextualGradient objects
    that were actually passed to generate_and_apply (regardless of success).
    The caller uses `attempted` to update its tried-fixes set on revert.
    """
    # Deduplicate: keep the highest-confidence gradient per unique edit target.
    seen: dict[tuple, TextualGradient] = {}
    for g in gradients:
        key = _gradient_target_key(g)
        if key not in seen or g.confidence > seen[key].confidence:
            seen[key] = g
    unique = list(seen.values())
    if len(unique) < len(gradients):
        print(
            f"[Optimizer] Deduplicated {len(gradients)} gradient(s) → "
            f"{len(unique)} unique edit target(s)."
        )

    results: list[ApplyResult] = []
    attempted: list[TextualGradient] = []
    applied = 0

    for gradient in sorted(unique, key=lambda g: g.confidence, reverse=True):
        if applied >= max_edits:
            break
        result = await generate_and_apply(gradient)
        # Record gradient as attempted only AFTER the call so that a crash or
        # exception mid-apply does not permanently blacklist the gradient.
        attempted.append(gradient)
        results.append(result)
        if result.success:
            applied += 1

    return results, attempted
