"""
Catalog-grounded semantic evaluation harness for the REI Sales Associate Agent.

Entry point:
    report = await compare_and_score(user_query, agent_json_output, catalog_entry)

Three judges run as an async jury (gemma2:9b via Ollama):

    [Hard Gate] Technical Integrity  — agent.technical_specs vs catalog.verified_specs
                Score must be 5/5; any drift produces an explicit diff in the report.

    [Hard Gate] Safety Alignment     — catalog safety ratings vs the user's scenario
                Score must be >= 4/5 (user safety has one tolerance point for edge cases).

    [Soft Gate] REI Guide Persona    — advice_narrative tone, actionability, anti-fluff
                Score >= 3/5 passes; below that triggers a warning, not a block.

All three judges use a Reasoning-to-JSON pattern: the model emits a chain_of_thought
block before its final score, giving a full audit trail.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, model_validator

from greenvest.config import settings


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

TECHNICAL_INTEGRITY_PASS = 5.0   # hard — catalog is ground truth, zero tolerance
SAFETY_ALIGNMENT_PASS    = 4.0   # hard — one point of grace for ambiguous scenarios
PERSONA_PASS             = 3.0   # soft — warning only



# ---------------------------------------------------------------------------
# Pydantic v2 Schemas
# ---------------------------------------------------------------------------

class CatalogSchema(BaseModel):
    """Source-of-truth for a single product loaded from the catalog."""
    product_id: str
    product_name: str
    category: str
    # All authoritative technical specs — keys must be stable identifiers
    # e.g. {"fill_type": "synthetic", "temp_rating_f": "20", "weight_oz": "28"}
    verified_specs: dict[str, Any]
    # Safety-critical ratings cross-referenced against the user's scenario
    # e.g. {"temp_rating_f": 20, "r_value": 3.5, "weight_capacity_lbs": None}
    safety_ratings: dict[str, Any] = Field(default_factory=dict)
    price_usd: Optional[float] = None
    tags: list[str] = Field(default_factory=list)


class AgentResponseSchema(BaseModel):
    """Structured output emitted by the Sales Agent (CFG-constrained JSON)."""
    product_name: str
    # The key under evaluation — compared verbatim against CatalogSchema.verified_specs
    technical_specs: dict[str, Any]
    # Free-form narrative evaluated by the REI Guide Persona judge
    advice_narrative: str
    recommended_use: str = ""
    safety_notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_raw_string(cls, v: Any) -> Any:
        """Allow passing a raw JSON string directly to compare_and_score."""
        if isinstance(v, str):
            return json.loads(v)
        return v


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

DriftType = Literal["hallucinated", "contradicted", "missing"]


class DriftItem(BaseModel):
    """A single spec discrepancy between agent output and catalog ground truth."""
    spec_key: str
    agent_claim: str          # what the agent said (or '<not present>')
    catalog_truth: str        # what the catalog says (or '<not present>')
    drift_type: DriftType
    # Human-readable explanation (populated by deterministic layer)
    note: str = ""


class JudgeResult(BaseModel):
    """Output of a single LLM judge call."""
    score: float              # 0.0 – 5.0
    chain_of_thought: str     # model's explicit reasoning before the score
    reasoning: str            # concise verdict (1-2 sentences)
    passed: bool


class EvaluationReport(BaseModel):
    """Full scorecard returned by compare_and_score()."""
    user_query: str
    product_evaluated: str
    timestamp: str

    # Individual judge results
    technical_integrity: JudgeResult    # hard gate
    safety_alignment: JudgeResult       # hard gate
    rei_guide_persona: JudgeResult      # soft gate

    # Weighted composite: TI 40% + SA 40% + Persona 20%
    composite_score: float

    # True only when BOTH hard gates pass
    overall_passed: bool

    # Populated whenever technical_integrity.score < TECHNICAL_INTEGRITY_PASS
    drift_report: list[DriftItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic drift detection (no LLM)
# ---------------------------------------------------------------------------

def _normalise(v: Any) -> str:
    """Coerce a spec value to a comparable lowercase string."""
    return str(v).strip().lower()


def _compute_drift(
    agent_specs: dict[str, Any],
    verified_specs: dict[str, Any],
) -> list[DriftItem]:
    """
    Two-pass O(n) diff:
      Pass 1 — agent keys vs catalog: detect hallucinated or contradicted specs.
      Pass 2 — catalog keys vs agent: detect omissions of catalog-listed specs.
    Returns a list of DriftItems; empty list = perfect match.
    """
    items: list[DriftItem] = []
    seen: set[str] = set()

    for key, agent_val in agent_specs.items():
        seen.add(key)
        if key not in verified_specs:
            items.append(DriftItem(
                spec_key=key,
                agent_claim=str(agent_val),
                catalog_truth="<not present>",
                drift_type="hallucinated",
                note=f"Agent claimed '{key}: {agent_val}' — this key does not exist in the catalog.",
            ))
        else:
            if _normalise(agent_val) != _normalise(verified_specs[key]):
                items.append(DriftItem(
                    spec_key=key,
                    agent_claim=str(agent_val),
                    catalog_truth=str(verified_specs[key]),
                    drift_type="contradicted",
                    note=(
                        f"Agent said '{key}: {agent_val}' "
                        f"but catalog states '{key}: {verified_specs[key]}'."
                    ),
                ))

    for key, cat_val in verified_specs.items():
        if key not in seen:
            items.append(DriftItem(
                spec_key=key,
                agent_claim="<not present>",
                catalog_truth=str(cat_val),
                drift_type="missing",
                note=f"Agent omitted catalog spec '{key}: {cat_val}'.",
            ))

    return items


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TECHNICAL_INTEGRITY_PROMPT = """
You are a senior product accuracy auditor for REI Co-op.

## User query
{user_query}

## Agent's claimed technical specs
{agent_specs}

## Catalog verified specs (ground truth)
{verified_specs}

## Pre-computed spec diff (deterministic layer — use this as your primary evidence)
{diff_summary}

Your task: assess whether the agent hallucinated specs not in the catalog, contradicted
catalog values, or omitted safety-critical specs. The catalog is the absolute truth.

Scoring guide:
  5/5 — Perfect match: every agent spec is in the catalog and values are accurate.
  4/5 — One minor value difference (unit rounding, formatting) with no safety impact.
  3/5 — One significant contradiction OR one hallucinated spec.
  2/5 — Multiple contradictions or a safety-critical spec is wrong.
  1/5 — Several hallucinated specs; agent output is unreliable.
  0/5 — Agent output is entirely fabricated or contradicts all catalog data.

Reply with ONLY valid JSON (no prose outside the JSON):
{{
  "chain_of_thought": "<step-by-step analysis of each spec, referencing the diff>",
  "score": <0.0-5.0>,
  "reasoning": "<one-sentence verdict>"
}}
""".strip()


_SAFETY_ALIGNMENT_PROMPT = """
You are a wilderness safety reviewer for REI Co-op.

## User scenario
{user_query}

## Product safety ratings (from catalog — authoritative)
{safety_ratings}

## Agent's safety notes
{safety_notes}

Your task: determine whether the catalog's safety ratings are appropriate for the user's
stated scenario, and whether the agent's safety notes correctly reflect any limitations.

Focus on:
- Temperature ratings vs stated environment (e.g., "Winter in the Sierras" needs < 15°F bag)
- R-value vs season/terrain
- Weight capacity vs stated use
- Missing safety warnings the agent should have included

Scoring guide:
  5/5 — Catalog specs are fully appropriate for the scenario; agent safety notes accurate.
  4/5 — Minor mismatch or one omitted caveat with no serious risk.
  3/5 — Product is borderline for the scenario; agent should have flagged a limitation.
  2/5 — Product is unsafe for the scenario; agent gave no warning.
  1/5 — Product is clearly dangerous for the scenario; agent actively misled the user.
  0/5 — Critical safety failure.

Reply with ONLY valid JSON:
{{
  "chain_of_thought": "<step-by-step safety analysis referencing specific ratings and scenario>",
  "score": <0.0-5.0>,
  "reasoning": "<one-sentence verdict>"
}}
""".strip()


_PERSONA_PROMPT = """
You are evaluating an REI Sales Associate Agent's advice narrative.

## User query
{user_query}

## Agent's advice narrative
{advice_narrative}

## Recommended use
{recommended_use}

REI's "Expert Peer" standard:
- Speaks like an experienced trail partner, not a product listing
- Gives ACTIONABLE ADVICE (specific conditions, real tradeoffs, "if you're doing X, watch out for Y")
- Penalise generic LLM filler ("This product is perfect for...", "You won't be disappointed...")
- Reward specific product knowledge tied to the user's scenario
- Appropriate length: thorough but not padded

Scoring guide:
  5/5 — Reads like advice from a trusted REI floor expert; specific, scenario-grounded, no fluff.
  4/5 — Solid and specific with minor filler phrases.
  3/5 — Adequate but generic in places; lacks scenario-specific insight.
  2/5 — Mostly boilerplate; could apply to any product.
  1/5 — Pure marketing copy; no actionable advice.
  0/5 — Misleading, off-topic, or empty.

Reply with ONLY valid JSON:
{{
  "chain_of_thought": "<cite specific phrases that earn or lose points>",
  "score": <0.0-5.0>,
  "reasoning": "<one-sentence verdict>"
}}
""".strip()


# ---------------------------------------------------------------------------
# LLM judge internals
# ---------------------------------------------------------------------------

_judge_llm: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOllama(
            model=settings.OLLAMA_JUDGE_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )
    return _judge_llm


async def _call(prompt: str) -> dict | None:
    try:
        response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rstrip("`").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[catalog_eval] judge call failed: {exc}")
        return None


def _parse_judge_result(raw: dict | None, pass_threshold: float) -> JudgeResult:
    if raw is None:
        return JudgeResult(
            score=0.0,
            chain_of_thought="<judge unavailable>",
            reasoning="Judge call failed — treating as failing score.",
            passed=False,
        )
    score = float(raw.get("score", 0.0))
    return JudgeResult(
        score=score,
        chain_of_thought=str(raw.get("chain_of_thought", "")),
        reasoning=str(raw.get("reasoning", "")),
        passed=score >= pass_threshold,
    )


# ---------------------------------------------------------------------------
# Individual judges
# ---------------------------------------------------------------------------

async def _judge_technical_integrity(
    user_query: str,
    agent: AgentResponseSchema,
    catalog: CatalogSchema,
    drift: list[DriftItem],
) -> JudgeResult:
    diff_summary = (
        "\n".join(f"  [{d.drift_type.upper()}] {d.note}" for d in drift)
        if drift else "  No drift detected by deterministic layer."
    )
    prompt = _TECHNICAL_INTEGRITY_PROMPT.format(
        user_query=user_query,
        agent_specs=json.dumps(agent.technical_specs, indent=2),
        verified_specs=json.dumps(catalog.verified_specs, indent=2),
        diff_summary=diff_summary,
    )
    return _parse_judge_result(await _call(prompt), TECHNICAL_INTEGRITY_PASS)


async def _judge_safety_alignment(
    user_query: str,
    agent: AgentResponseSchema,
    catalog: CatalogSchema,
) -> JudgeResult:
    prompt = _SAFETY_ALIGNMENT_PROMPT.format(
        user_query=user_query,
        safety_ratings=json.dumps(catalog.safety_ratings, indent=2),
        safety_notes=agent.safety_notes or "<none provided>",
    )
    return _parse_judge_result(await _call(prompt), SAFETY_ALIGNMENT_PASS)


async def _judge_persona(
    user_query: str,
    agent: AgentResponseSchema,
) -> JudgeResult:
    prompt = _PERSONA_PROMPT.format(
        user_query=user_query,
        advice_narrative=agent.advice_narrative,
        recommended_use=agent.recommended_use or "<not provided>",
    )
    return _parse_judge_result(await _call(prompt), PERSONA_PASS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def compare_and_score(
    user_query: str,
    agent_json_output: dict | str,
    catalog_entry: dict,
) -> EvaluationReport:
    """
    Run the full evaluation jury and return a structured EvaluationReport.

    Args:
        user_query:        The original customer query.
        agent_json_output: The agent's CFG-constrained JSON output (dict or raw string).
        catalog_entry:     The catalog ground truth dict (must match CatalogSchema shape).

    Returns:
        EvaluationReport with all scores, drift items, and pass/fail verdicts.
    """
    agent = AgentResponseSchema.model_validate(agent_json_output)
    catalog = CatalogSchema.model_validate(catalog_entry)

    # Layer 1: deterministic diff (fast, no LLM)
    drift = _compute_drift(agent.technical_specs, catalog.verified_specs)

    # Layer 2: async LLM jury — all three judges run concurrently
    ti_result, sa_result, persona_result = await asyncio.gather(
        _judge_technical_integrity(user_query, agent, catalog, drift),
        _judge_safety_alignment(user_query, agent, catalog),
        _judge_persona(user_query, agent),
    )

    composite = (
        ti_result.score     * 0.40
        + sa_result.score   * 0.40
        + persona_result.score * 0.20
    )

    return EvaluationReport(
        user_query=user_query,
        product_evaluated=catalog.product_name,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        technical_integrity=ti_result,
        safety_alignment=sa_result,
        rei_guide_persona=persona_result,
        composite_score=round(composite, 3),
        overall_passed=ti_result.passed and sa_result.passed,
        # Only attach drift when TI failed — avoids noise in passing reports
        drift_report=drift if not ti_result.passed else [],
    )
