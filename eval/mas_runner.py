"""
MAS (Multi-Agent Simulation) Monte Carlo runner.

Orchestrates conversations between BuyerAgent and the Greenvest graph,
running each persona scenario N times with natural phrasing variance.

Each run:
  1. BuyerAgent generates an initial query (randomly sampled from pool)
  2. Greenvest graph is invoked
  3. If REQUIRES_CLARIFICATION  → BuyerAgent responds → repeat (max MAX_TURNS)
  4. If READY_TO_SYNTHESIZE     → converged, record metrics
  5. If MAX_TURNS exceeded      → failed run

When enable_judge=True, each converged run is also scored by Claude Opus 4.6
across four dimensions (persona, accuracy, safety, relevance), and each
clarification question is scored for focus. Results are logged to Langfuse.

Usage (from scripts/run_mas.py):
    results = asyncio.run(run_monte_carlo(total_runs=500))
    results = asyncio.run(run_monte_carlo(total_runs=500, enable_judge=True))
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from greenvest.graph import graph
from greenvest.latency import LatencyCallback
from greenvest.retrieval.embeddings import warmup_models
from greenvest.state import initial_state
from eval.buyer_agent import BuyerAgent
from eval.personas import PERSONAS, Persona
from eval.langfuse_client import langfuse_client

log = structlog.get_logger(__name__)

MAX_TURNS = 4       # safety valve — clarification cap is 2, so 3 turns is the real max
CONCURRENCY = 5     # parallel conversations; raise if GPU memory allows


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: int
    persona_name: str
    converged: bool             # reached READY_TO_SYNTHESIZE with a recommendation
    turns: int                  # total buyer↔agent exchanges
    clarification_count: int    # how many clarification rounds occurred
    forced_forward: bool        # clarification cap (count=2) triggered
    spec_match: bool            # derived_specs contains all persona.required_specs
    intent_correct: bool        # intent_router returned persona.expected_intent
    recommendation_len: int     # char length of final recommendation (0 if not converged)
    recommendation: Optional[str]
    conversation: list[dict]    # [{role, content}, ...] full exchange log
    duration_seconds: float
    # Latency breakdown (first turn only — from query submission to first agent response)
    first_response_latency_s: Optional[float] = None
    node_timings_ms: dict = field(default_factory=dict)  # node_name -> ms (first turn)
    error: Optional[str] = None
    # Judge scores (populated when enable_judge=True and run converged)
    judge_composite: Optional[float] = None
    judge_persona: Optional[float] = None
    judge_accuracy: Optional[float] = None
    judge_safety: Optional[float] = None
    judge_relevance: Optional[float] = None
    judge_reasoning: Optional[str] = None
    # Clarification quality score (first clarification only)
    clarification_quality: Optional[float] = None
    # Spec translation quality score
    spec_quality: Optional[float] = None


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

async def _run_single(
    persona: Persona,
    run_id: int,
    sem: asyncio.Semaphore,
    enable_judge: bool = False,
) -> RunResult:
    async with sem:
        start = time.monotonic()
        buyer = BuyerAgent(persona)
        conversation: list[dict] = []
        result: dict | None = None
        error: str | None = None
        first_clarification_msg: str | None = None
        first_query: str = ""

        # --- initial query ---
        query = buyer.initial_query()
        first_query = query
        conversation.append({"role": "buyer", "content": query})

        state = initial_state(query=query)
        state["clarification_count"] = 0
        if persona.budget_usd:
            state["budget_usd"] = persona.budget_usd

        turns = 0
        first_response_latency_s: float | None = None
        node_timings_ms: dict = {}

        try:
            for turn in range(MAX_TURNS):
                turns = turn + 1
                latency_cb = LatencyCallback()
                t_query = time.monotonic()
                result = await graph.ainvoke(state, config={"callbacks": [latency_cb]})
                # Capture first-turn latency and node breakdown once
                if turn == 0:
                    first_response_latency_s = round(time.monotonic() - t_query, 3)
                    node_timings_ms = latency_cb.timings_ms
                flag = result.get("action_flag")

                if flag == "READY_TO_SYNTHESIZE":
                    if result.get("recommendation"):
                        conversation.append({
                            "role": "agent",
                            "content": result["recommendation"],
                        })
                    break

                if flag == "REQUIRES_CLARIFICATION":
                    clarification = result.get("clarification_message", "")
                    conversation.append({"role": "agent", "content": clarification})

                    # Record first clarification for quality scoring
                    if first_clarification_msg is None:
                        first_clarification_msg = clarification

                    buyer_response = await buyer.respond(clarification, conversation[:-1])
                    conversation.append({"role": "buyer", "content": buyer_response})

                    # Build next-turn state, carrying over extracted context
                    state = initial_state(query=buyer_response)
                    state["clarification_count"] = result.get("clarification_count", turn + 1)
                    state["activity"] = result.get("activity")
                    state["user_environment"] = result.get("user_environment")
                    state["experience_level"] = result.get("experience_level")
                    if persona.budget_usd:
                        state["budget_usd"] = persona.budget_usd

        except Exception as exc:
            error = str(exc)
            log.error("mas_runner.run_single", run_id=run_id, persona=persona.name, error=error)

        # --- compute metrics ---
        converged = (
            result is not None
            and result.get("action_flag") == "READY_TO_SYNTHESIZE"
            and bool(result.get("recommendation"))
            and error is None
        )

        derived: dict = result.get("derived_specs", {}) if result else {}
        spec_match = all(
            derived.get(k) == v
            for k, v in persona.required_specs.items()
        )

        intent_correct = (
            result.get("intent") == persona.expected_intent
            if result else False
        )

        clari_count = result.get("clarification_count", 0) if result else 0
        forced = clari_count >= 2

        log.info(
            "mas_runner.run_done",
            run_id=run_id,
            persona=persona.name,
            converged=converged,
            turns=turns,
            spec_match=spec_match,
        )

        # --- LLM-as-a-Judge (opt-in) ---
        judge_composite = judge_persona = judge_accuracy = None
        judge_safety = judge_relevance = None
        judge_reasoning = None
        clarification_quality = None
        spec_quality = None

        if enable_judge:
            from eval.judge import (
                judge_recommendation,
                judge_clarification,
                judge_spec_translation,
            )

            async def _noop():
                return None

            judge_tasks = []

            if converged and result.get("recommendation"):
                judge_tasks.append(
                    judge_recommendation(first_query, result["recommendation"])
                )
            else:
                judge_tasks.append(_noop())

            if first_clarification_msg:
                judge_tasks.append(judge_clarification(first_query, first_clarification_msg))
            else:
                judge_tasks.append(_noop())

            if derived:
                judge_tasks.append(
                    judge_spec_translation(
                        first_query,
                        result.get("activity") if result else None,
                        derived,
                    )
                )
            else:
                judge_tasks.append(_noop())

            rec_scores, clari_score, spec_score = await asyncio.gather(*judge_tasks)

            if rec_scores is not None:
                judge_composite = rec_scores.composite
                judge_persona = rec_scores.persona
                judge_accuracy = rec_scores.accuracy
                judge_safety = rec_scores.safety
                judge_relevance = rec_scores.relevance
                judge_reasoning = rec_scores.reasoning

            clarification_quality = clari_score
            spec_quality = spec_score

            # Log scores to Langfuse
            lf = langfuse_client()
            if lf and judge_composite is not None:
                try:
                    trace_name = f"mas_run_{run_id}_{persona.name}"
                    with lf.start_as_current_observation(
                        name=trace_name,
                        as_type="span",
                        input={"query": first_query, "persona": persona.name},
                        output={"recommendation": result.get("recommendation") if result else None},
                    ):
                        trace_id = lf.get_current_trace_id()

                    for score_name, score_val in [
                        ("judge_composite", judge_composite),
                        ("judge_persona", judge_persona),
                        ("judge_accuracy", judge_accuracy),
                        ("judge_safety", judge_safety),
                        ("judge_relevance", judge_relevance),
                    ]:
                        if score_val is not None and trace_id:
                            lf.create_score(
                                trace_id=trace_id,
                                name=score_name,
                                value=score_val,
                                data_type="NUMERIC",
                            )
                except Exception as exc:
                    log.warning("mas_runner.langfuse_error", error=str(exc))

        return RunResult(
            run_id=run_id,
            persona_name=persona.name,
            converged=converged,
            turns=turns,
            clarification_count=clari_count,
            forced_forward=forced,
            spec_match=spec_match,
            intent_correct=intent_correct,
            recommendation_len=len(result.get("recommendation") or "") if result else 0,
            recommendation=result.get("recommendation") if (result and converged) else None,
            conversation=conversation,
            duration_seconds=time.monotonic() - start,
            first_response_latency_s=first_response_latency_s,
            node_timings_ms=node_timings_ms,
            error=error,
            judge_composite=judge_composite,
            judge_persona=judge_persona,
            judge_accuracy=judge_accuracy,
            judge_safety=judge_safety,
            judge_relevance=judge_relevance,
            judge_reasoning=judge_reasoning,
            clarification_quality=clarification_quality,
            spec_quality=spec_quality,
        )


# ---------------------------------------------------------------------------
# Monte Carlo orchestrator
# ---------------------------------------------------------------------------

async def run_monte_carlo(
    personas: list[Persona] | None = None,
    total_runs: int = 500,
    concurrency: int = CONCURRENCY,
    enable_judge: bool = False,
) -> list[RunResult]:
    """
    Distribute `total_runs` evenly across personas and run all concurrently
    (bounded by `concurrency`).

    Set enable_judge=True to score each converged run with Claude Opus 4.6.
    """
    # Pre-warm embedding models once so no run pays the cold-start cost
    warmup_models()

    if personas is None:
        personas = PERSONAS

    runs_per_persona = total_runs // len(personas)
    remainder = total_runs % len(personas)

    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task] = []
    run_id = 0

    for i, persona in enumerate(personas):
        n = runs_per_persona + (1 if i < remainder else 0)
        for _ in range(n):
            tasks.append(asyncio.create_task(
                _run_single(persona, run_id, sem, enable_judge=enable_judge)
            ))
            run_id += 1

    log.info("mas_runner.start", total_runs=len(tasks), concurrency=concurrency, judge=enable_judge)

    results: list[RunResult] = []
    for i, fut in enumerate(asyncio.as_completed(tasks)):
        result = await fut
        results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
            converged = sum(1 for r in results if r.converged)
            log.info(
                "mas_runner.progress",
                completed=i + 1,
                total=len(tasks),
                convergence_rate=f"{converged / (i + 1):.1%}",
            )

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results: list[RunResult], output_dir: Path) -> Path:
    """Write results.jsonl + summary.json to output_dir. Returns output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results — one JSON object per line
    jsonl_path = output_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")

    # Aggregate summary
    import pandas as pd
    df = pd.DataFrame([asdict(r) for r in results])

    def _agg(subset: pd.DataFrame) -> dict:
        converged_subset = subset[subset["converged"]]
        agg: dict = {
            "n": len(subset),
            "convergence_rate": round(float(subset["converged"].mean()), 4),
            "mean_turns_to_close": (
                round(float(converged_subset["turns"].mean()), 2)
                if not converged_subset.empty else None
            ),
            "clarification_rate": round(float((subset["clarification_count"] >= 1).mean()), 4),
            "forced_forward_rate": round(float(subset["forced_forward"].mean()), 4),
            "spec_accuracy": round(float(subset["spec_match"].mean()), 4),
            "intent_accuracy": round(float(subset["intent_correct"].mean()), 4),
            "error_rate": round(float(subset["error"].notna().mean()), 4),
            "mean_recommendation_len": (
                round(float(converged_subset["recommendation_len"].mean()), 1)
                if not converged_subset.empty else 0
            ),
        }
        # Judge metrics (only include if we have any scores)
        if "judge_composite" in subset.columns:
            judged = subset["judge_composite"].notna()
            if judged.any():
                agg["judge_composite"] = round(float(subset.loc[judged, "judge_composite"].mean()), 4)
                agg["judge_persona"] = round(float(subset.loc[judged, "judge_persona"].mean()), 4)
                agg["judge_accuracy"] = round(float(subset.loc[judged, "judge_accuracy"].mean()), 4)
                agg["judge_safety"] = round(float(subset.loc[judged, "judge_safety"].mean()), 4)
                agg["judge_relevance"] = round(float(subset.loc[judged, "judge_relevance"].mean()), 4)
        if "first_response_latency_s" in subset.columns:
            lat = subset["first_response_latency_s"].dropna()
            if not lat.empty:
                agg["mean_first_response_latency_s"] = round(float(lat.mean()), 3)
                agg["p50_first_response_latency_s"] = round(float(lat.quantile(0.50)), 3)
                agg["p95_first_response_latency_s"] = round(float(lat.quantile(0.95)), 3)
        if "clarification_quality" in subset.columns:
            cq = subset["clarification_quality"].notna()
            if cq.any():
                agg["mean_clarification_quality"] = round(
                    float(subset.loc[cq, "clarification_quality"].mean()), 4
                )
        if "spec_quality" in subset.columns:
            sq = subset["spec_quality"].notna()
            if sq.any():
                agg["mean_spec_quality"] = round(float(subset.loc[sq, "spec_quality"].mean()), 4)
        return agg

    summary: dict = {"overall": _agg(df)}
    for persona_name in df["persona_name"].unique():
        summary[persona_name] = _agg(df[df["persona_name"] == persona_name])

    summary["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_runs": len(results),
        "personas": list(df["persona_name"].unique().tolist()),
        "judge_enabled": any(r.judge_composite is not None for r in results),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("mas_runner.saved", output_dir=str(output_dir))
    return output_dir
