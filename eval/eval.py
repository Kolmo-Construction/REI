"""
Greenvest evaluation script.

Loads scenario JSONs, runs graph.ainvoke for each scenario that reaches synthesis,
scores the recommendation via the local Ollama judge, logs traces to Langfuse,
and writes a results JSON.

Usage:
    uv run python eval/eval.py \\
        --dataset tests/fixtures/scenarios/ \\
        --output eval_results/run_001.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure mock LLM is on for agent pipeline (judge uses real API if key present)
os.environ.setdefault("USE_MOCK_LLM", "true")

from greenvest.graph import graph
from greenvest.state import initial_state
from eval.judge import judge_recommendation, composite_score
from eval.langfuse_client import langfuse_client


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_scenarios(dataset_dir: Path) -> list[dict]:
    scenarios = []
    for path in sorted(dataset_dir.glob("*.json")):
        try:
            scenarios.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
    return scenarios


def _scenario_reaches_synthesis(sc: dict) -> bool:
    gate = sc.get("expected_clarification_gate", {})
    return gate.get("action_flag") in ("READY_TO_SEARCH", "READY_TO_SYNTHESIZE")


async def _run_scenario(sc: dict) -> dict:
    inp = sc["input"]
    state = initial_state(
        query=inp["query"],
        session_id=inp["session_id"],
        store_id=inp["store_id"],
        member_number=inp.get("member_number"),
    )
    state["clarification_count"] = inp.get("clarification_count", 0)
    for key, val in inp.get("pre_seed_state", {}).items():
        state[key] = val
    if "budget_usd" in inp:
        state["budget_usd"] = tuple(inp["budget_usd"]) if isinstance(inp["budget_usd"], list) else inp["budget_usd"]
    return await graph.ainvoke(state)


async def _process_scenario(sc: dict, lf) -> tuple[dict, float, float, float, float, bool]:
    """Run one scenario through graph + judge.

    Returns
    -------
    (entry, persona, accuracy, safety, relevance, was_judged)
    """
    scenario_id = sc.get("scenario_id", "unknown")

    if not _scenario_reaches_synthesis(sc):
        print(f"  [{scenario_id}] skipped (does not reach synthesis)")
        return {"scenario_id": scenario_id, "skipped": True}, 0.0, 0.0, 0.0, 0.0, False

    print(f"  [{scenario_id}] running...")
    query = sc["input"]["query"]
    ground_truth = sc.get("judge_rubric")

    try:
        result = await _run_scenario(sc)
    except Exception as exc:
        print(f"  [{scenario_id}] FAILED: {exc}", file=sys.stderr)
        return {"scenario_id": scenario_id, "error": str(exc)}, 0.0, 0.0, 0.0, 0.0, False

    recommendation = result.get("recommendation") or ""
    scores = await judge_recommendation(query, recommendation, ground_truth=ground_truth)

    entry: dict = {"scenario_id": scenario_id}

    # Log to Langfuse (v4 API: start_observation returns a span object, not a context manager)
    if lf and scores:
        try:
            span = lf.start_observation(
                name=f"eval_{scenario_id}",
                as_type="evaluator",
                input={"query": query, "scenario_id": scenario_id},
                output={"recommendation": recommendation},
            )
            for score_name, score_val in [
                ("judge_composite", scores.composite),
                ("judge_persona", scores.persona),
                ("judge_accuracy", scores.accuracy),
                ("judge_safety", scores.safety),
                ("judge_relevance", scores.relevance),
            ]:
                try:
                    span.score_trace(name=score_name, value=score_val, data_type="NUMERIC")
                except Exception as exc:
                    print(f"  [WARN] Langfuse score failed ({score_name}): {exc}", file=sys.stderr)
            span.end()
        except Exception as exc:
            print(f"  [WARN] Langfuse trace failed: {exc}", file=sys.stderr)

    if scores:
        comp = scores.composite
        entry.update({
            "persona": scores.persona,
            "accuracy": scores.accuracy,
            "safety": scores.safety,
            "relevance": scores.relevance,
            "composite": comp,
            "reasoning": scores.reasoning,
        })
        print(
            f"  [{scenario_id}] composite={comp:.3f} "
            f"persona={scores.persona:.2f} accuracy={scores.accuracy:.2f} "
            f"safety={scores.safety:.2f} relevance={scores.relevance:.2f}"
        )
        return entry, scores.persona, scores.accuracy, scores.safety, scores.relevance, True
    else:
        entry.update({
            "persona": None, "accuracy": None, "safety": None,
            "relevance": None, "composite": None,
        })
        print(f"  [{scenario_id}] no judge scores (API key missing or call failed)")
        return entry, 0.0, 0.0, 0.0, 0.0, False


async def run_eval(dataset_dir: Path, output_path: Path) -> dict:
    scenarios = _load_scenarios(dataset_dir)
    print(f"Loaded {len(scenarios)} scenarios from {dataset_dir}")

    lf = langfuse_client()

    # Run all scenarios concurrently
    scenario_results = await asyncio.gather(
        *[_process_scenario(sc, lf) for sc in scenarios],
        return_exceptions=False,
    )

    per_scenario: list[dict] = []
    total_persona = total_accuracy = total_safety = total_relevance = 0.0
    judged_count = 0

    for entry, persona, accuracy, safety, relevance, was_judged in scenario_results:
        per_scenario.append(entry)
        if was_judged:
            total_persona += persona
            total_accuracy += accuracy
            total_safety += safety
            total_relevance += relevance
            judged_count += 1

    # Aggregate
    if judged_count > 0:
        avg_persona = total_persona / judged_count
        avg_accuracy = total_accuracy / judged_count
        avg_safety = total_safety / judged_count
        avg_relevance = total_relevance / judged_count
        avg_composite = composite_score({
            "persona": avg_persona,
            "accuracy": avg_accuracy,
            "safety": avg_safety,
            "relevance": avg_relevance,
        })
    else:
        avg_persona = avg_accuracy = avg_safety = avg_relevance = avg_composite = None

    output = {
        "composite": avg_composite,
        "persona": avg_persona,
        "accuracy": avg_accuracy,
        "safety": avg_safety,
        "relevance": avg_relevance,
        "scenario_count": len(per_scenario),
        "judged_count": judged_count,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": _get_git_sha(),
        "per_scenario": per_scenario,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}")
    if avg_composite is not None:
        print(f"Overall composite: {avg_composite:.3f}")
    else:
        print("Overall composite: N/A (no judge scores)")

    if lf:
        lf.flush()

    return output


def main():
    parser = argparse.ArgumentParser(description="Greenvest evaluation script")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/scenarios/"),
        help="Directory containing scenario JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/run.json"),
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: Dataset directory not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_eval(args.dataset, args.output))


if __name__ == "__main__":
    main()
