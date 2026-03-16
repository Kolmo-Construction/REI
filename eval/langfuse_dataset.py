"""
Push scenario JSONs to Langfuse as a dataset.

Replaces eval/langsmith_dataset.py — uses Langfuse datasets instead of LangSmith.

Usage:
    uv run python eval/langfuse_dataset.py \\
        --dataset greenvest-eval-v1 \\
        --scenarios tests/fixtures/scenarios/

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.
Gracefully skips if not set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.langfuse_client import langfuse_client


def _load_scenarios(scenarios_dir: Path) -> list[dict]:
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.json")):
        try:
            scenarios.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Push scenarios to Langfuse as a dataset")
    parser.add_argument("--dataset", required=True, help="Langfuse dataset name")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("tests/fixtures/scenarios/"),
        help="Directory containing scenario JSON files",
    )
    args = parser.parse_args()

    lf = langfuse_client()
    if lf is None:
        print(
            "[WARN] Langfuse not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing) "
            "— skipping dataset push.",
            file=sys.stderr,
        )
        sys.exit(0)

    if not args.scenarios.exists():
        print(f"ERROR: Scenarios directory not found: {args.scenarios}", file=sys.stderr)
        sys.exit(1)

    scenarios = _load_scenarios(args.scenarios)
    if not scenarios:
        print(f"[WARN] No scenario files found in {args.scenarios}")
        sys.exit(0)

    # Create dataset if it doesn't exist (Langfuse is idempotent on name)
    try:
        dataset = lf.create_dataset(
            name=args.dataset,
            description="Greenvest REI agent evaluation scenarios",
        )
        print(f"Dataset '{args.dataset}' ready (id={dataset.id})")
    except Exception as exc:
        print(f"ERROR: Failed to create/access dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    created = 0
    for sc in scenarios:
        scenario_id = sc.get("scenario_id", "unknown")
        inp = sc.get("input", {})
        rubric = sc.get("judge_rubric", {})
        gate = sc.get("expected_clarification_gate", {})

        example_input = {
            "query": inp.get("query", ""),
            "session_id": inp.get("session_id", "eval-session"),
            "store_id": inp.get("store_id", "REI-Seattle"),
            "member_number": inp.get("member_number"),
        }
        example_output = {
            "expected_intent": sc.get("expected_intent_router", {}).get("intent"),
            "expected_action_flag": gate.get("action_flag"),
            "expected_recommendation_non_null": rubric.get("recommendation_non_null", False),
            "top_recommended_sku": rubric.get("top_recommended_sku"),
        }

        try:
            lf.create_dataset_item(
                dataset_name=args.dataset,
                input=example_input,
                expected_output=example_output,
                metadata={"scenario_id": scenario_id},
            )
            created += 1
            print(f"  Created: {scenario_id}")
        except Exception as exc:
            print(f"  [WARN] Failed to push {scenario_id}: {exc}", file=sys.stderr)

    print(f"\nDone. Created={created} items in dataset '{args.dataset}'")


if __name__ == "__main__":
    main()
