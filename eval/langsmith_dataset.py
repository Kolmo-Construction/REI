"""
Push scenario JSONs to LangSmith as a dataset.

Usage:
    uv run python eval/langsmith_dataset.py \\
        --dataset greenvest-eval-v1 \\
        --scenarios tests/fixtures/scenarios/

Requires LANGCHAIN_API_KEY environment variable.
Gracefully skips if not set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_scenarios(scenarios_dir: Path) -> list[dict]:
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.json")):
        try:
            scenarios.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Push scenarios to LangSmith as a dataset")
    parser.add_argument("--dataset", required=True, help="LangSmith dataset name")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("tests/fixtures/scenarios/"),
        help="Directory containing scenario JSON files",
    )
    args = parser.parse_args()

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("[WARN] LANGCHAIN_API_KEY not set — skipping LangSmith dataset push.", file=sys.stderr)
        sys.exit(0)

    try:
        from langsmith import Client
    except ImportError:
        print("[WARN] langsmith package not installed — skipping LangSmith dataset push.", file=sys.stderr)
        sys.exit(0)

    if not args.scenarios.exists():
        print(f"ERROR: Scenarios directory not found: {args.scenarios}", file=sys.stderr)
        sys.exit(1)

    scenarios = _load_scenarios(args.scenarios)
    if not scenarios:
        print(f"[WARN] No scenario files found in {args.scenarios}")
        sys.exit(0)

    client = Client(api_key=api_key)
    dataset_name = args.dataset

    # Create or fetch existing dataset
    existing_datasets = {ds.name: ds for ds in client.list_datasets()}
    if dataset_name in existing_datasets:
        dataset = existing_datasets[dataset_name]
        print(f"Updating existing dataset '{dataset_name}' (id={dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Greenvest REI agent evaluation scenarios",
        )
        print(f"Created dataset '{dataset_name}' (id={dataset.id})")

    # Fetch existing example scenario_ids to avoid duplicates
    existing_examples = {
        ex.metadata.get("scenario_id"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
        if ex.metadata.get("scenario_id")
    }

    created = 0
    updated = 0
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
        metadata = {"scenario_id": scenario_id}

        if scenario_id in existing_examples:
            ex = existing_examples[scenario_id]
            client.update_example(
                example_id=ex.id,
                inputs=example_input,
                outputs=example_output,
                metadata=metadata,
            )
            updated += 1
            print(f"  Updated: {scenario_id}")
        else:
            client.create_example(
                inputs=example_input,
                outputs=example_output,
                dataset_id=dataset.id,
                metadata=metadata,
            )
            created += 1
            print(f"  Created: {scenario_id}")

    print(f"\nDone. Created={created} Updated={updated} in dataset '{dataset_name}'")


if __name__ == "__main__":
    main()
