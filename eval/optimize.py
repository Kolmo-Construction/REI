"""
Autonomous optimization loop for Greenvest agent.
Runs eval, compares against baseline, keeps or reverts changes.

Usage:
    uv run python eval/optimize.py --baseline eval_results/baseline.json
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

EVAL_SCRIPT = Path(__file__).parent / "eval.py"
SCENARIOS_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "scenarios"
EVAL_RESULTS_DIR = Path(__file__).parent.parent / "eval_results"
EXPERIMENTS_LOG = Path(__file__).parent.parent / "experiments" / "log.md"

EDITABLE_FILES = [
    "greenvest/nodes/synthesizer.py",
    "greenvest/nodes/clarification_gate.py",
    "greenvest/nodes/intent_router.py",
    "greenvest/nodes/query_translator.py",
    "greenvest/ontology/gear_ontology.yaml",
]

IMPROVEMENT_THRESHOLD = 0.01
SAFETY_FLOOR = 0.70


def _load_results(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: Results file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def _run_eval(output_path: Path) -> dict:
    """Run eval.py and return parsed results."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--dataset", str(SCENARIOS_DIR),
        "--output", str(output_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[WARN] eval.py exited with code {result.returncode}")
    return _load_results(output_path)


def _git_revert(files: list[str]) -> None:
    """Revert editable files to last committed state."""
    cmd = ["git", "checkout", "--"] + files
    print(f"Reverting: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def _append_log(entry: str) -> None:
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")


def _format_log_row(
    timestamp: str,
    description: str,
    file_hint: str,
    baseline: float | None,
    candidate: float | None,
    decision: str,
) -> str:
    b_str = f"{baseline:.4f}" if baseline is not None else "N/A"
    c_str = f"{candidate:.4f}" if candidate is not None else "N/A"
    if baseline is not None and candidate is not None:
        delta_str = f"{candidate - baseline:+.4f}"
    else:
        delta_str = "N/A"
    return f"| {timestamp} | {description} | {file_hint} | {b_str} | {c_str} | {delta_str} | {decision} |"


def main():
    parser = argparse.ArgumentParser(description="Greenvest optimization loop")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline results JSON. If not provided, runs eval first.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=20,
        help="Maximum number of experiments per session (default 20)",
    )
    args = parser.parse_args()

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate baseline
    if args.baseline and args.baseline.exists():
        baseline_data = _load_results(args.baseline)
        baseline_composite = baseline_data.get("composite")
        print(f"Loaded baseline from {args.baseline}: composite={baseline_composite}")
    else:
        print("No baseline found — running eval to establish baseline...")
        baseline_path = EVAL_RESULTS_DIR / "baseline.json"
        baseline_data = _run_eval(baseline_path)
        baseline_composite = baseline_data.get("composite")
        print(f"Baseline established: composite={baseline_composite}")

    experiment_count = 0

    print(f"\nReady. Edit agent files, then press Enter to evaluate.")
    print(f"Editable files: {', '.join(EDITABLE_FILES)}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"Improvement threshold: {IMPROVEMENT_THRESHOLD}")
    print(f"Safety floor: {SAFETY_FLOOR}")
    print()

    while experiment_count < args.max_experiments:
        try:
            input(f"[Experiment {experiment_count + 1}/{args.max_experiments}] Press Enter to evaluate (Ctrl+C to quit)...")
        except (KeyboardInterrupt, EOFError):
            print("\nOptimization loop ended by user.")
            break

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        candidate_filename = f"candidate_{timestamp.replace(':', '-').replace('T', '_')}.json"
        candidate_path = EVAL_RESULTS_DIR / candidate_filename

        # Get change description from user
        try:
            description = input("Describe your change (press Enter to skip): ").strip() or "undescribed change"
            file_hint = input("Which file(s) did you edit? ").strip() or "unknown"
        except (KeyboardInterrupt, EOFError):
            description = "undescribed change"
            file_hint = "unknown"

        print(f"\nRunning evaluation...")
        candidate_data = _run_eval(candidate_path)
        candidate_composite = candidate_data.get("composite")
        candidate_safety = candidate_data.get("safety")

        print(f"\nBaseline composite:  {baseline_composite}")
        print(f"Candidate composite: {candidate_composite}")

        # Safety floor check
        if candidate_safety is not None and candidate_safety < SAFETY_FLOOR:
            decision = "REVERT (safety below floor)"
            print(f"SAFETY ALERT: safety={candidate_safety:.3f} < floor={SAFETY_FLOOR} — reverting.")
            _git_revert(EDITABLE_FILES)
        elif (
            baseline_composite is not None
            and candidate_composite is not None
            and candidate_composite >= baseline_composite + IMPROVEMENT_THRESHOLD
        ):
            decision = "KEEP"
            print(f"KEEP: composite improved by {candidate_composite - baseline_composite:.4f}")
            baseline_composite = candidate_composite
            baseline_data = candidate_data
        elif (
            baseline_composite is None or candidate_composite is None
        ):
            decision = "KEEP (no scores to compare)"
            print("No composite scores to compare — keeping change.")
        else:
            delta = (candidate_composite or 0) - (baseline_composite or 0)
            decision = f"REVERT (delta={delta:.4f} < threshold={IMPROVEMENT_THRESHOLD})"
            print(f"REVERT: improvement {delta:.4f} did not meet threshold {IMPROVEMENT_THRESHOLD}")
            _git_revert(EDITABLE_FILES)

        log_row = _format_log_row(
            timestamp, description, file_hint, baseline_composite, candidate_composite, decision
        )
        _append_log(log_row)
        print(f"Logged to {EXPERIMENTS_LOG}")

        experiment_count += 1
        print()

    print(f"Optimization session complete. {experiment_count} experiment(s) run.")


if __name__ == "__main__":
    main()
