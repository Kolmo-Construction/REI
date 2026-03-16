"""
Multi-Agent Simulation (MAS) Monte Carlo runner.

Runs 500 persona-based buyer↔agent conversations against the Greenvest
graph and generates an interactive HTML report.

Usage:
    uv run python scripts/run_mas.py
    uv run python scripts/run_mas.py --runs 50          # quick smoke test
    uv run python scripts/run_mas.py --runs 500 --concurrency 10
    uv run python scripts/run_mas.py --report-only eval_results/mas_run_<ts>
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Greenvest agent must use real Ollama for meaningful MAS results
os.environ.setdefault("USE_MOCK_LLM", "false")

# Add repo root to path so eval.* imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mas_runner import run_monte_carlo, save_results
from eval.mas_report import generate_report
from eval.personas import PERSONAS


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def _main(args: argparse.Namespace) -> None:
    if args.report_only:
        run_dir = Path(args.report_only)
        if not run_dir.exists():
            print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        report_path = generate_report(run_dir)
        print(f"\nOpen in Chrome: {report_path}")
        return

    run_dir = Path("eval_results") / f"mas_run_{_timestamp()}"

    print(f"Starting MAS with {args.runs} runs, concurrency={args.concurrency}, judge={args.judge}")
    print(f"Personas: {[p.name for p in PERSONAS]}")
    print(f"Output:   {run_dir}\n")

    results = await run_monte_carlo(
        personas=PERSONAS,
        total_runs=args.runs,
        concurrency=args.concurrency,
        enable_judge=args.judge,
    )

    save_results(results, run_dir)

    converged = sum(1 for r in results if r.converged)
    errors = sum(1 for r in results if r.error)
    print(f"\nDone: {converged}/{len(results)} converged ({converged/len(results):.1%})")
    if errors:
        print(f"  {errors} runs errored — check results.jsonl for details")

    report_path = generate_report(run_dir)
    print(f"\nOpen in Chrome: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greenvest MAS Monte Carlo evaluation"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=500,
        help="Total number of Monte Carlo runs (default: 500)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max parallel conversations (default: 5, raise if GPU allows)",
    )
    parser.add_argument(
        "--report-only",
        metavar="RUN_DIR",
        help="Skip simulation, regenerate report from an existing run directory",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Score each converged run with Claude Opus 4.6 (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
