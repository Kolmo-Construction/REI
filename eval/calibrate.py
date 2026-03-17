"""
Judge calibration: run eval N times on HEAD with no changes to measure
per-scenario score variance.

Results are written to experiments/judge_calibration.json and include
per-scenario mean, std, and a reliability flag. The autonomous loop can
read this at startup to identify high-variance scenarios (flaky judge) and
to set a more realistic improvement threshold.

Usage
-----
    # 3 runs (default) — enough to detect high-variance scenarios
    uv run python -m eval.calibrate

    # 5 runs for tighter variance estimates
    uv run python -m eval.calibrate --runs 5

    # Custom output path
    uv run python -m eval.calibrate --runs 3 --output experiments/judge_calibration.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_OUTPUT = EXPERIMENTS_DIR / "judge_calibration.json"

EVAL_TIMEOUT_SECONDS = 600

# Std deviation above which a scenario is flagged as unreliable.
# At 0.10, a scenario that swings between 0.70 and 0.90 on the same code
# will be correctly classified as high-variance.
RELIABILITY_STD_THRESHOLD = 0.10


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


async def _run_eval_once(output_path: Path) -> dict:
    """Run eval.py once and return the parsed results JSON."""
    cmd = [
        sys.executable, "-m", "eval.eval",
        "--dataset", str(SCENARIOS_DIR),
        "--output", str(output_path),
    ]
    env = dict(os.environ)
    env["USE_MOCK_LLM"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"
    # Disable Langfuse to avoid duplicate traces during calibration runs
    env.setdefault("LANGFUSE_PUBLIC_KEY", "")
    env.setdefault("LANGFUSE_SECRET_KEY", "")

    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    try:
        await asyncio.wait_for(proc.wait(), timeout=EVAL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"eval timed out after {EVAL_TIMEOUT_SECONDS}s")
    if proc.returncode != 0:
        raise RuntimeError(f"eval exited with code {proc.returncode}")
    return json.loads(output_path.read_text(encoding="utf-8"))


async def run_calibration(runs: int, output_path: Path) -> None:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    git_sha = _get_git_sha()
    ts_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[Calibrate] git_sha={git_sha}  runs={runs}")
    print(f"[Calibrate] Running {runs} eval pass(es) on unchanged HEAD...")
    print(f"[Calibrate] (Langfuse disabled for calibration runs)")

    run_results: list[dict] = []
    for i in range(runs):
        out = EVAL_RESULTS_DIR / f"calibrate_{ts_start}_run{i + 1}.json"
        print(f"\n[Calibrate] Run {i + 1}/{runs}...")
        try:
            data = await _run_eval_once(out)
            run_results.append(data)
            print(f"[Calibrate] Run {i + 1} composite={data.get('composite')}")
        except RuntimeError as exc:
            print(f"[WARN] Run {i + 1} failed: {exc}", file=sys.stderr)

    if not run_results:
        print("[ERROR] All calibration runs failed — no output written.", file=sys.stderr)
        sys.exit(1)

    # Collect per-scenario composite scores across runs
    scenario_scores: dict[str, list[float]] = {}
    for run in run_results:
        for sc in run.get("per_scenario", []):
            if sc.get("skipped") or sc.get("error"):
                continue
            sid = sc["scenario_id"]
            c = sc.get("composite")
            if c is not None:
                scenario_scores.setdefault(sid, []).append(c)

    per_scenario_stats: dict[str, dict] = {}
    print(f"\n[Calibrate] Per-scenario variance ({len(run_results)} run(s)):\n")
    for sid, scores in sorted(scenario_scores.items()):
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores)   # population std (all runs are the population)
        reliable = std < RELIABILITY_STD_THRESHOLD
        per_scenario_stats[sid] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "reliable": reliable,
            "runs": len(scores),
            "scores": [round(s, 4) for s in scores],
        }
        status = "OK   " if reliable else "FLAKY"
        print(f"  [{status}]  {sid:<40}  mean={mean:.3f}  std={std:.3f}  n={len(scores)}")

    overall_scores = [r.get("composite") for r in run_results if r.get("composite") is not None]
    overall_mean = statistics.mean(overall_scores) if overall_scores else None
    overall_std = statistics.pstdev(overall_scores) if len(overall_scores) > 1 else 0.0

    output = {
        "git_sha": git_sha,
        "calibration_ts": ts_start,
        "runs_requested": runs,
        "runs_completed": len(run_results),
        "overall_mean": round(overall_mean, 4) if overall_mean is not None else None,
        "overall_std": round(overall_std, 4),
        "reliability_std_threshold": RELIABILITY_STD_THRESHOLD,
        "per_scenario": per_scenario_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n[Calibrate] Results written to {output_path}")
    print(
        f"[Calibrate] Overall: mean={overall_mean:.4f}  std={overall_std:.4f}  "
        f"({len(run_results)}/{runs} runs completed)"
    )

    unreliable = [sid for sid, s in per_scenario_stats.items() if not s["reliable"]]
    if unreliable:
        print(
            f"\n[Calibrate] WARNING — {len(unreliable)} unreliable scenario(s) "
            f"(std >= {RELIABILITY_STD_THRESHOLD}):"
        )
        for sid in unreliable:
            s = per_scenario_stats[sid]
            print(f"    {sid}  std={s['std']:.3f}  scores={s['scores']}")
        print(
            "\n  These scenarios have high judge variance. Any composite delta <= "
            f"{max(s['std'] for s in per_scenario_stats.values() if not s['reliable']):.3f} "
            "may be within noise.\n"
            "  Consider excluding them from the improvement threshold calculation or\n"
            "  raising IMPROVEMENT_THRESHOLD before running the autonomous loop."
        )
    else:
        print(
            f"\n[Calibrate] All scenarios reliable (std < {RELIABILITY_STD_THRESHOLD}). "
            "IMPROVEMENT_THRESHOLD=0.01 is meaningful."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge calibration: measure per-scenario score variance on HEAD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of eval passes to run on unchanged HEAD (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output path for calibration JSON (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()
    asyncio.run(run_calibration(args.runs, args.output))


if __name__ == "__main__":
    main()
