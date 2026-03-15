"""
Evaluation results dashboard.
Reads eval_results/*.json files from the last N days and prints a summary table.

Usage:
    uv run python eval/dashboard.py --days 7
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

EVAL_RESULTS_DIR = Path(__file__).parent.parent / "eval_results"
DIMENSIONS = ["composite", "persona", "accuracy", "safety", "relevance"]

try:
    import ctypes
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    _COLOUR = True
except Exception:
    _COLOUR = sys.platform != "win32"

GREEN = "\033[92m" if _COLOUR else ""
RED = "\033[91m" if _COLOUR else ""
RESET = "\033[0m" if _COLOUR else ""
BOLD = "\033[1m" if _COLOUR else ""


def _trend(prev: float | None, curr: float | None) -> str:
    if prev is None or curr is None:
        return "→"
    delta = curr - prev
    if delta > 0.005:
        return f"{GREEN}↑{RESET}"
    elif delta < -0.005:
        return f"{RED}↓{RESET}"
    return "→"


def _fmt(val: float | None, width: int = 7) -> str:
    if val is None:
        return " N/A   "[:width].ljust(width)
    return f"{val:.4f}".rjust(width)


def _load_results(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[WARN] Could not read {path.name}: {exc}", file=sys.stderr)
        return None


def _parse_timestamp(data: dict) -> datetime | None:
    ts = data.get("timestamp")
    if not ts:
        return None
    try:
        # Handle both "Z" and "+00:00" suffixes
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Greenvest eval results dashboard")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back (default 7)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EVAL_RESULTS_DIR,
        help="Directory containing eval result JSON files",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"[WARN] Results directory not found: {results_dir}")
        print("Run eval/eval.py first to generate results.")
        sys.exit(0)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # Load all result files
    entries = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        data = _load_results(path)
        if data is None:
            continue
        ts = _parse_timestamp(data)
        if ts is None:
            # Include files with no timestamp (might be baseline)
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if ts >= cutoff:
            entries.append((ts, path.name, data))

    if not entries:
        print(f"No evaluation results found in the last {args.days} day(s) in {results_dir}.")
        sys.exit(0)

    # Sort by timestamp ascending
    entries.sort(key=lambda x: x[0])

    print(f"\n{BOLD}Greenvest Evaluation Dashboard — Last {args.days} day(s){RESET}")
    print(f"Results directory: {results_dir}\n")

    # Header
    col_date = 20
    col_file = 35
    col_dim = 8
    col_trend = 3

    header = (
        f"{'Date (UTC)':<{col_date}} "
        f"{'File':<{col_file}} "
        f"{'Composite':>{col_dim}} "
        f"{'Persona':>{col_dim}} "
        f"{'Accuracy':>{col_dim}} "
        f"{'Safety':>{col_dim}} "
        f"{'Relevance':>{col_dim}} "
        f"{'Scen':>5} "
        f"{'T':>{col_trend}}"
    )
    print(BOLD + header + RESET)
    print("-" * len(header.replace("\033[1m", "").replace("\033[0m", "")))

    prev_composite: float | None = None

    for ts, filename, data in entries:
        date_str = ts.strftime("%Y-%m-%d %H:%M")
        composite = data.get("composite")
        persona = data.get("persona")
        accuracy = data.get("accuracy")
        safety = data.get("safety")
        relevance = data.get("relevance")
        scenario_count = data.get("scenario_count", "?")
        trend = _trend(prev_composite, composite)

        # Truncate filename if too long
        fn_display = filename if len(filename) <= col_file else filename[: col_file - 3] + "..."

        # Colour composite based on trend
        comp_str = _fmt(composite, col_dim)
        if composite is not None and prev_composite is not None:
            if composite > prev_composite + 0.005:
                comp_str = GREEN + comp_str + RESET
            elif composite < prev_composite - 0.005:
                comp_str = RED + comp_str + RESET

        # Safety colouring
        safety_str = _fmt(safety, col_dim)
        if safety is not None and safety < 0.70:
            safety_str = RED + safety_str + RESET

        row = (
            f"{date_str:<{col_date}} "
            f"{fn_display:<{col_file}} "
            f"{comp_str} "
            f"{_fmt(persona, col_dim)} "
            f"{_fmt(accuracy, col_dim)} "
            f"{safety_str} "
            f"{_fmt(relevance, col_dim)} "
            f"{str(scenario_count):>5} "
            f"{trend:>{col_trend}}"
        )
        print(row)

        if composite is not None:
            prev_composite = composite

    # Summary stats
    composites = [d.get("composite") for _, _, d in entries if d.get("composite") is not None]
    if composites:
        best = max(composites)
        worst = min(composites)
        latest = composites[-1]
        print()
        print(f"  Latest composite : {latest:.4f}")
        print(f"  Best composite   : {best:.4f}")
        print(f"  Worst composite  : {worst:.4f}")
        if len(composites) > 1:
            trend_overall = composites[-1] - composites[0]
            sign = "+" if trend_overall >= 0 else ""
            colour = GREEN if trend_overall >= 0 else RED
            print(f"  Trend (first→last): {colour}{sign}{trend_overall:.4f}{RESET}")


if __name__ == "__main__":
    main()
