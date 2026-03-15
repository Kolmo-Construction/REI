"""
Compare two evaluation run result files side by side.

Usage:
    uv run python eval/compare.py eval_results/baseline.json eval_results/candidate.json

Exit code 0 if composite improved or stayed the same.
Exit code 1 if composite declined.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DIMENSIONS = ["composite", "persona", "accuracy", "safety", "relevance"]

# ANSI colour codes (disabled on Windows if not supported)
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


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text())


def _fmt(value: float | None, width: int = 8) -> str:
    if value is None:
        return "  N/A   "
    return f"{value:.4f}".rjust(width)


def _delta_str(baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None:
        return "   N/A  "
    delta = candidate - baseline
    if delta > 0.0001:
        return f"{GREEN}+{delta:.4f}{RESET}"
    elif delta < -0.0001:
        return f"{RED}{delta:.4f}{RESET}"
    else:
        return f" {delta:.4f}"


def main():
    if len(sys.argv) != 3:
        print("Usage: compare.py <baseline.json> <candidate.json>", file=sys.stderr)
        sys.exit(2)

    baseline_path = Path(sys.argv[1])
    candidate_path = Path(sys.argv[2])

    baseline = _load(baseline_path)
    candidate = _load(candidate_path)

    print(f"\n{BOLD}Greenvest Eval Comparison{RESET}")
    print(f"  Baseline : {baseline_path}  (git: {baseline.get('git_sha', 'N/A')})")
    print(f"  Candidate: {candidate_path}  (git: {candidate.get('git_sha', 'N/A')})")
    print(f"  Baseline  timestamp: {baseline.get('timestamp', 'N/A')}")
    print(f"  Candidate timestamp: {candidate.get('timestamp', 'N/A')}")
    print()

    col_w = 10
    header = f"{'Dimension':<12} {'Baseline':>{col_w}} {'Candidate':>{col_w}} {'Delta':>{col_w}}"
    print(BOLD + header + RESET)
    print("-" * (12 + col_w * 3 + 6))

    for dim in DIMENSIONS:
        b_val = baseline.get(dim)
        c_val = candidate.get(dim)
        delta_s = _delta_str(b_val, c_val)
        is_composite = dim == "composite"
        label = BOLD + f"{dim:<12}" + RESET if is_composite else f"{dim:<12}"
        print(f"{label} {_fmt(b_val, col_w)} {_fmt(c_val, col_w)} {delta_s}")

    print()
    print(f"  Scenarios — baseline: {baseline.get('scenario_count', 'N/A')} | candidate: {candidate.get('scenario_count', 'N/A')}")
    print(f"  Judged    — baseline: {baseline.get('judged_count', 'N/A')} | candidate: {candidate.get('judged_count', 'N/A')}")

    # Per-scenario comparison
    b_per = {s["scenario_id"]: s for s in baseline.get("per_scenario", [])}
    c_per = {s["scenario_id"]: s for s in candidate.get("per_scenario", [])}
    all_ids = sorted(set(b_per) | set(c_per))

    if all_ids:
        print(f"\n{BOLD}Per-scenario composite:{RESET}")
        print(f"  {'Scenario':<35} {'Baseline':>8} {'Candidate':>9} {'Delta':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*8}")
        for sid in all_ids:
            b_s = b_per.get(sid, {})
            c_s = c_per.get(sid, {})
            b_comp = b_s.get("composite")
            c_comp = c_s.get("composite")
            delta_s = _delta_str(b_comp, c_comp).ljust(8)
            print(f"  {sid:<35} {_fmt(b_comp, 8)} {_fmt(c_comp, 9)} {delta_s}")

    # Decision
    b_composite = baseline.get("composite")
    c_composite = candidate.get("composite")

    print()
    if b_composite is None or c_composite is None:
        print("[INFO] Cannot determine winner — one or both runs have no composite score.")
        sys.exit(0)

    if c_composite >= b_composite:
        print(f"{GREEN}CANDIDATE improved or matched baseline composite.{RESET}")
        print(f"  {b_composite:.4f} → {c_composite:.4f} (Δ {c_composite - b_composite:+.4f})")
        sys.exit(0)
    else:
        print(f"{RED}CANDIDATE declined from baseline composite.{RESET}")
        print(f"  {b_composite:.4f} → {c_composite:.4f} (Δ {c_composite - b_composite:+.4f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
