"""
Inspect an eval results file and classify per-scenario outcomes.

Surfaces signals that composite averages hide: hard agent failures, per-scenario
improvements that were masked by another scenario collapsing, and misleading
composite deltas driven by a single flaky scenario.

Usage
-----
    # Single file — classify scenarios by score band
    uv run python -m eval.inspect eval_results/candidate.json

    # Two files — show per-scenario deltas with anomaly labels
    uv run python -m eval.inspect eval_results/candidate.json \\
        --baseline eval_results/baseline.json

    # Look up a specific experiment from history.jsonl
    uv run python -m eval.inspect --exp 5

    # Dump all history entries as a table
    uv run python -m eval.inspect --history
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
HISTORY_PATH = REPO_ROOT / "experiments" / "history.jsonl"

# Mirrors autonomous_optimize._HARD_FAILURE_KEYWORDS / _get_hard_failure_scenarios
_HARD_FAILURE_KEYWORDS = ("no response", "no recommendation", "did not provide")
_HARD_ERROR_MAX_SCORE = 0.35

# Delta thresholds for labelling
_IMPROVEMENT_MIN_DELTA = 0.02
_REGRESSION_MAX_DELTA = -0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_hard_error(sc: dict) -> bool:
    composite = sc.get("composite")
    if composite is None or composite > _HARD_ERROR_MAX_SCORE:
        return False
    reasoning = (sc.get("reasoning") or "").lower()
    return any(kw in reasoning for kw in _HARD_FAILURE_KEYWORDS)


def _score_band(composite: float | None) -> str:
    if composite is None:
        return "N/A "
    if composite >= 0.90:
        return "HIGH"
    if composite >= 0.80:
        return "MED "
    return "LOW "


# ---------------------------------------------------------------------------
# Single-file inspection
# ---------------------------------------------------------------------------

def inspect_single(data: dict, path: Path) -> None:
    """Print classified scenario breakdown for one eval result (no baseline)."""
    composite = data.get("composite")
    judged = data.get("judged_count", 0)
    total = data.get("scenario_count", 0)
    git_sha = data.get("git_sha", "unknown")
    ts = data.get("timestamp", "")

    print(f"\n{path.name}")
    print(f"  git_sha={git_sha}  ts={ts[:19]}  judged={judged}/{total}  composite={composite}\n")

    per_scenario = data.get("per_scenario", [])

    def sort_key(sc: dict) -> tuple:
        if sc.get("skipped"):
            return (3, 1.0)
        if sc.get("error"):
            return (4, 1.0)
        if _is_hard_error(sc):
            return (0, sc.get("composite") or 0.0)
        c = sc.get("composite") or 1.0
        return (1 if c < 0.80 else (2 if c < 0.90 else 2), 1.0 - c)

    counts: dict[str, int] = {"HARD_ERROR": 0, "LOW": 0, "MED": 0, "HIGH": 0,
                               "SKIPPED": 0, "ERROR": 0}

    for sc in sorted(per_scenario, key=sort_key):
        sid = sc["scenario_id"]
        if sc.get("skipped"):
            print(f"  [SKIPPED]     {sid}")
            counts["SKIPPED"] += 1
            continue
        if sc.get("error"):
            print(f"  [ERROR]       {sid}  {str(sc.get('error', ''))[:60]}")
            counts["ERROR"] += 1
            continue
        c = sc.get("composite")
        if _is_hard_error(sc):
            snippet = (sc.get("reasoning") or "")[:80]
            print(f"  [HARD_ERROR]  {sid:<42}  {c:.3f}  \"{snippet}\"")
            counts["HARD_ERROR"] += 1
        else:
            band = _score_band(c)
            c_str = f"{c:.3f}" if c is not None else " N/A"
            print(f"  [{band}]      {sid:<42}  {c_str}")
            counts[_score_band(c).strip()] += 1

    print(
        f"\n  {counts['HIGH']} HIGH  {counts['MED']} MED  {counts['LOW']} LOW  "
        f"{counts['HARD_ERROR']} hard_error  {counts['SKIPPED']} skipped"
        + (f"  {counts['ERROR']} error" if counts["ERROR"] else "")
    )


# ---------------------------------------------------------------------------
# Two-file comparison
# ---------------------------------------------------------------------------

def inspect_comparison(
    candidate_data: dict,
    baseline_data: dict,
    candidate_path: Path,
    baseline_path: Path,
) -> None:
    """Print per-scenario delta analysis: candidate vs baseline."""
    c_composite = candidate_data.get("composite")
    b_composite = baseline_data.get("composite")
    delta = (
        c_composite - b_composite
        if c_composite is not None and b_composite is not None
        else None
    )

    print(f"\nBaseline : {baseline_path.name}  composite={b_composite}")
    print(f"Candidate: {candidate_path.name}  composite={c_composite}")

    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
    print(f"\nComposite delta: {delta_str}\n")

    base_map = {sc["scenario_id"]: sc for sc in baseline_data.get("per_scenario", [])}
    cand_map = {sc["scenario_id"]: sc for sc in candidate_data.get("per_scenario", [])}
    all_ids = sorted(set(base_map) | set(cand_map))

    # label → (sid, b_val, c_val, delta, note)
    rows: list[tuple[str, str, float | None, float | None, float | None, str]] = []
    anomalies: list[str] = []

    for sid in all_ids:
        b_sc = base_map.get(sid, {})
        c_sc = cand_map.get(sid, {})

        if c_sc.get("skipped") or b_sc.get("skipped"):
            rows.append(("SKIPPED", sid, None, None, None, ""))
            continue
        if c_sc.get("error") or b_sc.get("error"):
            rows.append(("ERROR", sid, None, None, None,
                         str(c_sc.get("error") or b_sc.get("error") or "")[:60]))
            continue

        b_val = b_sc.get("composite")
        c_val = c_sc.get("composite")
        d = (c_val - b_val) if (b_val is not None and c_val is not None) else None

        if _is_hard_error(c_sc):
            snippet = (c_sc.get("reasoning") or "")[:80]
            rows.append(("HARD_ERROR", sid, b_val, c_val, d, f'"{snippet}"'))
            anomalies.append(f"HARD_ERROR:{sid}")
        elif d is not None and d >= _IMPROVEMENT_MIN_DELTA:
            rows.append(("IMPROVEMENT", sid, b_val, c_val, d, ""))
        elif d is not None and d <= _REGRESSION_MAX_DELTA:
            rows.append(("REGRESSION", sid, b_val, c_val, d, ""))
            anomalies.append(f"REGRESSION:{sid}")
        else:
            rows.append(("STABLE", sid, b_val, c_val, d, ""))

    _ORDER = {"HARD_ERROR": 0, "REGRESSION": 1, "IMPROVEMENT": 2,
               "STABLE": 3, "SKIPPED": 4, "ERROR": 5}
    counts: dict[str, int] = {k: 0 for k in _ORDER}

    for label, sid, b_val, c_val, d, note in sorted(rows, key=lambda r: _ORDER.get(r[0], 9)):
        counts[label] = counts.get(label, 0) + 1
        if label in ("SKIPPED", "ERROR"):
            suffix = f"  {note}" if note else ""
            print(f"  [{label:<11}]  {sid}{suffix}")
            continue
        b_str = f"{b_val:.3f}" if b_val is not None else " N/A "
        c_str = f"{c_val:.3f}" if c_val is not None else " N/A "
        d_str = f"({d:+.3f})" if d is not None else "(  N/A )"
        note_str = f"  {note}" if note else ""
        print(f"  [{label:<11}]  {sid:<42}  {b_str} → {c_str}  {d_str}{note_str}")

    n_hard = counts["HARD_ERROR"]
    n_improve = counts["IMPROVEMENT"]
    n_regress = counts["REGRESSION"]
    n_stable = counts["STABLE"]
    n_skipped = counts["SKIPPED"]

    print(
        f"\n  Summary: {n_improve} improved  {n_stable} stable  {n_regress} regressed  "
        f"{n_hard} hard_error  {n_skipped} skipped"
    )

    # Composite interpretation hint
    if delta is not None:
        if n_hard > 0 and n_improve > 0:
            print(
                f"\n  ↑ Composite {delta_str} is MISLEADING: "
                f"{n_hard} hard failure(s) and {n_improve} improvement(s) cancelled out in the average."
            )
        elif n_hard > 0 and delta < 0:
            print(
                f"\n  ↑ Composite {delta_str} is driven by {n_hard} hard agent failure(s), "
                "not a general regression. Other scenarios may be unchanged or improved."
            )
        elif n_improve > 0 and n_regress == 0 and n_hard == 0:
            print(f"\n  ↑ Clean improvement: {n_improve} scenario(s) better, none worse.")

    if anomalies:
        print(f"\n  Anomalies: {', '.join(anomalies)}")
    print()


# ---------------------------------------------------------------------------
# History inspection
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        print(f"ERROR: {HISTORY_PATH} not found. Run the optimizer loop first.", file=sys.stderr)
        sys.exit(1)
    entries = []
    with HISTORY_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] history.jsonl line {i}: {exc}", file=sys.stderr)
    return entries


def inspect_experiment(exp_num: int) -> None:
    """Print a stored experiment record from history.jsonl."""
    entries = _load_history()
    # Take the last entry matching exp_num (a crashed + resumed session may have duplicates)
    entry = next((e for e in reversed(entries) if e.get("exp") == exp_num), None)
    if entry is None:
        print(f"ERROR: experiment {exp_num} not found in {HISTORY_PATH}", file=sys.stderr)
        sys.exit(1)

    decision = entry.get("decision", "unknown")
    delta = entry.get("delta")
    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
    b = entry.get("baseline_composite")
    c = entry.get("candidate_composite")

    print(f"\nExperiment {exp_num}  [{entry.get('ts', 'unknown')}]")
    print(f"  Decision        : {decision}")
    print(f"  Edit type       : {entry.get('edit_type', 'unknown')}")
    print(f"  Target node     : {entry.get('target_node', 'unknown')}")
    print(f"  Scenario        : {entry.get('scenario_id', 'unknown')}")
    print(f"  Composite       : {b} → {c}  ({delta_str})")

    anomalies = entry.get("anomalies", [])
    if anomalies:
        print(f"  Anomalies       : {', '.join(anomalies)}")

    per_sc = entry.get("per_scenario", {})
    if per_sc:
        print(f"\n  Per-scenario deltas:")
        for sid, d in sorted(per_sc.items()):
            d_str = f"{d:+.3f}" if d is not None else " N/A "
            marker = "  [!hard]" if any(
                a.startswith("HARD_ERROR") and sid in a for a in anomalies
            ) else ("  [+]" if d is not None and d >= _IMPROVEMENT_MIN_DELTA
                    else ("  [-]" if d is not None and d <= _REGRESSION_MAX_DELTA else ""))
            print(f"    {sid:<42}  {d_str}{marker}")
    print()


def inspect_history_table() -> None:
    """Print all history entries as a compact table."""
    entries = _load_history()
    if not entries:
        print("history.jsonl is empty.")
        return

    col = "{:<5} {:<22} {:<18} {:<18} {:<9} {:<9} {:<8} {}"
    print(f"\n{HISTORY_PATH}")
    print(col.format("Exp", "Decision", "Edit type", "Target node",
                     "Baseline", "Candidate", "Delta", "Anomalies"))
    print("-" * 110)
    for e in entries:
        delta = e.get("delta")
        delta_str = f"{delta:+.4f}" if delta is not None else "   N/A"
        b = e.get("baseline_composite")
        c = e.get("candidate_composite")
        anomalies = ",".join(e.get("anomalies", []))
        print(col.format(
            str(e.get("exp", "?")),
            (e.get("decision") or "")[:21],
            (e.get("edit_type") or "")[:17],
            (e.get("target_node") or "")[:17],
            f"{b:.4f}" if b is not None else "  N/A",
            f"{c:.4f}" if c is not None else "  N/A",
            delta_str,
            anomalies[:50],
        ))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect eval results or experiment history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=None,
        metavar="RESULT_JSON",
        help="Eval results JSON file to inspect.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help="Baseline results JSON to compare against.",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        metavar="N",
        help="Look up experiment N from history.jsonl.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print all history.jsonl entries as a table.",
    )
    args = parser.parse_args()

    # Validate: exactly one mode must be active
    modes = sum([args.file is not None, args.exp is not None, args.history])
    if modes == 0:
        parser.print_help()
        sys.exit(1)
    if modes > 1:
        print("ERROR: provide exactly one of: RESULT_JSON, --exp N, --history", file=sys.stderr)
        sys.exit(1)

    if args.history:
        inspect_history_table()
        return

    if args.exp is not None:
        inspect_experiment(args.exp)
        return

    # File mode
    if not args.file.exists():
        print(f"ERROR: {args.file} not found", file=sys.stderr)
        sys.exit(1)
    candidate_data = json.loads(args.file.read_text(encoding="utf-8"))

    if args.baseline:
        if not args.baseline.exists():
            print(f"ERROR: baseline {args.baseline} not found", file=sys.stderr)
            sys.exit(1)
        baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
        inspect_comparison(candidate_data, baseline_data, args.file, args.baseline)
    else:
        inspect_single(candidate_data, args.file)


if __name__ == "__main__":
    main()
