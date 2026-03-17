"""
Promote a downvoted chat interaction from user_feedback.jsonl into an eval
scenario fixture, so the autonomous optimizer will target it on the next run.

Usage
-----
    # List all feedback entries (downvotes first)
    uv run python -m eval.promote_feedback --list

    # Promote entry 0 with interactive prompts for rubric fields
    uv run python -m eval.promote_feedback --index 0

    # Promote entry 0 non-interactively (all options on the command line)
    uv run python -m eval.promote_feedback --index 0 \\
        --expected-intent Out_of_Bounds \\
        --expected-flag   READY_TO_SYNTHESIZE \\
        --must-contain    "activity,gear,help" \\
        --must-not-contain "outside,professional" \\
        --expected-refusal false \\
        --id              greeting-oob-001

Writes to tests/fixtures/scenarios/<id>.json and prints the path.
You can edit the file by hand before the next optimizer run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT      = Path(__file__).parent.parent
FEEDBACK_PATH  = REPO_ROOT / "experiments" / "user_feedback.jsonl"
SCENARIOS_DIR  = REPO_ROOT / "tests" / "fixtures" / "scenarios"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        print(f"ERROR: {FEEDBACK_PATH} not found — no feedback recorded yet.", file=sys.stderr)
        sys.exit(1)
    entries: list[dict] = []
    with FEEDBACK_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] user_feedback.jsonl line {i}: {exc}", file=sys.stderr)
    return entries


def _slugify(text: str) -> str:
    """Turn arbitrary text into a lowercase-dashed slug (max 40 chars)."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    s = s.strip("-")[:40].rstrip("-")
    return s or "feedback"


def _next_id(base: str) -> str:
    """Return base-001, base-002 … choosing the first unused number."""
    existing = {p.stem for p in SCENARIOS_DIR.glob("*.json")}
    for n in range(1, 1000):
        candidate = f"{base}-{n:03d}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("Could not find an unused scenario ID.")


def _ask(prompt: str, default: str) -> str:
    """Interactive prompt with a default value."""
    shown = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{shown}: ").strip()
    except (EOFError, KeyboardInterrupt):
        val = ""
    return val if val else default


def _parse_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def cmd_list(entries: list[dict]) -> None:
    if not entries:
        print("user_feedback.jsonl is empty.")
        return

    col = "{:<4} {:<6} {:<22} {:<18} {:<30}"
    print(f"\n{FEEDBACK_PATH}\n")
    print(col.format("Idx", "Vote", "Intent", "Flag", "Query (truncated)"))
    print("-" * 84)
    # downvotes first, then upvotes
    for i, e in sorted(enumerate(entries), key=lambda x: (x[1].get("vote") != "down", x[0])):
        vote_str = "[-] down" if e.get("vote") == "down" else "[+] up  "
        print(col.format(
            str(i),
            vote_str,
            (e.get("intent") or "—")[:21],
            (e.get("action_flag") or "—")[:17],
            (e.get("query") or "")[:29],
        ))
    print()


# ---------------------------------------------------------------------------
# --index
# ---------------------------------------------------------------------------

def cmd_promote(
    entry: dict,
    *,
    expected_intent:     str | None,
    expected_flag:       str | None,
    must_contain:        list[str],
    must_not_contain:    list[str],
    expected_refusal:    bool | None,
    scenario_id:         str | None,
    interactive:         bool,
) -> None:

    print(f"\nPromoting feedback entry:")
    print(f"  query      : {entry.get('query')}")
    print(f"  vote       : {entry.get('vote')}")
    print(f"  intent     : {entry.get('intent')}")
    print(f"  action_flag: {entry.get('action_flag')}")
    print(f"  response   : {(entry.get('response') or '')[:120]}")
    print()

    # ── Fill in missing fields ──────────────────────────────────────────────
    if interactive:
        if expected_intent is None:
            expected_intent = _ask(
                "Expected intent the router should return",
                entry.get("intent") or "Out_of_Bounds",
            )
        if expected_flag is None:
            expected_flag = _ask(
                "Expected action_flag from the graph",
                entry.get("action_flag") or "READY_TO_SYNTHESIZE",
            )
        if not must_contain:
            raw = _ask("Response MUST contain (comma-separated keywords, or blank)", "")
            must_contain = _parse_keywords(raw)
        if not must_not_contain:
            raw = _ask("Response must NOT contain (comma-separated keywords, or blank)", "")
            must_not_contain = _parse_keywords(raw)
        if expected_refusal is None:
            ans = _ask("Is a refusal expected? (yes/no)", "no").lower()
            expected_refusal = ans in ("yes", "y", "true", "1")
        if scenario_id is None:
            default_id = _next_id(_slugify(entry.get("query") or "feedback"))
            scenario_id = _ask("Scenario ID", default_id)

    # ── Apply defaults for non-interactive / remaining blanks ───────────────
    if expected_intent is None:
        expected_intent = entry.get("intent") or "Out_of_Bounds"
    if expected_flag is None:
        expected_flag = entry.get("action_flag") or "READY_TO_SYNTHESIZE"
    if expected_refusal is None:
        expected_refusal = False
    if scenario_id is None:
        scenario_id = _next_id(_slugify(entry.get("query") or "feedback"))

    # ── Build scenario dict ─────────────────────────────────────────────────
    is_clarification = expected_flag == "REQUIRES_CLARIFICATION"
    is_product_search = expected_intent == "Product_Search"

    scenario: dict = {
        "scenario_id": scenario_id,
        "description": (
            f"Promoted from user feedback (vote={entry.get('vote')}) — "
            f"query: \"{entry.get('query')}\""
        ),
        "input": {
            "query":              entry.get("query", ""),
            "session_id":         f"test-{scenario_id}",
            "store_id":           entry.get("store_id", "REI-Seattle"),
            "member_number":      None,
            "clarification_count": 0,
        },
        "expected_intent_router": {
            "intent":           expected_intent,
            "activity":         entry.get("activity"),
            "user_environment": None,
        },
        "expected_clarification_gate": {
            "action_flag":                expected_flag,
            "clarification_count":        1 if is_clarification else 0,
            "clarification_message_non_null": is_clarification,
        },
        "catalog_assertions": {},
        "safety": {
            "requires_safety_disclaimer": False,
            "expected_refusal":           expected_refusal,
            "refusal_keywords":           must_not_contain if expected_refusal else [],
        },
        "judge_rubric": {
            "recommendation_non_null":    not is_clarification,
            "top_recommended_sku":        None,
            **({"response_must_contain_any": must_contain} if must_contain else {}),
            **({"response_must_not_contain": must_not_contain} if must_not_contain else {}),
        },
    }

    # Add promoted_from provenance
    scenario["promoted_from"] = {
        "ts":         entry.get("ts"),
        "session_id": entry.get("session_id"),
        "vote":       entry.get("vote"),
    }

    # ── Write ───────────────────────────────────────────────────────────────
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCENARIOS_DIR / f"{scenario_id}.json"

    if out_path.exists():
        print(f"WARNING: {out_path} already exists — overwriting.", file=sys.stderr)

    out_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"Written: {out_path}")
    print(
        f"\nNext steps:\n"
        f"  1. Review and edit {out_path.name} (especially judge_rubric)\n"
        f"  2. Run eval to confirm the scenario scores low:\n"
        f"       uv run python -m eval.eval --dataset tests/fixtures/scenarios --output eval_results/check.json\n"
        f"       uv run python -m eval.inspect eval_results/check.json\n"
        f"  3. Run the optimizer — it will now target this scenario:\n"
        f"       uv run python -m eval.autonomous_optimize\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a downvoted feedback entry into an eval scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list",  action="store_true", help="List all feedback entries.")
    parser.add_argument("--index", type=int, default=None, metavar="N",
                        help="Index of the feedback entry to promote (from --list).")
    parser.add_argument("--expected-intent",    default=None, metavar="TEXT")
    parser.add_argument("--expected-flag",      default=None, metavar="TEXT")
    parser.add_argument("--must-contain",       default="",   metavar="TEXT",
                        help="Comma-separated keywords the response MUST include.")
    parser.add_argument("--must-not-contain",   default="",   metavar="TEXT",
                        help="Comma-separated keywords the response must NOT include.")
    parser.add_argument("--expected-refusal",   default=None, metavar="BOOL",
                        help="'true' or 'false' — whether a refusal is expected.")
    parser.add_argument("--id",                 default=None, metavar="TEXT",
                        help="Scenario ID (default: auto-derived from query).")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip prompts; use CLI values and defaults only.")
    args = parser.parse_args()

    entries = _load_feedback()

    if args.list or args.index is None:
        cmd_list(entries)
        if args.index is None:
            sys.exit(0)

    if args.index < 0 or args.index >= len(entries):
        print(f"ERROR: index {args.index} out of range (0–{len(entries) - 1})", file=sys.stderr)
        sys.exit(1)

    expected_refusal: bool | None = None
    if args.expected_refusal is not None:
        expected_refusal = args.expected_refusal.lower() in ("true", "yes", "1")

    cmd_promote(
        entries[args.index],
        expected_intent=args.expected_intent,
        expected_flag=args.expected_flag,
        must_contain=_parse_keywords(args.must_contain),
        must_not_contain=_parse_keywords(args.must_not_contain),
        expected_refusal=expected_refusal,
        scenario_id=args.id,
        interactive=not args.no_interactive,
    )


if __name__ == "__main__":
    main()
