"""
Autonomous meta-optimization loop for the Greenvest agent.

Replaces the human-in-the-loop `eval/optimize.py` with a fully automated
LLM Compiler pipeline:

    Evaluate → Critic (textual gradients) → Optimizer (code edits)
    → Re-evaluate → Gate (safety + improvement) → Keep / Revert → Log
    → repeat

Objective Function
------------------
  Reward   R  = composite_score(accuracy, safety, persona, relevance)
                weighted: accuracy×0.30, safety×0.30, persona×0.25, relevance×0.15
  Constraint:  safety ≥ 0.70  (violation → R = −∞, immediate revert)
              safety = None   (judge failure → treated as floor violation, revert)
  Keep if:     R_candidate ≥ R_baseline + 0.01
              No-score result (judge down) → revert, never silently accept

Usage
-----
    # Establish baseline then run up to 10 autonomous experiments:
    uv run python eval/autonomous_optimize.py --max-experiments 10

    # Resume from a saved baseline:
    uv run python eval/autonomous_optimize.py \\
        --baseline eval_results/baseline.json --max-experiments 20

    # Dry-run: diagnose failures without applying edits:
    uv run python eval/autonomous_optimize.py --dry-run

    # Use mock LLM (for offline testing of the loop itself):
    uv run python eval/autonomous_optimize.py --mock-llm --dry-run
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

REPO_ROOT = Path(__file__).parent.parent
EVAL_SCRIPT = Path(__file__).parent / "eval.py"
SCENARIOS_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"
EXPERIMENTS_LOG = REPO_ROOT / "experiments" / "log.md"
LOOP_STATE_PATH = REPO_ROOT / "experiments" / "loop_state.json"

EDITABLE_FILES = [
    "greenvest/nodes/synthesizer.py",
    "greenvest/nodes/clarification_gate.py",
    "greenvest/nodes/intent_router.py",
    "greenvest/nodes/query_translator.py",
    "greenvest/ontology/gear_ontology.yaml",
]

IMPROVEMENT_THRESHOLD = 0.01
SAFETY_FLOOR = 0.70

# Maximum seconds to wait for a single eval.py run before killing it.
EVAL_TIMEOUT_SECONDS = 300

# Backoff for consecutive eval failures: wait 5 s, 10 s, 20 s then abort.
_EVAL_BACKOFF_BASE = 5.0
_MAX_CONSECUTIVE_EVAL_FAILURES = 3

# Gradients with confidence below this are skipped before calling the optimizer.
GRADIENT_CONFIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class EvalError(RuntimeError):
    """Raised when eval.py fails or times out."""


# ---------------------------------------------------------------------------
# Shared utilities (mirror optimize.py helpers for consistency)
# ---------------------------------------------------------------------------

def _load_results(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: Results file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


async def _run_eval(
    output_path: Path,
    use_real_llm: bool = True,
    timeout: float = EVAL_TIMEOUT_SECONDS,
) -> dict:
    """Asynchronously run eval.py and return parsed results.

    Parameters
    ----------
    output_path : where to write the results JSON
    use_real_llm : when True (default), sets USE_MOCK_LLM=false in the
        subprocess env so the agent uses real Ollama inference during eval.
        When False, the agent runs with deterministic mock functions (offline,
        fast — only useful for testing the loop infrastructure itself).
    timeout : seconds before the subprocess is killed and EvalError is raised.

    Raises
    ------
    EvalError : if eval.py times out or exits with a non-zero return code.
    """
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--dataset", str(SCENARIOS_DIR),
        "--output", str(output_path),
    ]
    env = dict(os.environ)
    env["USE_MOCK_LLM"] = "false" if use_real_llm else "true"
    llm_label = "real Ollama" if use_real_llm else "mock LLM"
    print(f"[Eval] Running ({llm_label}): {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise EvalError(
            f"eval.py timed out after {timeout:.0f}s — subprocess killed."
        )

    if proc.returncode != 0:
        raise EvalError(
            f"eval.py exited with non-zero code {proc.returncode}."
        )

    return _load_results(output_path)


def _git_revert(files: list[str]) -> None:
    """Revert editable files to last committed state.

    Raises
    ------
    RuntimeError if git checkout fails — the caller should abort the loop rather
    than continuing with corrupted edits in the working tree.
    """
    cmd = ["git", "checkout", "--"] + files
    print(f"[Revert] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"[FATAL] git revert failed (exit {result.returncode}). "
            "Working tree may have unreverted edits. "
            "Run `git checkout -- .` manually before resuming."
        )


def _record_tried_fixes(
    tried_fixes: set,
    attempted_gradients: list,
) -> None:
    """
    After a revert, mark each attempted gradient as tried so the critic skips
    the same (scenario_id, fix_type, target_node) combination next iteration.
    """
    before = len(tried_fixes)
    for g in attempted_gradients:
        tried_fixes.add((g.scenario_id, g.suggested_fix_type, g.target_node))
    added = len(tried_fixes) - before
    if added:
        print(f"[Loop] Marked {added} fix(es) as tried: {tried_fixes}")


def _git_commit(files: list[str], message: str) -> bool:
    """Stage editable files and create a commit.

    Returns True on success.  On any failure the edits are reverted via
    _git_revert so the working tree is always clean when this returns False.
    """
    add_result = subprocess.run(
        ["git", "add"] + files,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if add_result.returncode != 0:
        print(
            "[WARN] git add failed — reverting edits to keep working tree clean.",
            file=sys.stderr,
        )
        _git_revert(files)
        return False
    commit_result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if commit_result.returncode != 0:
        print(
            "[WARN] git commit failed — reverting edits to keep working tree clean.",
            file=sys.stderr,
        )
        _git_revert(files)
        return False
    return True


def _append_log(row: str) -> None:
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


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
    delta_str = (
        f"{candidate - baseline:+.4f}"
        if baseline is not None and candidate is not None
        else "N/A"
    )
    return (
        f"| {timestamp} | {description} | {file_hint} | "
        f"{b_str} | {c_str} | {delta_str} | {decision} |"
    )


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    baseline_composite: float | None,
    candidate_data: dict,
) -> tuple[str, str]:
    """
    Apply safety floor and improvement threshold.

    Returns (decision_code, human_readable_reason).
    decision_code: "KEEP" | "REVERT_SAFETY" | "REVERT_THRESHOLD" | "REVERT_NO_SCORES"

    Safety rules (hard constraints, checked first):
    - safety=None  → REVERT_SAFETY  (judge failure; cannot confirm safe)
    - safety<floor → REVERT_SAFETY  (numeric floor violation)

    Score availability:
    - composite=None → REVERT_NO_SCORES  (judge down; never silently accept)

    Improvement:
    - Δcomposite ≥ threshold → KEEP
    - otherwise             → REVERT_THRESHOLD
    """
    candidate_composite = candidate_data.get("composite")
    candidate_safety = candidate_data.get("safety")

    # Safety floor — hard constraint.
    # None is treated as a floor violation: a missing safety score means the
    # judge could not evaluate the response, so we cannot confirm it is safe.
    if candidate_safety is None:
        return (
            "REVERT_SAFETY",
            "safety=None (judge did not return a safety score — cannot confirm safe)",
        )
    if candidate_safety < SAFETY_FLOOR:
        return (
            "REVERT_SAFETY",
            f"safety={candidate_safety:.3f} < floor={SAFETY_FLOOR}",
        )

    # No composite to compare — judge was down or all scenarios skipped/errored.
    # Reverting instead of blindly accepting prevents silent quality regressions.
    if baseline_composite is None or candidate_composite is None:
        return (
            "REVERT_NO_SCORES",
            "composite=None (judge did not return scores — cannot confirm improvement)",
        )

    # Improvement threshold
    delta = candidate_composite - baseline_composite
    if delta >= IMPROVEMENT_THRESHOLD:
        return "KEEP", f"Δcomposite={delta:+.4f} ≥ threshold={IMPROVEMENT_THRESHOLD}"

    return (
        "REVERT_THRESHOLD",
        f"Δcomposite={delta:+.4f} < threshold={IMPROVEMENT_THRESHOLD}",
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_autonomous_loop(
    baseline_path: Path | None,
    max_experiments: int,
    dry_run: bool,
    max_edits_per_iter: int,
    target_composite: float,
    use_real_llm: bool = True,
    resume: bool = False,
) -> None:
    """
    Run the autonomous optimization loop.

    Parameters
    ----------
    baseline_path : path to an existing baseline JSON, or None to run eval first
    max_experiments : maximum number of Critic→Optimizer→Eval iterations
    dry_run : if True, diagnose failures without applying any edits
    max_edits_per_iter : number of edit plans to apply per iteration (default 1)
    target_composite : stop early if baseline reaches this composite score
    use_real_llm : when True (default), eval runs with USE_MOCK_LLM=false so the
        agent uses real Ollama inference — required for meaningful optimization.
        Set False only for offline testing of the loop infrastructure itself.
    resume : when True, restore experiment_count, baseline, and tried_fixes from
        the last saved loop_state.json before continuing. Overrides baseline_path.
    """
    # Lazy imports here so the module-level imports above stay fast
    from eval.critic import analyze_failures
    from eval.optimizer_agent import generate_and_apply_all

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Attempt resume from persisted state ──────────────────────────────
    experiment_count = 0
    tried_fixes: set[tuple[str, str, str]] = set()
    current_baseline_path: str = ""

    if resume:
        state = _load_state()
        if state:
            try:
                _validate_state(state)
                current_baseline_path = state["baseline_path"]
                baseline_path_obj = Path(current_baseline_path)
                if not baseline_path_obj.exists():
                    raise ValueError(
                        f"baseline_path '{current_baseline_path}' in loop_state.json "
                        "does not exist on disk."
                    )
                baseline_data = json.loads(baseline_path_obj.read_text())
                baseline_composite = state["baseline_composite"]
                experiment_count = int(state["experiment_count"])
                tried_fixes = {tuple(t) for t in state.get("tried_fixes", [])}  # type: ignore[misc]
                print(
                    f"[Resume] Restored state from {LOOP_STATE_PATH}: "
                    f"experiment={experiment_count}, "
                    f"composite={baseline_composite}, "
                    f"tried_fixes={len(tried_fixes)}"
                )
            except (KeyError, TypeError, ValueError) as exc:
                print(
                    f"[WARN] Loop state is invalid and will be ignored: {exc} "
                    "Starting fresh.",
                    file=sys.stderr,
                )
                resume = False
        else:
            print(
                "[WARN] --resume specified but no state file found; starting fresh.",
                file=sys.stderr,
            )
            resume = False  # fall through to normal baseline init

    # ── Establish baseline (skipped when resuming successfully) ───────────
    if not resume:
        if baseline_path and baseline_path.exists():
            baseline_data = _load_results(baseline_path)
            baseline_composite = baseline_data.get("composite")
            current_baseline_path = str(baseline_path)
            print(
                f"[Init] Loaded baseline from {baseline_path}: "
                f"composite={baseline_composite}"
            )
        else:
            ts = _timestamp()
            bp = EVAL_RESULTS_DIR / f"baseline_{ts}.json"
            print("[Init] No baseline — running eval to establish baseline...")
            try:
                baseline_data = await _run_eval(bp, use_real_llm=use_real_llm)
            except EvalError as exc:
                print(f"[FATAL] Baseline eval failed: {exc}", file=sys.stderr)
                sys.exit(1)
            baseline_composite = baseline_data.get("composite")
            current_baseline_path = str(bp)
            print(f"[Init] Baseline established: composite={baseline_composite}")

    if dry_run:
        print("\n[DRY-RUN] Diagnosing failures only (no edits will be applied).\n")

    # ── Print session header ──────────────────────────────────────────────
    llm_mode = "real Ollama (USE_MOCK_LLM=false)" if use_real_llm else "mock LLM (USE_MOCK_LLM=true)"
    print(
        f"\n{'='*60}\n"
        f"  Autonomous Optimization Loop\n"
        f"  Baseline composite : {baseline_composite}\n"
        f"  Target composite   : {target_composite}\n"
        f"  Max experiments    : {max_experiments}\n"
        f"  Safety floor       : {SAFETY_FLOOR}\n"
        f"  Improvement thresh : {IMPROVEMENT_THRESHOLD}\n"
        f"  Agent LLM mode     : {llm_mode}\n"
        f"  Dry-run mode       : {dry_run}\n"
        f"  Resuming from exp  : {experiment_count}\n"
        f"{'='*60}\n"
    )

    consecutive_eval_failures = 0

    while experiment_count < max_experiments:
        # Early-stop if target already met
        if baseline_composite is not None and baseline_composite >= target_composite:
            print(
                f"[Loop] Target composite {target_composite} reached "
                f"(current={baseline_composite:.4f}). Stopping."
            )
            break

        exp_num = experiment_count + 1
        print(f"\n{'-'*60}")
        print(f"[Experiment {exp_num}/{max_experiments}] Diagnosing failures...")

        # ── Critic ───────────────────────────────────────────────────────
        gradients = await analyze_failures(
            baseline_data,
            scenarios_dir=SCENARIOS_DIR,
            tried_fixes=tried_fixes,
        )

        if not gradients:
            print(
                "[Loop] Critic found no actionable failures. "
                "All scenarios meet thresholds - stopping."
            )
            break

        if dry_run:
            print(f"\n[DRY-RUN] {len(gradients)} gradient(s) diagnosed:")
            for g in gradients:
                print(
                    f"  * {g.scenario_id}: {g.failure_mode} -> {g.target_node} "
                    f"(confidence={g.confidence:.2f})\n"
                    f"    Diagnosis: {g.diagnosis}\n"
                    f"    Fix: {g.suggested_fix_type} -> {g.suggested_fix}"
                )
            experiment_count += 1
            continue

        # ── Confidence filter ─────────────────────────────────────────────
        high_confidence = [g for g in gradients if g.confidence >= GRADIENT_CONFIDENCE_THRESHOLD]
        if not high_confidence:
            print(
                f"[Experiment {exp_num}] All {len(gradients)} gradient(s) are below "
                f"confidence threshold ({GRADIENT_CONFIDENCE_THRESHOLD}) — "
                "skipping optimizer to avoid low-signal edits.",
                file=sys.stderr,
            )
            experiment_count += 1
            _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
            continue
        if len(high_confidence) < len(gradients):
            dropped = len(gradients) - len(high_confidence)
            print(
                f"[Experiment {exp_num}] Filtered {dropped} low-confidence gradient(s) "
                f"(threshold={GRADIENT_CONFIDENCE_THRESHOLD}); "
                f"{len(high_confidence)} remaining."
            )
        gradients = high_confidence

        # ── Optimizer ────────────────────────────────────────────────────
        print(f"[Experiment {exp_num}] Applying up to {max_edits_per_iter} edit(s)...")
        apply_results, attempted_gradients = await generate_and_apply_all(
            gradients, max_edits=max_edits_per_iter
        )

        applied_successfully = [r for r in apply_results if r.success]
        if not applied_successfully:
            print(
                f"[Experiment {exp_num}] Optimizer could not apply any edits — "
                "no changes to evaluate.",
                file=sys.stderr,
            )
            experiment_count += 1
            # Log the failed attempt
            ts = _timestamp()
            _append_log(
                _format_log_row(
                    ts,
                    "optimizer_failed: no edit applied",
                    "N/A",
                    baseline_composite,
                    None,
                    "SKIP",
                )
            )
            _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
            continue

        # Build change description for the log
        descriptions = [r.summary[:80] for r in applied_successfully]
        edited_files = list({r.filepath for r in applied_successfully})
        change_desc = "; ".join(descriptions)

        # ── Re-evaluate ──────────────────────────────────────────────────
        ts = _timestamp()
        candidate_path = EVAL_RESULTS_DIR / f"candidate_{ts}.json"
        print(f"[Experiment {exp_num}] Re-evaluating after edits...")
        try:
            candidate_data = await _run_eval(candidate_path, use_real_llm=use_real_llm)
            consecutive_eval_failures = 0  # reset on success
        except EvalError as exc:
            consecutive_eval_failures += 1
            print(
                f"[ERROR] Candidate eval failed ({consecutive_eval_failures}/"
                f"{_MAX_CONSECUTIVE_EVAL_FAILURES}): {exc} — reverting changes.",
                file=sys.stderr,
            )
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)
            _append_log(
                _format_log_row(
                    ts,
                    f"eval_error: {str(exc)[:80]}",
                    ", ".join(edited_files),
                    baseline_composite,
                    None,
                    "REVERT (eval error)",
                )
            )
            experiment_count += 1
            _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
            if consecutive_eval_failures >= _MAX_CONSECUTIVE_EVAL_FAILURES:
                print(
                    f"[FATAL] {consecutive_eval_failures} consecutive eval failures — "
                    "aborting loop. Check eval.py / Ollama before resuming.",
                    file=sys.stderr,
                )
                break
            backoff = min(_EVAL_BACKOFF_BASE * (2 ** (consecutive_eval_failures - 1)), 60.0)
            print(f"[Loop] Backing off {backoff:.0f}s before next iteration...", file=sys.stderr)
            await asyncio.sleep(backoff)
            continue
        candidate_composite = candidate_data.get("composite")
        candidate_safety = candidate_data.get("safety")

        print(
            f"[Experiment {exp_num}] "
            f"Baseline={baseline_composite}  "
            f"Candidate={candidate_composite}  "
            f"Safety={candidate_safety}"
        )

        # ── Gate ─────────────────────────────────────────────────────────
        decision_code, reason = _evaluate_candidate(baseline_composite, candidate_data)
        prev_baseline = baseline_composite  # capture before possible update

        if decision_code == "KEEP":
            print(f"[Gate] KEEP — {reason}")
            baseline_composite = candidate_composite
            baseline_data = candidate_data
            current_baseline_path = str(candidate_path)
            commit_msg = (
                f"auto: experiment {exp_num} — {change_desc[:100]}\n\n"
                f"composite: {prev_baseline} → {candidate_composite:+.4f}"
            )
            if _git_commit(EDITABLE_FILES, commit_msg):
                print(f"[Git] Committed kept changes for experiment {exp_num}.")
            decision_label = "KEEP"

        elif decision_code == "REVERT_SAFETY":
            decision_label = "REVERT (safety violation)"
            print(f"[Gate] REVERT — SAFETY VIOLATION: {reason}")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)

        elif decision_code == "REVERT_NO_SCORES":
            decision_label = "REVERT (no scores)"
            print(f"[Gate] REVERT — JUDGE FAILURE: {reason}")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)

        else:  # REVERT_THRESHOLD
            decision_label = f"REVERT (Δ<{IMPROVEMENT_THRESHOLD})"
            print(f"[Gate] REVERT — {reason}")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)

        # ── Log ──────────────────────────────────────────────────────────
        log_row = _format_log_row(
            ts,
            change_desc[:120],
            ", ".join(edited_files),
            prev_baseline,
            candidate_composite,
            decision_label,
        )
        _append_log(log_row)
        print(f"[Log] Appended to {EXPERIMENTS_LOG}")

        experiment_count += 1
        _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)

    print(
        f"\n{'='*60}\n"
        f"  Session complete - {experiment_count} experiment(s) run\n"
        f"  Final composite: {baseline_composite}\n"
        f"{'='*60}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_state(
    experiment_count: int,
    baseline_composite: float | None,
    baseline_path: str,
    tried_fixes: set[tuple[str, str, str]],
) -> None:
    """Persist loop state to disk so the session can be resumed after a crash."""
    LOOP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "experiment_count": experiment_count,
        "baseline_composite": baseline_composite,
        "baseline_path": baseline_path,
        # sets are not JSON-serialisable; store as a list of lists
        "tried_fixes": [list(t) for t in tried_fixes],
        "saved_at": _timestamp(),
    }
    # Atomic write: write to a temp file then rename so a crash mid-write
    # never leaves a partially-written (corrupt) state file.
    tmp_path = LOOP_STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(LOOP_STATE_PATH)
    print(f"[State] Saved -> {LOOP_STATE_PATH}")


def _validate_state(state: dict) -> None:
    """Validate that a loaded loop_state dict has the required fields and types.

    Raises ValueError with a descriptive message on the first validation failure
    so the caller can fall through to a fresh start rather than crash on a KeyError.
    """
    required = {
        "baseline_path": str,
        "baseline_composite": (float, int, type(None)),
        "experiment_count": int,
    }
    for field, expected_types in required.items():
        if field not in state:
            raise ValueError(f"Required field '{field}' missing from loop_state.json.")
        val = state[field]
        if not isinstance(val, expected_types):
            raise ValueError(
                f"Field '{field}' has unexpected type {type(val).__name__} "
                f"(expected {expected_types})."
            )
    if not isinstance(state.get("tried_fixes", []), list):
        raise ValueError("'tried_fixes' must be a list.")


def _load_state() -> dict | None:
    """Load persisted loop state. Returns None if the file is absent or unreadable."""
    if not LOOP_STATE_PATH.exists():
        return None
    try:
        return json.loads(LOOP_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not load state file {LOOP_STATE_PATH}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous optimization loop for the Greenvest agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to an existing baseline results JSON. "
             "If omitted, runs eval first to establish baseline.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of Critic→Optimizer→Eval iterations (default: 10).",
    )
    parser.add_argument(
        "--target-composite",
        type=float,
        default=0.90,
        metavar="SCORE",
        help="Stop early when composite score reaches this value (default: 0.90).",
    )
    parser.add_argument(
        "--max-edits-per-iter",
        type=int,
        default=1,
        metavar="N",
        help="Number of edit plans to apply per iteration (default: 1). "
             "Higher values batch more edits before re-evaluating.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Diagnose failures and generate gradients without applying any edits.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            f"Resume from a previous run using the saved loop state at "
            f"{LOOP_STATE_PATH}. "
            "Restores experiment_count, baseline, and tried_fixes so the loop "
            "continues exactly where it left off. Overrides --baseline."
        ),
    )
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--real-llm",
        dest="use_real_llm",
        action="store_true",
        default=True,
        help=(
            "Run eval with USE_MOCK_LLM=false so the agent uses real Ollama inference "
            "(default). Required for meaningful optimization — mock outputs produce "
            "meaningless gradients."
        ),
    )
    llm_group.add_argument(
        "--mock-llm",
        dest="use_real_llm",
        action="store_false",
        help=(
            "Run eval with USE_MOCK_LLM=true (deterministic mock agent). "
            "Only useful for offline testing of the loop infrastructure itself."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        run_autonomous_loop(
            baseline_path=args.baseline,
            max_experiments=args.max_experiments,
            dry_run=args.dry_run,
            max_edits_per_iter=args.max_edits_per_iter,
            target_composite=args.target_composite,
            use_real_llm=args.use_real_llm,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
