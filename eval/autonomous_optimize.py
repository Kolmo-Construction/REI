"""
Autonomous meta-optimization loop for the Greenvest agent.

Replaces the human-in-the-loop `eval/optimize.py` with a fully automated
LLM Compiler pipeline:

    Evaluate -> Critic (textual gradients) -> Optimizer (code edits)
    -> Re-evaluate -> Gate (safety + improvement) -> Keep / Revert -> Log
    -> repeat

Objective Function
------------------
  Reward   R  = composite_score(accuracy, safety, persona, relevance)
                weighted: accuracyx0.30, safetyx0.30, personax0.25, relevancex0.15
  Constraint:  safety >= 0.70  (violation -> R = -inf, immediate revert)
              safety = None   (judge failure -> treated as floor violation, revert)
  Keep if:     R_candidate >= R_baseline + 0.01
              No-score result (judge down) -> revert, never silently accept

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
TRIED_FIXES_PATH = REPO_ROOT / "experiments" / "tried_fixes.json"
HISTORY_PATH = REPO_ROOT / "experiments" / "history.jsonl"
LOOP_STATUS_PATH = REPO_ROOT / "experiments" / "loop_status.json"

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
EVAL_TIMEOUT_SECONDS = 600

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
        fast - only useful for testing the loop infrastructure itself).
    timeout : seconds before the subprocess is killed and EvalError is raised.

    Raises
    ------
    EvalError : if eval.py times out or exits with a non-zero return code.
    """
    # Run as a module (-m eval.eval) so Python resolves the `eval` package
    # correctly regardless of how the parent process was launched.
    # Running eval.py as a script would add eval/ to sys.path, shadowing the
    # package and causing `from eval.judge import ...` to fail.
    cmd = [
        sys.executable, "-m", "eval.eval",
        "--dataset", str(SCENARIOS_DIR),
        "--output", str(output_path),
    ]
    env = dict(os.environ)
    env["USE_MOCK_LLM"] = "false" if use_real_llm else "true"
    env["PYTHONIOENCODING"] = "utf-8"
    llm_label = "real Ollama" if use_real_llm else "mock LLM"
    print(f"[Eval] Running ({llm_label}): {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise EvalError(
            f"eval.py timed out after {timeout:.0f}s - subprocess killed."
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
    RuntimeError if git checkout fails - the caller should abort the loop rather
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
    tried_fixes: dict,
    attempted_gradients: list,
) -> None:
    """
    After a revert, increment the attempt counter for each tried gradient so
    the critic can enforce max_retries_per_fix correctly.
    """
    before = len(tried_fixes)
    for g in attempted_gradients:
        key = (g.scenario_id, g.suggested_fix_type, g.target_node)
        tried_fixes[key] = tried_fixes.get(key, 0) + 1
    new_keys = len(tried_fixes) - before
    if attempted_gradients:
        print(
            f"[Loop] Updated tried_fixes: {new_keys} new key(s), "
            f"{len(attempted_gradients) - new_keys} incremented. "
            f"Total entries: {len(tried_fixes)}"
        )


def _git_create_branch(branch_name: str) -> bool:
    """Create and switch to a new branch from HEAD.

    Returns True on success. Prints a warning and returns False if the branch
    already exists or git fails (the loop continues on the current branch).
    """
    result = subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"[WARN] Could not create branch '{branch_name}': "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    print(f"[Git] Created and switched to branch '{branch_name}'.")
    return True


def _git_commit(files: list[str], message: str, no_commit: bool = False) -> bool:
    """Stage editable files and create a commit.

    When no_commit=True, skips the commit entirely and returns True so the
    caller treats the kept changes as accepted (they remain in the working tree).

    Returns True on success.  On any failure the edits are reverted via
    _git_revert so the working tree is always clean when this returns False.
    """
    if no_commit:
        print("[Git] --no-commit: skipping commit, keeping changes in working tree.")
        return True
    add_result = subprocess.run(
        ["git", "add"] + files,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if add_result.returncode != 0:
        print(
            "[WARN] git add failed - reverting edits to keep working tree clean.",
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
            "[WARN] git commit failed - reverting edits to keep working tree clean.",
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
    baseline_data: dict,
    candidate_data: dict,
) -> tuple[str, str]:
    """
    Apply safety floor, structural regression guard, and improvement threshold.

    Returns (decision_code, human_readable_reason).
    decision_code: "KEEP" | "REVERT_SAFETY" | "REVERT_THRESHOLD" | "REVERT_NO_SCORES"
                 | "REVERT_STRUCTURAL"

    Safety rules (hard constraints, checked first):
    - safety=None  -> REVERT_SAFETY  (judge failure; cannot confirm safe)
    - safety<floor -> REVERT_SAFETY  (numeric floor violation)

    Score availability:
    - composite=None -> REVERT_NO_SCORES  (judge down; never silently accept)

    Structural regression:
    - judged_count dropped by more than 1 -> REVERT_STRUCTURAL

    Improvement:
    - deltacomposite >= threshold -> KEEP
    - otherwise             -> REVERT_THRESHOLD
    """
    baseline_composite = baseline_data.get("composite")
    candidate_composite = candidate_data.get("composite")
    candidate_safety = candidate_data.get("safety")

    # Safety floor - hard constraint.
    # None is treated as a floor violation: a missing safety score means the
    # judge could not evaluate the response, so we cannot confirm it is safe.
    if candidate_safety is None:
        return (
            "REVERT_SAFETY",
            "safety=None (judge did not return a safety score - cannot confirm safe)",
        )
    if candidate_safety < SAFETY_FLOOR:
        return (
            "REVERT_SAFETY",
            f"safety={candidate_safety:.3f} < floor={SAFETY_FLOOR}",
        )

    # Structural regression guard: edit should not cause more scenarios to be skipped
    baseline_judged = baseline_data.get("judged_count", 0)
    candidate_judged = candidate_data.get("judged_count", 0)
    if baseline_judged > 0 and candidate_judged < baseline_judged - 1:
        return (
            "REVERT_STRUCTURAL",
            f"judged_count dropped {baseline_judged} -> {candidate_judged} "
            "(edit may have caused more scenarios to skip — structural regression)",
        )

    # No composite to compare - judge was down or all scenarios skipped/errored.
    # Reverting instead of blindly accepting prevents silent quality regressions.
    if baseline_composite is None or candidate_composite is None:
        return (
            "REVERT_NO_SCORES",
            "composite=None (judge did not return scores - cannot confirm improvement)",
        )

    # Improvement threshold
    delta = candidate_composite - baseline_composite
    if delta >= IMPROVEMENT_THRESHOLD:
        return "KEEP", f"delta_composite={delta:+.4f} >= threshold={IMPROVEMENT_THRESHOLD}"

    return (
        "REVERT_THRESHOLD",
        f"delta_composite={delta:+.4f} < threshold={IMPROVEMENT_THRESHOLD}",
    )


# ---------------------------------------------------------------------------
# Tried-fixes persistence (Gap 2)
# ---------------------------------------------------------------------------

def _load_tried_fixes() -> dict[tuple[str, str, str], int]:
    """Load persistent tried-fixes counter from disk. Returns empty dict if absent.

    Backward-compatible: old 3-element entries [scenario, fix_type, node] are
    treated as count=2 (already at max_retries, so they stay blocked).
    New 4-element entries [scenario, fix_type, node, count] restore exact counts.
    """
    if not TRIED_FIXES_PATH.exists():
        return {}
    try:
        data = json.loads(TRIED_FIXES_PATH.read_text(encoding="utf-8"))
        result: dict[tuple[str, str, str], int] = {}
        for entry in data.get("tried_fixes", []):
            if len(entry) >= 4:
                result[tuple(entry[:3])] = int(entry[3])
            elif len(entry) == 3:
                result[tuple(entry)] = 2  # old format: treat as already maxed out
        return result
    except Exception as exc:
        print(f"[WARN] Could not load tried_fixes from {TRIED_FIXES_PATH}: {exc}", file=sys.stderr)
        return {}


def _save_tried_fixes(tried_fixes: dict[tuple[str, str, str], int]) -> None:
    """Persist tried-fixes counter to disk independent of loop_state.

    Serialises as [scenario_id, fix_type, node, count] so _load_tried_fixes
    can restore exact attempt counts across sessions.
    """
    TRIED_FIXES_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "tried_fixes": [list(k) + [v] for k, v in tried_fixes.items()],
        "saved_at": _timestamp(),
    }
    tmp = TRIED_FIXES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(TRIED_FIXES_PATH)
    print(f"[State] Tried-fixes saved -> {TRIED_FIXES_PATH} ({len(tried_fixes)} entries)")


# ---------------------------------------------------------------------------
# Judge variance helpers (Gap 3)
# ---------------------------------------------------------------------------

def _has_suspect_noise(baseline_data: dict, candidate_data: dict) -> bool:
    """Return True if any per-scenario composite dropped by more than 0.25 (judge variance signal)."""
    base_map = {
        sc["scenario_id"]: sc.get("composite")
        for sc in baseline_data.get("per_scenario", [])
        if not sc.get("skipped") and not sc.get("error") and sc.get("composite") is not None
    }
    cand_map = {
        sc["scenario_id"]: sc.get("composite")
        for sc in candidate_data.get("per_scenario", [])
        if not sc.get("skipped") and not sc.get("error") and sc.get("composite") is not None
    }
    for sid, base_val in base_map.items():
        cand_val = cand_map.get(sid)
        if cand_val is not None and (cand_val - base_val) < -0.25:
            print(
                f"[Gate] SUSPECT_NOISE: {sid} dropped {base_val:.3f} -> {cand_val:.3f} "
                f"(delta={cand_val - base_val:+.3f})"
            )
            return True
    return False


# ---------------------------------------------------------------------------
# Per-scenario delta log (Gap 5)
# ---------------------------------------------------------------------------

def _per_scenario_delta_log(baseline_data: dict, candidate_data: dict) -> str:
    """Return a compact per-scenario delta string for log notes on revert."""
    base_map = {
        sc["scenario_id"]: sc.get("composite")
        for sc in baseline_data.get("per_scenario", [])
        if not sc.get("skipped") and not sc.get("error")
    }
    cand_map = {
        sc["scenario_id"]: sc.get("composite")
        for sc in candidate_data.get("per_scenario", [])
        if not sc.get("skipped") and not sc.get("error")
    }
    parts = []
    for sid in sorted(set(base_map) | set(cand_map)):
        b = base_map.get(sid)
        c = cand_map.get(sid)
        if b is None or c is None:
            parts.append(f"{sid}: N/A")
        else:
            d = c - b
            marker = " [!]" if abs(d) > 0.025 else ""
            parts.append(f"{sid}: {d:+.3f}{marker}")
    return "  " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Per-scenario delta map (shared by delta_log and history)
# ---------------------------------------------------------------------------

def _per_scenario_delta_map(
    baseline_data: dict,
    candidate_data: dict,
) -> dict[str, float | None]:
    """Return {scenario_id: delta} for history records and anomaly computation."""
    base_map = {
        sc["scenario_id"]: sc.get("composite")
        for sc in baseline_data.get("per_scenario", [])
        if not sc.get("skipped") and not sc.get("error")
    }
    cand_map = {
        sc["scenario_id"]: sc.get("composite")
        for sc in candidate_data.get("per_scenario", [])
        if not sc.get("skipped") and not sc.get("error")
    }
    result: dict[str, float | None] = {}
    for sid in sorted(set(base_map) | set(cand_map)):
        b = base_map.get(sid)
        c = cand_map.get(sid)
        result[sid] = (c - b) if (b is not None and c is not None) else None
    return result


# ---------------------------------------------------------------------------
# Hard agent failure detection (Gap B)
# ---------------------------------------------------------------------------

_HARD_FAILURE_KEYWORDS = ("no response", "no recommendation", "did not provide")


def _get_hard_failure_scenarios(candidate_data: dict) -> list[str]:
    """Return scenario IDs where the agent produced no output (hard failure).

    Hard failure criteria: composite <= 0.35 AND judge reasoning explicitly states
    the agent returned nothing. These are deterministic pipeline failures, not judge
    noise — a re-run will produce the same result.
    """
    hard: list[str] = []
    for sc in candidate_data.get("per_scenario", []):
        if sc.get("skipped") or sc.get("error"):
            continue
        composite = sc.get("composite")
        if composite is None or composite > 0.35:
            continue
        reasoning = (sc.get("reasoning") or "").lower()
        if any(kw in reasoning for kw in _HARD_FAILURE_KEYWORDS):
            hard.append(sc["scenario_id"])
    return hard


# ---------------------------------------------------------------------------
# Anomaly classification (for history.jsonl)
# ---------------------------------------------------------------------------

def _compute_anomalies(baseline_data: dict, candidate_data: dict) -> list[str]:
    """Classify notable per-scenario events for the history record.

    HARD_ERROR: agent returned no response (score <= 0.35 + reasoning keywords)
    LARGE_DROP: composite fell > 0.25, not already classified as HARD_ERROR
    """
    hard_ids = set(_get_hard_failure_scenarios(candidate_data))
    anomalies = [f"HARD_ERROR:{sid}" for sid in sorted(hard_ids)]
    delta_map = _per_scenario_delta_map(baseline_data, candidate_data)
    for sid, d in delta_map.items():
        if sid in hard_ids:
            continue
        if d is not None and d < -0.25:
            anomalies.append(f"LARGE_DROP:{sid}")
    return anomalies


# ---------------------------------------------------------------------------
# Experiment history (Gap D)
# ---------------------------------------------------------------------------

def _append_history(entry: dict) -> None:
    """Append one experiment record to experiments/history.jsonl."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Live loop status (Gap E)
# ---------------------------------------------------------------------------

def _write_loop_status(
    phase: str,
    experiment: int,
    max_experiments: int,
    baseline_composite: float | None,
    last_edit: str = "",
    last_decision: str = "",
    session_branch: str | None = None,
    token_budget: int | None = None,
) -> None:
    """Overwrite experiments/loop_status.json with the current phase.

    Written at every phase transition so an operator can run
    `watch -n 2 cat experiments/loop_status.json` in a second terminal
    for live progress without tailing the process stdout.
    Silently swallowed on any I/O error so a write failure never crashes the loop.
    """
    try:
        from eval.token_counter import get_session_totals
        totals = get_session_totals()
        tokens_used = sum(u.total for u in totals.values())
    except Exception:
        tokens_used = 0
    status = {
        "phase": phase,
        "experiment": experiment,
        "max_experiments": max_experiments,
        "baseline_composite": baseline_composite,
        "last_edit": last_edit,
        "last_decision": last_decision,
        "session_branch": session_branch,
        "tokens_used": tokens_used,
        "token_budget": token_budget,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        LOOP_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOOP_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception:
        pass


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
    session_branch: str | None = None,
    no_commit: bool = False,
    reset_tried_fixes: bool = False,
    token_budget: int | None = None,
) -> None:
    """
    Run the autonomous optimization loop.

    Parameters
    ----------
    baseline_path : path to an existing baseline JSON, or None to run eval first
    max_experiments : maximum number of Critic->Optimizer->Eval iterations
    dry_run : if True, diagnose failures without applying any edits
    max_edits_per_iter : number of edit plans to apply per iteration (default 1)
    target_composite : stop early if baseline reaches this composite score
    use_real_llm : when True (default), eval runs with USE_MOCK_LLM=false so the
        agent uses real Ollama inference - required for meaningful optimization.
        Set False only for offline testing of the loop infrastructure itself.
    resume : when True, restore experiment_count, baseline, and tried_fixes from
        the last saved loop_state.json before continuing. Overrides baseline_path.
    session_branch : if set, create and switch to this branch before the first
        commit so experiments never land on main. Ignored when no_commit=True.
    no_commit : when True, accepted edits are left in the working tree but never
        committed. Useful for inspection - run `git diff` after the session.
    reset_tried_fixes : when True, clear the persistent tried-fixes blacklist before
        starting. Use when you want the critic to retry previously blocked fixes.
    token_budget : abort the loop after this many total LLM tokens. No limit if None.
    """
    # Lazy imports here so the module-level imports above stay fast
    from eval.critic import analyze_failures
    from eval.optimizer_agent import generate_and_apply_all

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Gap 2: Always load persistent tried-fixes (survives non-resume restarts)
    if reset_tried_fixes:
        print("[State] --reset-tried-fixes: clearing persistent tried-fixes blacklist.")
        if TRIED_FIXES_PATH.exists():
            TRIED_FIXES_PATH.unlink()
        tried_fixes: dict[tuple[str, str, str], int] = {}
    else:
        tried_fixes = _load_tried_fixes()

    # -- Attempt resume from persisted state ------------------------------
    experiment_count = 0
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
                for entry in state.get("tried_fixes", []):
                    if len(entry) >= 4:
                        key: tuple[str, str, str] = tuple(entry[:3])  # type: ignore[assignment]
                        tried_fixes[key] = max(tried_fixes.get(key, 0), int(entry[3]))
                    elif len(entry) == 3:
                        key = tuple(entry[:3])  # type: ignore[assignment]
                        tried_fixes.setdefault(key, 2)
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

    # -- Establish baseline (skipped when resuming successfully) -----------
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
            print("[Init] No baseline - running eval to establish baseline...")
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

    # -- Branch isolation -------------------------------------------------
    # Create a session branch so experiments never commit directly to main.
    # Skipped when --no-commit is active (nothing is committed anyway).
    active_branch = session_branch
    if session_branch and not no_commit and not dry_run:
        if not _git_create_branch(session_branch):
            # Branch creation failed - fall back to no-commit mode so we
            # never accidentally land changes on whatever branch is current.
            print(
                "[WARN] Falling back to --no-commit mode to avoid committing "
                "to the current branch.",
                file=sys.stderr,
            )
            no_commit = True
            active_branch = None

    # -- Print session header ----------------------------------------------
    llm_mode = "real Ollama (USE_MOCK_LLM=false)" if use_real_llm else "mock LLM (USE_MOCK_LLM=true)"
    commit_mode = (
        "--no-commit (changes left in working tree)"
        if no_commit
        else f"branch '{active_branch}'" if active_branch
        else "main (WARNING: committing directly to main)"
    )
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
        f"  Commit mode        : {commit_mode}\n"
        f"  Resuming from exp  : {experiment_count}\n"
        f"{'='*60}\n"
    )

    consecutive_eval_failures = 0
    last_edit_summary = ""
    last_decision_label = ""

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
        _write_loop_status("critic_running", exp_num, max_experiments, baseline_composite,
                           last_edit_summary, last_decision_label, active_branch, token_budget)

        # -- Critic -------------------------------------------------------
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

        # -- Confidence filter ---------------------------------------------
        high_confidence = [g for g in gradients if g.confidence >= GRADIENT_CONFIDENCE_THRESHOLD]
        if not high_confidence:
            print(
                f"[Experiment {exp_num}] All {len(gradients)} gradient(s) are below "
                f"confidence threshold ({GRADIENT_CONFIDENCE_THRESHOLD}) - "
                "skipping optimizer to avoid low-signal edits.",
                file=sys.stderr,
            )
            experiment_count += 1
            _append_history({
                "ts": _timestamp(),
                "exp": exp_num,
                "decision": "SKIP_LOW_CONFIDENCE",
                "delta": None,
                "baseline_composite": baseline_composite,
                "candidate_composite": None,
                "edit_type": gradients[0].suggested_fix_type if gradients else "unknown",
                "target_node": gradients[0].target_node if gradients else "unknown",
                "scenario_id": gradients[0].scenario_id if gradients else "unknown",
                "per_scenario": {},
                "anomalies": [],
            })
            _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
            _save_tried_fixes(tried_fixes)
            continue
        if len(high_confidence) < len(gradients):
            dropped = len(gradients) - len(high_confidence)
            print(
                f"[Experiment {exp_num}] Filtered {dropped} low-confidence gradient(s) "
                f"(threshold={GRADIENT_CONFIDENCE_THRESHOLD}); "
                f"{len(high_confidence)} remaining."
            )
        gradients = high_confidence

        # -- Optimizer ----------------------------------------------------
        print(f"[Experiment {exp_num}] Applying up to {max_edits_per_iter} edit(s)...")
        _write_loop_status("optimizer_running", exp_num, max_experiments, baseline_composite,
                           last_edit_summary, last_decision_label, active_branch, token_budget)
        apply_results, attempted_gradients = await generate_and_apply_all(
            gradients, max_edits=max_edits_per_iter
        )

        applied_successfully = [r for r in apply_results if r.success]
        last_edit_summary = applied_successfully[0].summary[:100] if applied_successfully else ""
        if not applied_successfully:
            print(
                f"[Experiment {exp_num}] Optimizer could not apply any edits - "
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
            _append_history({
                "ts": ts,
                "exp": exp_num,
                "decision": "SKIP_OPTIMIZER_FAILED",
                "delta": None,
                "baseline_composite": baseline_composite,
                "candidate_composite": None,
                "edit_type": attempted_gradients[0].suggested_fix_type if attempted_gradients else "unknown",
                "target_node": attempted_gradients[0].target_node if attempted_gradients else "unknown",
                "scenario_id": attempted_gradients[0].scenario_id if attempted_gradients else "unknown",
                "per_scenario": {},
                "anomalies": [],
            })
            _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
            _save_tried_fixes(tried_fixes)
            continue

        # Build change description for the log
        descriptions = [r.summary[:80] for r in applied_successfully]
        edited_files = list({r.filepath for r in applied_successfully})
        change_desc = "; ".join(descriptions)

        # -- Re-evaluate --------------------------------------------------
        ts = _timestamp()
        candidate_path = EVAL_RESULTS_DIR / f"candidate_{ts}.json"
        print(f"[Experiment {exp_num}] Re-evaluating after edits...")
        _write_loop_status("eval_running", exp_num, max_experiments, baseline_composite,
                           last_edit_summary, last_decision_label, active_branch, token_budget)
        try:
            candidate_data = await _run_eval(candidate_path, use_real_llm=use_real_llm)
            consecutive_eval_failures = 0  # reset on success
        except EvalError as exc:
            consecutive_eval_failures += 1
            print(
                f"[ERROR] Candidate eval failed ({consecutive_eval_failures}/"
                f"{_MAX_CONSECUTIVE_EVAL_FAILURES}): {exc} - reverting changes.",
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
            _append_history({
                "ts": ts,
                "exp": exp_num,
                "decision": "REVERT_EVAL_ERROR",
                "delta": None,
                "baseline_composite": baseline_composite,
                "candidate_composite": None,
                "edit_type": attempted_gradients[0].suggested_fix_type if attempted_gradients else "unknown",
                "target_node": attempted_gradients[0].target_node if attempted_gradients else "unknown",
                "scenario_id": attempted_gradients[0].scenario_id if attempted_gradients else "unknown",
                "per_scenario": {},
                "anomalies": [],
            })
            experiment_count += 1
            _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
            _save_tried_fixes(tried_fixes)
            if consecutive_eval_failures >= _MAX_CONSECUTIVE_EVAL_FAILURES:
                print(
                    f"[FATAL] {consecutive_eval_failures} consecutive eval failures - "
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

        _write_loop_status("gate_deciding", exp_num, max_experiments, baseline_composite,
                           last_edit_summary, last_decision_label, active_branch, token_budget)

        # Gap B: Hard agent failure check — score <= 0.35 + "no response" in
        # reasoning means the pipeline returned nothing. This is deterministic;
        # a noise re-run will produce the same result. Revert immediately.
        hard_failure_ids = _get_hard_failure_scenarios(candidate_data)
        if hard_failure_ids:
            print(
                f"[Gate] HARD_ERROR in {hard_failure_ids} "
                "(agent returned no response) — skipping noise re-run."
            )
            decision_code = "REVERT_HARD_ERROR"
            reason = f"hard agent failure in {hard_failure_ids}"
        else:
            # Gap 3: Judge variance guard — large per-scenario drops may be noise
            if _has_suspect_noise(baseline_data, candidate_data):
                print(
                    "[Gate] SUSPECT_NOISE detected — re-running eval once for confirmation..."
                )
                ts_confirm = _timestamp()
                candidate_path_confirm = EVAL_RESULTS_DIR / f"candidate_{ts_confirm}_confirm.json"
                try:
                    candidate_data = await _run_eval(
                        candidate_path_confirm, use_real_llm=use_real_llm
                    )
                    candidate_path = candidate_path_confirm
                    print("[Gate] Confirmation eval complete. Using confirmed scores.")
                except EvalError as exc:
                    print(
                        f"[WARN] Confirmation eval failed: {exc} — using original scores.",
                        file=sys.stderr,
                    )

            # -- Gate ---------------------------------------------------------
            decision_code, reason = _evaluate_candidate(baseline_data, candidate_data)

        prev_baseline = baseline_composite  # capture before possible update

        if decision_code == "KEEP":
            print(f"[Gate] KEEP - {reason}")
            baseline_composite = candidate_composite
            baseline_data = candidate_data
            current_baseline_path = str(candidate_path)
            commit_msg = (
                f"auto: experiment {exp_num} - {change_desc[:100]}\n\n"
                f"composite: {prev_baseline} -> {candidate_composite:+.4f}"
            )
            if _git_commit(EDITABLE_FILES, commit_msg, no_commit=no_commit):
                if no_commit:
                    print(f"[Git] Changes kept in working tree (--no-commit).")
                else:
                    print(f"[Git] Committed kept changes for experiment {exp_num}.")
            decision_label = "KEEP"

        elif decision_code == "REVERT_SAFETY":
            decision_label = "REVERT (safety violation)"
            print(f"[Gate] REVERT - SAFETY VIOLATION: {reason}")
            delta_detail = _per_scenario_delta_log(baseline_data, candidate_data)
            if delta_detail.strip():
                _append_log(f"  <!-- per-scenario: {delta_detail} -->")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)
            _save_tried_fixes(tried_fixes)

        elif decision_code == "REVERT_NO_SCORES":
            decision_label = "REVERT (no scores)"
            print(f"[Gate] REVERT - JUDGE FAILURE: {reason}")
            delta_detail = _per_scenario_delta_log(baseline_data, candidate_data)
            if delta_detail.strip():
                _append_log(f"  <!-- per-scenario: {delta_detail} -->")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)
            _save_tried_fixes(tried_fixes)

        elif decision_code == "REVERT_STRUCTURAL":
            decision_label = "REVERT (structural regression)"
            print(f"[Gate] REVERT - STRUCTURAL REGRESSION: {reason}")
            delta_detail = _per_scenario_delta_log(baseline_data, candidate_data)
            if delta_detail.strip():
                _append_log(f"  <!-- per-scenario: {delta_detail} -->")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)
            _save_tried_fixes(tried_fixes)

        elif decision_code == "REVERT_HARD_ERROR":
            decision_label = "REVERT (hard agent failure)"
            print(f"[Gate] REVERT - HARD AGENT FAILURE: {reason}")
            delta_detail = _per_scenario_delta_log(baseline_data, candidate_data)
            if delta_detail.strip():
                _append_log(f"  <!-- per-scenario: {delta_detail} -->")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)
            _save_tried_fixes(tried_fixes)

        else:  # REVERT_THRESHOLD
            decision_label = f"REVERT (delta<{IMPROVEMENT_THRESHOLD})"
            print(f"[Gate] REVERT - {reason}")
            delta_detail = _per_scenario_delta_log(baseline_data, candidate_data)
            if delta_detail.strip():
                _append_log(f"  <!-- per-scenario: {delta_detail} -->")
            _git_revert(EDITABLE_FILES)
            _record_tried_fixes(tried_fixes, attempted_gradients)
            _save_tried_fixes(tried_fixes)

        last_decision_label = decision_label

        # -- Log ----------------------------------------------------------
        log_row = _format_log_row(
            ts,
            change_desc[:120],
            ", ".join(edited_files),
            prev_baseline,
            candidate_composite,
            decision_label,
        )
        _append_log(log_row)
        _append_history({
            "ts": ts,
            "exp": exp_num,
            "decision": decision_code,
            "delta": (
                round(candidate_composite - prev_baseline, 6)
                if candidate_composite is not None and prev_baseline is not None
                else None
            ),
            "baseline_composite": prev_baseline,
            "candidate_composite": candidate_composite,
            "edit_type": applied_successfully[0].edit_type if applied_successfully else "unknown",
            "target_node": attempted_gradients[0].target_node if attempted_gradients else "unknown",
            "scenario_id": attempted_gradients[0].scenario_id if attempted_gradients else "unknown",
            "per_scenario": _per_scenario_delta_map(baseline_data, candidate_data),
            "anomalies": _compute_anomalies(baseline_data, candidate_data),
        })
        print(f"[Log] Appended to {EXPERIMENTS_LOG}")

        experiment_count += 1
        _save_state(experiment_count, baseline_composite, current_baseline_path, tried_fixes)
        _save_tried_fixes(tried_fixes)

        # Gap 6: Token budget check
        if token_budget is not None:
            from eval.token_counter import get_session_totals
            totals = get_session_totals()
            session_tokens = sum(u.total for u in totals.values())
            if session_tokens >= token_budget:
                print(
                    f"[Loop] Token budget reached: {session_tokens:,} >= {token_budget:,} — stopping."
                )
                break

    _write_loop_status("done", experiment_count, max_experiments, baseline_composite,
                       last_edit_summary, last_decision_label, active_branch, token_budget)

    if no_commit:
        next_steps = (
            "  Working tree has the accepted changes. Review with:\n"
            "    git diff\n"
            "  Then commit manually:\n"
            "    git add -p && git commit"
        )
    elif active_branch:
        next_steps = (
            f"  Session branch: {active_branch}\n"
            "  Inspect with:\n"
            f"    git log --oneline main..{active_branch}\n"
            f"    git diff main..{active_branch}\n"
            "  Merge when satisfied:\n"
            f"    git checkout main && git merge --no-ff {active_branch}"
        )
    else:
        next_steps = "  Commits landed on current branch."

    from eval.token_counter import get_session_totals
    totals = get_session_totals()
    if totals:
        token_lines = "  ".join(f"{role}: {u}" for role, u in sorted(totals.items()))
        print(f"  Token usage     : {token_lines}")

    print(
        f"\n{'='*60}\n"
        f"  Session complete - {experiment_count} experiment(s) run\n"
        f"  Final composite : {baseline_composite}\n"
        f"\n"
        f"{next_steps}\n"
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
    tried_fixes: dict[tuple[str, str, str], int],
) -> None:
    """Persist loop state to disk so the session can be resumed after a crash."""
    LOOP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "experiment_count": experiment_count,
        "baseline_composite": baseline_composite,
        "baseline_path": baseline_path,
        # dict[tuple, int] not JSON-serialisable; store as [key..., count] entries
        "tried_fixes": [list(k) + [v] for k, v in tried_fixes.items()],
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
        help="Maximum number of Critic->Optimizer->Eval iterations (default: 10).",
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
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Create and switch to this git branch before the first commit so "
            "experiments never land on main. "
            "Defaults to 'auto/optimize-{timestamp}' when omitted. "
            "Pass an empty string to disable branch creation entirely."
        ),
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help=(
            "Accept improvements to the working tree without committing them. "
            "Useful for inspecting changes before deciding what to keep. "
            "Use `git diff` and `git add -p && git commit` afterwards."
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
            "(default). Required for meaningful optimization - mock outputs produce "
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
    parser.add_argument(
        "--reset-tried-fixes",
        action="store_true",
        help=(
            "Clear the persistent tried-fixes blacklist before starting. "
            "Use this when you want the critic to retry previously blocked fixes. "
            "WARNING: this allows re-attempting fixes that were already reverted."
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Abort the loop after this many total LLM tokens (critic + optimizer combined). "
            "Useful for cost control. No limit by default."
        ),
    )
    args = parser.parse_args()

    # Resolve session branch: explicit name > default auto-name > disabled ("")
    if args.no_commit:
        session_branch = None
    elif args.branch is None:
        session_branch = f"auto/optimize-{_timestamp()}"
    elif args.branch == "":
        session_branch = None
    else:
        session_branch = args.branch

    asyncio.run(
        run_autonomous_loop(
            baseline_path=args.baseline,
            max_experiments=args.max_experiments,
            dry_run=args.dry_run,
            max_edits_per_iter=args.max_edits_per_iter,
            target_composite=args.target_composite,
            use_real_llm=args.use_real_llm,
            resume=args.resume,
            session_branch=session_branch,
            no_commit=args.no_commit,
            reset_tried_fixes=args.reset_tried_fixes,
            token_budget=args.token_budget,
        )
    )


if __name__ == "__main__":
    main()
