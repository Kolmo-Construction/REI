"""
Grid-search optimizer for the Greenvest agent.

Generates N candidate edits in parallel, evaluates each one in an isolated
git worktree, then presents a ranked comparison table. No commits are made -
the user inspects the table and applies their chosen candidate manually.

Pipeline
--------
  1. Critic  -> N gradients (no tried-fixes blacklist, all failing scenarios)
  2. Optimizer LLM -> N edit plans (parallel calls, no file writes yet)
  3. For each plan:
       apply to main working tree -> capture `git diff` -> revert immediately
  4. For each diff:
       git worktree add --detach -> git apply -> eval.py -> worktree remove
  5. Ranked table + per-scenario delta breakdown + git diff + apply command
  6. Saves .patch files and session JSON to experiments/

Usage
-----
    # Use an existing baseline JSON (fastest):
    uv run python -m eval.grid_search --baseline eval_results/baseline.json

    # Run a fresh baseline eval first:
    uv run python -m eval.grid_search

    # More candidates, lower the failure bar to surface near-misses:
    uv run python -m eval.grid_search --max-candidates 8 --failure-threshold 0.85

    # Run candidate evals in parallel (requires Ollama to handle concurrent requests):
    uv run python -m eval.grid_search --parallel 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EVAL_SCRIPT = Path(__file__).parent / "eval.py"
SCENARIOS_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

EVAL_TIMEOUT_SECONDS = 600


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidatePlan:
    """A generated (but not yet applied) edit plan for one gradient."""
    gradient_idx: int           # position in the gradient list (for labelling)
    scenario_id: str
    target_node: str
    fix_type: str
    confidence: float
    plan: dict                  # raw optimizer LLM output
    edit_summary: str           # human-readable description of the edit
    diff: str                   # git diff patch text
    patch_path: Path | None = field(default=None)   # saved .patch file


@dataclass
class CandidateResult:
    """Outcome of evaluating one CandidatePlan."""
    candidate: CandidatePlan
    candidate_data: dict | None = field(default=None)   # eval results JSON
    error: str = field(default="")

    @property
    def candidate_composite(self) -> float | None:
        if self.candidate_data is None:
            return None
        return self.candidate_data.get("composite")

    def delta(self, baseline_composite: float | None) -> float | None:
        c = self.candidate_composite
        if c is None or baseline_composite is None:
            return None
        return c - baseline_composite

    def per_scenario_deltas(self, baseline_data: dict) -> dict[str, float | None]:
        """Return {scenario_id: delta} for all judged scenarios."""
        if self.candidate_data is None:
            return {}
        base_map = {
            sc["scenario_id"]: sc.get("composite")
            for sc in baseline_data.get("per_scenario", [])
            if not sc.get("skipped") and not sc.get("error")
        }
        cand_map = {
            sc["scenario_id"]: sc.get("composite")
            for sc in self.candidate_data.get("per_scenario", [])
            if not sc.get("skipped") and not sc.get("error")
        }
        all_ids = sorted(set(base_map) | set(cand_map))
        result: dict[str, float | None] = {}
        for sid in all_ids:
            b = base_map.get(sid)
            c = cand_map.get(sid)
            result[sid] = (c - b) if (b is not None and c is not None) else None
        return result


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

class EvalError(RuntimeError):
    pass


async def _run_baseline_eval(output_path: Path, use_real_llm: bool) -> dict:
    cmd = [sys.executable, "-m", "eval.eval",
           "--dataset", str(SCENARIOS_DIR),
           "--output", str(output_path)]
    env = dict(os.environ)
    env["USE_MOCK_LLM"] = "false" if use_real_llm else "true"
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[Grid] Running baseline eval...")
    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    try:
        await asyncio.wait_for(proc.wait(), timeout=EVAL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise EvalError("Baseline eval timed out.")
    if proc.returncode != 0:
        raise EvalError(f"Baseline eval exited with code {proc.returncode}.")
    return json.loads(output_path.read_text())


async def _run_eval_in_worktree(
    worktree_path: Path,
    output_path: Path,
    use_real_llm: bool,
    candidate_label: str,
) -> dict:
    """Run eval.py inside an isolated git worktree.

    Sets PYTHONPATH to the worktree root so that modified greenvest/ and
    eval/ packages are used instead of any editable-installed versions.
    """
    cmd = [sys.executable, "-m", "eval.eval",
           "--dataset", str(SCENARIOS_DIR),
           "--output", str(output_path)]
    env = dict(os.environ)
    env["USE_MOCK_LLM"] = "false" if use_real_llm else "true"
    env["PYTHONIOENCODING"] = "utf-8"
    # Prepend worktree to PYTHONPATH so modified source files take precedence
    # over any editable install of the same package in site-packages.
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(worktree_path) + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else str(worktree_path)
    )
    # Disable Langfuse telemetry in worktree evals to avoid duplicate traces.
    env.setdefault("LANGFUSE_PUBLIC_KEY", "")
    env.setdefault("LANGFUSE_SECRET_KEY", "")

    print(f"[Grid] Evaluating candidate {candidate_label} in worktree...")
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env, cwd=str(worktree_path)
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=EVAL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise EvalError(f"Candidate {candidate_label} eval timed out.")
    if proc.returncode != 0:
        raise EvalError(
            f"Candidate {candidate_label} eval exited with code {proc.returncode}."
        )
    return json.loads(output_path.read_text())


# ---------------------------------------------------------------------------
# Edit plan capture: apply -> diff -> revert
# ---------------------------------------------------------------------------

def _capture_diff_for_plan(
    plan: dict,
    gradient,  # TextualGradient
    editable_files: list[str],
) -> tuple[str, str] | tuple[None, str]:
    """
    Apply plan to the main working tree, capture git diff, then revert.

    Returns (diff_text, edit_summary) on success, (None, error_message) on failure.
    The working tree is always reverted before returning.
    """
    from eval.optimizer_agent import apply_plan

    result = apply_plan(plan, gradient)
    if not result.success:
        return None, f"apply failed: {result.error}"

    diff_result = subprocess.run(
        ["git", "diff", "--"] + editable_files,
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    diff = diff_result.stdout

    # Always revert immediately - the diff is all we need
    revert_cmd = ["git", "checkout", "--"] + editable_files
    subprocess.run(revert_cmd, cwd=str(REPO_ROOT), check=False)

    if not diff.strip():
        return None, "edit produced no diff (no change to any editable file)"

    return diff, result.summary


# ---------------------------------------------------------------------------
# Worktree isolation
# ---------------------------------------------------------------------------

async def _eval_candidate_in_worktree(
    candidate: CandidatePlan,
    output_path: Path,
    use_real_llm: bool,
    semaphore: asyncio.Semaphore,
) -> CandidateResult:
    """Apply candidate diff to an isolated worktree, run eval, clean up."""
    async with semaphore:
        worktree_path = Path(tempfile.mkdtemp(prefix=f"rei-gs-{candidate.gradient_idx}-"))
        label = f"{candidate.gradient_idx + 1} ({candidate.scenario_id})"
        try:
            # Create detached worktree at HEAD
            add_result = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path)],
                cwd=str(REPO_ROOT), capture_output=True, text=True,
            )
            if add_result.returncode != 0:
                return CandidateResult(
                    candidate=candidate,
                    error=f"git worktree add failed: {add_result.stderr.strip()}",
                )

            # Apply the captured diff
            apply_result = subprocess.run(
                ["git", "apply", "--whitespace=fix"],
                input=candidate.diff, text=True,
                cwd=str(worktree_path), capture_output=True,
            )
            if apply_result.returncode != 0:
                return CandidateResult(
                    candidate=candidate,
                    error=f"git apply failed: {apply_result.stderr.strip()}",
                )

            # Run eval
            candidate_data = await _run_eval_in_worktree(
                worktree_path, output_path, use_real_llm, label
            )
            return CandidateResult(candidate=candidate, candidate_data=candidate_data)

        except EvalError as exc:
            return CandidateResult(candidate=candidate, error=str(exc))

        except Exception as exc:
            return CandidateResult(candidate=candidate, error=f"unexpected: {exc}")

        finally:
            # Always clean up worktree - force removal even if dirty
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=str(REPO_ROOT), capture_output=True,
            )
            shutil.rmtree(worktree_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _decision_label(delta: float | None) -> str:
    if delta is None:
        return "ERROR"
    if delta >= 0.01:
        return "IMPROVE"
    if delta <= -0.01:
        return "REGRESS"
    return "NO CHANGE"


def _print_report(
    baseline_data: dict,
    baseline_composite: float | None,
    results: list[CandidateResult],
    session_ts: str,
) -> None:
    W = 78
    print(f"\n{'=' * W}")
    print(f"  Grid Search Results  -  {session_ts}")
    print(f"  Baseline composite : {baseline_composite}")
    print(f"  Candidates         : {len(results)}")
    print(f"{'=' * W}")

    # Sort: IMPROVE first (by delta desc), then NO CHANGE, then REGRESS/ERROR
    def sort_key(r: CandidateResult):
        d = r.delta(baseline_composite)
        return (0 if d is None else -d,)

    sorted_results = sorted(results, key=sort_key)

    # ── Summary table ─────────────────────────────────────────────────────
    col = "{:<4} {:<25} {:<16} {:<16} {:<8} {:<10} {:<10} {:<10}"
    print(col.format("Rank", "Scenario", "Target Node", "Fix Type",
                     "Conf", "Baseline", "Cand", "Δ"))
    print("-" * W)
    for rank, r in enumerate(sorted_results, 1):
        c = r.candidate
        delta = r.delta(baseline_composite)
        cand_str = f"{r.candidate_composite:.4f}" if r.candidate_composite is not None else "N/A"
        base_str = f"{baseline_composite:.4f}" if baseline_composite is not None else "N/A"
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        decision = _decision_label(delta)
        icon = "[+]" if decision == "IMPROVE" else ("[-]" if decision == "REGRESS" else "~")
        print(col.format(
            f"{rank}.",
            c.scenario_id[:24],
            c.target_node[:15],
            c.fix_type[:15],
            f"{c.confidence:.2f}",
            base_str,
            cand_str,
            f"{icon} {delta_str}",
        ))
    print()

    # ── Per-candidate detail ──────────────────────────────────────────────
    for rank, r in enumerate(sorted_results, 1):
        c = r.candidate
        delta = r.delta(baseline_composite)
        decision = _decision_label(delta)
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{'─' * W}")
        print(
            f"  Candidate {rank}  [{decision}  Δ{delta_str}]  "
            f"{c.scenario_id} -> {c.target_node} ({c.fix_type})"
        )
        print(f"  {c.edit_summary}")

        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            # Per-scenario delta breakdown
            deltas = r.per_scenario_deltas(baseline_data)
            if deltas:
                print("\n  Per-scenario delta:")
                for sid, d in sorted(deltas.items()):
                    d_str = f"{d:+.3f}" if d is not None else "N/A"
                    marker = " [+]" if d is not None and d > 0.005 else (
                        " [-]" if d is not None and d < -0.005 else ""
                    )
                    print(f"    {sid:<35} {d_str}{marker}")

        # Show the diff (truncated if large)
        if c.diff:
            diff_lines = c.diff.splitlines()
            shown = diff_lines[:40]
            print(f"\n  Diff ({len(diff_lines)} lines"
                  f"{', truncated' if len(diff_lines) > 40 else ''}):")
            for line in shown:
                print(f"    {line}")
            if len(diff_lines) > 40:
                print(f"    ... ({len(diff_lines) - 40} more lines)")

        # Apply instructions
        if c.patch_path and not r.error and decision == "IMPROVE":
            print(f"\n  To apply this candidate:")
            print(f"    git apply {c.patch_path}")
            print(f"    git add -p && git commit -m 'opt: {c.edit_summary[:60]}'")
        print()

    print(f"{'=' * W}")
    print(f"  Patch files saved to: {EXPERIMENTS_DIR}/")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------

async def run_grid_search(
    baseline_path: Path | None,
    max_candidates: int,
    failure_threshold: float,
    use_real_llm: bool,
    parallel: int,
) -> None:
    from eval.critic import analyze_failures
    from eval.optimizer_agent import generate_edit_plan
    from eval.edit_tools import EDITABLE_FILES_REL

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    session_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    editable_files = list(EDITABLE_FILES_REL)

    # ── 1. Baseline ───────────────────────────────────────────────────────
    if baseline_path and baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text())
        baseline_composite = baseline_data.get("composite")
        print(f"[Grid] Loaded baseline: composite={baseline_composite}  ({baseline_path})")
    else:
        bp = EVAL_RESULTS_DIR / f"gs_baseline_{session_ts}.json"
        baseline_data = await _run_baseline_eval(bp, use_real_llm=use_real_llm)
        baseline_composite = baseline_data.get("composite")
        print(f"[Grid] Baseline established: composite={baseline_composite}")

    # ── 2. Critic - no tried-fixes filter so all hypotheses are visible ───
    print(f"\n[Grid] Running critic (failure_threshold={failure_threshold})...")
    gradients = await analyze_failures(
        baseline_data,
        scenarios_dir=SCENARIOS_DIR,
        tried_fixes=None,
        failure_threshold=failure_threshold,
    )
    if not gradients:
        print("[Grid] No failing scenarios - nothing to optimize. Done.")
        return

    gradients = gradients[:max_candidates]
    print(f"[Grid] {len(gradients)} gradient(s) to explore.")

    # ── 3. Generate edit plans (parallel LLM calls) ───────────────────────
    print(f"\n[Grid] Generating {len(gradients)} edit plan(s) in parallel...")
    plans_raw: list[dict | None] = list(await asyncio.gather(
        *[generate_edit_plan(g) for g in gradients],
        return_exceptions=False,
    ))

    # ── 4. Apply each plan to main tree, capture diff, revert ─────────────
    print(f"\n[Grid] Capturing diffs (apply -> diff -> revert for each plan)...")
    candidates: list[CandidatePlan] = []
    for idx, (gradient, plan) in enumerate(zip(gradients, plans_raw)):
        if plan is None:
            print(
                f"[Grid] Candidate {idx + 1}: optimizer returned no plan "
                f"for {gradient.scenario_id} - skipping.",
                file=sys.stderr,
            )
            continue

        diff, summary = _capture_diff_for_plan(plan, gradient, editable_files)
        if diff is None:
            print(
                f"[Grid] Candidate {idx + 1} ({gradient.scenario_id}): {summary}",
                file=sys.stderr,
            )
            continue

        # Save patch file for later manual application
        patch_path = EXPERIMENTS_DIR / f"gs_{session_ts}_candidate_{idx}.patch"
        patch_path.write_text(diff, encoding="utf-8")

        candidates.append(CandidatePlan(
            gradient_idx=idx,
            scenario_id=gradient.scenario_id,
            target_node=gradient.target_node,
            fix_type=gradient.suggested_fix_type,
            confidence=gradient.confidence,
            plan=plan,
            edit_summary=summary,
            diff=diff,
            patch_path=patch_path,
        ))
        print(f"[Grid] Candidate {idx + 1}: {gradient.scenario_id} -> {gradient.target_node} "
              f"| patch saved to {patch_path.name}")

    if not candidates:
        print("[Grid] No valid candidates generated. Done.")
        return

    # ── 5. Evaluate each candidate in an isolated worktree ────────────────
    print(f"\n[Grid] Evaluating {len(candidates)} candidate(s) "
          f"(parallel={parallel}, timeout={EVAL_TIMEOUT_SECONDS}s each)...")

    semaphore = asyncio.Semaphore(parallel)
    eval_tasks = []
    for i, candidate in enumerate(candidates):
        output_path = EVAL_RESULTS_DIR / f"gs_{session_ts}_candidate_{candidate.gradient_idx}.json"
        eval_tasks.append(
            _eval_candidate_in_worktree(candidate, output_path, use_real_llm, semaphore)
        )
    results: list[CandidateResult] = list(await asyncio.gather(*eval_tasks))

    # ── 6. Save session JSON ──────────────────────────────────────────────
    session_path = EXPERIMENTS_DIR / f"gs_{session_ts}_session.json"
    session_data = {
        "session_ts": session_ts,
        "baseline_composite": baseline_composite,
        "failure_threshold": failure_threshold,
        "candidates": [
            {
                "gradient_idx": r.candidate.gradient_idx,
                "scenario_id": r.candidate.scenario_id,
                "target_node": r.candidate.target_node,
                "fix_type": r.candidate.fix_type,
                "confidence": r.candidate.confidence,
                "edit_summary": r.candidate.edit_summary,
                "patch_file": str(r.candidate.patch_path.name) if r.candidate.patch_path else None,
                "candidate_composite": r.candidate_composite,
                "delta": r.delta(baseline_composite),
                "decision": _decision_label(r.delta(baseline_composite)),
                "error": r.error,
                "per_scenario_deltas": r.per_scenario_deltas(baseline_data),
            }
            for r in results
        ],
    }
    session_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

    # ── 7. Print report ───────────────────────────────────────────────────
    _print_report(baseline_data, baseline_composite, results, session_ts)
    print(f"Session results saved to: {session_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search optimizer: evaluate N candidates in isolated worktrees.",
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
        "--max-candidates",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of candidates to generate and evaluate (default: 5).",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.90,
        metavar="SCORE",
        help=(
            "Composite score below which a scenario is considered failing "
            "(default: 0.90 - slightly higher than the loop's 0.80 to surface "
            "near-misses as candidates)."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of candidate evals to run in parallel (default: 1). "
            "Each parallel eval uses its own git worktree and Ollama request. "
            "Increase only if your machine and Ollama can handle concurrent load."
        ),
    )
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--real-llm",
        dest="use_real_llm",
        action="store_true",
        default=True,
        help="Use real Ollama inference for agent eval (default).",
    )
    llm_group.add_argument(
        "--mock-llm",
        dest="use_real_llm",
        action="store_false",
        help="Use mock LLM for agent eval (offline testing only).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_grid_search(
            baseline_path=args.baseline,
            max_candidates=args.max_candidates,
            failure_threshold=args.failure_threshold,
            use_real_llm=args.use_real_llm,
            parallel=args.parallel,
        )
    )


if __name__ == "__main__":
    main()
