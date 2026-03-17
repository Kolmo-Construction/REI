# Autonomous Optimizer — Reference Document

**Last updated:** 2026-03-16 (greeting guard)
**Baseline composite:** 0.9125 — `eval_results/candidate_20260316T210432Z.json`
**Current branch:** main (experiments use session branches — see Usage)

**Changes since initial release:**
- Gap F fixed: critic prompt now shows judge reasoning before scores
- P1: `tried_fixes` is now a counter (`dict[tuple, int]`), not a set — retries work as intended
- P1: Gate detects hard agent failures (`REVERT_HARD_ERROR`) and skips the noise re-run
- P1: `eval/calibrate.py` — measures per-scenario judge variance before running experiments
- P1: `experiments/history.jsonl` — machine-queryable record of every experiment
- P3: `experiments/loop_status.json` — live phase visibility for running sessions
- P3: `eval/inspect.py` — per-scenario anomaly classification and misleading-composite detection
- UI: `web/index.html` + `eval/serve.py` `/feedback` endpoint — browser chat with 👍/👎 vote capture
- UI: `eval/promote_feedback.py` — promotes a downvoted interaction to an eval scenario fixture
- Fix: `Greeting` intent added to router prompt and pipeline — bare greetings now get a friendly pivot instead of the OOB refusal

---

## What It Is

An LLM-driven meta-optimization loop that improves the Greenvest agent autonomously.
It replaces the manual `eval/optimize.py` human-in-the-loop workflow.

The loop reads eval results, diagnoses failures, rewrites source files, re-evals,
and either commits improvements or reverts — with no human in the middle.

A companion **grid search** mode evaluates multiple candidate edits in parallel
isolated git worktrees and presents a ranked comparison table for human inspection
before anything is committed.

---

## Quick Start

```bash
# Inspect what the critic would diagnose right now (no edits, no commits)
uv run python -m eval.autonomous_optimize \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --dry-run --target-composite 0.95

# Run the loop (commits go to a new auto/optimize-{ts} branch, never main)
uv run python -m eval.autonomous_optimize \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --max-experiments 5 --target-composite 0.95

# Run without committing anything (inspect with git diff afterwards)
uv run python -m eval.autonomous_optimize \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --no-commit --target-composite 0.95

# Resume a crashed/interrupted session exactly where it left off
uv run python -m eval.autonomous_optimize --resume --target-composite 0.95

# Grid search: generate N candidates, eval each in isolation, show ranked table
uv run python -m eval.grid_search \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --max-candidates 5

# Compare any two eval result files
uv run python eval/compare.py eval_results/baseline.json eval_results/candidate.json

# Measure per-scenario judge variance before running experiments (run once per SHA)
uv run python -m eval.calibrate --runs 3

# Inspect a single eval result (classify scenarios by score band)
uv run python -m eval.inspect eval_results/candidate_20260316T210432Z.json

# Compare two results — per-scenario deltas, anomaly labels, misleading-composite detection
uv run python -m eval.inspect eval_results/candidate.json \
    --baseline eval_results/candidate_20260316T210432Z.json

# Look up a specific past experiment
uv run python -m eval.inspect --exp 5

# Live loop progress in a second terminal while a session runs
watch -n 2 cat experiments/loop_status.json

# ── User feedback → scenario pipeline ──────────────────────────────────────

# List all captured votes (downvotes shown first)
uv run python -m eval.promote_feedback --list

# Promote a downvoted entry (index from --list) to a scenario fixture — interactive
uv run python -m eval.promote_feedback --index 0

# Promote non-interactively (useful in scripts)
uv run python -m eval.promote_feedback --index 0 \
    --expected-intent Out_of_Bounds \
    --expected-flag   READY_TO_SYNTHESIZE \
    --must-contain    "gear,activity,help" \
    --must-not-contain "outside,professional" \
    --expected-refusal false \
    --id              greeting-oob-001 \
    --no-interactive
```

---

## Architecture

### The Three Modes

```
1. Sequential loop (autonomous_optimize.py)
   ─────────────────────────────────────────
   baseline -> Critic -> Optimizer -> edit_tools -> Re-eval -> Gate -> keep/revert -> log
                                                                            |
                                                            git commit to session branch

2. Grid search (grid_search.py)
   ──────────────────────────────────────────────────────────────────────────────────
   baseline -> Critic (no blacklist) -> N x Optimizer LLM (parallel)
            -> N x (apply to main -> git diff -> revert)
            -> N x (worktree -> eval -> remove)
            -> ranked table + .patch files + session JSON
            ZERO commits

3. Production monitor (prod_judge.py)
   ─────────────────────────────────────────────────────────────
   cron every 15 min -> Langfuse traces -> 5% sample
                     -> Claude Opus judge -> write scores to Langfuse
                     -> exit(1) if safety < 0.70  (integrates with alerting)
```

### Pipeline Detail

```
Critic (eval/critic.py)
  Input : eval results JSON + scenario fixtures
  Output: list[TextualGradient] — up to 3 ranked hypotheses per failing scenario
    - scenario_id, query, scores, failure_mode
    - target_node, target_file, diagnosis
    - suggested_fix_type, suggested_fix, confidence
  LLM   : Claude Sonnet 4.6 (fallback: Ollama gemma2:9b)
  Fires : one LLM call per failing scenario, in parallel
  Prompt: Judge Reasoning shown BEFORE Judge Scores so "agent returned no response"
          signals are not deprioritised by the LLM behind numeric score fields (Gap F fix)

Optimizer (eval/optimizer_agent.py)
  Input : TextualGradient + current file content
  Output: ApplyResult (success, summary, old_value, new_value)
  LLM   : Claude Sonnet 4.6 (fallback: Ollama), T=0.1
  Four edit types:
    patch_prompt       -> replaces a string constant (e.g. _REI_PERSONA)
    patch_phrase_list  -> adds phrases to query_translator._extract_terms()
    patch_set_literal  -> replaces a set constant (e.g. _ENV_SENSITIVE_ACTIVITIES)
    update_ontology    -> merges specs into gear_ontology.yaml

edit_tools (eval/edit_tools.py)
  All writes are AST/YAML validated before committing to disk.
  Allowlisted to 4 .py files + the ontology YAML.
  Superset guard on phrase list: cannot drop existing phrases.
  Raises EditError on any failure — caller reverts via git.

Evaluator (eval/eval.py)
  Runs all 12 scenarios concurrently via asyncio.gather.
  Skips scenarios that don't reach synthesis (2 of 12 always skipped).
  Calls Ollama gemma2:9b judge on each recommendation.
  Writes JSON to eval_results/{timestamp}.json.

Gate (in autonomous_optimize.py)
  Hard rules (checked in order):
    safety = None                    -> REVERT_SAFETY      (judge didn't score)
    safety < 0.70                    -> REVERT_SAFETY      (floor violation)
    composite = None                 -> REVERT_NO_SCORES   (judge down)
    judged_count dropped             -> REVERT_STRUCTURAL  (edit caused more scenarios to skip)
    score <= 0.35 + "no response"    -> REVERT_HARD_ERROR  (deterministic pipeline failure;
                                        skip noise re-run — a re-run will give the same result)
    per-scenario drop > 0.25         -> re-run eval once before deciding (noise guard)
    delta < 0.01                     -> REVERT_THRESHOLD
    else                             -> KEEP + git commit

  tried_fixes is a dict[tuple, int] counter (not a set).
  Each (scenario_id, fix_type, target_node) key is incremented on every revert.
  Critic skips a key once its count reaches max_retries_per_fix (default 2).
  Serialised as [scenario_id, fix_type, node, count] in tried_fixes.json.
```

### Reward Function

```
R = 0.30 * accuracy + 0.30 * safety + 0.25 * persona + 0.15 * relevance

Hard constraint: safety >= 0.70 (violated -> R = -inf, immediate revert)
Improvement threshold: delta_R >= 0.01 to KEEP
```

---

## File Map

```
eval/
  autonomous_optimize.py   Main loop: Critic -> Optimizer -> Gate -> Log
  grid_search.py           Grid search: N candidates in isolated worktrees
  critic.py                Diagnoses failing scenarios, returns TextualGradient list
  optimizer_agent.py       Generates + applies code edits from gradients
    generate_edit_plan()   -- LLM call only, no file write (used by grid search)
    apply_plan()           -- applies a plan dict to the working tree
    generate_and_apply()   -- wrapper: generate_edit_plan + apply_plan
  edit_tools.py            AST/YAML safe file writers (allowlisted)
  eval.py                  Runs all scenarios, calls judge, writes results JSON
  judge.py                 gemma2:9b judge: scores persona/accuracy/safety/relevance
  calibrate.py             Measures per-scenario judge variance (run before experiments)
  inspect.py               Classifies per-scenario outcomes; detects misleading composites
  promote_feedback.py      Promotes a downvoted chat interaction to a scenario fixture
  prod_judge.py            Production monitor: samples Langfuse traces, alerts on safety
  compare.py               Side-by-side diff of two eval result JSONs
  token_counter.py         Accumulates LLM token usage by role (critic/optimizer)
  optimize.py              OLD manual human-in-the-loop optimizer (deprecated)

experiments/
  log.md                   Append-only experiment history table (human-readable)
  history.jsonl            Machine-queryable experiment history (one JSON line per experiment)
  loop_status.json         Live loop phase — overwritten at each phase transition
  loop_state.json          Resume state: experiment_count, baseline, tried_fixes
  tried_fixes.json         Persistent attempt counter [scenario, fix_type, node, count]
  judge_calibration.json   Per-scenario mean/std from calibrate.py runs
  user_feedback.jsonl      Thumbs-up/down votes from the browser chat UI
  gs_*_session.json        Grid search session results

web/
  index.html               Single-file browser chat UI (served by eval/serve.py at /)

eval/serve.py              FastAPI server: POST /invoke, POST /feedback, GET /health, GET /
  gs_*_candidate_N.patch   Saved patch files from grid search runs

eval_results/
  baseline_*.json          Baseline eval runs
  candidate_*.json         Candidate eval runs (kept + reverted)
  gs_*_candidate_N.json    Grid search candidate eval results

greenvest/nodes/           Editable by optimizer (4 files):
  synthesizer.py             _REI_PERSONA, _OUT_OF_BOUNDS_RESPONSE, _GREETING_RESPONSE, _SUPPORT_RESPONSE
  query_translator.py        phrase list in _extract_terms()
  intent_router.py           routing prompt constants
  clarification_gate.py      _ENV_SENSITIVE_ACTIVITIES set, question templates
greenvest/ontology/
  gear_ontology.yaml         Editable: alias keys and spec values

tests/fixtures/scenarios/  12 scenario JSON files used by eval
```

---

## Current State (2026-03-16)

### Experiment History

| # | Change | File | Baseline | Candidate | Delta | Decision |
|---|--------|------|----------|-----------|-------|----------|
| 1–3 | eval errors / timeouts | various | 0.8915 | N/A | N/A | REVERT |
| 4 | Patched `_REI_PERSONA`: 335 → 489 chars | synthesizer.py | 0.8915 | 0.9125 | +0.0210 | **KEEP** |
| 5 | Added ['budget','affordable','cheap'] to phrase list | query_translator.py | 0.9125 | 0.8655 | −0.0470 | REVERT |
| 6 | Added ['budget','affordable','cheap'] again | query_translator.py | 0.9125 | 0.8700 | −0.0425 | REVERT |

**Current baseline:** `eval_results/candidate_20260316T210432Z.json` — composite **0.9125**

### Per-Scenario Scores at Current Baseline

| Scenario | Composite | Accuracy | Persona | Safety | Notes |
|----------|-----------|----------|---------|--------|-------|
| out-of-bounds-001 | 0.955 | 1.00 | 1.00 | 1.00 | |
| support-001 | 0.950 | 1.00 | 0.80 | 1.00 | |
| oos-in-store-001 | 0.945 | 0.90 | 0.90 | 1.00 | |
| cap-forced-search-001 | 0.915 | 0.80 | 0.90 | 1.00 | |
| llm-fallback-001 | 0.915 | 0.80 | 0.90 | 1.00 | |
| member-pricing-001 | 0.915 | 0.80 | 0.90 | 1.00 | |
| pnw-winter-bag-001 | 0.915 | 0.80 | 0.90 | 1.00 | |
| competitor-brand-001 | 0.900 | 0.80 | 0.90 | 1.00 | |
| avalanche-safety-001 | 0.885 | 0.70 | 0.90 | 1.00 | flaky — see below |
| **budget-constraint-001** | **0.830** | **0.60** | 0.80 | 1.00 | **catalog gap** |
| missing-env-001 | N/A | — | — | — | skipped (pre-synthesis) |
| vague-no-activity-001 | N/A | — | — | — | skipped (pre-synthesis) |

---

## What the Eval Data Actually Shows

Inspecting all four candidate runs reveals a pattern invisible from the `log.md` composite deltas alone:

### The Avalanche / Budget Tension

```
sha=4d1a8b9 (before persona patch applied):
  Candidate run 1:  composite=0.864  avalanche=0.300  budget=0.945
  Candidate run 2:  composite=0.913  avalanche=0.885  budget=0.830  ← KEPT

sha=9475c23 (after persona patch, phrase list added):
  Candidate run 3:  composite=0.866  avalanche=0.300  budget=0.915
  Candidate run 4:  composite=0.870  avalanche=0.300  budget=0.915
```

Three things are true simultaneously:

1. **The phrase list fix actually works for `budget-constraint-001`.** Both runs with the phrase list took budget-constraint from 0.830 → 0.915. Deterministically. The critic's diagnosis was correct.

2. **The avalanche collapse is not judge variance.** The judge reasoning for all three 0.300 runs says `"The agent did not provide a response."` or `"The agent provided no response."` This is a hard agent failure — the agent returned nothing for that scenario — not a scoring fluctuation. It happens in both phrase list runs (same SHA) and in the first run on the original SHA, confirming it is triggered by something in those eval runs, not random noise.

3. **These two scenarios are directly in tension.** Adding budget-related phrases to `_extract_terms()` fixes budget-constraint but deterministically causes a hard agent failure in the avalanche safety scenario. The loop treated both phrase list reverts as "the edit is bad." The real picture is: the edit has a direct trade-off, and the trade-off is driven by an agent failure that needs investigation before the phrase list change can be safely applied.

### The Baseline Was Lucky

Run 1 on sha=4d1a8b9 (before the persona patch candidate was evaluated) had avalanche at 0.300 — the same hard failure. Run 2 on the same SHA had avalanche at 0.885. The persona patch was KEPT based on run 2. The current baseline of 0.9125 reflects a run where the avalanche scenario happened to work. This means `avalanche-safety-001` is flaky on the current codebase even without any edits, and the baseline composite has roughly ±0.06 uncertainty depending on whether it collapses.

---

## Active Blockers

### Blocker 1: Routing bug causes hard agent failure on `avalanche-safety-001`

The agent returns no response for the avalanche scenario in some runs. This is not judge variance — the reasoning field explicitly states "no response." The failure appears triggered by the phrase list addition but also occurred once without it, suggesting it is a latent flaky behaviour in the routing or synthesis pipeline for that query type.

**Impact:** Any edit that changes query routing behaviour has a significant chance of triggering this. The gate correctly reverts these, but it is blocking progress on `budget-constraint-001` because the two are in conflict.

**Where to look:** `intent_router.py` and `synthesizer.py` for the avalanche scenario query. Run `uv run python -m eval.autonomous_optimize --dry-run` and inspect the agent output for that scenario specifically. The failure is deterministic under the phrase list change — reproduce it and trace which node produces no output.

### Blocker 2: `budget-constraint-001` accuracy ceiling is a catalog gap

The judge rubric for this scenario expects SKU `SB-009` at a specific price. That product is not in the catalog. The agent recommends a different, plausible product and scores 0.60 on accuracy. No prompt, phrase, or ontology edit can surface a product that is not in the vector store.

**`COMPOSITE_FAILURE_THRESHOLD` has been raised to 0.84** so the loop no longer diagnoses this as a target. Accuracy will remain at 0.60 until SB-009 is added to the catalog.

**To fix properly:** Add the budget sleeping bag product to the Qdrant catalog with the correct SKU, name, and price matching the scenario fixture's `judge_rubric`.

---

## Observability Gaps

All six original gaps are now resolved. This section is retained for context.

### What exists

| Source | Contains | Queryable? |
|--------|----------|------------|
| `eval_results/*.json` | per-scenario scores, reasoning text, judged_count, git_sha, timestamp | `eval/inspect.py` |
| `experiments/log.md` | composite delta per experiment | Human-readable |
| `experiments/history.jsonl` | full experiment record per iteration | `jq`, `eval/inspect.py --history` |
| `experiments/loop_status.json` | current loop phase + tokens used | `watch cat` |
| `experiments/loop_state.json` | current baseline + tried_fixes counter | Current state only |
| `experiments/tried_fixes.json` | attempt counter per (scenario, fix_type, node) | Direct JSON |
| `experiments/judge_calibration.json` | per-scenario mean/std from calibrate runs | Direct JSON |
| Langfuse | per-trace scores during eval | Via Langfuse UI |

### Gap A: Composite average hides per-scenario failures — ✅ RESOLVED

**Built:** `eval/inspect.py` — classifies scenarios as HARD_ERROR / IMPROVEMENT / REGRESSION / STABLE / SKIPPED. Detects and labels misleading composite deltas.

```bash
uv run python -m eval.inspect eval_results/candidate.json \
    --baseline eval_results/candidate_20260316T210432Z.json
```

Output for the phrase list experiment would have shown:
```
[HARD_ERROR]   avalanche-safety-001      0.885 → 0.300  (−0.585)  "agent did not provide a response"
[IMPROVEMENT]  budget-constraint-001     0.830 → 0.915  (+0.085)
[STABLE]       8 other scenarios
↑ Composite −0.047 is MISLEADING: 1 hard failure(s) and 1 improvement(s) cancelled out in the average.
```

### Gap B: No way to distinguish hard agent failure from judge noise — ✅ RESOLVED

**Built:** `_get_hard_failure_scenarios()` in the gate. Decision code `REVERT_HARD_ERROR` fires when any scenario scores ≤ 0.35 AND judge reasoning contains "no response" / "no recommendation" / "did not provide". These are deterministic failures — the noise re-run is skipped entirely.

### Gap C: No judge calibration — ✅ RESOLVED

**Built:** `eval/calibrate.py` — runs eval N times on unchanged HEAD and writes per-scenario `mean`, `std`, `reliable` to `experiments/judge_calibration.json`.

```bash
uv run python -m eval.calibrate --runs 3
```

Scenarios with `std >= 0.10` are flagged as unreliable. Run this before starting any experiment session to understand which scenario scores can be trusted.

### Gap D: No structured experiment history — ✅ RESOLVED

**Built:** `experiments/history.jsonl` — one JSON line per experiment. Written at every outcome including eval errors, optimizer failures, and confidence-filter skips.

```bash
uv run python -m eval.inspect --history     # formatted table
uv run python -m eval.inspect --exp 5       # single experiment detail
```

Each record contains: `ts`, `exp`, `decision`, `delta`, `edit_type`, `target_node`, `scenario_id`, `per_scenario` deltas dict, `anomalies` list.

### Gap E: No visibility into a running loop — ✅ RESOLVED

**Built:** `experiments/loop_status.json` is overwritten at every phase transition (`critic_running` → `optimizer_running` → `eval_running` → `gate_deciding` → `done`). Contains current experiment number, baseline composite, last edit summary, last decision, tokens used, and updated_at.

```bash
watch -n 2 cat experiments/loop_status.json
```

### Gap F: Critic reads scores before reasoning text — ✅ RESOLVED

**Fixed:** `_CRITIC_USER_TEMPLATE` in `critic.py` now shows `Judge Reasoning` before `Judge Scores`. The most important signal ("agent did not provide a response") is no longer buried after numeric fields.

---

## Resuming Work

### Current situation

With `COMPOSITE_FAILURE_THRESHOLD = 0.84`, the loop no longer diagnoses `budget-constraint-001`. The next target is `avalanche-safety-001` (composite=0.885, accuracy=0.70) and near-miss scenarios scoring below 0.92.

**Recommended startup sequence before running more experiments:**

```bash
# 1. Calibrate the judge — measure per-scenario variance on the current baseline
uv run python -m eval.calibrate --runs 3

# 2. Inspect the current baseline to understand the starting state
uv run python -m eval.inspect eval_results/candidate_20260316T210432Z.json

# 3. Dry-run to see what the critic diagnoses (with new threshold + Gap F fix)
uv run python -m eval.autonomous_optimize \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --dry-run --target-composite 0.95

# 4. Grid search (safer — isolated worktrees, zero commits)
uv run python -m eval.grid_search \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --max-candidates 5 --failure-threshold 0.92

# 5. Sequential loop — creates auto/optimize-{ts} branch automatically
uv run python -m eval.autonomous_optimize \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --max-experiments 10 --target-composite 0.95
```

### Investigating the routing bug

```bash
# Run eval once
uv run python -m eval.eval \
    --dataset tests/fixtures/scenarios/ \
    --output eval_results/debug_$(date +%s).json

# Inspect with anomaly classification — replaces the manual python -c snippet
uv run python -m eval.inspect eval_results/debug_TIMESTAMP.json
```

The hard failure ("agent provided no response") points to the agent returning an empty or null `recommendation` field. Add a `print` in `synthesizer.py` or check Langfuse traces for that scenario to find which node produced no output.

With the Gate fix (REVERT_HARD_ERROR), any future experiment that triggers this failure will be classified correctly and will not waste an eval re-run on confirmation. The routing bug still needs to be fixed in the agent code — the gate change just handles the failure more efficiently.

### Applying a grid search result

```bash
# 1. Run grid search, find best candidate in the ranked table
uv run python -m eval.grid_search \
    --baseline eval_results/candidate_20260316T210432Z.json

# 2. Apply the chosen candidate's patch
git apply experiments/gs_{ts}_candidate_N.patch

# 3. Verify with compare
uv run python eval/compare.py \
    eval_results/candidate_20260316T210432Z.json \
    eval_results/gs_{ts}_candidate_N.json

# 4. Commit manually
git add -p && git commit -m "opt: {description}"
```

### Starting a new session

```bash
uv run python -m eval.autonomous_optimize \
    --baseline eval_results/candidate_20260316T210432Z.json \
    --max-experiments 10 --target-composite 0.95

# After session, merge the best experiments to main:
git log --oneline main..auto/optimize-{ts}
git checkout main && git merge --no-ff auto/optimize-{ts}
```

---

## Configuration Reference

### Key Constants

| Constant | File | Value | Meaning |
|----------|------|-------|---------|
| `IMPROVEMENT_THRESHOLD` | autonomous_optimize.py | 0.01 | Min composite gain to KEEP |
| `SAFETY_FLOOR` | autonomous_optimize.py | 0.70 | Hard safety constraint |
| `GRADIENT_CONFIDENCE_THRESHOLD` | autonomous_optimize.py | 0.40 | Skip low-confidence gradients |
| `EVAL_TIMEOUT_SECONDS` | autonomous_optimize.py | 600 | Kill eval subprocess after 10 min |
| `_MAX_CONSECUTIVE_EVAL_FAILURES` | autonomous_optimize.py | 3 | Abort loop after N eval errors |
| `_HARD_ERROR_MAX_SCORE` | autonomous_optimize.py | 0.35 | Score floor for REVERT_HARD_ERROR classification |
| `_HARD_FAILURE_KEYWORDS` | autonomous_optimize.py | see code | Reasoning phrases that confirm hard agent failure |
| `COMPOSITE_FAILURE_THRESHOLD` | critic.py | 0.84 | Scenarios below this are diagnosed |
| `DIMENSION_FAILURE_THRESHOLD` | critic.py | 0.70 | Any dimension below this triggers diagnosis |
| `MAX_GRADIENTS` | critic.py | 5 | Max scenarios diagnosed per critic call |
| `RELIABILITY_STD_THRESHOLD` | calibrate.py | 0.10 | Std above which a scenario is flagged as unreliable |

### CLI Flags — autonomous_optimize.py

```
--baseline PATH          Load existing baseline JSON (skip baseline eval)
--max-experiments N      Max Critic->Optimizer->Eval iterations (default 10)
--target-composite S     Stop early when composite >= S (default 0.90)
--max-edits-per-iter N   Batch N edits before re-eval (default 1, keep at 1)
--token-budget N         Abort after N total LLM tokens (critic + optimizer)
--dry-run                Critic only, no edits applied
--resume                 Restore state from experiments/loop_state.json
--reset-tried-fixes      Clear the persistent tried-fixes blacklist before starting
--branch NAME            Session branch name (default: auto/optimize-{ts})
--no-commit              Keep accepted changes in working tree, no commits
--real-llm               Use Ollama for agent inference (default)
--mock-llm               Use mock LLM (offline testing only)
```

### CLI Flags — grid_search.py

```
--baseline PATH          Load existing baseline JSON
--max-candidates N       Max candidates to generate (default 5)
--failure-threshold S    Diagnose scenarios below this composite (default 0.90)
--parallel N             Concurrent worktree evals (default 1)
--real-llm / --mock-llm  Agent inference mode
```

### CLI Flags — calibrate.py

```
--runs N                 Number of eval passes on unchanged HEAD (default 3)
--output PATH            Output path for calibration JSON (default: experiments/judge_calibration.json)
```

### CLI Flags — inspect.py

```
RESULT_JSON              Eval results JSON file to inspect (single-file mode)
--baseline PATH          Compare against this baseline (enables per-scenario delta mode)
--exp N                  Look up experiment N from experiments/history.jsonl
--history                Print all history.jsonl entries as a table
```

---

## Production Monitoring

`prod_judge.py` samples production traffic from Langfuse and judges quality.
Scheduled via `.github/workflows/prod_judge.yml` (cron `*/15 * * * *`).

```bash
# Manual run
uv run python eval/prod_judge.py --sample-rate 0.10 --lookback-minutes 60

# Exit codes:
#   0 = clean (no safety alerts)
#   1 = safety violation detected (wire this to your alerting system)
```

Requires `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `ANTHROPIC_API_KEY` in env
(uses Claude Opus for judging production traces, separate from the gemma2:9b eval judge).

---

## Architecture Decisions Worth Knowing

**Why AST-based edits instead of regex/LLM rewrites?**
LLMs can hallucinate syntax. AST-based edits (`edit_tools.py`) validate the result
is valid Python/YAML before writing. If validation fails, `EditError` is raised and
the caller reverts via `git checkout --`. The working tree is always clean.

**Why the superset guard on phrase lists?**
The LLM optimizer sometimes returns a "simplified" list that drops existing phrases.
Dropping phrases silently regresses other scenarios. The guard prevents this.

**Why gemma2:9b for the eval judge and Claude Sonnet for critic/optimizer?**
Eval judge runs on every scenario in every eval run — local Ollama keeps costs near
zero and removes API dependency for the inner eval loop. Critic/optimizer calls are
infrequent (once per experiment iteration) and quality matters more there.

**Why git worktrees for grid search?**
Each grid search candidate needs to run eval with different source files. Worktrees
give each candidate a complete isolated copy of the repo at HEAD, apply the patch
there, then run eval with `PYTHONPATH={worktree}` to shadow installed packages.
The main working tree is never touched during grid search.

**Why does the loop commit to a branch, not stash?**
Stash is not resumable across sessions. A named branch (`auto/optimize-{ts}`) gives
a clean audit trail of what the optimizer accepted, and can be merged or discarded
as a unit. The `--no-commit` flag exists for when you want to inspect before committing.

**Why does REVERT_HARD_ERROR skip the noise re-run?**
The noise re-run guard (`_has_suspect_noise`) exists because gemma2:9b judge scores
can fluctuate by ±0.05–0.10 between runs on identical output. A hard agent failure
(score ≤ 0.35, reasoning = "agent provided no response") is not judge variance — the
agent returned nothing. Re-running eval on the same code produces the same failure.
Skipping the re-run saves a full 10-minute eval cycle and avoids a second false signal.

**Why is `tried_fixes` a counter rather than a set?**
The old set API marked every reverted gradient as permanently blocked after one failure.
The counter allows `max_retries_per_fix` (default 2) retries before blocking, which
gives the optimizer a second attempt with a different edit plan before giving up on a
gradient. The counter is serialised as `[scenario_id, fix_type, node, count]` and is
backward-compatible with old 3-element entries (treated as count=2, already maxed).

**Why `calibrate.py` instead of just raising IMPROVEMENT_THRESHOLD?**
Raising the threshold globally would suppress real improvements on stable scenarios.
`calibrate.py` measures variance per scenario so you know *which* scenarios are
reliable judges of improvement. A scenario with std=0.00 (always scores the same) is
a much stronger improvement signal than one with std=0.29 even if their means are equal.

**Why is `COMPOSITE_FAILURE_THRESHOLD` 0.84 and not 0.80?**
`budget-constraint-001` scores 0.830 accuracy=0.60 because SKU `SB-009` is not in
the catalog. No code edit can fix a missing catalog entry. At 0.80 the critic
diagnosed this scenario every session, generating phrase list fixes that triggered
a routing bug in `avalanche-safety-001`. Raising the threshold to 0.84 stops the
critic from targeting an unfixable scenario and lets the loop focus on improvable ones.
