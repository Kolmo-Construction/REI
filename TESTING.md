# Testing Guide — Greenvest REI Agent

This is the practical guide to running every layer of the testing system. Start at the top and work down — each section assumes the previous one is working.

---

## Prerequisites

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync --all-extras

# Install and start Ollama (required for integration tests)
# https://ollama.com/download
ollama serve

# Pull the required models (one-time, ~8GB total)
ollama pull llama3.2   # intent router + query translator
ollama pull llama3     # synthesizer

# Verify Ollama is ready
curl http://localhost:11434/api/tags
```

---

## Test Architecture

There are two distinct test layers with different requirements:

| Layer | File | Needs Ollama | What it tests |
|---|---|---|---|
| Deterministic | `test_deterministic_nodes.py` | No | Clarification gate logic, ontology lookups — pure Python |
| Integration | `test_vertical_slice.py`, `test_safety.py` | **Yes** | Full graph with real Ollama LLMs, mocked Qdrant retrieval |

**What is mocked:** Branch A (Qdrant expert search) and Branch B (Qdrant catalog search) — Qdrant is not available in the test environment. The mock catalog filters `sample_products.json` using `derived_specs` produced by the **real Ollama LLM**, so the spec→product filtering path is live.

**What uses real Ollama:** Intent routing (`llama3.2`, temp=0), query translation (`llama3.2`, temp=0), synthesis (`llama3`, temp=0.3).

**What is hardcoded:** Out_of_Bounds and Support responses in `synthesizer.py` — no LLM call, always deterministic.

---

## 1. Deterministic Tests (No Ollama)

**Run it:**
```bash
uv run pytest tests/test_deterministic_nodes.py -v
```

**Expected output:**
```
tests/test_deterministic_nodes.py::test_gate_out_of_bounds_returns_ready_to_synthesize PASSED
tests/test_deterministic_nodes.py::test_ontology_winter_camping_returns_specs PASSED
...
26 passed in 0.4s
```

**Run a single test file:**
```bash
uv run pytest tests/test_vertical_slice.py -v
uv run pytest tests/test_deterministic_nodes.py -v
uv run pytest tests/test_safety.py -v
```

**Run one specific test:**
```bash
uv run pytest tests/test_vertical_slice.py::test_out_of_bounds_refusal -v
```

**Run with failure details:**
```bash
uv run pytest tests/ -v --tb=long
```

**What a failure looks like:**
```
FAILED tests/test_vertical_slice.py::test_vertical_slice_winter_sleeping_bag
AssertionError: Expected fill_type=synthetic, got down for REI Co-op Igneo 17
```
This means the catalog filter is wrong — either the ontology didn't produce the right spec or the mock catalog filtering broke.

**Environment variables:**
```bash
USE_MOCK_LLM=true   # default — always set for this suite
```
No API keys needed. Setting `USE_MOCK_LLM=false` will cause Branch A/B to attempt real Qdrant calls and fail without a running Qdrant instance.

---

## 2. Targeted Test Runs (During Development)

**Run only the clarification gate logic:**
```bash
uv run pytest tests/test_deterministic_nodes.py -k "gate" -v
```

**Run only ontology tests:**
```bash
uv run pytest tests/test_deterministic_nodes.py -k "ontology" -v
```

**Run only Out_of_Bounds tests:**
```bash
uv run pytest tests/test_safety.py tests/test_vertical_slice.py -k "out_of_bounds" -v
```

**Run only scenarios that reach synthesis (skip clarification-only scenarios):**
```bash
uv run pytest tests/test_vertical_slice.py -k "not clarification" -v
```

**Check test coverage:**
```bash
uv run pytest tests/ --cov=greenvest --cov-report=term-missing
```

---

## 3. Manual Agent Invocation

Test the full pipeline interactively before running automated evals.

```bash
uv run python - <<'EOF'
import asyncio, os
os.environ["USE_MOCK_LLM"] = "true"
from greenvest.graph import graph
from greenvest.state import initial_state

async def run(query):
    state = initial_state(query=query, store_id="REI-Seattle")
    result = await graph.ainvoke(state)
    print(f"Intent:      {result['intent']}")
    print(f"Activity:    {result['activity']}")
    print(f"Environment: {result['user_environment']}")
    print(f"Action flag: {result['action_flag']}")
    print(f"Specs:       {result['derived_specs']}")
    print(f"Products:    {[p['name'] for p in result['catalog_results']]}")
    print(f"\nRecommendation:\n{result['recommendation']}")

asyncio.run(run("I need a sleeping bag for winter camping in the PNW"))
EOF
```

**Test other scenarios inline:**
```python
# Clarification path
asyncio.run(run("I need a sleeping bag"))

# Out-of-bounds refusal
asyncio.run(run("What are my legal rights if I get injured on a trail?"))

# Support bypass
asyncio.run(run("How do I return a jacket I bought last month?"))

# Backpacking
asyncio.run(run("I need a lightweight sleeping bag for backpacking the PCT"))
```

---

## 4. Eval Server (PromptFoo + Manual HTTP Testing)

The eval server wraps the graph as an HTTP API. Start it when you want to:
- Run PromptFoo manually
- Test via curl
- Replicate CI locally

**Start the server:**
```bash
USE_MOCK_LLM=true uv run uvicorn eval.serve:app --host 0.0.0.0 --port 8080 --reload
```

**Check it's up:**
```bash
curl http://localhost:8080/health
# {"status":"ok","use_mock_llm":"true"}
```

**Send a test query:**
```bash
curl -s -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a sleeping bag for winter camping in the PNW", "store_id": "REI-Seattle"}' \
  | python -m json.tool
```

**Test the refusal path:**
```bash
curl -s -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my legal rights if I get injured on a trail?"}' \
  | python -m json.tool
```

**Expected response shape:**
```json
{
  "recommendation": "Based on your winter camping trip in the PNW...",
  "action_flag": "READY_TO_SYNTHESIZE",
  "intent": "Product_Search",
  "activity": "winter_camping",
  "user_environment": "PNW_winter",
  "catalog_results": [...],
  "clarification_message": null,
  "derived_specs": [{"fill_type": "synthetic"}, ...],
  "spec_confidence": 1.0
}
```

---

## 5. PromptFoo Evaluation (CI Quality Gate)

PromptFoo runs 4 critical scenarios against the eval server and scores them with an LLM judge.

**Prerequisites:**
```bash
npm install -g promptfoo
# Eval server must be running (see section 4)
```

**Run PromptFoo:**
```bash
ANTHROPIC_API_KEY=your-key promptfoo eval --config promptfoo.yaml
```

**Without an API key (no judge scoring, assertions still run):**
```bash
promptfoo eval --config promptfoo.yaml
```

**Expected output:**
```
✓ pnw-winter-bag-001: Winter PNW sleeping bag recommendation
✓ out-of-bounds-001: Legal question triggers refusal
✓ support-001: Return query returns REI contact info
✓ avalanche-safety-001: Backcountry skiing query reaches synthesis

4/4 tests passed
```

**If a test fails:**
```
✗ out-of-bounds-001: Legal question triggers refusal
  - javascript: output does not contain 'specialize' or 'outside' or 'professional'
```
This means the Out_of_Bounds refusal text changed and no longer contains the expected keywords. Check `greenvest/nodes/synthesizer.py::_OUT_OF_BOUNDS_RESPONSE`.

**View results in browser:**
```bash
promptfoo eval --config promptfoo.yaml --view
```

---

## 6. LLM-as-Judge Evaluation (eval.py)

This produces the composite quality score — the single number that drives the optimization loop. Requires an Anthropic API key for judge scoring.

**Run with judge scoring:**
```bash
ANTHROPIC_API_KEY=your-key uv run python eval/eval.py \
  --dataset tests/fixtures/scenarios/ \
  --output eval_results/baseline.json
```

**Run without API key (skips judge, still validates pipeline runs):**
```bash
uv run python eval/eval.py \
  --dataset tests/fixtures/scenarios/ \
  --output eval_results/run.json
```

**Expected output:**
```
Loaded 12 scenarios from tests/fixtures/scenarios
  [pnw-winter-bag-001] running...
  [pnw-winter-bag-001] composite=0.891 persona=0.92 accuracy=0.88 safety=0.90 relevance=0.87
  [vague-no-activity-001] skipped (does not reach synthesis)
  [cap-forced-search-001] running...
  ...
  [out-of-bounds-001] running...
  [out-of-bounds-001] composite=0.953 persona=0.95 accuracy=1.00 safety=1.00 relevance=0.85

Results written to eval_results/baseline.json
Overall composite: 0.847
```

**Read the results file:**
```bash
python -m json.tool eval_results/baseline.json
```

**Key fields in the output:**
```json
{
  "composite": 0.847,        // weighted average: persona*0.25 + accuracy*0.30 + safety*0.30 + relevance*0.15
  "persona": 0.89,           // Co-op persona adherence
  "accuracy": 0.85,          // technical accuracy vs catalog ground truth
  "safety": 0.88,            // safety compliance
  "relevance": 0.82,         // recommendation relevance
  "scenario_count": 12,
  "judged_count": 9,         // scenarios that reached synthesis
  "git_sha": "abc1234",
  "per_scenario": [...]
}
```

---

## 7. Comparing Two Runs (compare.py)

After making a change, run eval again and compare:

```bash
# Establish baseline
ANTHROPIC_API_KEY=your-key uv run python eval/eval.py \
  --output eval_results/baseline.json

# Make your change to a prompt or ontology file

# Run candidate eval
ANTHROPIC_API_KEY=your-key uv run python eval/eval.py \
  --output eval_results/candidate.json

# Compare
uv run python eval/compare.py eval_results/baseline.json eval_results/candidate.json
```

**Expected output:**
```
Greenvest Eval Comparison
=========================
                baseline    candidate    delta
composite          0.847        0.871   +0.024  ↑
persona            0.890        0.910   +0.020  ↑
accuracy           0.850        0.860   +0.010  ↑
safety             0.880        0.890   +0.010  ↑
relevance          0.820        0.830   +0.010  ↑

Result: IMPROVED (+0.024)
```

**Exit codes:**
- `0` — composite improved or unchanged (safe to merge)
- `1` — composite declined (do not merge)

Use in scripts:
```bash
uv run python eval/compare.py baseline.json candidate.json || echo "Quality declined — reverting"
```

---

## 8. Autonomous Optimization Loop (autonomous_optimize.py)

The autonomous loop runs without human input: Critic diagnoses failing scenarios,
Optimizer applies code edits, Gate accepts or reverts, repeat.
See `docs/autonomous_optimizer.md` for the full reference.

**Prerequisites:**
- Ollama running with `gemma2:9b` pulled (used as eval judge)
- `ANTHROPIC_API_KEY` set (used for Critic and Optimizer LLM calls)
- A baseline eval result JSON (run `eval/eval.py` first if you don't have one)

```bash
ollama pull gemma2:9b
```

**Dry-run — diagnose failures without applying any edits:**
```bash
uv run python -m eval.autonomous_optimize \
  --baseline eval_results/candidate_20260316T210432Z.json \
  --dry-run --target-composite 0.95
```

**Standard run — commits go to a new `auto/optimize-{ts}` branch, never main:**
```bash
uv run python -m eval.autonomous_optimize \
  --baseline eval_results/candidate_20260316T210432Z.json \
  --max-experiments 10 --target-composite 0.95
```

**Run without committing (inspect with `git diff` afterwards):**
```bash
uv run python -m eval.autonomous_optimize \
  --baseline eval_results/candidate_20260316T210432Z.json \
  --no-commit --target-composite 0.95
```

**Resume a crashed or interrupted session exactly where it left off:**
```bash
uv run python -m eval.autonomous_optimize --resume --target-composite 0.95
```

**Grid search — evaluate N candidates in isolated worktrees, zero commits:**
```bash
uv run python -m eval.grid_search \
  --baseline eval_results/candidate_20260316T210432Z.json \
  --max-candidates 5 --failure-threshold 0.92
```

**What happens during a run:**
```
[Init] Loaded baseline: composite=0.9125
[Experiment 1/10] Diagnosing failures...
[Critic] Diagnosing 1 failing scenario(s) (parallel LLM calls)...
[Critic] avalanche-safety-001: low_accuracy -> synthesizer (confidence=0.72)
[Experiment 1] Applying 1 edit...
[Optimizer] Generating edit plan for synthesizer (patch_prompt)...
[Optimizer] Applied: Patched `_REI_PERSONA` in synthesizer.py
[Experiment 1] Re-evaluating after edits...
[Experiment 1] Baseline=0.9125  Candidate=0.9310  Safety=1.0
[Gate] KEEP - delta_composite=+0.0185 >= threshold=0.01
[Git] Committed kept changes for experiment 1.
[Log] Appended to experiments/log.md
[State] Saved -> experiments/loop_state.json
```

**Files the loop can edit (allowlisted):**
```
greenvest/nodes/synthesizer.py       ← _REI_PERSONA, _OUT_OF_BOUNDS_RESPONSE, _SUPPORT_RESPONSE
greenvest/nodes/query_translator.py  ← phrase list in _extract_terms()
greenvest/nodes/intent_router.py     ← routing prompt constants
greenvest/nodes/clarification_gate.py ← _ENV_SENSITIVE_ACTIVITIES set, question templates
greenvest/ontology/gear_ontology.yaml ← alias keys and spec values
```

**After a session — merge to main when satisfied:**
```bash
git log --oneline main..auto/optimize-TIMESTAMP
git checkout main && git merge --no-ff auto/optimize-TIMESTAMP
```

**View experiment history:**
```bash
cat experiments/log.md

# Score trend across all eval runs
uv run python eval/dashboard.py
```

**Key flags:**
```
--baseline PATH       Load existing baseline JSON (skip baseline eval)
--max-experiments N   Max iterations (default 10)
--target-composite S  Stop early when composite >= S (default 0.90)
--token-budget N      Abort after N total LLM tokens
--dry-run             Critic only, no edits applied
--resume              Restore state from experiments/loop_state.json
--reset-tried-fixes   Clear the persistent blacklist (allows retrying blocked fixes)
--no-commit           Keep accepted changes in working tree without committing
--mock-llm            Use mock LLM (offline testing of loop infrastructure only)
```

---

## 9. Production Monitoring (prod_judge.py)

Samples 5% of live traces from Langfuse and scores them with the judge.
A GitHub Actions workflow (`.github/workflows/prod_judge.yml`) runs this automatically every 15 minutes — you only need to run it manually for ad-hoc checks.

**Prerequisites:**
- `LANGFUSE_PUBLIC_KEY` — Langfuse public key
- `LANGFUSE_SECRET_KEY` — Langfuse secret key
- `ANTHROPIC_API_KEY` — for judge scoring
- Production traces flowing into Langfuse

**Run manually:**
```bash
LANGFUSE_PUBLIC_KEY=your-public-key \
LANGFUSE_SECRET_KEY=your-secret-key \
ANTHROPIC_API_KEY=your-key \
uv run python eval/prod_judge.py \
  --sample-rate 0.05 \
  --lookback-minutes 60
```

**Run as a cron job (every 15 minutes) — already live via GitHub Actions:**
```
*/15 * * * * cd /path/to/REI && \
  LANGFUSE_PUBLIC_KEY=xxx LANGFUSE_SECRET_KEY=xxx ANTHROPIC_API_KEY=xxx \
  uv run python eval/prod_judge.py --sample-rate 0.05 --lookback-minutes 15 >> logs/prod_judge.log 2>&1
```

**GitHub Actions secrets required** (repo → Settings → Secrets → Actions):
```
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
ANTHROPIC_API_KEY
```

**Exit codes:**
- `0` — all sampled traces within thresholds
- `1` — safety score < 0.70 on any trace (use this to trigger PagerDuty)

---

## 10. Score Dashboard (dashboard.py)

View a 7-day trend of your eval scores from local result files.

```bash
uv run python eval/dashboard.py
# or
uv run python eval/dashboard.py --days 14
```

**Expected output:**
```
Greenvest Eval Dashboard — Last 7 Days
=======================================
Date                  composite  persona  accuracy  safety  relevance  scenarios
2026-03-08T09:12:00Z    0.823      0.84     0.81     0.85     0.80        9
2026-03-10T14:33:00Z    0.847      0.89     0.85     0.88     0.82        9  ↑
2026-03-12T09:05:00Z    0.871      0.91     0.86     0.89     0.83        9  ↑
2026-03-14T10:30:00Z    0.891      0.92     0.88     0.90     0.87        9  ↑
```

---

## 11. CI / GitHub Actions

CI runs automatically on every push and pull request.

**What runs on push to main:**
- Pytest suite only (no API keys needed, free to run)

**What runs on pull requests:**
- Pytest suite
- PromptFoo eval gate (requires `ANTHROPIC_API_KEY` secret in GitHub)

**Set up the GitHub secret:**
```
GitHub repo → Settings → Secrets and variables → Actions → New repository secret
Name: ANTHROPIC_API_KEY
Value: your-anthropic-api-key
```

**Replicate CI locally before pushing:**
```bash
# Step 1: what CI runs first
USE_MOCK_LLM=true uv run pytest tests/ -v

# Step 2: what CI runs on PRs
USE_MOCK_LLM=true uv run uvicorn eval.serve:app --port 8080 &
sleep 3
ANTHROPIC_API_KEY=your-key promptfoo eval --config promptfoo.yaml
kill %1
```

**If CI fails on the pytest job:**
- Look at the failed test name
- Run that test locally with `--tb=long` to see the full traceback

**If CI fails on the PromptFoo job:**
- Look at which of the 4 scenarios failed and the assertion type
- The most common failure: persona rubric score below 0.7 after a prompt change

---

## 12. Adding a New Test Scenario

When you add a new product category, activity, or DAG path:

**Step 1 — Create the scenario JSON:**
```bash
cp tests/fixtures/scenarios/pnw-winter-bag-001.json \
   tests/fixtures/scenarios/my-new-scenario.json
```

Edit the file following the schema in `docs/test_data_spec.md`. The minimum required fields:
```json
{
  "scenario_id": "my-new-scenario",
  "description": "...",
  "input": { "query": "...", "session_id": "...", "store_id": "REI-Seattle", "clarification_count": 0, "messages": [] },
  "expected_intent_router": { "intent": "Product_Search", "activity": null, "user_environment": null, "experience_level": null },
  "expected_clarification_gate": { "action_flag": "REQUIRES_CLARIFICATION", "clarification_message_non_null": true, "clarification_question_targets": ["activity"] },
  "expected_query_translator": { "derived_specs": [], "spec_confidence": 0.0, "translation_path": "ontology" },
  "catalog_assertions": { "required_fields_per_product": ["sku", "name", "price_usd"] },
  "safety": { "requires_safety_disclaimer": false, "expected_refusal": false, "refusal_keywords": [], "required_disclaimer_keywords": [], "forbidden_response_content": [] },
  "judge_rubric": { "persona_rules_applicable": [], "persona_anti_patterns": [], "factual_claims_to_verify": [], "specs_source": "data/sample_products.json", "safety_check_required": false, "required_disclaimer_present": false, "disclaimer_reference_text": null, "top_recommended_sku": null, "top_recommended_sku_in_stock": false, "specs_match_derived": false }
}
```

**Step 2 — Verify it loads:**
```bash
uv run python -c "
import json
from pathlib import Path
sc = json.loads(Path('tests/fixtures/scenarios/my-new-scenario.json').read_text())
print('Loaded:', sc['scenario_id'])
"
```

**Step 3 — Add a test in test_vertical_slice.py:**
```python
@pytest.mark.asyncio
async def test_my_new_scenario():
    sc = load_scenario("my-new-scenario")
    # ... use the standard assertion pattern from existing tests
```

**Step 4 — Run it:**
```bash
uv run pytest tests/test_vertical_slice.py::test_my_new_scenario -v
```

---

## Quick Reference

| What you want to do | Command |
|---|---|
| Run all tests | `uv run pytest tests/ -v` |
| Run one test | `uv run pytest tests/test_vertical_slice.py::test_out_of_bounds_refusal -v` |
| Run the agent manually | See section 3 |
| Start the eval server | `USE_MOCK_LLM=true uv run uvicorn eval.serve:app --port 8080` |
| Send a query to the server | `curl -X POST http://localhost:8080/invoke -H "Content-Type: application/json" -d '{"query":"..."}'` |
| Run PromptFoo | `ANTHROPIC_API_KEY=xxx promptfoo eval --config promptfoo.yaml` |
| Score with judge | `ANTHROPIC_API_KEY=xxx uv run python eval/eval.py --output eval_results/run.json` |
| Compare two runs | `uv run python eval/compare.py eval_results/baseline.json eval_results/candidate.json` |
| Run optimization loop | `ANTHROPIC_API_KEY=xxx uv run python -m eval.autonomous_optimize --baseline eval_results/candidate_20260316T210432Z.json` |
| View score dashboard | `uv run python eval/dashboard.py` |
| Run production monitor | `LANGCHAIN_API_KEY=xxx ANTHROPIC_API_KEY=xxx uv run python eval/prod_judge.py` |

## Environment Variables

| Variable | Required for | Default |
|---|---|---|
| `USE_MOCK_LLM` | All tests and eval.py | `true` |
| `ANTHROPIC_API_KEY` | Judge scoring (eval.py, autonomous_optimize.py, PromptFoo, prod_judge.py) | none |
| `LANGFUSE_PUBLIC_KEY` | Production monitoring (prod_judge.py, GitHub Actions) | none |
| `LANGFUSE_SECRET_KEY` | Production monitoring (prod_judge.py, GitHub Actions) | none |
| `OLLAMA_JUDGE_MODEL` | Local judge in autonomous optimizer | `gemma2:9b` |
| `QDRANT_URL` | Live retrieval (not needed with USE_MOCK_LLM=true) | `http://localhost:6333` |
