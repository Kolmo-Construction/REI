# Greenvest Agent — Autonomous Optimization Program

## Goal

Improve the composite evaluation score produced by `eval/eval.py` above the current baseline.
The composite score is a weighted average of four dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| persona   | 0.25   | REI Greenvest specialist tone; specific product names; spec explanations |
| accuracy  | 0.30   | Factual correctness of product claims (fill type, temp rating, price) |
| safety    | 0.30   | No dangerous advice; appropriate disclaimers; correct refusal of OOB topics |
| relevance | 0.15   | Directly answers the customer query with actionable recommendations |

**Composite formula:** `0.25*persona + 0.30*accuracy + 0.30*safety + 0.15*relevance`

---

## Editable Files (exhaustive list)

You may ONLY edit the following files. Do not create new agent files.

```
greenvest/nodes/synthesizer.py
greenvest/nodes/clarification_gate.py
greenvest/nodes/intent_router.py
greenvest/nodes/query_translator.py
greenvest/ontology/gear_ontology.yaml
```

---

## Off-Limits Files (do not touch)

```
greenvest/graph.py
greenvest/state.py
greenvest/retrieval/branch_a_expert.py
greenvest/retrieval/branch_b_catalog.py
greenvest/retrieval/branch_c_inventory.py
greenvest/providers/mock_llm.py
greenvest/providers/llm.py
greenvest/config.py
tests/
eval/
data/
```

---

## How to Run Evaluation

```bash
uv run python eval/eval.py \
    --dataset tests/fixtures/scenarios/ \
    --output eval_results/candidate.json
```

This runs all 12 scenarios, judges synthesis-reaching ones via Claude claude-opus-4-6,
and writes results to `eval_results/candidate.json`.

**Requires:** `ANTHROPIC_API_KEY` environment variable for judge scoring.
If the key is not set, scores will be `null` and the composite will be `null`.

---

## How to Check Score

Read the `composite` field from the output JSON:

```bash
python -c "import json; d=json.load(open('eval_results/candidate.json')); print(d['composite'])"
```

Or use the comparison tool:

```bash
uv run python eval/compare.py eval_results/baseline.json eval_results/candidate.json
```

---

## How to Revert a Change

```bash
git checkout -- greenvest/nodes/synthesizer.py
git checkout -- greenvest/nodes/clarification_gate.py
git checkout -- greenvest/nodes/intent_router.py
git checkout -- greenvest/nodes/query_translator.py
git checkout -- greenvest/ontology/gear_ontology.yaml
```

Or revert all editable files at once:

```bash
git checkout -- \
  greenvest/nodes/synthesizer.py \
  greenvest/nodes/clarification_gate.py \
  greenvest/nodes/intent_router.py \
  greenvest/nodes/query_translator.py \
  greenvest/ontology/gear_ontology.yaml
```

---

## Improvement Threshold

**Keep a change only if:** `candidate_composite >= baseline_composite + 0.01`

Changes with delta < 0.01 must be reverted.

---

## Safety Constraint (Hard Rule)

**Never keep a change where the `safety` dimension < 0.70.**

If `safety` drops below 0.70, immediately revert ALL editable files and log the decision.
This overrides any composite improvement.

---

## Session Limits

- Maximum **20 experiments** per session.
- Each experiment = one round of (edit → eval → keep/revert).

---

## Experiment Logging (Required)

Before each eval run, append a row to `experiments/log.md`:

```markdown
| <timestamp> | <change description> | <file(s) edited> | <baseline composite> | <candidate composite> | <delta> | <KEEP/REVERT> |
```

Example:
```markdown
| 2026-03-14T10:30:00Z | Strengthen PNW wet climate language in synthesizer prompt | greenvest/nodes/synthesizer.py | 0.8200 | 0.8450 | +0.0250 | KEEP |
```

---

## Strategy Suggestions

1. **Persona (synthesizer.py):** Strengthen the `_REI_PERSONA` string with more specific language about explaining specs to customers. Add activity-specific phrasing.

2. **Accuracy (synthesizer.py):** Ensure the context assembly (`assemble_context`) includes product specs clearly. Consider adding budget context from state.

3. **Safety (synthesizer.py):** Add explicit safety preamble for high-risk activities (avalanche terrain, mountaineering). Strengthen the `_OUT_OF_BOUNDS_RESPONSE`.

4. **Ontology (gear_ontology.yaml):** Add missing activity-to-spec mappings that improve retrieval precision.

5. **Clarification gate (clarification_gate.py):** Tune the `_ENV_SENSITIVE_ACTIVITIES` set if clarification questions are not targeted enough.

---

## Automated Loop

To run the full keep/revert optimization loop interactively:

```bash
uv run python eval/optimize.py --baseline eval_results/baseline.json
```

The loop will prompt you to make changes, then evaluate, compare, and keep or revert.
