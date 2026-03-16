# Multi-Agent Simulation (MAS) Testing

## Overview

Standard regression tests verify that a fixed input produces a fixed output. MAS testing goes further: it deploys a **Buyer Agent** (an LLM tuned to a specific buyer archetype) that interacts with the Greenvest Sales Agent across many conversations, measuring whether the agent reliably reaches a good recommendation under realistic variance.

This is sometimes called **Monte Carlo-style evaluation**: a single scenario is played out hundreds of times with slight variation in the buyer's phrasing and mood, and we measure the agent's **Convergence Rate** — how often it successfully closes.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  MAS Runner (asyncio)                │
│                                                      │
│   ┌─────────────┐    query     ┌──────────────────┐ │
│   │ BuyerAgent  │─────────────▶│ Greenvest Graph  │ │
│   │ (llama3.2)  │◀─────────────│ (LangGraph DAG)  │ │
│   │ persona card│  clarif. msg └──────────────────┘ │
│   └─────────────┘                                    │
│          │  ×500 runs (bounded concurrency)          │
│          ▼                                           │
│   ┌─────────────┐                                    │
│   │  RunResult  │  converged, turns, spec_match, ... │
│   └─────────────┘                                    │
└──────────────────────────────────────────────────────┘
```

**BuyerAgent** (`eval/buyer_agent.py`)
- Backed by `llama3.2` via Ollama with `temperature ≈ 0.7`
- Given a persona system prompt (psychographics, budget, constraints)
- Temperature is varied ±0.05 per run to produce phrasing variance

**Greenvest Graph** — unchanged production code (`greenvest/graph.py`)
- `USE_MOCK_LLM=false` so real Ollama inference runs throughout

**MAS Runner** (`eval/mas_runner.py`)
- Orchestrates the conversation loop: buyer sends message → Greenvest responds → buyer answers clarifications → until close or timeout
- Runs all conversations concurrently, bounded by a semaphore

---

## Personas

Five buyer archetypes, each run 100 times = 500 total:

| Persona | Archetype | Required Specs | Budget |
|---|---|---|---|
| `vague_newbie` | First-timer, vague answers, eventually "camping" | `fill_type=synthetic` | $150 |
| `pnw_winter_expert` | Experienced, knows PNW winter, direct | `fill_type=synthetic, temp_rating_f=<=15` | — |
| `budget_constrained` | Car camper, strict $120 ceiling | `fill_type=synthetic` | $120 |
| `technical_expert` | Mountaineer, speaks specs (EN 13537, fill power) | `temp_rating_f=<=15` | — |
| `skeptical_backpacker` | Hesitant, had bad experience, backpacking | `fill_type=down` | — |

Persona definitions live in `eval/personas.py`. Add new personas by appending to the `PERSONAS` list.

---

## Conversation Loop

Each run plays out as follows:

```
Turn 0  Buyer sends initial query (randomly sampled from pool)
         → graph.ainvoke(state)
         → READY_TO_SYNTHESIZE?  ──yes──▶ converged ✓
         → REQUIRES_CLARIFICATION?

Turn 1  BuyerAgent responds to clarification (in-persona, via Ollama)
         → graph.ainvoke(state, clarification_count=1, carry activity/env)
         → READY_TO_SYNTHESIZE?  ──yes──▶ converged ✓
         → REQUIRES_CLARIFICATION?

Turn 2  BuyerAgent responds again
         → graph.ainvoke(state, clarification_count=2)
         → clarification cap hit → forced READY_TO_SEARCH → synthesizer runs
         → converged (forced_forward=True) ✓

Turn 3+ safety valve — MAX_TURNS=4, marks as failed
```

State is carried across turns: `activity`, `user_environment`, and `experience_level` extracted in earlier turns are pre-seeded into the next turn's state so the graph doesn't re-ask for information it already has.

---

## Metrics

### Run-level (one value per run)

| Metric | Type | Definition |
|---|---|---|
| `converged` | bool | Reached `READY_TO_SYNTHESIZE` with a non-empty recommendation |
| `turns` | int | Total buyer↔agent exchanges before close or timeout |
| `clarification_count` | int | How many clarification rounds occurred (0–2) |
| `forced_forward` | bool | Clarification cap (count=2) triggered |
| `spec_match` | bool | `derived_specs` contains all `persona.required_specs` key-value pairs |
| `intent_correct` | bool | `intent_router` returned `persona.expected_intent` |
| `recommendation_len` | int | Character length of final recommendation (proxy for quality) |
| `duration_seconds` | float | Wall-clock time for the run |
| `error` | str\|null | Exception message if the run errored |

### Aggregate (computed per-persona and overall)

| Metric | Definition |
|---|---|
| **Convergence Rate** | `converged.mean()` — the headline number |
| **Mean Turns to Close** | Mean `turns` over converged runs only |
| **Clarification Rate** | % runs with `clarification_count >= 1` |
| **Forced Forward Rate** | % runs where cap triggered |
| **Spec Accuracy** | `spec_match.mean()` |
| **Intent Accuracy** | `intent_correct.mean()` |
| **Error Rate** | % runs that threw an exception |

---

## Output Files

Each run creates a timestamped directory:

```
eval_results/
  mas_run_20260315_120000/
    results.jsonl     ← one JSON object per run (raw data, all fields)
    summary.json      ← aggregate metrics per persona + overall
    report.html       ← standalone Plotly dashboard (open in Chrome)
```

`report.html` is entirely self-contained — no server required.

---

## Report Charts

| Chart | What it shows |
|---|---|
| **Convergence Rate by Persona** | Horizontal bar, headline metric per archetype |
| **Turn Distribution** | Histogram of turns, converged (green) vs. failed (red) |
| **Clarification Funnel** | Stacked bar — proportion of runs with 0/1/2/forced clarifications |
| **Spec Accuracy by Persona** | How reliably the agent matched the persona's technical need |
| **Rolling Convergence Rate** | Line chart over run index (50-run window) — shows stability |
| **Summary Scorecard** | Full metrics table, all personas + overall row |

---

## Running

**Full 500-run evaluation:**
```bash
uv run python scripts/run_mas.py
```

**Quick smoke test (50 runs):**
```bash
uv run python scripts/run_mas.py --runs 50
```

**Higher concurrency (if your GPU allows):**
```bash
uv run python scripts/run_mas.py --runs 500 --concurrency 10
```

**Regenerate report from existing results (no re-simulation):**
```bash
uv run python scripts/run_mas.py --report-only eval_results/mas_run_20260315_120000
```

**Expected runtime** at default concurrency=5:

| Runs | Concurrency | Estimated Time |
|---|---|---|
| 50  | 5 | ~3–5 min |
| 500 | 5 | ~30–45 min |
| 500 | 10 | ~15–25 min |

Times depend on Ollama inference speed. The BuyerAgent and Greenvest both call `llama3.2`; each turn takes roughly 5–10 seconds.

---

## Interpreting Results

**Convergence Rate < 80%** — the agent is failing to close for some personas. Check which personas have the lowest rate. Low convergence on `vague_newbie` is expected (it requires clarification before search); low convergence on `pnw_winter_expert` (who provides everything upfront) indicates a graph routing problem.

**High Forced Forward Rate** — buyers are consistently hitting the clarification cap. This means the clarification gate is asking too many questions, or the buyer's responses aren't giving the gate enough information.

**Low Spec Accuracy** — the query translator is mis-mapping persona needs to specs. Compare with ontology entries in `greenvest/ontology/gear_ontology.yaml`.

**High Rolling Variance** — the convergence line chart shows spiky behavior. This indicates the agent is sensitive to phrasing variation, which could be addressed by tuning the `intent_router` prompt or expanding the ontology.

---

## Adding New Personas

Edit `eval/personas.py` and append a new `Persona` to `PERSONAS`:

```python
Persona(
    name="alpine_climber",
    archetype="Technical alpine climber, weight-obsessed",
    system_prompt=(
        "You are an alpine climber who values lightweight gear above all else. "
        "You are planning a mountaineering expedition. Weight is your primary concern. "
        "When asked about activity, say 'alpine climbing and mountaineering'. "
        "When asked about environment, say 'alpine, 4000m+, cold and wet'. "
    ),
    initial_queries=[
        "I need the lightest possible sleeping bag for alpine climbing",
        "what's your lightest bag for mountaineering?",
    ],
    required_specs={"weight_oz": "<32", "temp_rating_f": "<=15"},
    expected_intent="Product_Search",
    runs=100,
)
```

The runner distributes runs evenly across all personas in `PERSONAS`.
