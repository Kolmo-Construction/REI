# Greenvest Agent — Phase 10: Real LLM via Ollama

## What This Phase Does

Phases 1–9 ran entirely on a mock LLM — deterministic functions that returned hardcoded responses without touching any network or AI model. Phase 10 replaces those mocks with real local inference via [Ollama](https://ollama.com), making the pipeline produce genuine AI-generated intent classification, spec translation, and product recommendations.

No external API keys are required. All inference runs locally on your machine.

---

## Model Selection

Two models are used, each chosen for a different role in the pipeline:

| Role | Model | Why |
|---|---|---|
| Intent Router | `llama3.2:latest` (2.0 GB) | Low latency, strong instruction following for structured JSON output |
| Query Translator | `llama3.2:latest` (2.0 GB) | Same — fast, cheap, handles constrained vocabulary well |
| Synthesizer | `llama3:latest` (4.7 GB) | Stronger reasoning and language quality for the customer-facing recommendation |

**Why split models at all?**

The intent router and query translator are classification tasks with a constrained output format (JSON with a fixed key set). They run at every request, need to be fast, and are evaluated on structural correctness, not prose quality. `llama3.2` handles this well.

The synthesizer produces the final recommendation the customer reads. It needs to maintain the REI Co-op persona, structure a coherent argument for a specific product, and handle the nuance of multiple competing products. `llama3` produces noticeably better output here at the cost of being larger and slower — acceptable because synthesis runs once at the end of the pipeline, not on every node.

**What was rejected and why:**

- `qwen2.5-coder:7b` — code-focused model, not well-suited for natural language recommendation tasks
- `deepseek-coder:6.7b` — same reason
- `nemotron-3-super:cloud` — requires NVIDIA cloud credentials, defeats the no-API-key goal
- All on `llama3` — the extra latency on every routing call isn't justified; `llama3.2` handles structured JSON equally well
- All on `llama3.2` — lower prose quality for the synthesizer; recommendations are noticeably less coherent

---

## Architecture: How the Provider System Works

The LLM provider system uses a single feature flag (`USE_MOCK_LLM`) to switch between mock and real implementations. The key design principle is that **no calling code knows which provider it's using** — the nodes in the LangGraph pipeline call `get_intent_router()`, `get_query_translator()`, and `get_synthesizer()` from `providers/llm.py` and receive callable functions with identical signatures regardless of the flag.

```
greenvest/providers/
├── llm.py          ← Factory: returns mock or Ollama callable based on USE_MOCK_LLM
├── mock_llm.py     ← Phase 1–9: deterministic functions, no network calls
└── ollama_llm.py   ← Phase 10+: real Ollama inference
```

```
USE_MOCK_LLM=true                    USE_MOCK_LLM=false
      │                                      │
      ▼                                      ▼
mock_intent_router()            ollama_intent_router()
mock_query_translator()         ollama_query_translator()
mock_synthesizer()              ollama_synthesizer()
      │                                      │
      └──────────── identical signature ─────┘
                  ↓
         intent_router node
         query_translator node
         synthesizer node
```

This design means tests written against the mock continue to pass unchanged (Phase 9's 7 tests still pass). The real implementation is swapped in only by changing the environment variable.

---

## What Was Built

### `greenvest/providers/ollama_llm.py`

Three functions, each calling Ollama via `langchain-ollama`'s `ChatOllama`:

#### `ollama_intent_router(query: str) -> dict`

Uses `llama3.2:latest` in JSON mode (`format="json"`, `temperature=0`).

Sends a structured prompt that:
1. Defines the four intent classes (`Product_Search`, `Education`, `Support`, `Out_of_Bounds`) with examples
2. Defines the entity fields (`activity`, `user_environment`, `experience_level`) with example values
3. Instructs the model to return `null` for any field it cannot determine

Returns a dict with keys: `intent`, `activity`, `user_environment`, `experience_level`.

**JSON mode** (`format="json"`) forces `llama3.2` to produce valid JSON regardless of its tendency to add preamble or markdown fences. The `_parse_json()` helper also strips markdown fences defensively.

**Temperature 0** is correct here — intent classification should be deterministic. The same query should always produce the same intent.

Example:
```
Input:  "I need a sleeping bag for winter camping in the PNW"
Output: {"intent": "Product_Search", "activity": "winter_camping",
         "user_environment": "PNW_winter", "experience_level": null}
```

#### `ollama_query_translator(state: GreenvestState) -> dict`

Also uses `llama3.2:latest` in JSON mode. Receives the full state and uses `activity`, `user_environment`, `experience_level`, and `query` to produce filterable specs.

The prompt defines:
- The six spec keys (`fill_type`, `temp_rating_f`, `weight_oz`, `r_value`, `water_resistance`) and their value formats
- The `spec_confidence` field (0.0–1.0) and what drives high vs. low confidence
- The constraint that vague queries should return `spec_confidence < 0.7`, which triggers a re-clarification pass

This function is the LLM **fallback** in the query translation pipeline — the ontology runs first (deterministically, at confidence=1.0) and only unmatched residue falls through to this function. In practice, for common queries like "winter camping in the PNW", the ontology hits fully and this function is never called.

#### `ollama_synthesizer(prompt: str) -> str`

Uses `llama3:latest` with `temperature=0.3`. Receives the fully assembled synthesis prompt (built by `synthesizer.py`'s `assemble_context()`) and returns the raw string recommendation.

Temperature 0.3 produces slightly varied phrasing across calls while remaining coherent and factual. Temperature 0 would produce identical recommendations for the same inputs; temperature > 0.5 risks hallucinating product details.

---

### `greenvest/config.py` — New Settings Fields

Three new fields added to the `Settings` dataclass:

```python
OLLAMA_BASE_URL: str        # default: "http://localhost:11434"
OLLAMA_ROUTER_MODEL: str    # default: "llama3.2:latest"
OLLAMA_SYNTHESIZER_MODEL: str  # default: "llama3:latest"
```

All three are overridable via environment variables. The defaults match the chosen configuration (llama3.2 for routing, llama3 for synthesis) so no `.env` file changes are required to run Phase 10 — only `USE_MOCK_LLM=false` needs to be set.

The `ollama_llm.py` reads these via `settings.OLLAMA_ROUTER_MODEL` and `settings.OLLAMA_SYNTHESIZER_MODEL`, meaning the model choice can be changed at runtime without touching code.

---

### `pyproject.toml` — New Dependency

```
langchain-ollama>=0.2.0
```

This is the LangChain integration package for Ollama. It provides the `ChatOllama` class, which wraps the Ollama HTTP API with LangChain's `BaseMessage` interface. The package resolves to `langchain-ollama==1.0.1`, which also installs `ollama==0.6.1` as a transitive dependency (the Ollama Python SDK).

Install: `uv add langchain-ollama`

---

### `.env.example` — New Variables

```env
# Ollama (local, no API key needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_ROUTER_MODEL=llama3.2:latest
OLLAMA_SYNTHESIZER_MODEL=llama3:latest
```

These document the available Ollama settings for anyone setting up the project. Since they match the defaults in `config.py`, they only need to be set if overriding (e.g., using a different Ollama host, or swapping models).

---

## Bug Fixed: Ontology `r_value` in Sleeping Bags

### The Problem

During end-to-end testing with the real LLM and Qdrant, Branch B returned 0 results for the query `"I need a sleeping bag for winter camping in the PNW"` despite the collection having 25 indexed products.

**Root cause:** `gear_ontology.yaml` had `r_value: ">=4.5"` under `sleeping_bags."winter camping"`:

```yaml
sleeping_bags:
  "winter camping / winter camp":
    fill_type: "synthetic"
    temp_rating_f: "<=15"
    r_value: ">=4.5"    ← wrong: r_value is a sleeping PAD spec
```

R-value measures thermal resistance in sleeping **pads**, not bags. In `sample_products.json`, every sleeping bag has `"r_value": null`. The Qdrant filter `r_value >= 4.5` therefore excluded every sleeping bag in the collection — none passed the filter, returning 0 results.

This spec was a copy-paste error from the `sleeping_pads` section of the ontology, where `r_value: ">=4.5"` is correct for `alpine / winter camping`.

**Why it didn't surface during Phase 9:**

Phase 9 tests ran against `branch_b_catalog.py` in its Phase 6 implementation — a flat JSON scorer that iterated over `sample_products.json` directly. The flat scorer applied `r_value: ">=4.5"` as a missing-field tolerance (products with `r_value=null` scored 0 on that spec but were not excluded). The Qdrant `FieldCondition` with `Range(gte=4.5)` is a strict pre-filter: products with `r_value=null` are definitively excluded.

### The Fix

Removed `r_value: ">=4.5"` from `sleeping_bags."winter camping"`:

```yaml
# Before
sleeping_bags:
  "winter camping / winter camp":
    fill_type: "synthetic"
    temp_rating_f: "<=15"
    r_value: ">=4.5"

# After
sleeping_bags:
  "winter camping / winter camp":
    fill_type: "synthetic"
    temp_rating_f: "<=15"
```

The `r_value` spec remains correct under `sleeping_pads."alpine / winter camping"` — that is where it belongs.

**After the fix:** Branch B returned 4 products for the same query, all correctly matching the filter (synthetic fill, temperature rating ≤ 15°F, tagged PNW):

```
SB-006  NEMO Disco 15 Sleeping Bag          $229
SB-004  REI Co-op Magma 0 Sleeping Bag      $249
SB-001  REI Co-op Magma 15 Sleeping Bag     $199
SB-005  Marmot Trestles 15 Sleeping Bag     $159
```

### Lesson: Spec Fields Must Be Category-Specific

The ontology maps customer language to filterable specs. Each spec key must only be included in a category's entry if that field is non-null for products in that category. Cross-category specs silently over-constrain the filter and cause zero results.

A guard to add before Phase 12: an ontology validation script that cross-references every spec key in each category against the product catalog to verify that at least N products would match the filter. Zero-result filters should fail the validation.

---

## End-to-End Pipeline Trace (Phase 10, Real Ollama)

Query: `"I need a sleeping bag for winter camping in the PNW"`

```
[llama3.2] intent_router
  → intent=Product_Search, activity=winter_camping,
    user_environment=PNW_winter, experience_level=null
  → ~0ms (< 1s, model already warm)

clarification_gate
  → activity=winter_camping (not null) ✓
  → user_environment=PNW_winter (not null, env-sensitive activity) ✓
  → decision: READY_TO_SEARCH

query_translator
  → terms extracted: ["winter camping", "PNW", "PNW_winter"]
  → ontology_hits=3 (all matched deterministically)
  → derived_specs:
      [{"fill_type": "synthetic"},
       {"temp_rating_f": "<=15"},
       {"water_resistance": "hydrophobic_down OR synthetic"}]
  → spec_confidence=1.0 (ontology, no LLM call)

retrieval_dispatcher (asyncio.gather)
  ├─ branch_a_expert  → [] (stub)
  ├─ branch_b_catalog → 4 products (Qdrant hybrid RRF, filter active)
  └─ branch_c_inventory → [] (stub)

[llama3] synthesizer
  → recommendation: 1,138 chars
  → recommends SB-001 (REI Co-op Magma 15) with accurate spec reasoning
  → ~14 seconds (llama3 cold inference on CPU)
```

**Key observation:** The query translator hit the ontology for all terms and never called the LLM (`spec_confidence=1.0`). For well-covered queries, Phase 10 only makes two LLM calls: `llama3.2` for intent routing and `llama3` for synthesis. Query translation is free.

---

## How to Run

### Prerequisites

1. Ollama installed and running: `ollama serve`
2. Required models pulled:
   ```bash
   ollama pull llama3.2
   ollama pull llama3
   ```
3. Qdrant running with the catalog indexed:
   ```bash
   docker run -d -p 6333:6333 qdrant/qdrant
   uv run python scripts/index_catalog.py
   ```

### Running the pipeline

```bash
# Real LLM mode
USE_MOCK_LLM=false uv run python -c "
import asyncio
from greenvest.graph import graph
import uuid

async def main():
    state = {
        'session_id': str(uuid.uuid4()),
        'store_id': 'REI-Seattle',
        'member_number': None,
        'query': 'I need a sleeping bag for winter camping in the PNW',
        'intent': None, 'activity': None, 'user_environment': None,
        'experience_level': None, 'budget_usd': None,
        'derived_specs': [], 'spec_confidence': 0.0,
        'expert_context': [], 'catalog_results': [], 'inventory_snapshot': [],
        'action_flag': 'READY_TO_SEARCH', 'clarification_count': 0,
        'clarification_message': None, 'messages': [], 'compressed_summary': None,
        'recommendation': None,
    }
    result = await graph.ainvoke(state)
    for p in result['catalog_results']:
        print(p['sku'], '-', p['name'])
    print()
    print(result['recommendation'])

asyncio.run(main())
"
```

### Switching between mock and real

```bash
USE_MOCK_LLM=true  uv run pytest tests/test_vertical_slice.py -v  # fast, offline
USE_MOCK_LLM=false uv run python ...                               # real inference
```

The test suite is not re-run against the real LLM — Ollama inference introduces latency and non-determinism that breaks the existing deterministic assertions. The mock tests remain the regression baseline.

---

## Latency Expectations (Local CPU)

| Node | Model | Observed |
|---|---|---|
| Intent router | llama3.2 | < 1s (warm) |
| Query translator | llama3.2 | < 1s (ontology hit, no LLM call) |
| Branch B catalog | Qdrant + FastEmbed | ~5s (dense+sparse embedding, cold model load) |
| Synthesizer | llama3 | ~14s (cold CPU inference) |

These are local CPU numbers — far above the production latency budget (≤ 2,000ms P95). In production, this system runs against a cloud LLM (Anthropic Claude Sonnet 4.6 for synthesis, Gemini 2.0 Flash for routing) with GPU-accelerated embedding. Ollama on local CPU is a development-only configuration.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `USE_MOCK_LLM` | `true` | Set to `false` to enable Ollama inference |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_ROUTER_MODEL` | `llama3.2:latest` | Model for intent routing and query translation |
| `OLLAMA_SYNTHESIZER_MODEL` | `llama3:latest` | Model for final recommendation synthesis |

---

## PLAN.md Status After Phase 10

| Phase | Status |
|---|---|
| 10 — Real LLM (Ollama) | ✅ Done |
| 11 — Weaviate / Branch B | ⬜ Qdrant already implemented (Phase 6 was upgraded); this phase is complete |
| 12 — Branch A: Expert Advice | ⬜ Needs vector DB + content ingestion |
| 13 — Branch C: PostgreSQL Inventory | ⬜ Needs DB |
| 14 — Redis Caching | ⬜ Needs Redis |
| 15 — Guardrails | ⬜ Needs core pipeline stable |
| 16 — FastAPI Server + Streaming | ⬜ Needs pipeline complete |
| 17 — Observability: LangSmith | ⬜ Needs server running |
| 18 — LLM-as-a-Judge | ⬜ Needs production traffic |

**Note on Phase 11:** The plan references Weaviate as the Phase 11 target, but Qdrant hybrid search was implemented as part of Phase 6's upgrade (see `branch_b_catalog.py`). The Branch B interface contract was always the same; the backing implementation is already Qdrant. Phase 11 as written is effectively complete.

---

## Files Changed in Phase 10

| File | Change |
|---|---|
| `greenvest/config.py` | Added `OLLAMA_BASE_URL`, `OLLAMA_ROUTER_MODEL`, `OLLAMA_SYNTHESIZER_MODEL` fields |
| `greenvest/providers/ollama_llm.py` | New file — three Ollama-backed callable functions |
| `greenvest/providers/llm.py` | Updated factory to import and return Ollama callables when `USE_MOCK_LLM=false` |
| `pyproject.toml` | Added `langchain-ollama>=0.2.0` dependency |
| `.env.example` | Documented three new `OLLAMA_*` variables |
| `greenvest/ontology/gear_ontology.yaml` | Removed incorrect `r_value: ">=4.5"` from `sleeping_bags.winter_camping` |
