# Greenvest Agent — Running and Testing

## Overview

The pipeline has two execution modes that coexist without any code changes — only environment variables differ:

| Mode | When to use | LLM calls | Qdrant required | Speed |
|---|---|---|---|---|
| **Mock** (`USE_MOCK_LLM=true`) | Development, CI, regression testing | None (deterministic functions) | No | < 1s |
| **Ollama** (`USE_MOCK_LLM=false`) | Manual exploration, validating real inference | llama3.2 + llama3 locally | Yes | 15–30s |

The test suite always runs in mock mode. Ollama mode is for manual runs only.

---

## Prerequisites

### Python Environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if not already installed
pip install uv

# Install all project dependencies (including dev)
uv sync --dev

# Verify
uv run python -c "import greenvest; print('OK')"
```

### For Mock Mode Only (no further setup needed)

Mock mode requires zero external services. `USE_MOCK_LLM=true` is the default. You can run the full test suite immediately after `uv sync --dev`.

### For Ollama Mode

**1. Install and start Ollama:**
```bash
# Windows: download installer from https://ollama.com
# Then start the server
ollama serve
```

**2. Pull the required models:**
```bash
ollama pull llama3.2   # 2.0 GB — intent router + query translator
ollama pull llama3     # 4.7 GB — synthesizer
```

**3. Start Qdrant:**
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**4. Index the product catalog into Qdrant (one-time):**
```bash
uv run python scripts/index_catalog.py
```

Expected output:
```
Loaded 25 products.
Generating dense embeddings with BAAI/bge-large-en-v1.5 @ 1024 dims...
Dense embeddings done.
Generating sparse embeddings via SPLADE (FastEmbed)...
Sparse embeddings done.
Collection 'rei_products' created with payload indexes.
Done. 25 products indexed into 'rei_products'.
Qdrant dashboard: http://localhost:6333/dashboard
```

This downloads two FastEmbed models on first run (~500 MB total). Subsequent runs use the cached ONNX models.

**5. Verify Qdrant has data:**
```bash
curl http://localhost:6333/collections/rei_products
# Look for: "points_count": 25
```

---

## Running the Test Suite

### The standard command

```bash
uv run pytest tests/test_vertical_slice.py -v
```

Expected output:
```
tests/test_vertical_slice.py::test_vertical_slice_winter_sleeping_bag PASSED
tests/test_vertical_slice.py::test_clarification_gate_needs_activity PASSED
tests/test_vertical_slice.py::test_clarification_cap_forces_search PASSED
tests/test_vertical_slice.py::test_out_of_bounds_refusal PASSED
tests/test_vertical_slice.py::test_catalog_results_have_required_fields PASSED
tests/test_vertical_slice.py::test_ontology_lookup_produces_specs PASSED
tests/test_vertical_slice.py::test_no_network_calls_in_mock_mode PASSED

7 passed in 0.26s
```

All 7 tests run in under a second with zero network calls and zero external services.

### Running a single test

```bash
uv run pytest tests/test_vertical_slice.py::test_vertical_slice_winter_sleeping_bag -v
```

### Running with stdout visible (useful for debugging)

```bash
uv run pytest tests/test_vertical_slice.py -v -s
```

### Running with extra verbosity (shows log output)

```bash
uv run pytest tests/test_vertical_slice.py -v -s --log-cli-level=INFO
```

---

## What Each Test Covers

### `test_vertical_slice_winter_sleeping_bag`

**The primary end-to-end test.** Runs the full pipeline for the canonical query.

```
Input:  "I need a sleeping bag for winter camping in the PNW"

Asserts:
  - intent == "Product_Search"              (intent router classified correctly)
  - action_flag == "READY_TO_SYNTHESIZE"    (pipeline completed, not stuck in clarification)
  - len(catalog_results) > 0               (retrieval returned products)
  - catalog_results[0]["fill_type"] == "synthetic"  (top result matches PNW wet-climate spec)
  - recommendation is not None             (synthesizer produced output)
  - len(recommendation) > 0               (non-empty output)
```

This test is the regression baseline for the whole pipeline. If this fails, something is broken at a fundamental level.

**Why `fill_type == "synthetic"` for the top result?**
The ontology maps both "winter camping" and "PNW" to `fill_type: synthetic`. Down loses loft when wet; the PNW is chronically wet. The retrieval should surface synthetic bags first. If a down bag ranks first, the ontology or filter logic has regressed.

---

### `test_clarification_gate_needs_activity`

**Tests that vague queries trigger clarification rather than a bad recommendation.**

```
Input:  "I need a sleeping bag"  (no activity, no environment)

Asserts:
  - action_flag == "REQUIRES_CLARIFICATION"
  - clarification_message is not None and non-empty
  - clarification_count == 1
```

The clarification gate's single-question rule: when `activity` is null, the gate must ask for activity first (not budget, not experience level — activity is the highest-value missing field for sleeping bag queries). This prevents the synthesizer from guessing and recommending the wrong product.

---

### `test_clarification_cap_forces_search`

**Tests that the system never gets stuck in an infinite clarification loop.**

```
Input:  "I need a sleeping bag"  with  clarification_count=2 pre-set

Asserts:
  - action_flag == "READY_TO_SYNTHESIZE"   (forced to complete, not another clarification)
  - recommendation is not None
```

The clarification cap is 2. On the third vague query from the same customer, the system stops asking questions and produces its best-guess recommendation using whatever context it has. This is a UX safety valve — customers who don't answer clarification questions should still get help.

---

### `test_out_of_bounds_refusal`

**Tests the deterministic refusal path for off-topic queries.**

```
Input:  "What are my legal rights if I get injured on a trail?"

Asserts:
  - intent == "Out_of_Bounds"
  - "specialize" or "outside" in recommendation.lower()
```

Out-of-bounds handling bypasses retrieval entirely. No products are searched, no LLM synthesis is attempted. The refusal message is generated deterministically by the synthesizer's intent-routing logic. The word check (`"specialize"` or `"outside"`) is intentionally loose — it verifies a polite deflection, not an exact string.

Other out-of-bounds triggers in the mock: queries containing `medical`, `legal`, `financial`, `tax`.

---

### `test_catalog_results_have_required_fields`

**Tests the schema contract of catalog results.**

```
Input:  "I need a sleeping bag for winter camping in the PNW"

Asserts (for every product in catalog_results):
  - "sku" in product
  - "name" in product
  - "price_usd" in product
```

This is a contract test. The synthesizer and any downstream UI depend on `sku`, `name`, and `price_usd` always being present. If `branch_b_catalog.py` returns payloads missing these fields (e.g., from a schema change in the Qdrant collection or a change to `sample_products.json`), this test catches it before the synthesizer fails silently with a KeyError.

---

### `test_ontology_lookup_produces_specs`

**Tests the ontology in isolation — no pipeline, no LangGraph.**

```
Input:  lookup_all(["winter camping", "PNW"])

Asserts:
  - len(specs) > 0
  - "fill_type" or "r_value" in spec keys
```

This test exercises the ontology module directly. If `gear_ontology.yaml` is malformed, a key is renamed, or the lookup logic is broken, this test fails before any graph node runs. It serves as a unit test for the deterministic spec translation path.

Note: `r_value` is included in the assertion (`"fill_type" in keys or "r_value" in keys`) because `mock_query_translator` still injects `r_value` into specs for winter camping. This assertion would remain true even if the ontology returned only `fill_type`. The ontology fix from Phase 10 (removing `r_value` from `sleeping_bags.winter_camping`) does not break this test.

---

### `test_no_network_calls_in_mock_mode`

**Smoke test that verifies the pipeline is actually offline.**

```
Asserts:
  - settings.USE_MOCK_LLM is True
  - graph.ainvoke(state) completes without exception
```

In CI environments, there are no credentials and no Ollama server. If any node accidentally makes a network call (e.g., someone wires a real LLM provider without checking the flag), the call will fail with a connection error. This test exists to catch that regression: it confirms `USE_MOCK_LLM` is `True` at runtime, and verifies the pipeline completes — any connection error would surface as an exception.

---

## Running the Pipeline Manually (Ollama Mode)

### Minimal script

```python
# run_pipeline.py
import asyncio
import uuid
from greenvest.graph import graph
from greenvest.state import initial_state

async def main(query: str):
    state = initial_state(query=query, session_id=str(uuid.uuid4()))
    result = await graph.ainvoke(state)

    print(f"\nIntent:    {result['intent']}")
    print(f"Activity:  {result['activity']}")
    print(f"Env:       {result['user_environment']}")
    print(f"Specs:     {result['derived_specs']}")
    print(f"\nCatalog results ({len(result['catalog_results'])}):")
    for p in result["catalog_results"]:
        print(f"  {p['sku']}  {p['name']}  ${p['price_usd']}")
    print(f"\nRecommendation:\n{result['recommendation']}")

asyncio.run(main("I need a sleeping bag for winter camping in the PNW"))
```

```bash
USE_MOCK_LLM=false uv run python run_pipeline.py
```

### Inline one-liner

```bash
USE_MOCK_LLM=false uv run python -c "
import asyncio, uuid
from greenvest.graph import graph
from greenvest.state import initial_state

async def run(q):
    r = await graph.ainvoke(initial_state(query=q, session_id=str(uuid.uuid4())))
    [print(p['sku'], '-', p['name']) for p in r['catalog_results']]
    print('\n' + r['recommendation'])

asyncio.run(run('I need a sleeping bag for winter camping in the PNW'))
"
```

---

## Manual Test Scenarios

These are not automated — they exercise real Ollama inference and verify the system handles varied inputs correctly.

### Scenario 1: Primary happy path (sleeping bag, specific context)

```
Query: "I need a sleeping bag for winter camping in the PNW"

Expected:
  - intent: Product_Search
  - activity: winter_camping
  - user_environment: PNW_winter
  - catalog: 4 products, all synthetic fill, temp_rating ≤ 15°F
  - recommendation: recommends synthetic bag, mentions PNW wet conditions
```

### Scenario 2: Vague query → clarification

```
Query: "I need a sleeping bag"

Expected:
  - action_flag: REQUIRES_CLARIFICATION
  - clarification_message: asks about activity (not budget — activity comes first)
  - recommendation: None (no synthesis until context is gathered)
```

### Scenario 3: Brand query (tests sparse weight switching)

```
Query: "Do you have any Osprey backpacks?"

Expected:
  - intent: Product_Search
  - brand_query detected: True (logged in branch_b_catalog)
  - sparse_weight=2.0, dense_weight=1.0 (brand mode — token match dominates)
  - catalog: BP-002 Osprey Atmos AG 65 surfaces first
```

### Scenario 4: Sleeping pad query (verifies r_value filter)

```
Query: "I need a sleeping pad for winter camping"

Expected:
  - derived_specs includes r_value >= 4.5 (from sleeping_pads ontology)
  - catalog: SP-001, SP-002, SP-003 (all high r_value pads)
  - no sleeping bags in results
```

This is the counterpart to the Phase 10 bug fix: `r_value >= 4.5` is correct and should work for pad queries.

### Scenario 5: Out-of-bounds deflection

```
Query: "What supplements should I take for altitude sickness?"

Expected:
  - intent: Out_of_Bounds
  - recommendation: polite deflection, no product mention
  - catalog_results: [] (retrieval skipped)
```

### Scenario 6: Education intent (no retrieval)

```
Query: "How do I choose between down and synthetic fill?"

Expected:
  - intent: Education
  - action_flag: READY_TO_SYNTHESIZE (bypasses retrieval)
  - recommendation: educational answer, no specific product recommendation
  - catalog_results: [] (retrieval skipped)
```

### Scenario 7: Ultralight backpacking (weight filter)

```
Query: "I need an ultralight sleeping bag for thru-hiking"

Expected:
  - derived_specs includes weight_oz < 20 and fill_type: down
  - catalog: SB-007 (Sea to Summit, 14oz) or SB-008 (Western Mountaineering, 19oz)
  - recommendation: emphasizes pack weight
```

---

## Inspecting State at Each Node

To see the exact state after each node runs, use LangGraph's `astream_events`:

```python
import asyncio, uuid
from greenvest.graph import graph
from greenvest.state import initial_state

async def trace(query: str):
    state = initial_state(query=query, session_id=str(uuid.uuid4()))
    async for event in graph.astream_events(state, version="v2"):
        kind = event["event"]
        name = event.get("name", "")
        if kind == "on_chain_end" and name in (
            "intent_router", "clarification_gate",
            "query_translator", "retrieval_dispatcher", "synthesizer"
        ):
            data = event["data"]["output"]
            print(f"\n{'='*40}")
            print(f"NODE: {name}")
            for k, v in data.items():
                if v:  # skip None/empty
                    print(f"  {k}: {v}")

asyncio.run(trace("I need a sleeping bag for winter camping in the PNW"))
```

```bash
USE_MOCK_LLM=false uv run python trace.py
```

This shows the exact `GreenvestState` diff produced by each node — useful for debugging when a node is setting unexpected values.

---

## Checking the Qdrant Collection

### Via the dashboard

```
http://localhost:6333/dashboard
```

The Qdrant web dashboard shows collections, point counts, and allows manual search queries.

### Via curl

```bash
# Collection info (point count, vector config)
curl http://localhost:6333/collections/rei_products | python -m json.tool

# Count points
curl http://localhost:6333/collections/rei_products/points/count \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool

# Scroll all payloads (see what's stored)
curl http://localhost:6333/collections/rei_products/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 5, "with_payload": true, "with_vector": false}' \
  | python -m json.tool
```

### Verify filters work correctly

This checks that the `temp_rating_f <= 15` filter returns the right products:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

results = client.scroll(
    collection_name="rei_products",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="temp_rating_f",
                range=models.Range(lte=15)
            ),
            models.FieldCondition(
                key="fill_type",
                match=models.MatchValue(value="synthetic")
            ),
        ]
    ),
    with_payload=True,
    with_vectors=False,
    limit=10,
)

for point in results[0]:
    p = point.payload
    print(f"{p['sku']}  {p['name']}  temp={p['temp_rating_f']}°F  fill={p['fill_type']}")
```

Expected: 5 products (SB-001, SB-004, SB-005, SB-006, SB-010) — all synthetic, all rated ≤ 15°F.

---

## Verifying the Ollama Models

### Check models are available

```bash
ollama list
# Should show llama3.2:latest and llama3:latest
```

### Test each model directly

```bash
# Test llama3.2 (router model) — should return valid JSON
ollama run llama3.2 \
  'Return JSON only: {"intent": "Product_Search", "activity": "winter_camping", "user_environment": "PNW_winter", "experience_level": null}'

# Test llama3 (synthesizer model) — should return natural language
ollama run llama3 \
  'You are an REI gear expert. In one sentence, why is synthetic fill better than down for PNW winter camping?'
```

### Test via the provider functions directly

```python
# Test intent router (llama3.2)
from greenvest.providers.ollama_llm import ollama_intent_router
result = ollama_intent_router("I need a sleeping bag for winter camping in the PNW")
print(result)
# Expected: {'intent': 'Product_Search', 'activity': 'winter_camping',
#            'user_environment': 'PNW_winter', 'experience_level': None}

# Test query translator (llama3.2)
from greenvest.providers.ollama_llm import ollama_query_translator
state = {
    "activity": "winter_camping",
    "user_environment": "PNW_winter",
    "experience_level": None,
    "query": "I need a sleeping bag for winter camping in the PNW"
}
result = ollama_query_translator(state)
print(result)
# Expected: {'derived_specs': [...], 'spec_confidence': 0.85+}

# Test synthesizer (llama3)
from greenvest.providers.ollama_llm import ollama_synthesizer
result = ollama_synthesizer(
    "You are an REI gear expert. Recommend the REI Co-op Magma 15 Sleeping Bag "
    "for winter camping in the PNW. Be concise."
)
print(result[:200])
```

```bash
USE_MOCK_LLM=false uv run python test_providers.py
```

---

## Environment Variable Reference

| Variable | Default | Effect |
|---|---|---|
| `USE_MOCK_LLM` | `true` | `true` = mock functions, offline. `false` = Ollama inference |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_ROUTER_MODEL` | `llama3.2:latest` | Model for intent router and query translator |
| `OLLAMA_SYNTHESIZER_MODEL` | `llama3:latest` | Model for synthesizer |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (required in Ollama mode) |

### Setting variables without a `.env` file

```bash
# Inline for a single command
USE_MOCK_LLM=false uv run python ...

# For the session
export USE_MOCK_LLM=false
export OLLAMA_BASE_URL=http://localhost:11434
uv run python ...
```

### Using a `.env` file

```bash
cp .env.example .env
# Edit .env: set USE_MOCK_LLM=false
uv run --env-file .env python ...
```

---

## Common Failures and Fixes

### `uv run pytest` fails with `ModuleNotFoundError`

```bash
uv sync --dev   # re-install dependencies
```

### `Connection refused` on Qdrant (Ollama mode)

```bash
docker ps | grep qdrant   # check if container is running
docker run -d -p 6333:6333 qdrant/qdrant   # restart if needed
```

### `matched=0` from Branch B (Ollama mode, Qdrant running)

1. Check the collection has data: `curl http://localhost:6333/collections/rei_products`
2. If `points_count` is 0, re-run: `uv run python scripts/index_catalog.py`
3. If `points_count` is 25, the filter is over-constraining. Print `derived_specs` from state and verify against the ontology fix (no `r_value` in `sleeping_bags.winter_camping`).

### `ollama: command not found` or Ollama connection error

```bash
ollama serve   # start Ollama server
ollama list    # verify models are pulled
ollama pull llama3.2 && ollama pull llama3   # pull if missing
```

### LangGraph `asyncio` errors in pytest

The project uses `pytest-asyncio` with `asyncio_mode = "auto"` (set in `pyproject.toml`). All async tests are handled automatically. If you see event loop errors, verify:

```bash
uv run pytest --version   # should show pytest 8.x
uv run python -c "import pytest_asyncio; print(pytest_asyncio.__version__)"
```

### JSON parse error from Ollama (intent router or query translator)

`llama3.2` occasionally wraps JSON in markdown fences (` ```json ... ``` `) even in JSON mode. The `_parse_json()` helper in `ollama_llm.py` strips these defensively. If a parse error still occurs:

1. Run the provider function directly and print the raw response:
   ```python
   from langchain_ollama import ChatOllama
   llm = ChatOllama(model="llama3.2:latest", format="json", temperature=0)
   response = llm.invoke("Your prompt here")
   print(repr(response.content))
   ```
2. If the model consistently fails to produce valid JSON, switch `OLLAMA_ROUTER_MODEL` to `llama3:latest` — it's more reliable but slower.

---

## CI Integration Notes

The test suite (`USE_MOCK_LLM=true`) is CI-safe: no services, no credentials, no network calls, deterministic output. The recommended CI command:

```bash
uv sync --dev && uv run pytest tests/test_vertical_slice.py -v
```

Do not run with `USE_MOCK_LLM=false` in CI — Ollama is not available in typical CI environments and the test assertions assume deterministic mock behavior.

---

## Autonomous Optimization Loop

For running and operating the autonomous LLM-driven optimization pipeline, see:

**`docs/autonomous_optimizer.md`** — Full reference covering architecture, CLI flags, configuration, observability gaps, and active blockers.

**Quick start:**
```bash
# Dry run — diagnose failures without applying edits
ANTHROPIC_API_KEY=xxx uv run python -m eval.autonomous_optimize \
  --baseline eval_results/candidate_20260316T210432Z.json \
  --dry-run

# Standard run — commits to auto/optimize-{ts} branch
ANTHROPIC_API_KEY=xxx uv run python -m eval.autonomous_optimize \
  --baseline eval_results/candidate_20260316T210432Z.json \
  --max-experiments 10 --target-composite 0.95
```

See **`TESTING.md` Section 8** for the full command reference including grid search, resume, and post-session merge.
