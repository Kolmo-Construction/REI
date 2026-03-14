# Greenvest Agent — Build Plan

## Overview

Vertical slice approach: one thin, testable end-to-end path through the system before adding breadth. All external dependencies (LLMs, Redis, Weaviate, PostgreSQL) are mocked in Phase 1 so the pipeline runs and validates before any credentials or infrastructure are provisioned.

**Vertical slice target:** `"I need a sleeping bag for winter camping in the PNW"` → parsed intent → translated specs → catalog results → streamed recommendation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 0: API Gateway & Cache Interceptor                       │
│  Redis Exact Match → Semantic Cache → Rate Limiter → Auth       │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Orchestration & State (GreenvestState FSM)            │
│  Intent Router → Query Translator → Clarification Gate          │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Parallel Retrieval Dispatcher                         │
│  Branch A: Expert Advice (Vector)                               │
│  Branch B: Product Catalog (Hybrid RRF)                         │
│  Branch C: Inventory API (Parameterized SQL, store-localized)   │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Synthesis & Streaming                                 │
│  Context Assembly → Frontier Model → Guardrail Check → Stream   │
└─────────────────────────────────────────────────────────────────┘
```

### DAG Topology

```
START → intent_router → clarification_gate
    ├─ REQUIRES_CLARIFICATION → END  (clarification_message in state)
    ├─ READY_TO_SYNTHESIZE   → synthesizer → END  (Out_of_Bounds / Support)
    └─ READY_TO_SEARCH       → query_translator → retrieval_dispatcher → synthesizer → END
```

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project scaffold | ✅ Done |
| 2 | State schema | ✅ Done |
| 3 | Mock LLM provider | ✅ Done |
| 4 | Gear ontology | ✅ Done |
| 5 | Layer 1 nodes | ✅ Done |
| 6 | Branch B catalog retrieval | ✅ Done |
| 7 | LangGraph graph wiring | ✅ Done |
| 8 | Context assembly + synthesizer | ✅ Done |
| 9 | End-to-end test | ✅ Done |
| 10 | Real Anthropic API | ⬜ Needs API keys |
| 11 | Weaviate / pgvector — Branch B | ⬜ Needs vector DB |
| 12 | Branch A: expert advice index | ⬜ Needs vector DB |
| 13 | Branch C: PostgreSQL inventory | ⬜ Needs DB |
| 14 | Redis caching layer | ⬜ Needs Redis |
| 15 | Guardrails | ⬜ Needs core pipeline stable |
| 16 | FastAPI server + streaming | ⬜ Needs pipeline complete |
| 17 | Observability: LangSmith tracing | ⬜ Needs server running |
| 18 | LLM-as-a-Judge async eval cron | ⬜ Needs production traffic |

---

## Phase 1 — Project Scaffold ✅

**Goal:** Runnable Python project with zero external dependencies.

```
greenvest/
├── __init__.py
├── config.py               Settings dataclass; USE_MOCK_LLM=True by default
├── state.py
├── graph.py
├── nodes/
│   ├── intent_router.py
│   ├── query_translator.py
│   ├── clarification_gate.py
│   └── synthesizer.py
├── retrieval/
│   ├── branch_b_catalog.py
│   ├── branch_a_expert.py    (stub)
│   └── branch_c_inventory.py (stub)
├── ontology/
│   ├── __init__.py
│   └── gear_ontology.yaml
└── providers/
    ├── llm.py
    └── mock_llm.py
data/
└── sample_products.json
tests/
└── test_vertical_slice.py
```

**Dependencies:** `langgraph`, `langchain-core`, `pydantic`, `httpx`, `structlog`, `pyyaml`, `pytest`, `pytest-asyncio`

---

## Phase 2 — State Schema ✅

`GreenvestState` is the contract for all components. Uses `TypedDict` (required by LangGraph).

```python
class GreenvestState(TypedDict):
    # Session
    session_id: str
    store_id: str
    member_number: Optional[str]

    # Intent & context
    query: str
    intent: Optional[Literal["Product_Search", "Education", "Support", "Out_of_Bounds"]]
    activity: Optional[str]
    user_environment: Optional[str]
    experience_level: Optional[Literal["beginner", "intermediate", "expert"]]
    budget_usd: Optional[tuple]

    # Derived specs
    derived_specs: list           # e.g., [{"fill_type": "synthetic"}, {"r_value": ">=4.5"}]
    spec_confidence: float        # 0.0–1.0; below 0.7 triggers re-clarification

    # Retrieval results
    expert_context: list          # Branch A
    catalog_results: list         # Branch B
    inventory_snapshot: list      # Branch C

    # Control flow
    action_flag: Literal["REQUIRES_CLARIFICATION", "READY_TO_SEARCH", "READY_TO_SYNTHESIZE"]
    clarification_count: int      # Capped at 2 before forcing READY_TO_SEARCH
    clarification_message: Optional[str]
    messages: list
    compressed_summary: Optional[str]

    # Output
    recommendation: Optional[str]
```

---

## Phase 3 — Mock LLM Provider ✅

**`mock_llm.py`** — deterministic, no network calls:
- `mock_intent_router(query)` → classifies intent, extracts activity + environment from query text
- `mock_query_translator(state)` → returns derived specs for known activity/environment combinations
- `mock_synthesizer(prompt)` → returns a static REI-persona recommendation string

**`llm.py`** — single flag, two paths:
```python
def get_intent_router() -> Callable:
    if settings.USE_MOCK_LLM:
        return mock_intent_router
    # Phase 10: wire Gemini Flash / GPT-4o-mini
```

---

## Phase 4 — Gear Ontology ✅

`gear_ontology.yaml` maps subjective terms to filterable specs. Keys use `" / "` to separate aliases.

```yaml
sleeping_bags:
  "winter camping / winter camp":
    fill_type: "synthetic"
    temp_rating_f: "<=15"
    r_value: ">=4.5"
  "wet climate / PNW / coastal / pacific northwest":
    fill_type: "synthetic"
    water_resistance: "hydrophobic_down OR synthetic"
```

`ontology/__init__.py` exposes:
- `lookup(term)` — O(n) scan, returns spec list or `None` (falls through to LLM)
- `lookup_all(terms)` — batched, deduplicates keys

---

## Phase 5 — Layer 1 Nodes ✅

### Intent Router
Calls `llm.get_intent_router()(query)`. Populates `intent`, `activity`, `user_environment`, `experience_level`.

### Clarification Gate
Pure logic, no LLM. Implements the decision tree:

```
Evaluate GreenvestState
    │
    ├─ intent == Out_of_Bounds  → READY_TO_SYNTHESIZE (deterministic refusal)
    ├─ intent == Support        → READY_TO_SYNTHESIZE (bypass retrieval)
    ├─ clarification_count >= 2 → READY_TO_SEARCH (cap enforced)
    ├─ activity IS NULL         → REQUIRES_CLARIFICATION (ask activity first)
    ├─ env-sensitive activity AND user_environment IS NULL → REQUIRES_CLARIFICATION
    └─ otherwise                → READY_TO_SEARCH
```

Single question rule: one question per turn, targeting the highest-value missing field (`activity > environment > experience_level > budget`). Budget is never asked first.

### Query Translator
1. Extract terms from `activity`, `user_environment`, and raw query
2. **Step 1:** ontology lookup (deterministic, confidence = 1.0)
3. **Step 2:** LLM fallback for unmatched residue
4. If `spec_confidence < 0.7` and cap not hit → set `REQUIRES_CLARIFICATION`

---

## Phase 6 — Branch B Catalog Retrieval ✅

Vertical slice uses `data/sample_products.json` (25 fake REI products). Interface matches production signature so Weaviate can be swapped in Phase 11 with no callers changed.

```python
async def search_catalog(state: GreenvestState) -> list[dict]:
    # Score each product against derived_specs
    # Supports: numeric constraints (>=4.5, <32), string match, "A OR B" alternatives
    # Returns top 5 by score
```

Branches A and C are stubbed to return empty lists.

---

## Phase 7 — LangGraph Graph ✅

Compiled DAG. `retrieval_dispatcher` fans out all three branches concurrently:

```python
results = await asyncio.gather(
    search_expert_advice(state),
    search_catalog(state),
    search_inventory(state),
    return_exceptions=True,   # branch failure = empty result, not crash
)
```

Note: `query_translator` runs **before** `retrieval_dispatcher` so `derived_specs` are populated when the catalog scores products.

---

## Phase 8 — Context Assembly + Synthesizer ✅

`assemble_context()` builds the synthesis prompt respecting the token budget:

| Section | Max Tokens |
|---|---|
| System prompt (persona + rules) | 800 |
| Compressed history summary | 400 |
| Recent turns (last 4) | 1,200 |
| Expert advice chunks | 600 |
| Product catalog results | 800 |
| Inventory data | 400 |
| User query | 200 |
| **Total input** | **≤ 4,400** |
| **Reserved for output** | **≤ 600** |

`Out_of_Bounds` and `Support` intents are handled deterministically before the LLM is called.

---

## Phase 9 — End-to-End Test ✅

```
uv run pytest tests/test_vertical_slice.py -v
7 passed in 0.26s
```

Tests covered:
1. Primary vertical slice — winter sleeping bag → recommendation
2. Clarification gate — vague query triggers question on turn 1
3. Clarification cap — `clarification_count=2` pre-set → forced to `READY_TO_SEARCH`
4. Out-of-bounds refusal — deterministic response, no product recommendation
5. Catalog result fields — SKU, name, price always present
6. Ontology lookup — `winter camping + PNW` → `fill_type` and `r_value` without LLM
7. No network calls — `USE_MOCK_LLM=True` confirmed, pipeline completes offline

---

## Phases 10–18 — Roadmap

Each phase swaps a mock for a real implementation. Interfaces are unchanged.

### Phase 10 — Real LLM (unblocked by: API keys)
Wire `greenvest/providers/llm.py` with real clients:
- Intent Router + Query Translator: **Gemini 2.0 Flash** or GPT-4o-mini (structured JSON output, low latency)
- Synthesizer: **Claude Sonnet 4.6** (persona adherence, streaming)
- Set `USE_MOCK_LLM=false` in `.env`

### Phase 11 — Weaviate / Branch B (unblocked by: vector DB)
Replace `branch_b_catalog.py` flat JSON scoring with Weaviate hybrid search (RRF).

**RRF tuning:**
- `k = 60`, `w_dense = 0.6`, `w_sparse = 0.4` for activity/feature queries
- `w_dense = 0.2`, `w_sparse = 0.8` for brand/SKU queries (exact token match dominates)
- Weights dynamically selected by brand token detection against REI brand allowlist

### Phase 12 — Branch A: Expert Advice (unblocked by: vector DB)
Load REI expert content into pgvector / Weaviate. Replace `branch_a_expert.py` stub.
- Top 3 chunks passed to synthesizer (token budget: 600 tokens)

### Phase 13 — Branch C: Inventory (unblocked by: PostgreSQL)
Parameterized query against `inventory_view`. No text-to-SQL — template only:
```sql
SELECT sku, product_name, store_stock_qty, online_stock_qty, price_usd, member_price_usd
FROM inventory_view
WHERE sku = ANY(:sku_list) AND store_id = :store_id
  AND (store_stock_qty > 0 OR online_stock_qty > 0)
```
- Read-only role (`greenvest_inventory_ro`), 100ms server-side timeout, result TTL 60s

### Phase 14 — Redis Caching (unblocked by: Redis)
Three-tier cache intercepted before the LLM pipeline:
- **T1 Exact:** SHA256 of `session_id + normalized_query`, TTL 5 min
- **T2 Semantic:** RedisVL ANN on query embedding, cosine ≥ 0.92, TTL 30 min (bucketed by intent class)
- **T3 Product card:** SKU-keyed hash, TTL 2 hr

### Phase 15 — Guardrails (unblocked by: core pipeline stable)
**Hard (pre-synthesis, synchronous):**
- Avalanche gear → prepend mandatory safety disclaimer + REI Avalanche Safety link
- Climbing protection → prepend professional instruction disclaimer
- `Out_of_Bounds` → deterministic refusal (already implemented in Phase 9)
- Competitor brand → acknowledge, pivot to REI equivalent without disparaging

**Soft (post-synthesis, asynchronous):**
- Guardrail model evaluates output after streaming begins
- Violations appended — stream never interrupted

### Phase 16 — FastAPI Server + Streaming (unblocked by: pipeline complete)
- `POST /chat` endpoint → `astream_events` from LangGraph
- SSE streaming with TTFT ≤ 800ms target
- Session management via Redis (stateless pods)

### Phase 17 — Observability (unblocked by: server running)
LangSmith tracing on every DAG execution:
- Node entry/exit timestamps (actual vs. budgeted latency per node)
- `GreenvestState` diffs at each transition
- Model call metadata (tokens in/out, model version, finish reason)
- PagerDuty alert on any node exceeding its latency ceiling at P95 over 5-minute windows

### Phase 18 — LLM-as-a-Judge (unblocked by: production traffic)
Async cron job sampling 5% of production traces, evaluated with **Claude Opus 4.6**:

| Dimension | Weight |
|---|---|
| Co-op Persona Adherence | 25% |
| Technical Accuracy | 30% |
| Safety Compliance | 30% |
| Recommendation Relevance | 15% |

- Score < 0.7 on Safety → auto-flag for human review within 1 hour
- Score < 0.6 overall → model rollback review

---

## Critical Files

| File | Purpose |
|---|---|
| `solution.md` | Authoritative architecture spec |
| `greenvest/state.py` | Contract for all components — change with care |
| `greenvest/graph.py` | DAG wiring — update when adding nodes |
| `greenvest/ontology/gear_ontology.yaml` | Deterministic term lookup — expand continuously |
| `greenvest/providers/llm.py` | Single point to swap mock → real LLM |
| `data/sample_products.json` | Mock catalog — expand to cover all gear categories |

---

## Latency Budget (P95 targets)

| Stage | Component | Model / Tech | Ceiling | P95 |
|---|---|---|---|---|
| L0 | Exact-match cache | Redis | 10ms | 5ms |
| L0 | Semantic cache | RedisVL + pgvector | 60ms | 40ms |
| L1 | Intent router | Gemini 2.0 Flash | 350ms | 250ms |
| L1 | Query translator | Gemini 2.0 Flash | 400ms | 300ms |
| L1 | Clarification generator | Gemini 2.0 Flash | 300ms | 200ms |
| L2 | Expert advice search | pgvector / Weaviate | 150ms | 80ms |
| L2 | Product catalog search | Weaviate RRF | 200ms | 120ms |
| L2 | Inventory query | PostgreSQL replica | 150ms | 80ms |
| L3 | Context assembly | In-process | 50ms | 30ms |
| L3 | Synthesis (TTFT) | Claude Sonnet 4.6 | 800ms | 600ms |
| **Total** | **Happy path (cache miss)** | | **≤ 2,000ms** | **≤ 1,400ms** |

L2 branches run fully in parallel — total L2 cost is `max(A, B, C)`, not their sum.

---

## Verification Checklist

After Phase 9 (complete):
- [x] `uv run pytest tests/test_vertical_slice.py -v` — 7/7, zero network calls, 0.26s
- [x] Clarification gate test: `"I need a sleeping bag"` → `REQUIRES_CLARIFICATION` on turn 1
- [x] Clarification cap test: same query with `clarification_count=2` → forced to `READY_TO_SEARCH`
- [x] Out-of-bounds refusal: deterministic response, no LLM call

After Phase 16 (server):
- [ ] Load test: P95 ≤ 2,000ms at 500 concurrent sessions
- [ ] Manual trace: inspect each state transition via `graph.ainvoke()`
- [ ] Ontology coverage audit: last 90 days of REI search logs vs. query translator
