# Greenvest Agent — Production Architecture
### REI Co-op AI Sales Associate

---

## 1. System Architecture Overview

The Greenvest Agent is a compiled, state-driven LangGraph pipeline organized into four discrete layers. Each layer has a hard latency ceiling that feeds into a P95 end-to-end SLA of **≤ 2,000ms** (TTFT ≤ 800ms for streamed responses).

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

---

## 2. End-to-End Latency Budget

Every node in the DAG is assigned a hard ceiling. Exceeding a ceiling triggers the fallback protocol for that node — never a blocking wait.

| Stage | Component | Model / Tech | Latency Ceiling | P95 Target |
|---|---|---|---|---|
| L0 | Exact-match cache | Redis (in-memory) | 10ms | 5ms |
| L0 | Semantic cache lookup | RedisVL + pgvector | 60ms | 40ms |
| L1 | Intent classification + entity extraction | Gemini 2.0 Flash / GPT-4o-mini | 350ms | 250ms |
| L1 | Query translation (subjective → objective) | Gemini 2.0 Flash | 400ms | 300ms |
| L1 | Clarification question generation (if needed) | Gemini 2.0 Flash | 300ms | 200ms |
| L2 | Expert Advice vector search | pgvector / Pinecone | 150ms | 80ms |
| L2 | Product Catalog hybrid search (RRF) | Weaviate / Elasticsearch | 200ms | 120ms |
| L2 | Inventory API call (parameterized) | PostgreSQL read-replica | 150ms | 80ms |
| L3 | Context assembly + compression | Synchronous, in-process | 50ms | 30ms |
| L3 | Final synthesis (TTFT) | Claude Sonnet / GPT-4o | 800ms | 600ms |
| **Total** | **Happy path (cache miss)** | | **≤ 2,000ms** | **≤ 1,400ms** |

**Note:** L2 branches run fully in parallel. The total L2 cost equals `max(Branch A, Branch B, Branch C)`, not their sum.

---

## 3. Model Cascading Strategy

Tasks are routed to the smallest model capable of meeting quality requirements. Frontier model calls are reserved exclusively for final synthesis.

| Component | Model | Rationale |
|---|---|---|
| Intent Router + Entity Extractor | Gemini 2.0 Flash or GPT-4o-mini | Structured JSON output, low latency, cheap per-token cost |
| Query Translator | Gemini 2.0 Flash | Needs REI ontology grounding; SLM alone drifts without fine-tuning |
| Clarification Generator | Gemini 2.0 Flash | Single-sentence output; quality bar is low |
| Final Synthesizer | Claude Sonnet 4.6 or GPT-4o | Persona adherence, nuanced trade-off reasoning, streaming |
| LLM-as-a-Judge (async, offline) | Claude Opus 4.6 | Highest accuracy evaluation; runs off critical path |

---

## 4. State Management & The Clarification Gate

### 4.1 GreenvestState Schema

The FSM does not permit downstream retrieval until `action_flag` is `READY_TO_SEARCH`. All fields have explicit null states — an absent field is not the same as an unknown.

```python
class GreenvestState(TypedDict):
    # Session identity
    session_id: str
    store_id: str                    # REI store location, resolved at session init
    member_number: Optional[str]     # REI Co-op member ID if authenticated

    # Intent & context
    intent: Literal[
        "Product_Search",
        "Education",
        "Support",
        "Out_of_Bounds"
    ]
    activity: Optional[str]          # e.g., "alpine_climbing", "car_camping", "thru_hiking"
    user_environment: Optional[str]  # e.g., "PNW_winter", "desert_summer", "alpine"
    experience_level: Optional[Literal["beginner", "intermediate", "expert"]]
    budget_usd: Optional[tuple[int, int]]  # (min, max) — None means unspecified

    # Derived technical specs (populated by Query Translator)
    derived_specs: list[dict]        # e.g., [{"R_value": ">4.0"}, {"weight_oz": "<48"}]
    spec_confidence: float           # 0.0–1.0; below 0.7 triggers re-clarification

    # Retrieval results
    expert_context: list[str]        # Branch A results
    catalog_results: list[dict]      # Branch B results (post-RRF)
    inventory_snapshot: list[dict]   # Branch C results (store-localized, TTL 60s)

    # Control flow
    action_flag: Literal["REQUIRES_CLARIFICATION", "READY_TO_SEARCH", "READY_TO_SYNTHESIZE"]
    clarification_count: int         # Increment each turn; cap at 2 before forcing synthesis
    messages: list[BaseMessage]      # Compressed conversation history
    compressed_summary: Optional[str]  # Summary of turns > 4 (synchronous, in-path)
```

### 4.2 Clarification Gate Logic

The gate evaluates state completeness before allowing retrieval. Critically, it caps clarification rounds at 2 — a user who won't answer gets a best-effort recommendation rather than an infinite loop.

```
Evaluate GreenvestState
    │
    ├─ intent == Out_of_Bounds ──→ Deterministic refusal ("I specialize in gear...")
    │
    ├─ intent == Support ──→ Bypass retrieval, route to Support KB directly
    │
    ├─ activity IS NULL OR (user_environment IS NULL AND activity requires env) ──→ REQUIRES_CLARIFICATION
    │    └─ clarification_count >= 2 ──→ Override to READY_TO_SEARCH with available data
    │
    └─ derived_specs populated AND spec_confidence >= 0.7 ──→ READY_TO_SEARCH
```

**Single question rule:** Each clarification turn generates exactly one question, targeting the highest-value missing field (activity > environment > experience_level > budget). Budget is never asked first — it signals distrust.

---

## 5. Query Translator — Subjective to Objective Mapping

This is the hardest component in the system. The Query Translator converts natural language descriptions into filterable technical specifications using a **two-stage approach**: a deterministic ontology lookup first, an LLM fallback second.

### 5.1 REI Gear Ontology (Partial — expand with product catalog metadata)

```yaml
sleeping_bags:
  "sleeps cold" / "cold sleeper":
    temp_rating_f: "subtract 10 from stated rating"
    note: "recommend EN-tested ratings, not comfort ratings"
  "backpacking":
    fill_type: ["down", "synthetic"]
    weight_oz: "<32"
  "car camping":
    fill_type: ["synthetic"]  # moisture-tolerant
    weight_oz: unconstrained
  "wet climate" / "PNW" / "coastal":
    fill_type: "synthetic"
    water_resistance: "hydrophobic_down OR synthetic"

sleeping_pads:
  "side sleeper":
    r_value: "+1.0 over activity baseline"
    width_in: ">=25"
  "alpine" / "winter camping":
    r_value: ">=4.5"
  "thru-hiking":
    r_value: ">=2.0"
    weight_oz: "<16"

footwear:
  "waterproof":
    technology: ["GORE-TEX", "eVent", "brand waterproof membrane"]
    note: "clarify use case — waterproof reduces breathability"
```

### 5.2 Translation Pipeline

```
Raw user input
    │
    ├─ Step 1: Tokenize against ontology dictionary (O(n), deterministic, ~5ms)
    │           Matched terms → populate derived_specs directly, confidence = 1.0
    │
    ├─ Step 2: Unmatched residue → Gemini Flash with structured output prompt
    │           Output: { spec_key, operator, value, confidence }[]
    │           confidence < 0.7 on any spec → flag for clarification
    │
    └─ Step 3: Validate derived_specs against catalog metadata schema
                Invalid keys dropped, logged for ontology expansion
```

This prevents hallucinated specs (e.g., `"warmth_level": "high"`) from reaching the catalog query.

---

## 6. Parallel Retrieval Pipeline

### 6.1 Dispatcher

When `action_flag == READY_TO_SEARCH`, the dispatcher fans out all three branches concurrently using `asyncio.gather`. A per-branch timeout enforces the ceiling from §2 — a branch that misses its deadline returns an empty result, never blocks synthesis.

```python
async def dispatch(state: GreenvestState) -> GreenvestState:
    results = await asyncio.gather(
        branch_a_expert_advice(state),
        branch_b_product_catalog(state),
        branch_c_inventory(state),
        return_exceptions=True  # branch failure = empty result, not crash
    )
    # Unpack, log exceptions, merge into state
    ...
```

### 6.2 Branch B: Hybrid Search with Tuned RRF

Reciprocal Rank Fusion merges the dense (semantic) and sparse (BM25/keyword) rankings:

$$RRF(d) = \sum_{r \in R} \frac{w_r}{k + r(d)}$$

**Production tuning:**
- `k = 60` (standard default; reduces sensitivity to top-rank volatility)
- `w_dense = 0.6`, `w_sparse = 0.4` for activity/feature queries ("warm jacket for wet weather")
- `w_dense = 0.2`, `w_sparse = 0.8` for brand/SKU queries ("Patagonia Nano Puff") — exact token match must dominate
- Weights are dynamically selected based on whether the query contains known brand tokens (from an allowlist of REI-stocked brands)

### 6.3 Branch C: Inventory — Parameterized Queries Only

Text-to-SQL against a production database is **prohibited**. All inventory lookups use a predefined query template with parameter injection:

```sql
-- Read-only replica. Connection user has SELECT on inventory_view only.
SELECT
    sku,
    product_name,
    store_stock_qty,
    online_stock_qty,
    price_usd,
    member_price_usd
FROM inventory_view
WHERE
    sku = ANY(:sku_list)          -- populated from Branch B results, max 20
    AND store_id = :store_id      -- from session state, not user input
    AND (store_stock_qty > 0 OR online_stock_qty > 0)
ORDER BY store_stock_qty DESC;
```

**Security constraints:**
- Dedicated read-only PostgreSQL role (`greenvest_inventory_ro`) — no INSERT, UPDATE, DELETE, no raw table access
- Query timeout: 100ms server-side; connection via PgBouncer pool (max 10 connections per pod)
- Result TTL: 60 seconds in Redis; stale reads are acceptable for in-stock signals

---

## 7. Context Assembly & Compression

Context assembly is **synchronous and in-path** — it runs before synthesis and must stay under 50ms. This eliminates the async race condition where an oversized history could reach the synthesizer under load.

### 7.1 History Compression (Synchronous)

```python
def assemble_context(state: GreenvestState) -> str:
    # Compress turns older than 4 exchanges synchronously
    # before building the synthesis prompt
    recent_turns = state["messages"][-8:]  # last 4 exchanges = 8 messages

    if len(state["messages"]) > 8:
        # Use compressed_summary if available; otherwise summarize inline
        # with a fast SLM call (≤100ms budget)
        summary = state.get("compressed_summary") or compress_history(
            state["messages"][:-8]
        )
    else:
        summary = None

    return build_synthesis_prompt(
        summary=summary,
        recent_turns=recent_turns,
        expert_context=state["expert_context"][:3],  # top 3 chunks only
        catalog_results=state["catalog_results"][:5],  # top 5 products
        inventory=state["inventory_snapshot"],
        store_id=state["store_id"],
    )
```

### 7.2 Context Budget

The synthesis prompt must fit within a predictable token budget to protect TTFT:

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

---

## 8. Speculative Execution

If the user's initial message contains a high-confidence brand or SKU signal (matched against the REI brand allowlist), the inventory lookup fires **before** intent classification completes:

```
User message arrives
    │
    ├─ Parallel fork A: Brand token detection (regex, ~2ms)
    │       match found → fire Branch C asynchronously, store future in state
    │
    └─ Parallel fork B: Intent Router (Gemini Flash, ~250ms)
            intent == Product_Search AND Branch C future exists → await future (already running)
            intent != Product_Search → cancel future, discard result
```

This recovers ~150ms on direct product queries (the most common high-intent pattern), which represent an estimated 30–40% of traffic.

---

## 9. Caching Strategy

### Layer 0: API Gateway Cache (pre-LLM)

| Tier | Mechanism | Match Condition | TTL | Hit Rate Target |
|---|---|---|---|---|
| T1: Exact | Redis string key (SHA256 of `session_id + normalized_query`) | Identical query in same session | 5 min | ~15% |
| T2: Semantic | RedisVL ANN search on query embedding | Cosine similarity ≥ 0.92 (same intent_class bucket) | 30 min | ~25% |
| T3: Product card | Redis hash keyed on SKU | Same product referenced across sessions | 2 hr | varies |

T2 semantic matching is **bucketed by intent class** before similarity comparison — a `Product_Search` query never matches an `Education` query even at high cosine similarity.

---

## 10. Safety & Guardrails

### 10.1 Hard Guardrails (Synchronous, Pre-Synthesis)

Certain topics require deterministic handling before the frontier model runs:

| Trigger | Action |
|---|---|
| Avalanche terrain (beacon, probe, shovel, AIARE, rescue) | Prepend mandatory safety disclaimer + link to REI Avalanche Safety guide |
| Climbing protection (trad gear, anchor building) | Prepend instruction-to-seek-professional-instruction disclaimer |
| Out_of_Bounds intent (medical, legal, financial) | Hard stop — deterministic refusal, no LLM call |
| Competitor brand mentioned | Acknowledge, pivot to REI-stocked equivalent without disparaging |

### 10.2 Soft Guardrails (Asynchronous Post-Synthesis)

A guardrail model evaluates the synthesis output after streaming begins. If a violation is detected mid-stream, a correction is appended — the stream is never interrupted (too damaging to UX).

---

## 11. Fallback Protocol

Fallback triggers are ordered by severity:

| Trigger | Fallback Action |
|---|---|
| Synthesis exceeds 1,500ms | Stream pre-cached response for the detected activity + generic top-3 products |
| Branch B returns 0 results | Widen derived_specs (relax one constraint) and retry once; if still 0, offer category browse |
| Branch C timeout (>150ms) | Serve catalog results with "Check in-store availability" CTA; no inventory data shown |
| Safety guardrail tripped | Deterministic response: "Let me connect you with a Greenvest at your local REI — [store_name] — they'll get this exactly right." (store_name resolved from session state) |
| `clarification_count >= 2` with no env data | Best-effort recommendation with explicit uncertainty: "Based on what you've shared, here are strong options — a Greenvest can help dial this in further." |

**The fallback for safety/uncertainty always resolves the local store from `store_id` in session state.** It never says "your local store" without a specific name — vague fallbacks erode trust.

---

## 12. Observability & Governance

### 12.1 Tracing

Every DAG execution emits a structured trace (LangSmith or Phoenix) capturing:
- Node entry/exit timestamps (derive actual vs. budgeted latency per node)
- `GreenvestState` diffs at each transition
- Tool payloads (inputs + outputs) for every retrieval branch
- Model call metadata (tokens in/out, model version, finish reason)

Latency SLA breaches (any node exceeding its ceiling) trigger a PagerDuty alert at P95 measured over 5-minute windows.

### 12.2 LLM-as-a-Judge (Async, Offline)

A background **cron job** (not "chron") samples 5% of production traces and runs an evaluator prompt against the full DAG trace using Claude Opus 4.6:

**Evaluation rubric:**

| Dimension | Description | Weight |
|---|---|---|
| Co-op Persona Adherence | Knowledgeable, approachable, non-pushy; never disparages competitors | 25% |
| Technical Accuracy | Specs cited match product data; R-values, weights, temperatures are correct | 30% |
| Safety Compliance | Mandatory disclaimers present where required | 30% |
| Recommendation Relevance | Top recommendation matches derived_specs and inventory availability | 15% |

Scores below 0.7 on Safety Compliance auto-flag for human review within 1 hour. Scores below 0.6 overall trigger a model rollback review.

### 12.3 Metrics Dashboard (Key Signals)

| Metric | Alert Threshold |
|---|---|
| P95 end-to-end latency | > 2,000ms |
| Cache hit rate (T1+T2) | < 30% (indicates query diversity spike or cache invalidation bug) |
| Clarification rate | > 40% of sessions (indicates ontology gaps or poor entity extraction) |
| Branch C timeout rate | > 5% (indicates DB performance degradation) |
| Safety judge score < 0.7 | > 1% of sampled traces |
| Fallback invocation rate | > 8% of requests |

---

## 13. Deployment Topology

```
Client (REI.com / kiosk / mobile)
    │
    ▼
API Gateway (Kong / AWS API GW)
    ├─ Rate limiting: 10 req/s per session, 1000 req/s global
    ├─ Auth: REI member JWT validation (optional — guest sessions allowed)
    └─ Cache interceptor (Redis L0)
    │
    ▼
Greenvest Agent Service (containerized, horizontally scaled)
    ├─ LangGraph compiled graph (stateless pod; state lives in Redis)
    ├─ Target: p99 < 2,000ms at 500 concurrent sessions
    └─ Auto-scales on CPU + custom metric (LLM queue depth)
    │
    ├──▶ Gemini / OpenAI / Anthropic API (external, via private endpoints)
    ├──▶ Vector DB (Weaviate / Pinecone, dedicated cluster)
    ├──▶ PostgreSQL read-replica (inventory, `greenvest_inventory_ro` role)
    └──▶ Redis cluster (cache + session state)
```

---

## 14. Open Questions / Pre-Launch Checklist

- [ ] **Ontology coverage audit:** Run the last 90 days of REI search logs against the Query Translator ontology to identify top-10 unmatched subjective terms before launch.
- [ ] **Brand allowlist maintenance:** Who owns updates when REI adds/drops a brand? Needs an ops runbook.
- [ ] **Store ID resolution:** Confirm store_id is set at session init via geolocation or user preference — the fallback protocol depends on it.
- [ ] **Member pricing display:** Confirm legal/UX approval for showing `member_price_usd` to non-authenticated sessions (even as a teaser).
- [ ] **Avalanche disclaimer copy:** Legal sign-off required before hardcoding in guardrails.
- [ ] **Load test:** Validate P95 ≤ 2,000ms at 500 concurrent sessions with Branch C hitting the read-replica under realistic query distribution.
- [ ] **Cron job scheduling:** LLM-as-a-Judge evaluation pipeline needs an SLA for how quickly flagged traces reach human reviewers.
