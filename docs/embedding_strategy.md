# Greenvest Agent — Embedding Strategy

## The Fundamental Constraint

**Query vectors and document vectors must come from the same model.**

Cosine similarity only produces meaningful results between vectors that live in the same vector space. A vector from `text-embedding-3-large` and a vector from `text-embedding-3-small` are not comparable — they are two completely different coordinate systems. Even if you truncate `3-large` to 1536 dims and compare it against a native 1536-dim `3-small` vector, the result is numerically undefined. They happen to have the same shape, but they mean completely different things.

This constraint applies to **content retrieval** (Branches A and B): whatever model you use to index documents, you must use to embed queries at search time.

The one exception is the **semantic cache** (Job 3): the cache compares queries against other queries, never queries against documents. Both sides of that comparison use the same model, so a different (smaller/faster) model is safe there.

```
Content index (Branch A + B)          Semantic cache (T2)
─────────────────────────────          ──────────────────────────
Documents → text-embedding-3-large    Past queries → text-embedding-3-small
             at 1536 dims (MRL)                       at 256 dims (MRL)
                  ↕  same model                            ↕  same model
Queries   → text-embedding-3-large    New query   → text-embedding-3-small
             at 1536 dims (MRL)                       at 256 dims (MRL)
```

The speed optimisation comes from **Matryoshka Representation Learning (MRL)**, not from mixing models. MRL lets you call the same model with a smaller `dimensions` parameter and get a high-quality truncation of the full vector — in the same vector space, at a fraction of the cost and latency.

---

## Overview

Three distinct embedding jobs. Each is optimised independently within the model constraint above.

| Job | Where | Latency budget | Optimise for | Model |
|---|---|---|---|---|
| Content indexing | Branch A + B, offline | Unconstrained | Quality | `text-embedding-3-large` @ 1536 dims |
| Retrieval queries | ANN search, request time | < 20ms | Speed (same model as content) | `text-embedding-3-large` @ 1536 dims |
| Semantic cache | T2 cache lookup, every request | < 10ms | Speed | `text-embedding-3-small` @ 256 dims |

---

## Job 1 — Content Embeddings (Offline, Index Time)

### What gets embedded

**Branch B — Product catalog:**
Concatenate structured fields into a single natural-language document per product. Don't embed raw JSON — the model needs natural language context to produce useful vectors.

```python
def build_product_document(product: dict) -> str:
    parts = [product["name"]]
    if product.get("category"):
        parts.append(f"Category: {product['category'].replace('_', ' ')}")
    if product.get("fill_type"):
        parts.append(f"Fill: {product['fill_type']}")
    if product.get("temp_rating_f") is not None:
        parts.append(f"Temperature rating: {product['temp_rating_f']}°F")
    if product.get("weight_oz") is not None:
        parts.append(f"Weight: {product['weight_oz']} oz")
    if product.get("r_value") is not None:
        parts.append(f"R-value: {product['r_value']}")
    if product.get("water_resistance"):
        parts.append(f"Water resistance: {product['water_resistance']}")
    if product.get("tags"):
        parts.append(f"Use cases: {', '.join(product['tags'])}")
    return ". ".join(parts)
```

Example output: `"REI Co-op Magma 15 Sleeping Bag. Category: sleeping bags. Fill: synthetic. Temperature rating: 15°F. Weight: 46 oz. Water resistance: synthetic. Use cases: winter_camping, PNW, wet_climate."`

**Branch A — Expert content:**
Chunk REI gear guides and activity articles into **256–512 token passages** before embedding. Smaller chunks retrieve more precisely; larger chunks provide more context. For gear guides, 384 tokens is a reasonable default — it fits one complete gear recommendation or one "how to choose" section without splitting mid-argument.

Use a sliding window overlap of ~50 tokens so passages at chunk boundaries don't lose surrounding context.

### Recommended model: `text-embedding-3-large` (OpenAI)

- Native Weaviate module: `text2vec-openai` — configure once in schema, Weaviate handles indexing automatically
- Best general-purpose retrieval quality on MTEB benchmarks
- Supports Matryoshka truncation — set `dimensions=1536` in Weaviate to get near-full quality at half the storage
- Cost: $0.13 / 1M tokens — negligible for a catalog of tens of thousands of products

**Strong alternative: `voyage-large-2` (Voyage AI, 1536 dims)**
- Marginally outperforms `text-embedding-3-large` on retrieval-specific MTEB tasks
- No native Weaviate module — requires a custom vectorizer or pre-embedding pipeline, then use `text2vec-none` in Weaviate (bring-your-own-vector mode)
- If you go with Voyage for documents, all retrieval queries must also use Voyage — no OpenAI mixing

**Do not use for content indexing:**
- `text-embedding-3-small` — quality gap matters at index time when there's no latency pressure
- `ada-002` — superseded; worse quality at higher cost than `3-large`
- Local sentence transformers (`all-MiniLM-L6-v2` etc.) — fast and free but noticeably worse on domain-specific retrieval. Acceptable for prototyping only

### Matryoshka dimension tradeoff

`text-embedding-3-large` supports MRL: passing `dimensions=1536` to the API returns a truncated but high-quality vector in the same vector space. This is not lossy compression — it is by design.

| Dimensions | Memory per 50k products | ANN latency | Quality vs 3072 |
|---|---|---|---|
| 3072 | ~600 MB | Baseline | 100% |
| 1536 | ~300 MB | ~30% faster | ~98% |
| 256 | ~50 MB | ~70% faster | ~90% |

**Use 1536 for both content and retrieval queries.** The 2% quality drop from 3072 is not worth the engineering complexity of managing different dimension counts, and 1536 dims comfortably fits the ANN latency budget. Only drop to 3072 if an eval set shows meaningful precision loss.

Configure in Weaviate schema:
```json
{
  "vectorIndexConfig": {
    "distance": "cosine"
  },
  "moduleConfig": {
    "text2vec-openai": {
      "model": "text-embedding-3-large",
      "dimensions": 1536,
      "type": "text"
    }
  }
}
```

---

## Job 2 — Retrieval Query Embeddings (Online, Request Time)

### The model constraint in practice

Because documents are indexed with `text-embedding-3-large` at 1536 dims, every retrieval query must also be embedded with `text-embedding-3-large` at 1536 dims. You cannot use a different model to save latency here — the vector spaces are incompatible.

The speed saving comes from Matryoshka: generating a 1536-dim vector from `3-large` is significantly faster than generating 3072 dims, and Weaviate's `text2vec-openai` module handles this automatically for nearText queries. You don't write a separate embedding call — Weaviate embeds the query for you using the same model and dimension configured in the schema.

### When this runs

Every `READY_TO_SEARCH` request triggers ANN search in Branches A and B. Latency ceiling for the full Branch B call is **200ms** — the embedding step (handled by Weaviate's nearText internally) should cost under 20ms.

### What to embed

Don't embed the raw user query. Embed the enriched query built from state — this narrows the vector space search to semantically relevant products before the ANN index is even touched.

```python
def build_query_document(state: GreenvestState) -> str:
    parts = [state["query"]]
    if state.get("activity"):
        parts.append(state["activity"].replace("_", " "))
    if state.get("user_environment"):
        parts.append(state["user_environment"].replace("_", " "))
    for spec in state.get("derived_specs", []):
        for k, v in spec.items():
            parts.append(f"{k}: {v}")
    return ". ".join(parts)
```

This grounds the embedding in translated specs, not surface phrasing. "I need something warm for the PNW" and "I need a sleeping bag for winter camping in the Pacific Northwest" produce nearly identical vectors after enrichment — even if their raw query embeddings are far apart.

### Weaviate nearText (automatic query embedding)

When using Weaviate's `text2vec-openai` module, you pass the enriched query string directly — Weaviate embeds it server-side using the configured model and dimension, so your query and document vectors are always guaranteed to match:

```python
result = (
    client.query
    .get("Product", ["sku", "name", "fill_type", ...])
    .with_hybrid(
        query=query_doc,   # Weaviate embeds this with text-embedding-3-large @ 1536
        alpha=alpha,
    )
    .with_limit(5)
    .do()
)
```

If you pre-compute query embeddings yourself (e.g., for Voyage or a custom model), use `.with_near_vector({"vector": query_embedding})` instead of `.with_near_text()` — but then you own ensuring the model and dimension match the index.

---

## Job 3 — Semantic Cache Embeddings (Phase 14, T2 Cache)

### Why a different model is safe here

The T2 semantic cache compares **queries against other queries**, not queries against documents. Both sides of the comparison use the same smaller model — the constraint is satisfied, and the incompatibility with the content index is irrelevant.

The T2 cache intercepts requests before the LangGraph pipeline runs. It embeds the incoming query and runs a RedisVL ANN search against previously cached queries. If cosine similarity ≥ 0.92 within the same intent class bucket, it returns the cached recommendation directly — no LLM calls, no retrieval, no Weaviate.

This is the most latency-sensitive embedding job in the system. The entire lookup — embed query + ANN search + retrieve cached response — must complete in **60ms total**.

### Recommended model: `text-embedding-3-small` at 256 dims (MRL)

`text-embedding-3-small` natively supports Matryoshka truncation. Passing `dimensions=256` to the API:
- Embedding call: ~8ms
- RedisVL ANN search on a 256-dim index: ~2ms
- Total: ~10ms, well within the 60ms ceiling

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed_for_cache(text: str) -> list[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=256,   # MRL truncation
    )
    return response.data[0].embedding
```

The quality tradeoff is acceptable — the cache only needs to answer "are these queries roughly the same intent?" not "which product ranks first?" At cosine ≥ 0.92 with 256 dims, it reliably distinguishes "sleeping bag for PNW winter" from "sleeping bag for desert summer."

### Intent class bucketing

Maintain a separate RedisVL index per intent class. Never run a global similarity search across all intents:

```
redis_cache:Product_Search   → ANN index (text-embedding-3-small @ 256 dims)
redis_cache:Education        → ANN index (text-embedding-3-small @ 256 dims)
redis_cache:Support          → ANN index (text-embedding-3-small @ 256 dims)
```

This prevents cross-intent false hits (an `Education` query about sleeping bag insulation should never return a cached `Product_Search` recommendation even at high cosine similarity) and keeps each index small enough for sub-5ms ANN search.

### Cache similarity threshold

**0.92 is the right starting point.** Don't tune this before you have real traffic data:
- Too low (0.85): serves stale recommendations for semantically related but intent-different queries
- Too high (0.97): hit rate drops to near zero, negating the cache's value

After Phase 18 (LLM-as-a-Judge) is running, instrument false positive rate and false negative rate and tune from data.

---

## Model Compatibility Map

```
                  ┌────────────────────────────────────────┐
                  │  CONTENT INDEX (Weaviate)              │
                  │  text-embedding-3-large @ 1536 dims    │
                  └──────────────┬─────────────────────────┘
                                 │ same model, same dims
                                 ▼
                  ┌────────────────────────────────────────┐
                  │  RETRIEVAL QUERIES (Branches A + B)    │
                  │  text-embedding-3-large @ 1536 dims    │
                  │  (Weaviate nearText handles this)      │
                  └────────────────────────────────────────┘

                  ┌────────────────────────────────────────┐
                  │  SEMANTIC CACHE INDEX (RedisVL)        │
                  │  text-embedding-3-small @ 256 dims     │
                  └──────────────┬─────────────────────────┘
                                 │ same model, same dims
                                 ▼
                  ┌────────────────────────────────────────┐
                  │  CACHE LOOKUP QUERIES (T2)             │
                  │  text-embedding-3-small @ 256 dims     │
                  └────────────────────────────────────────┘

  These two groups are completely isolated from each other.
  Cross-group comparison is never performed.
```

---

## Full Model Reference

| Model | Dims | Quality (MTEB) | Latency | Cost / 1M tokens | Weaviate native | Correct use |
|---|---|---|---|---|---|---|
| `text-embedding-3-large` | 1536 (MRL) | ★★★★★ | Medium | $0.13 | ✅ | Content index + retrieval queries |
| `text-embedding-3-large` | 3072 | ★★★★★ | Slow | $0.13 | ✅ | Content index only if eval shows 1536 is insufficient |
| `voyage-large-2` | 1536 | ★★★★★ | Slow | $0.12 | ❌ (bring-your-own-vector) | Content index + retrieval queries (A/B test vs 3-large) |
| `text-embedding-3-small` | 256 (MRL) | ★★★☆☆ | Fastest | $0.02 | ✅ | Semantic cache only |
| `text-embedding-3-small` | 1536 | ★★★★☆ | Fast | $0.02 | ✅ | Only valid if content is also indexed with `3-small` |
| `text-embedding-004` (Google) | 768 | ★★★★☆ | Fast | ~$0.025 | ❌ | Content + retrieval if Google-only stack |
| `all-MiniLM-L6-v2` | 384 | ★★★☆☆ | Fastest (local) | Free | ❌ | Prototyping only |
| `ada-002` | 1536 | ★★★☆☆ | Medium | $0.10 | ✅ | Do not use — superseded |

---

## Domain Adaptation

General-purpose embeddings handle outdoor gear vocabulary well enough for launch — terms like "R-value", "fill power", "crampon-compatible", and "GORE-TEX" appear in their training data.

**Where general embeddings fail and what to do instead:**

| Failure mode | Example | Fix |
|---|---|---|
| Subjective terms don't map to products | "I sleep cold" → no relevant results | Expand `gear_ontology.yaml` — translate to `temp_rating_f: subtract 10` before embedding |
| Colloquial synonyms miss | "bombproof jacket" → no results | Add alias to ontology |
| Brand/model name queries | "Nano Puff" → wrong Patagonia products | Boost sparse weight (`alpha=0.2`) for brand token queries — already specced in RRF tuning |
| Numeric spec queries | "r-value 4.5 or higher sleeping pad" | This is a metadata filter, not a semantic query — apply as a Weaviate `where` filter, not ANN |

The ontology is the first line of defence against embedding failures. Every time a query returns poor results, ask "should this be an ontology entry?" before reaching for embedding fine-tuning.

**When fine-tuning is actually warranted:**
- You have >1,000 labelled (query → relevant product) pairs from real traffic
- The ontology + general embeddings miss a systematic category (e.g., all "feel" descriptors: "packable", "stiff", "crunchy")
- Phase 18 LLM-as-a-Judge scores show Recommendation Relevance consistently below 0.7

Fine-tune on `text-embedding-3-small` first (cheaper to iterate), then validate on the eval set before promoting to `3-large` for the content index. Keep the train/validation split time-based — don't let future queries leak into training data.

---

## Weaviate Schema Design

```python
product_class = {
    "class": "Product",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-large",
            "dimensions": 1536,      # MRL truncation — same dimension used for query nearText
            "vectorizeClassName": False,
        }
    },
    "properties": [
        # Vectorized — semantic value, included in the embedding document
        {"name": "name",        "dataType": ["text"],   "moduleConfig": {"text2vec-openai": {"skip": False}}},
        {"name": "description", "dataType": ["text"],   "moduleConfig": {"text2vec-openai": {"skip": False}}},
        {"name": "tags",        "dataType": ["text[]"], "moduleConfig": {"text2vec-openai": {"skip": False}}},
        # Not vectorized — used as hard filters only, not similarity signals
        {"name": "sku",              "dataType": ["text"],   "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "category",         "dataType": ["text"],   "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "fill_type",        "dataType": ["text"],   "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "temp_rating_f",    "dataType": ["number"], "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "weight_oz",        "dataType": ["number"], "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "r_value",          "dataType": ["number"], "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "price_usd",        "dataType": ["number"], "moduleConfig": {"text2vec-openai": {"skip": True}}},
        {"name": "member_price_usd", "dataType": ["number"], "moduleConfig": {"text2vec-openai": {"skip": True}}},
    ]
}
```

Key decisions:
- **Only vectorize fields with semantic value.** SKU, price, and numeric specs are filters — not meaning. Including them in the embedding dilutes vector quality.
- **Numeric specs as Weaviate properties** enable pre-filtering before ANN search (e.g., `temp_rating_f <= 15`). Pre-filtering is dramatically faster than post-filtering a large ANN result set.
- **`dimensions: 1536` in the schema** ensures Weaviate embeds both documents and nearText queries at 1536 dims automatically — model and dimension are always guaranteed to match.

---

## Hybrid Search Query (Phase 11 Implementation Reference)

```python
async def search_catalog(state: GreenvestState) -> list[dict]:
    query_doc = build_query_document(state)

    # Determine RRF weights based on brand signal
    has_brand_token = _detect_brand_token(state["query"])
    alpha = 0.2 if has_brand_token else 0.6  # 0=sparse only, 1=dense only

    # Numeric specs become hard filters, not similarity signals
    where_filter = _build_where_filter(state.get("derived_specs", []))

    result = (
        client.query
        .get("Product", ["sku", "name", "fill_type", "temp_rating_f",
                         "weight_oz", "r_value", "price_usd", "member_price_usd"])
        .with_hybrid(
            query=query_doc,   # Weaviate nearText embeds this at 1536 dims automatically
            alpha=alpha,
            fusion_type=weaviate.gql.get.HybridFusion.RELATIVE_SCORE,
        )
        .with_where(where_filter)
        .with_limit(5)
        .do()
    )

    return result["data"]["Get"]["Product"]
```

---

## Eval Checklist Before Phase 11 Launch

- [ ] Build 50+ (query → expected top-3 products) pairs covering: activity queries, environment queries, spec queries, brand queries, subjective terms ("I sleep cold", "ultralight", "bombproof")
- [ ] Measure Recall@5: what fraction of expected products appear in the top 5 results
- [ ] Measure MRR (Mean Reciprocal Rank): does the best product rank first?
- [ ] Confirm flat JSON scoring baseline — establish the floor to beat
- [ ] Verify RRF weight switching: brand token queries shift to `alpha=0.2`
- [ ] Verify spec pre-filters don't over-constrain: `temp_rating_f <= 15` should still return 3+ results

**Target:** Recall@5 ≥ 0.85 and MRR ≥ 0.70 before Phase 11 ships to production.
