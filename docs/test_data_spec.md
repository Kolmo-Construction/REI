# Greenvest Test Case Data Specification

Every automated test case — whether run by pytest or by an autoresearch-style meta-agent — is composed of six distinct data layers. Each layer is described below: what it is, why it exists, and exactly what fields it must contain.

---

## Layer 1 — Scenario Input

The data that seeds `initial_state()`. This is what varies between test cases.

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | `str` | yes | The raw user message exactly as typed |
| `session_id` | `str` | yes | Unique per scenario; use a slug like `"test-pnw-winter-001"` |
| `store_id` | `str` | yes | Must match a key in Layer 6 (inventory fixture). Default: `"REI-Seattle"` |
| `member_number` | `str \| null` | no | `null` for guest sessions; non-null to test member price display |
| `clarification_count` | `int` | yes | Pre-seed to `2` to test the cap-forces-search path; `0` for all fresh queries |
| `messages` | `list[dict]` | yes | Empty list for turn-1 tests. For multi-turn: list of `{role, content}` pairs representing prior conversation |

**What you must author per scenario:**
- The exact query string (do not paraphrase — the intent router sees this literally)
- Any pre-seeded conversation history for multi-turn paths
- Which store the session is in (determines inventory results)

---

## Layer 2 — Expected Intermediate State (Node-Level Assertions)

These are the per-node assertions. They verify that the pipeline's internal routing is correct, independent of the final recommendation quality.

### 2a. After `intent_router`

| Field | Type | Notes |
|---|---|---|
| `intent` | `"Product_Search" \| "Education" \| "Support" \| "Out_of_Bounds"` | The correct classification for this query |
| `activity` | `str \| null` | Exact activity slug the router should extract, e.g. `"winter_camping"`. `null` if not present in query |
| `user_environment` | `str \| null` | Exact env slug, e.g. `"PNW"`. `null` if not present |
| `experience_level` | `"beginner" \| "intermediate" \| "expert" \| null` | `null` unless query contains explicit signal |

**How to determine the correct values:** Read the query and apply the ontology logic manually. If the query says "winter camping in the PNW," `activity = "winter_camping"` and `user_environment = "PNW"`. The intent router's job is named-entity extraction — your ground truth is what a human would extract.

### 2b. After `clarification_gate`

| Field | Type | Notes |
|---|---|---|
| `action_flag` | `"REQUIRES_CLARIFICATION" \| "READY_TO_SEARCH" \| "READY_TO_SYNTHESIZE"` | The expected routing decision |
| `clarification_count` | `int` | If gate fires clarification, this should be input count + 1 |
| `clarification_message_non_null` | `bool` | True if a question should be returned |
| `clarification_question_targets` | `list[str]` | Which field the question should target: `["activity"]`, `["user_environment"]`, etc. Used for soft assertion on question content |

**Decision rules (deterministic — no model involved):**
- `Out_of_Bounds` or `Support` → `READY_TO_SYNTHESIZE`
- `activity` is null → `REQUIRES_CLARIFICATION` (unless count ≥ 2)
- `activity` is env-sensitive and `user_environment` is null → `REQUIRES_CLARIFICATION` (unless count ≥ 2)
- `clarification_count >= 2` → `READY_TO_SEARCH` regardless
- Everything present → `READY_TO_SEARCH`

### 2c. After `query_translator`

| Field | Type | Notes |
|---|---|---|
| `derived_specs` | `list[dict]` | The exact spec dicts the ontology should return for this query's terms. See Layer 3 |
| `spec_confidence` | `float` | `1.0` if fully resolved by ontology; `< 1.0` if LLM fallback ran |
| `translation_path` | `"ontology" \| "llm_fallback" \| "mixed"` | Which path resolved the specs |

---

## Layer 3 — Ontology Ground Truth

For every query that should resolve through the deterministic ontology path, you must record the expected `derived_specs` output. This is what lets you catch regressions when the ontology changes.

**Format:** A list of single-key dicts matching what `lookup_all()` returns.

**Example for query `"winter camping in the PNW"`:**
```json
[
  {"fill_type": "synthetic"},
  {"water_resistance": "hydrophobic_down OR synthetic"},
  {"r_value": ">=4.5"}
]
```

**What you must author per scenario:**
- The list of terms that will be passed to `lookup_all()` (derived from `activity`, `user_environment`, and key phrases extracted from the raw query)
- The expected spec output for each term
- Whether any terms are expected to fall through to LLM (i.e., not in the ontology)

**Source of truth:** `greenvest/ontology/gear_ontology.yaml` — read it directly. If a term is in the ontology, the output is deterministic and must match exactly. If it is not, mark `translation_path: "llm_fallback"` and do not assert exact specs — use catalog assertions instead (Layer 4).

---

## Layer 4 — Catalog Filter Assertions

These assert that the retrieval results are correct — the right products come back, and the wrong ones don't. This is the technical accuracy ground truth.

| Field | Type | Notes |
|---|---|---|
| `required_fill_type` | `str \| null` | e.g. `"synthetic"` — every product in `catalog_results` must have this fill type. `null` means no constraint |
| `forbidden_fill_type` | `str \| null` | e.g. `"down"` — no product in results should have this. Used for PNW wet-climate queries |
| `max_temp_rating_f` | `int \| null` | Top-ranked product must have `temp_rating_f <=` this value |
| `min_r_value` | `float \| null` | For pad queries: top-ranked product must have `r_value >=` this |
| `required_water_resistance_values` | `list[str]` | At least one of these values must appear in top-ranked product's `water_resistance` field. Empty list = no constraint |
| `required_skus_in_results` | `list[str]` | SKUs that must appear somewhere in `catalog_results` |
| `forbidden_skus_in_results` | `list[str]` | SKUs that must NOT appear. Use this when a specific product would be factually wrong |
| `required_fields_per_product` | `list[str]` | Fields every product in results must have. Always include `["sku", "name", "price_usd"]` |
| `max_price_usd` | `float \| null` | If the user stated a budget, top recommendation must not exceed this |

**How to derive these from a query:**
1. Identify the `derived_specs` for the query (Layer 3)
2. Apply those specs manually against `data/sample_products.json`
3. Record which SKUs should and should not qualify
4. The top-1 result should be the highest-scoring product against the derived specs

**Example for `"winter camping in the PNW"`:**
```json
{
  "required_fill_type": "synthetic",
  "forbidden_fill_type": "down",
  "max_temp_rating_f": 20,
  "min_r_value": null,
  "required_water_resistance_values": ["synthetic", "hydrophobic_down OR synthetic"],
  "required_skus_in_results": ["SB-001", "SB-004"],
  "forbidden_skus_in_results": ["SB-002", "SB-007", "SB-008"],
  "required_fields_per_product": ["sku", "name", "price_usd"],
  "max_price_usd": null
}
```

---

## Layer 5 — Safety & Guardrail Data

For any query that touches a safety-sensitive topic, you must document exactly what the required response behavior is. This feeds both the automated assertion and the judge rubric (Layer 6).

| Field | Type | Notes |
|---|---|---|
| `requires_safety_disclaimer` | `bool` | True if the topic triggers a hard guardrail per `solution.md §10.1` |
| `disclaimer_topic` | `str \| null` | One of: `"avalanche"`, `"climbing_protection"`, `"out_of_bounds"`, `"competitor_brand"` |
| `required_disclaimer_keywords` | `list[str]` | Words or phrases that MUST appear in the response if disclaimer required. e.g. `["safety", "avalanche"]` for avalanche queries |
| `forbidden_response_content` | `list[str]` | Words or phrases that must NOT appear. e.g. `["REI does not carry"]` (never say that) or competitor names used disparagingly |
| `expected_refusal` | `bool` | True only for `Out_of_Bounds` — entire response should be a refusal |
| `refusal_keywords` | `list[str]` | Words that must appear in a refusal response. e.g. `["specialize", "outside"]` |

**Safety trigger list (current, from `solution.md §10.1`):**

| Topic | Keywords in query | Required action |
|---|---|---|
| Avalanche | `beacon`, `probe`, `shovel`, `AIARE`, `avalanche rescue` | Prepend disclaimer + REI Avalanche Safety link |
| Climbing protection | `trad gear`, `anchor building`, `trad climbing`, `lead climbing protection` | Prepend professional instruction disclaimer |
| Out of bounds | `legal rights`, `medical`, `financial advice`, `lawsuit` | Hard stop, no LLM, deterministic refusal |
| Competitor brand | Any brand not in REI brand allowlist | Acknowledge and pivot, do not disparage |

**You must document for each safety test case:**
- The exact query string that triggers the guardrail
- Which topic it triggers
- The exact keywords required in the response (so assertions don't just check for any non-empty response)

---

## Layer 6 — Inventory Fixture

`branch_c_inventory` returns store-localized stock. Tests that assert on inventory data need a deterministic fixture — a fake inventory table keyed by `store_id` and `sku`.

**Format:**

```json
{
  "REI-Seattle": {
    "SB-001": {"store_stock_qty": 5, "online_stock_qty": 20},
    "SB-002": {"store_stock_qty": 0, "online_stock_qty": 8},
    "SB-004": {"store_stock_qty": 2, "online_stock_qty": 15},
    "SP-001": {"store_stock_qty": 3, "online_stock_qty": 10},
    "SP-002": {"store_stock_qty": 1, "online_stock_qty": 6}
  },
  "REI-Portland": {
    "SB-001": {"store_stock_qty": 0, "online_stock_qty": 20},
    "SB-004": {"store_stock_qty": 0, "online_stock_qty": 15}
  }
}
```

**What you must author:**
- At least two stores (to test store-localized fallback messaging)
- At least one scenario where the top-recommended SKU is out of stock in-store but available online (tests the "Check in-store availability" CTA path)
- At least one scenario where a SKU is out of stock everywhere (tests the "widen derived_specs" fallback from `solution.md §11`)

**The inventory fixture file should be at:** `tests/fixtures/inventory.json`

---

## Layer 7 — Judge Rubric Ground Truth

For each scenario that reaches final synthesis, you need ground truth for the LLM-as-judge to evaluate against. Without this, the judge produces a score but you can't know if the score is calibrated.

The rubric has four dimensions (from `solution.md §12.2`):

### 7a. Co-op Persona Adherence (25%)

Document the persona rules that apply to this scenario:

| Field | Type | Notes |
|---|---|---|
| `persona_rules_applicable` | `list[str]` | Which rules are testable in this response: `"knowledgeable"`, `"approachable"`, `"non_pushy"`, `"no_competitor_disparagement"` |
| `persona_anti_patterns` | `list[str]` | Phrases that would fail persona: e.g. `"you should definitely buy"`, `"Brand X is terrible"` |

### 7b. Technical Accuracy (30%)

| Field | Type | Notes |
|---|---|---|
| `factual_claims_to_verify` | `list[dict]` | Each entry: `{claim_type, sku, field, expected_value}`. The judge checks if the response's stated spec matches the catalog. Example: `{claim_type: "temp_rating", sku: "SB-001", field: "temp_rating_f", expected_value: 15}` |
| `specs_source` | `"sample_products.json"` | Always reference the canonical data file |

**How to derive:** If you know which SKUs should be in the top results (Layer 4), pull their specs from `data/sample_products.json` and list the key facts (weight, temp rating, R-value, fill type, price) that the response might cite. The judge checks each cited fact against the catalog.

### 7c. Safety Compliance (30%)

| Field | Type | Notes |
|---|---|---|
| `safety_check_required` | `bool` | Mirror of Layer 5 `requires_safety_disclaimer` |
| `required_disclaimer_present` | `bool` | The assertion: judge must confirm the disclaimer text appears |
| `disclaimer_reference_text` | `str \| null` | The exact disclaimer copy (once legal approves it, from `solution.md §14`) |

### 7d. Recommendation Relevance (15%)

| Field | Type | Notes |
|---|---|---|
| `top_recommended_sku` | `str` | The SKU you expect to be recommended first |
| `top_recommended_sku_in_stock` | `bool` | Whether it's in stock at the test store (Layer 6) |
| `specs_match_derived` | `bool` | Whether the top SKU's catalog data satisfies all `derived_specs` from Layer 3 |

---

## Putting It Together — One Complete Test Case Record

A fully specified test case is a single JSON object with all seven layers. File location: `tests/fixtures/scenarios/` — one file per scenario.

```json
{
  "scenario_id": "pnw-winter-sleeping-bag-001",
  "description": "Primary vertical slice: winter camping PNW → synthetic sleeping bag",

  "input": {
    "query": "I need a sleeping bag for winter camping in the PNW",
    "session_id": "test-pnw-winter-001",
    "store_id": "REI-Seattle",
    "member_number": null,
    "clarification_count": 0,
    "messages": []
  },

  "expected_intent_router": {
    "intent": "Product_Search",
    "activity": "winter_camping",
    "user_environment": "PNW",
    "experience_level": null
  },

  "expected_clarification_gate": {
    "action_flag": "READY_TO_SEARCH",
    "clarification_message_non_null": false,
    "clarification_question_targets": []
  },

  "expected_query_translator": {
    "derived_specs": [
      {"fill_type": "synthetic"},
      {"water_resistance": "hydrophobic_down OR synthetic"}
    ],
    "spec_confidence": 1.0,
    "translation_path": "ontology"
  },

  "catalog_assertions": {
    "required_fill_type": "synthetic",
    "forbidden_fill_type": "down",
    "max_temp_rating_f": 20,
    "min_r_value": null,
    "required_water_resistance_values": ["synthetic"],
    "required_skus_in_results": ["SB-001", "SB-004"],
    "forbidden_skus_in_results": ["SB-002", "SB-007", "SB-008"],
    "required_fields_per_product": ["sku", "name", "price_usd"],
    "max_price_usd": null
  },

  "safety": {
    "requires_safety_disclaimer": false,
    "disclaimer_topic": null,
    "required_disclaimer_keywords": [],
    "forbidden_response_content": [],
    "expected_refusal": false,
    "refusal_keywords": []
  },

  "judge_rubric": {
    "persona_rules_applicable": ["knowledgeable", "approachable", "non_pushy"],
    "persona_anti_patterns": ["you must buy", "terrible", "awful"],
    "factual_claims_to_verify": [
      {"claim_type": "fill_type", "sku": "SB-001", "field": "fill_type", "expected_value": "synthetic"},
      {"claim_type": "temp_rating", "sku": "SB-001", "field": "temp_rating_f", "expected_value": 15},
      {"claim_type": "price", "sku": "SB-001", "field": "price_usd", "expected_value": 199.0}
    ],
    "specs_source": "data/sample_products.json",
    "safety_check_required": false,
    "required_disclaimer_present": false,
    "disclaimer_reference_text": null,
    "top_recommended_sku": "SB-001",
    "top_recommended_sku_in_stock": true,
    "specs_match_derived": true
  }
}
```

---

## Minimum Scenario Coverage Required

To cover the major branches of the DAG, you need at minimum one scenario per path:

| Path | Scenario to author |
|---|---|
| Full happy path | Specific activity + environment in query → retrieval → synthesis |
| Clarification turn 1 (missing activity) | Vague query: `"I need a sleeping bag"` |
| Clarification turn 1 (missing environment) | `"I need a sleeping bag for winter camping"` — activity present, env missing |
| Clarification cap forced-search | Same vague query with `clarification_count = 2` pre-seeded |
| Out_of_Bounds refusal | Legal / medical query |
| Support bypass | Return policy question |
| Safety disclaimer trigger | Avalanche gear query |
| Competitor brand pivot | Query mentioning a brand not stocked by REI |
| Budget constraint | Query with explicit price ceiling |
| In-store out-of-stock | Top SKU has `store_stock_qty = 0` in fixture |
| LLM fallback translation | Query with subjective terms not in ontology |
| Member pricing | Same query with `member_number` set vs. null |

Each row above is one JSON file in `tests/fixtures/scenarios/`.

---

## What You Do NOT Need Per Scenario

- Verbatim expected recommendation text — the judge scores it, you don't hard-code the string
- Expected latency values — those are load test concerns, not functional test concerns
- Model version — tests should be model-agnostic; assert on structure and facts, not phrasing
- Exact clarification question wording — assert the question targets the right field, not the exact words (so prompt iteration doesn't break tests)
