# Code Review Guide

Read files in this order — each one builds on the previous.

| # | File | Why |
|---|---|---|
| 1 | `greenvest/state.py` | The data contract everything passes around. Understand this first or nothing else makes sense. |
| 2 | `greenvest/graph.py` | The DAG wiring — shows the full flow and how nodes connect. |
| 3 | `greenvest/nodes/intent_router.py` | Entry point of every request. |
| 4 | `greenvest/nodes/clarification_gate.py` | The decision tree — when to ask, when to search, when to refuse. |
| 5 | `greenvest/nodes/query_translator.py` | How a vague activity becomes filterable specs. |
| 6 | `greenvest/providers/mock_llm.py` | What the LLM does — easier to read the mock first than the Ollama prompts. |
| 7 | `greenvest/retrieval/branch_b_catalog.py` | The only live retrieval branch right now. |
| 8 | `greenvest/nodes/synthesizer.py` | How context is assembled and the final response generated. |
| 9 | `greenvest/ontology/gear_ontology.yaml` | The deterministic term→spec mapping. Short file, high leverage. |
| 10 | `scripts/run_agent.py` | How to drive the whole thing end to end. |

## Skip for now

- `greenvest/retrieval/branch_a_expert.py` — stub, returns empty list until Phase 12
- `greenvest/retrieval/branch_c_inventory.py` — stub, returns empty list until Phase 13
- `greenvest/providers/ollama_llm.py` — only relevant when `USE_MOCK_LLM=false` and Ollama is running locally
