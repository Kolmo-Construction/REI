# Optimizer Stack: DSPy + Aider vs. Custom Implementation

## What We Built and Why

The autonomous optimizer in `eval/autonomous_optimize.py` is a custom implementation of a pattern the research community calls **automatic prompt optimization with textual gradients**. It was built from scratch because:

1. No existing framework handles edits to non-prompt artifacts (YAML ontologies, Python phrase lists, set literals)
2. The eval harness (`eval.py`) is domain-specific and couldn't be replaced by a generic framework
3. The git-safety requirements (AST validation, atomic writes, branch isolation) needed explicit control

This document describes how the same system could be rebuilt on top of **DSPy** (optimizer loop) and **Aider** (edit engine), what that would look like in practice, and a concrete path forward.

---

## Current Architecture → Framework Mapping

| Custom component | Framework equivalent | Notes |
|---|---|---|
| `eval/critic.py` | DSPy `MIPROv2` internal proposer | DSPy generates candidate prompt rewrites; Critic generates structured diagnoses |
| `eval/optimizer_agent.py` | DSPy optimizer + Aider edit engine | DSPy decides *what* to change; Aider applies it safely |
| `eval/autonomous_optimize.py` loop | DSPy `teleprompter.compile()` | The keep/revert/retry loop |
| `eval/edit_tools.py` | Aider `Coder` headless mode | AST validation, git commit, revert on failure |
| `eval/eval.py` metric | DSPy `Metric` function | Always custom — no framework provides domain metrics |
| `experiments/tried_fixes.json` | DSPy's internal candidate deduplication | DSPy tracks what it has tried across compile runs |
| Gate `REVERT_STRUCTURAL` | DSPy's built-in regression guard | DSPy re-evaluates on multiple bootstrap samples before accepting |

**What has no framework equivalent (always custom):**
- `eval/eval.py` — the composite metric and judge rubrics
- `tests/fixtures/scenarios/` — the scenario dataset
- The `patch_set_literal` / `update_ontology` edit types — DSPy only optimizes prompts

---

## The Target Stack

```
┌─────────────────────────────────────────────────────────┐
│                    DSPy Optimizer Loop                   │
│  MIPROv2 / COPRO: proposes changes, tracks candidates,  │
│  handles resume, dedup, variance-aware evaluation        │
└──────────────┬──────────────────────────────────────────┘
               │ "change synthesizer prompt to X"
               ▼
┌─────────────────────────────────────────────────────────┐
│                   Adapter Layer (thin)                   │
│  Translates DSPy's prompt rewrite instructions into     │
│  Aider edit commands targeting specific files/symbols   │
└──────────────┬──────────────────────────────────────────┘
               │ file path + edit instruction
               ▼
┌─────────────────────────────────────────────────────────┐
│              Aider Edit Engine (headless)                │
│  Applies edits with AST validation, git commit,         │
│  auto-revert on syntax failure                          │
└──────────────┬──────────────────────────────────────────┘
               │ git SHA of applied change
               ▼
┌─────────────────────────────────────────────────────────┐
│              eval.py (always custom)                     │
│  Runs scenarios, scores with judge, returns composite   │
│  Langfuse traces logged per run                         │
└─────────────────────────────────────────────────────────┘
```

---

## DSPy: The Optimizer Loop

### What it replaces

DSPy's `MIPROv2` optimizer replaces the entire `autonomous_optimize.py` loop:
- Critic diagnosis → DSPy's internal "instruction proposer" (uses an LLM to propose rewritten instructions)
- Keep/revert logic → DSPy evaluates candidates on a bootstrap sample before accepting
- Tried-fixes deduplication → DSPy tracks candidates internally
- Judge variance guard → MIPROv2 uses multiple eval samples by default
- Token budget → `max_bootstrapped_demos` and `max_labeled_demos` control LLM call volume

### What it looks like

```python
import dspy
from dspy.teleprompt import MIPROv2

# 1. Define your pipeline as DSPy modules
class SynthesizerModule(dspy.Module):
    def __init__(self):
        self.synthesize = dspy.ChainOfThought("query, catalog_results -> recommendation")

    def forward(self, query, catalog_results):
        return self.synthesize(query=query, catalog_results=catalog_results)

# 2. Wrap your eval.py metric as a DSPy metric
def composite_metric(example, prediction, trace=None) -> float:
    from eval.eval import judge_recommendation
    scores = asyncio.run(judge_recommendation(
        query=example.query,
        recommendation=prediction.recommendation,
        ground_truth=example.ground_truth,
    ))
    return scores.composite

# 3. Load your scenarios as a DSPy dataset
trainset = [
    dspy.Example(
        query=sc["input"]["query"],
        ground_truth=sc["judge_rubric"],
    ).with_inputs("query")
    for sc in load_scenarios("tests/fixtures/scenarios/")
]

# 4. Run the optimizer
teleprompter = MIPROv2(
    metric=composite_metric,
    auto="medium",          # controls how many candidates to try
    num_threads=4,
)
optimized_pipeline = teleprompter.compile(
    SynthesizerModule(),
    trainset=trainset,
    num_trials=10,
    minibatch_size=5,
)

# 5. Save the optimized prompts
optimized_pipeline.save("experiments/dspy_optimized.json")
```

### DSPy limitations for this project

1. **Only optimizes prompts and few-shot examples** — cannot touch `gear_ontology.yaml`, phrase lists, or `_ENV_SENSITIVE_ACTIVITIES`
2. **Assumes LangChain-compatible LLMs** — requires a DSPy LM adapter for Ollama (exists, but adds a dependency)
3. **Black-box optimization** — doesn't produce structured diagnoses like the Critic; harder to understand *why* a candidate was accepted
4. **No file-level edit safety** — DSPy rewrites in-memory prompt strings, not files on disk; requires separate git management

---

## Aider: The Edit Engine

### What it replaces

Aider's headless `Coder` replaces `edit_tools.py`:
- Diff application with AST/syntax validation before committing
- Auto-git-commit of accepted changes
- Revert on failure (Aider uses git to restore state)
- Multiple edit formats (search/replace is the most reliable for targeted edits)

### Installation

```bash
uv add aider-chat
```

### Headless usage pattern

```python
import os
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

def apply_edit(file_path: str, instruction: str, model: str = "claude-sonnet-4-6") -> bool:
    """
    Apply a natural-language edit instruction to a file.
    Returns True if the edit was applied and committed, False if it failed.
    """
    io = InputOutput(
        yes=True,           # non-interactive: auto-accept all prompts
        chat_history_file=None,
    )
    coder = Coder.create(
        main_model=Model(model),
        fnames=[file_path],
        io=io,
        auto_commits=True,  # commit successful edits
        dirty_commits=False,
        git=True,
    )
    try:
        coder.run(instruction)
        return True
    except Exception:
        # Aider will have reverted via git if it failed mid-edit
        return False
```

### Example: rewriting a prompt constant

```python
success = apply_edit(
    file_path="greenvest/nodes/synthesizer.py",
    instruction=(
        "In the _REI_PERSONA constant, add a sentence emphasizing that "
        "the assistant always recommends checking current trail conditions "
        "before backcountry skiing trips."
    )
)
```

Aider handles:
- Finding `_REI_PERSONA` in the file
- Generating the edit in search/replace format
- Validating Python syntax before writing
- `git commit` with a descriptive message

### Aider limitations for this project

1. **Requires an API key for the edit LLM** — Aider drives an LLM to generate the diff; you pay per edit. Mitigate by pointing it at your local Ollama with `Model("ollama/qwen2.5-coder:7b")`.
2. **Edit quality depends on instruction clarity** — vague instructions produce wrong edits. The Critic's structured diagnosis (`fix_type`, `target_node`, `diagnosis`) maps cleanly to precise instructions, which helps.
3. **Slower than direct string manipulation** — adds one LLM round-trip per edit. Acceptable for the optimizer loop where each iteration already takes minutes.
4. **YAML editing is less reliable** — Aider works best on Python. For `gear_ontology.yaml` edits, the custom `update_ontology()` in `edit_tools.py` is more reliable.

---

## The Adapter Layer

The non-trivial piece is translating DSPy's output ("here is the new prompt string") into Aider's input ("here is the file and what to change"). This is the thin layer you'd write yourself.

### Current Critic output → Aider instruction

The Critic already produces structured diagnoses that map cleanly:

```python
# Critic produces:
{
    "scenario_id": "avalanche-safety-001",
    "target_node": "synthesizer",
    "fix_type": "patch_prompt",
    "confidence": 0.72,
    "diagnosis": "Response lacks safety disclaimer for backcountry activities"
}

# Adapter translates to:
def critic_to_aider_instruction(gradient: dict) -> tuple[str, str]:
    """Returns (file_path, instruction) for Aider."""
    node_files = {
        "synthesizer": "greenvest/nodes/synthesizer.py",
        "query_translator": "greenvest/nodes/query_translator.py",
        "intent_router": "greenvest/nodes/intent_router.py",
        "clarification_gate": "greenvest/nodes/clarification_gate.py",
    }
    file_path = node_files[gradient["target_node"]]
    instruction = (
        f"Fix the following issue in this file: {gradient['diagnosis']}. "
        f"Edit type required: {gradient['fix_type']}."
    )
    return file_path, instruction
```

This adapter is ~30 lines. It's the only glue code needed between DSPy/Critic and Aider.

---

## Way Forward: Migration Options

### Option A — Incremental: Replace edit_tools.py with Aider (lowest risk)

Keep the entire current optimizer loop. Only swap `edit_tools.py` for Aider's edit engine.

**What changes:**
- `_apply_patch_prompt()`, `_apply_patch_phrase_list()` → delegated to Aider
- `update_ontology()` and `patch_set_literal()` — keep custom (Aider is unreliable on YAML)
- Remove manual AST validation (`ast.parse()` checks) — Aider handles this

**What stays the same:** everything else — Critic, Gate, loop, eval, Langfuse logging.

**Effort:** 1–2 days. Low risk. The optimizer continues to work during migration.

**When to do this:** When `edit_tools.py` becomes a maintenance burden or when you want to support editing a wider range of files without writing new edit type handlers.

---

### Option B — Partial: Replace the loop with DSPy, keep edit tools

Replace `autonomous_optimize.py` with DSPy's `MIPROv2`, keep `edit_tools.py` for non-prompt edits.

**Architecture:**
```
DSPy MIPROv2 (prompt optimization)
    ↓ optimized prompt strings
Adapter layer
    ↓ file + instruction
edit_tools.py (for YAML/phrase list/set literal edits)
    ↓
eval.py metric
```

**What you gain:** DSPy's variance-aware evaluation, better candidate deduplication, community support, research-grade optimization algorithms.

**What you lose:** The structured Critic diagnosis (harder to debug *why* a change was made). DSPy is a black-box optimizer — it doesn't tell you "avalanche-safety-001 fails because the synthesizer lacks a safety disclaimer."

**Effort:** 1–2 weeks. Medium risk. Requires wrapping the pipeline in DSPy modules, which touches every node.

---

### Option C — Full: DSPy + Aider as the complete stack

Full rewrite of the optimizer layer on top of DSPy + Aider.

```
DSPy MIPROv2 optimizer
    ↓
Adapter (Critic-style diagnosis → Aider instruction)
    ↓
Aider edit engine (Python files)
    + custom edit_tools.py (YAML/set literal only)
    ↓
eval.py metric + Langfuse
```

**What you gain:** Production-grade optimization loop with minimal custom maintenance, active open-source development on both DSPy and Aider.

**What you lose:** The Critic's structured diagnoses (explainability), full control over the loop behavior, the tight integration between optimizer and git history.

**Effort:** 3–4 weeks. High risk during transition — the current system is working and scoring 0.91+.

---

## Recommendation

**Now:** Do nothing. The current system works, scores well, and the eval→Langfuse pipeline is now live. The custom implementation has advantages DSPy doesn't: structured Critic diagnoses, explicit fix types, YAML editing, tight git integration.

**Next natural trigger for Option A (Aider):** When you want to add a new edit type (e.g., editing the `clarification_gate` decision logic) and don't want to write another handler in `edit_tools.py`. At that point, delegating Python file edits to Aider removes the per-edit-type boilerplate.

**Next natural trigger for Option B/C (DSPy):** When the scenario set grows beyond 20–30 scenarios and the optimizer's single-pass Critic starts missing patterns that would benefit from DSPy's bootstrapped few-shot construction. DSPy shines at scale; below ~20 scenarios the custom Critic is simpler and more explainable.

---

## Dependency Reference

```bash
# DSPy
uv add dspy-ai

# Aider (headless edit engine)
uv add aider-chat

# Aider with local Ollama (no API key needed for edits)
# Point model at: ollama/qwen2.5-coder:7b  (already in ollama list)
```

**DSPy + Ollama LM adapter:**
```python
import dspy
lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="ollama")
dspy.configure(lm=lm)
```

**Aider + Ollama:**
```python
from aider.models import Model
model = Model("ollama/qwen2.5-coder:7b")  # uses OLLAMA_BASE_URL from env
```
