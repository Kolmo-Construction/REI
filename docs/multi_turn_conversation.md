# Multi-Turn Conversation & Session Memory

## Current State

The `GreenvestState` schema was designed for multi-turn from the start — `messages`, `clarification_count`, and `compressed_summary` are all present. However the system currently:

- Runs as a single-invocation script (`run_agent.py` exits after one response)
- Uses no checkpointer — state is lost between invocations
- Never populates `messages` or `compressed_summary`

The REPL loop added in Phase 10.x carries state forward manually by merging the previous state dict with the new query. This works but bypasses LangGraph's native persistence mechanism.

---

## Checkpointer Options

A **checkpointer** is LangGraph's built-in mechanism for persisting state between turns. Each conversation is identified by a `thread_id`. On every turn, LangGraph reads the thread's saved state, runs the graph, and writes the updated state back.

| Option | Latency | Scales | Production? | Notes |
|---|---|---|---|---|
| `MemorySaver` | ~0ms | Single process | No | Dev/testing only. Lost on restart. |
| `AsyncSqliteSaver` | ~2ms | Single instance | Limited | Simple single-server deployments. |
| `AsyncPostgresSaver` | ~5–20ms | Horizontal | **Yes** | LangGraph's official production path. `POSTGRES_URL` already in config. |
| `RedisSaver` | ~1–5ms | Horizontal | Yes | Community lib. Fastest for high-frequency short sessions. `REDIS_URL` already in config. |

**Recommended:** `AsyncPostgresSaver`. The project already has `POSTGRES_URL` configured and PostgreSQL is provisioned in Phase 13. No new infrastructure needed.

### How to wire it

```python
# graph.py
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def build_graph():
    async with AsyncPostgresSaver.from_conn_string(settings.POSTGRES_URL) as checkpointer:
        await checkpointer.setup()  # creates checkpoint tables if absent
        return workflow.compile(checkpointer=checkpointer)
```

```python
# per-turn invocation
config = {"configurable": {"thread_id": state["session_id"]}}
result = await graph.ainvoke({"query": user_input}, config=config)
```

No manual state merging. LangGraph handles load and save automatically.

---

## Latency Impact

Each turn adds two DB operations:

| Operation | Postgres | Redis |
|---|---|---|
| Read thread state (turn start) | ~5–15ms | ~1–3ms |
| Write updated state (turn end) | ~5–15ms | ~1–3ms |
| **Total per turn** | **~10–30ms** | **~2–6ms** |

LLM inference takes 200ms–2s. Checkpointing is **not a bottleneck**.

---

## Context Window Growth Problem

Every turn appends to `messages`. After many turns the full conversation history is sent to the LLM on every invocation, which means:

- Increasing cost per turn (more tokens billed)
- Growing latency (larger prompt = slower TTFT)
- Eventually hitting the model's context limit

The state already has `compressed_summary` with the design note `"Summary of turns > 4 (synchronous, in-path)"`. This field was planned but never implemented.

---

## Summarize-and-Trim Pattern

The standard production solution is to **summarize old turns and drop them from `messages`**, keeping the active window bounded regardless of session length.

```
Turn 1–4:  messages = [T1, T2, T3, T4]
Turn 5:    LLM summarizes T1–T3 → compressed_summary
           messages = [T4, T5]   (T1–T3 dropped)

Turn 8:    LLM appends T6–T8 to compressed_summary
           messages = [T7, T8]   (rolling window of last 2 full turns)
```

What the synthesizer always sees:

```
[System prompt]
[compressed_summary]   ← all history condensed, bounded ~400 tokens
[last N full turns]    ← recent context for coherence, bounded ~1,200 tokens
[retrieval context]
[current query]
```

Total input stays within the Phase 8 token budget (≤ 4,400 tokens) regardless of session length.

### Trigger condition

```python
# In synthesizer or a new memory_compressor node
if len(state["messages"]) > 4:
    # summarize messages[:-2], keep last 2 full turns
    # append summary to state["compressed_summary"]
    # truncate state["messages"] to last 2
```

---

## Phase 8 Token Budget (reference)

| Section | Max Tokens |
|---|---|
| System prompt | 800 |
| `compressed_summary` | 400 |
| Recent turns (last 4) | 1,200 |
| Expert advice chunks | 600 |
| Product catalog results | 800 |
| Inventory data | 400 |
| User query | 200 |
| **Total input** | **≤ 4,400** |
| **Reserved for output** | **≤ 600** |

---

## Implementation Checklist (Phase 16.5)

- [ ] Add `AsyncPostgresSaver` to `graph.py` — swap `workflow.compile()` for checkpointer-backed compile
- [ ] Update `run_agent.py` REPL to pass `thread_id` in config instead of manually merging state dicts
- [ ] Update `POST /chat` endpoint (Phase 16) to pass `session_id` as `thread_id`
- [ ] Implement `memory_compressor` node — triggers when `len(messages) > 4`
- [ ] Wire `memory_compressor` into the graph before `synthesizer`
- [ ] Add integration test: 6-turn session → assert `compressed_summary` is populated and `len(messages) <= 2`
- [ ] Load test: P95 latency with checkpointer enabled at 500 concurrent sessions
