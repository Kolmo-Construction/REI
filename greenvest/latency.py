"""
Latency instrumentation for the Greenvest graph.

LatencyCallback is an async LangGraph/LangChain callback that times each node
by hooking into on_chain_start / on_chain_end without touching any node code.

Usage:
    cb = LatencyCallback()
    await graph.ainvoke(state, config={"callbacks": [cb]})
    print(cb.timings_ms)          # {"intent_router": 312.4, "synthesizer": 4821.0, ...}
    print(cb.total_ms)            # sum of all tracked nodes
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

# Nodes we care about — everything else (the graph wrapper itself) is ignored
_TRACKED_NODES = frozenset({
    "intent_router",
    "clarification_gate",
    "query_translator",
    "retrieval_dispatcher",
    "synthesizer",
})


class LatencyCallback(AsyncCallbackHandler):
    """
    Collects per-node wall-clock timing for a single graph.ainvoke() call.
    Create a fresh instance per invocation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending: dict[str, tuple[str, float]] = {}  # run_id -> (node_name, t0)
        self.timings_ms: dict[str, float] = {}            # node_name -> duration_ms

    # LangGraph fires on_chain_start for each node (and for the graph wrapper).
    # In current LangGraph versions serialized is None — node name is in kwargs["name"]
    # and confirmed by kwargs["metadata"]["langgraph_node"].
    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or ""
        if name in _TRACKED_NODES:
            self._pending[str(run_id)] = (name, time.monotonic())

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        entry = self._pending.pop(str(run_id), None)
        if entry:
            name, t0 = entry
            self.timings_ms[name] = round((time.monotonic() - t0) * 1000, 1)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Still record partial timing so failures are visible in the report
        entry = self._pending.pop(str(run_id), None)
        if entry:
            name, t0 = entry
            self.timings_ms[f"{name}__error"] = round((time.monotonic() - t0) * 1000, 1)

    @property
    def total_ms(self) -> float:
        return round(sum(v for k, v in self.timings_ms.items() if not k.endswith("__error")), 1)
