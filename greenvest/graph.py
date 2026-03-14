"""
Greenvest Agent — compiled LangGraph DAG.

Topology:
    START → intent_router → clarification_gate
        ├─ REQUIRES_CLARIFICATION → END  (state carries clarification_message)
        ├─ READY_TO_SYNTHESIZE   → synthesizer → END  (Out_of_Bounds / Support)
        └─ READY_TO_SEARCH       → query_translator → retrieval_dispatcher → synthesizer → END
"""
import asyncio
from typing import Literal

import structlog
from langgraph.graph import StateGraph, START, END

from greenvest.state import GreenvestState
from greenvest.nodes.intent_router import intent_router
from greenvest.nodes.clarification_gate import clarification_gate
from greenvest.nodes.query_translator import query_translator
from greenvest.nodes.synthesizer import synthesizer
from greenvest.retrieval.branch_a_expert import search_expert_advice
from greenvest.retrieval.branch_b_catalog import search_catalog
from greenvest.retrieval.branch_c_inventory import search_inventory

log = structlog.get_logger(__name__)


async def retrieval_dispatcher(state: GreenvestState) -> dict:
    """
    Fan out all three retrieval branches concurrently.
    Per solution.md §6.1: asyncio.gather with return_exceptions=True.
    A branch failure returns an empty result — never crashes the pipeline.
    """
    results = await asyncio.gather(
        search_expert_advice(state),
        search_catalog(state),
        search_inventory(state),
        return_exceptions=True,
    )

    expert_context, catalog_results, inventory_snapshot = [], [], []

    for i, result in enumerate(results):
        branch = ["branch_a_expert", "branch_b_catalog", "branch_c_inventory"][i]
        if isinstance(result, Exception):
            log.error("retrieval_dispatcher", branch=branch, error=str(result))
        elif i == 0:
            expert_context = result
        elif i == 1:
            catalog_results = result
        else:
            inventory_snapshot = result

    log.info(
        "retrieval_dispatcher",
        session_id=state["session_id"],
        expert_chunks=len(expert_context),
        catalog_hits=len(catalog_results),
        inventory_items=len(inventory_snapshot),
    )

    return {
        "expert_context": expert_context,
        "catalog_results": catalog_results,
        "inventory_snapshot": inventory_snapshot,
    }


def _route_after_clarification_gate(
    state: GreenvestState,
) -> Literal["query_translator", "synthesizer", "__end__"]:
    flag = state.get("action_flag")
    if flag == "READY_TO_SEARCH":
        return "query_translator"
    if flag == "READY_TO_SYNTHESIZE":
        # Out_of_Bounds or Support — skip retrieval
        return "synthesizer"
    # REQUIRES_CLARIFICATION — clarification_message is set in state; exit immediately
    return "__end__"


def _route_after_query_translator(
    state: GreenvestState,
) -> Literal["retrieval_dispatcher", "__end__"]:
    # Re-clarification requested by query_translator (low confidence)
    flag = state.get("action_flag")
    if flag == "REQUIRES_CLARIFICATION":
        return "__end__"
    return "retrieval_dispatcher"


# Build the graph
workflow = StateGraph(GreenvestState)

workflow.add_node("intent_router", intent_router)
workflow.add_node("clarification_gate", clarification_gate)
workflow.add_node("query_translator", query_translator)
workflow.add_node("retrieval_dispatcher", retrieval_dispatcher)
workflow.add_node("synthesizer", synthesizer)

workflow.add_edge(START, "intent_router")
workflow.add_edge("intent_router", "clarification_gate")

workflow.add_conditional_edges(
    "clarification_gate",
    _route_after_clarification_gate,
    {
        "query_translator": "query_translator",
        "synthesizer": "synthesizer",
        "__end__": END,
    },
)

workflow.add_conditional_edges(
    "query_translator",
    _route_after_query_translator,
    {
        "retrieval_dispatcher": "retrieval_dispatcher",
        "__end__": END,
    },
)

workflow.add_edge("retrieval_dispatcher", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile — enables checkpointing and runtime optimizations
graph = workflow.compile()
