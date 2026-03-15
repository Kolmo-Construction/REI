import structlog
from greenvest.state import GreenvestState
from greenvest.providers.llm import get_synthesizer

log = structlog.get_logger(__name__)

# Token budget constants (characters ≈ tokens * 4 for English prose)
_MAX_EXPERT_CHUNKS = 3
_MAX_CATALOG_RESULTS = 5

_REI_PERSONA = (
    "You are a Greenvest — an REI Co-op outdoor gear specialist. "
    "You are knowledgeable, approachable, and non-pushy. "
    "You never disparage competitors. You recommend specific products by name and explain "
    "exactly why each spec matters for the customer's stated conditions. "
    "Always resolve inventory to the customer's local store when available."
)

_OUT_OF_BOUNDS_RESPONSE = (
    "That's outside my area — I specialize in outdoor gear and REI products. "
    "For this question, I'd recommend reaching out to a relevant professional. "
    "Happy to help you find the right gear for your next adventure!"
)

_SUPPORT_RESPONSE = (
    "For order and warranty questions, the REI customer service team is your best path: "
    "rei.com/help or call 1-800-426-4840. They can pull up your account and get this resolved quickly."
)


def assemble_context(state: GreenvestState) -> str:
    """
    Build synthesis prompt from state, respecting the token budget from solution.md §7.2.
    Runs synchronously and in-path.
    """
    expert_chunks = state.get("expert_context", [])[:_MAX_EXPERT_CHUNKS]
    catalog = state.get("catalog_results", [])[:_MAX_CATALOG_RESULTS]
    inventory = state.get("inventory_snapshot", [])
    query = state.get("query", "")
    store_id = state.get("store_id", "your local REI")

    sections = [_REI_PERSONA, ""]

    # Compressed history summary (if present)
    summary = state.get("compressed_summary")
    if summary:
        sections.append(f"[Conversation summary]\n{summary}\n")

    # Recent messages
    messages = state.get("messages", [])
    if messages:
        sections.append("[Recent conversation]")
        for msg in messages[-8:]:
            role = getattr(msg, "type", "user")
            content = getattr(msg, "content", str(msg))
            sections.append(f"{role}: {content}")
        sections.append("")

    # Expert advice chunks
    if expert_chunks:
        sections.append("[Expert advice]")
        for chunk in expert_chunks:
            if isinstance(chunk, dict):
                sections.append(f"- {chunk.get('section', '')}: {chunk.get('chunk_text', '')}")
            else:
                sections.append(f"- {chunk}")
        sections.append("")

    # Catalog results
    if catalog:
        sections.append("[Matching products]")
        for p in catalog:
            line = (
                f"- {p['name']} (SKU: {p['sku']}) | "
                f"Fill: {p.get('fill_type', 'N/A')} | "
                f"Temp: {p.get('temp_rating_f', 'N/A')}°F | "
                f"Weight: {p.get('weight_oz', 'N/A')} oz | "
                f"Price: ${p.get('price_usd', 'N/A')} (member: ${p.get('member_price_usd', 'N/A')})"
            )
            if p.get("r_value"):
                line += f" | R-value: {p['r_value']}"
            sections.append(line)
        sections.append("")

    # Inventory
    if inventory:
        sections.append(f"[Inventory at {store_id}]")
        for item in inventory:
            sections.append(f"- {item.get('product_name', item.get('sku'))}: {item.get('store_stock_qty', '?')} in store")
        sections.append("")

    sections.append(f"[Customer query]\n{query}")
    sections.append("\nProvide a specific, helpful recommendation in the REI Greenvest persona.")

    return "\n".join(sections)


def synthesizer(state: GreenvestState) -> dict:
    """
    Layer 3 node: assembles context and calls LLM to generate the final recommendation.
    Handles Out_of_Bounds and Support intents deterministically.
    """
    intent = state.get("intent")

    if intent == "Out_of_Bounds":
        log.info("synthesizer", session_id=state["session_id"], intent="out_of_bounds")
        return {
            "recommendation": _OUT_OF_BOUNDS_RESPONSE,
            "action_flag": "READY_TO_SYNTHESIZE",
        }

    if intent == "Support":
        log.info("synthesizer", session_id=state["session_id"], intent="support")
        return {
            "recommendation": _SUPPORT_RESPONSE,
            "action_flag": "READY_TO_SYNTHESIZE",
        }

    prompt = assemble_context(state)
    synthesizer_fn = get_synthesizer()
    recommendation = synthesizer_fn(prompt)

    log.info(
        "synthesizer",
        session_id=state["session_id"],
        intent=intent,
        catalog_results=len(state.get("catalog_results", [])),
        recommendation_length=len(recommendation),
    )

    return {
        "recommendation": recommendation,
        "action_flag": "READY_TO_SYNTHESIZE",
    }
