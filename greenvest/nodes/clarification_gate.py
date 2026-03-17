import structlog
from greenvest.state import GreenvestState

log = structlog.get_logger(__name__)

def clarification_gate(state: GreenvestState) -> dict:
    """
    Pure logic, no LLM call.
    Implements the decision tree from solution.md §4.2.

    Sets action_flag to:
      - REQUIRES_CLARIFICATION  → one question back to user
      - READY_TO_SEARCH         → proceed to retrieval
    """
    intent = state.get("intent")
    activity = state.get("activity")
    user_environment = state.get("user_environment")
    clarification_count = state.get("clarification_count", 0)
    requires_env: bool = state.get("requires_environment_context", False)

    # Out_of_Bounds — deterministic refusal (handled downstream in synthesizer)
    if intent == "Out_of_Bounds":
        log.info("clarification_gate", decision="out_of_bounds", session_id=state["session_id"])
        return {"action_flag": "READY_TO_SYNTHESIZE"}

    # Greeting — bypass retrieval, respond with friendly pivot
    if intent == "Greeting":
        log.info("clarification_gate", decision="greeting_bypass", session_id=state["session_id"])
        return {"action_flag": "READY_TO_SYNTHESIZE"}

    # Support — bypass retrieval, route straight to synthesizer
    if intent == "Support":
        log.info("clarification_gate", decision="support_bypass", session_id=state["session_id"])
        return {"action_flag": "READY_TO_SYNTHESIZE"}

    # Cap clarification rounds — force forward after 2 turns
    if clarification_count >= 2:
        log.info(
            "clarification_gate",
            decision="cap_reached_force_search",
            clarification_count=clarification_count,
            session_id=state["session_id"],
        )
        return {"action_flag": "READY_TO_SEARCH"}

    # Missing activity — highest priority question
    if not activity:
        question = _build_activity_question(state["query"])
        log.info("clarification_gate", decision="needs_activity", session_id=state["session_id"])
        return {
            "action_flag": "REQUIRES_CLARIFICATION",
            "clarification_message": question,
            "clarification_count": clarification_count + 1,
        }

    # Missing environment for activities that require it (flag set by LLM)
    if requires_env and not user_environment:
        question = _build_environment_question(activity or "your trip")
        log.info(
            "clarification_gate",
            decision="needs_environment",
            activity=activity,
            session_id=state["session_id"],
        )
        return {
            "action_flag": "REQUIRES_CLARIFICATION",
            "clarification_message": question,
            "clarification_count": clarification_count + 1,
        }

    # All required fields present — proceed to search
    log.info("clarification_gate", decision="ready_to_search", session_id=state["session_id"])
    return {"action_flag": "READY_TO_SEARCH"}


def _build_activity_question(query: str) -> str:
    return (
        "What activity are you gearing up for? For example: backpacking, "
        "winter camping, car camping, thru-hiking, or day hiking?"
    )


def _build_environment_question(activity: str) -> str:
    activity_label = activity.replace("_", " ")
    return (
        f"Where will you be {activity_label}? Knowing your environment — "
        f"Pacific Northwest, desert, alpine, or somewhere else — helps me match "
        f"the right gear to actual conditions."
    )
