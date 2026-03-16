import structlog
from greenvest.state import GreenvestState
from greenvest.providers.llm import get_intent_router

log = structlog.get_logger(__name__)


def intent_router(state: GreenvestState) -> dict:
    """
    Layer 1 node: classify intent and extract entities from the user query.
    Populates: intent, activity, user_environment, experience_level.
    """
    router_fn = get_intent_router()
    result = router_fn(state["query"])

    log.info(
        "intent_router",
        session_id=state["session_id"],
        intent=result.get("intent"),
        activity=result.get("activity"),
        user_environment=result.get("user_environment"),
    )

    def _clean(val):
        """Coerce LLM-emitted string 'null'/'none'/'unknown' to None."""
        return None if (val is None or str(val).lower() in ("null", "none", "unknown")) else val

    return {
        "intent": result["intent"],
        "activity": _clean(result.get("activity")),
        "user_environment": _clean(result.get("user_environment")),
        "experience_level": _clean(result.get("experience_level")),
    }
