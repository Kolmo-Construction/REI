import structlog
from greenvest.state import GreenvestState
from greenvest.providers.llm import get_query_translator
from greenvest.ontology import lookup_all

log = structlog.get_logger(__name__)


def query_translator(state: GreenvestState) -> dict:
    """
    Layer 1 node: converts natural language context into filterable technical specs.

    Step 1: Deterministic ontology lookup (O(n), ~5ms, confidence = 1.0)
    Step 2: LLM fallback for residue not matched by ontology
    """
    # Step 1: build candidate terms from extracted context
    terms = _extract_terms(state)
    # Ontology returns list of single-key dicts — flatten to {key: value}
    ontology_specs: dict[str, str] = {
        k: v for spec in lookup_all(terms) for k, v in spec.items()
    }

    log.info(
        "query_translator",
        session_id=state["session_id"],
        terms=terms,
        ontology_hits=len(ontology_specs),
    )

    if ontology_specs:
        # Ontology matched — high confidence, no LLM needed
        confidence = 1.0
        specs = ontology_specs
    else:
        # Step 2: LLM fallback — returns flat dict via DerivedSpecs.model_dump(exclude_none=True)
        translator_fn = get_query_translator()
        result = translator_fn(state)
        specs = result["derived_specs"]
        confidence = result["spec_confidence"]
        log.info(
            "query_translator",
            session_id=state["session_id"],
            source="llm_fallback",
            spec_confidence=confidence,
        )

    # Trigger re-clarification if confidence too low (and cap not hit)
    if confidence < 0.7 and state.get("clarification_count", 0) < 2:
        return {
            "derived_specs": specs,
            "spec_confidence": confidence,
            "action_flag": "REQUIRES_CLARIFICATION",
            "clarification_message": (
                "I want to make sure I match the right gear — could you tell me a bit more "
                "about where or how you'll be using it?"
            ),
            "clarification_count": state.get("clarification_count", 0) + 1,
        }

    return {
        "derived_specs": specs,
        "spec_confidence": confidence,
    }


def _extract_terms(state: GreenvestState) -> list[str]:
    """Build list of candidate lookup terms from state fields and raw query."""
    terms = []
    if state.get("activity"):
        terms.append(state["activity"].replace("_", " "))
    if state.get("user_environment"):
        terms.append(state["user_environment"].replace("_", " ").replace("winter", "").strip())
        terms.append(state["user_environment"])
    # Also pull key phrases from the raw query
    query = state.get("query", "").lower()
    for phrase in ["PNW", "winter camping", "backpacking", "car camping", "thru-hiking",
                   "thru hiking", "alpine", "mountaineering", "alpine climbing",
                   "coastal", "wet climate", "waterproof", "side sleeper",
                   "ultralight", "cold sleeper"]:
        if phrase.lower() in query:
            terms.append(phrase)
    return terms
