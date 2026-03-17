from typing import Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class GreenvestState(TypedDict):
    # Session identity
    session_id: str
    store_id: str                    # REI store location, resolved at session init
    member_number: Optional[str]     # REI Co-op member ID if authenticated

    # Intent & context
    query: str                       # Raw user query
    intent: Optional[Literal[
        "Product_Search",
        "Education",
        "Support",
        "Out_of_Bounds",
        "Greeting"
    ]]
    activity: Optional[str]          # e.g., "alpine_climbing", "car_camping", "thru_hiking"
    user_environment: Optional[str]  # e.g., "PNW_winter", "desert_summer", "alpine"
    experience_level: Optional[Literal["beginner", "intermediate", "expert"]]
    budget_usd: Optional[tuple]      # (min, max) — None means unspecified

    # Derived technical specs (populated by Query Translator)
    derived_specs: dict              # e.g., {"fill_type": "synthetic", "temp_rating_f": "<=15"}
    spec_confidence: float           # 0.0–1.0; below 0.7 triggers re-clarification

    # Retrieval results
    expert_context: list             # Branch A results (list[str])
    catalog_results: list            # Branch B results (list[dict], post-RRF)
    inventory_snapshot: list         # Branch C results (list[dict], store-localized, TTL 60s)

    # Control flow
    action_flag: Literal["REQUIRES_CLARIFICATION", "READY_TO_SEARCH", "READY_TO_SYNTHESIZE"]
    clarification_count: int         # Increment each turn; cap at 2 before forcing synthesis
    clarification_message: Optional[str]  # Question to ask user when clarification needed
    messages: list                   # Compressed conversation history (list[BaseMessage])
    compressed_summary: Optional[str]  # Summary of turns > 4 (synchronous, in-path)

    # LLM-driven routing flags
    requires_environment_context: bool  # Set by intent_router; True if gear depends on weather/terrain

    # Output
    recommendation: Optional[str]   # Final synthesized recommendation


def initial_state(
    query: str,
    session_id: str = "session-001",
    store_id: str = "REI-Seattle",
    member_number: Optional[str] = None,
) -> GreenvestState:
    return GreenvestState(
        session_id=session_id,
        store_id=store_id,
        member_number=member_number,
        query=query,
        intent=None,
        activity=None,
        user_environment=None,
        experience_level=None,
        budget_usd=None,
        derived_specs={},
        spec_confidence=0.0,
        expert_context=[],
        catalog_results=[],
        inventory_snapshot=[],
        action_flag="REQUIRES_CLARIFICATION",
        clarification_count=0,
        clarification_message=None,
        messages=[],
        compressed_summary=None,
        recommendation=None,
        requires_environment_context=False,
    )
