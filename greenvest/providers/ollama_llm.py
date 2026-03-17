"""
Ollama LLM provider — calls local Ollama models, no external API keys required.
Used when USE_MOCK_LLM=false.

Structured output (with_structured_output) passes the Pydantic schema to Ollama
as a GBNF grammar. The token sampler is constrained — invalid values cannot be
emitted. No JSON parsing or validation needed.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from greenvest.config import settings

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


# ---------------------------------------------------------------------------
# Output schemas — these become the grammar Ollama enforces at token level
# ---------------------------------------------------------------------------

class IntentRouterOutput(BaseModel):
    intent: Literal["Product_Search", "Education", "Support", "Out_of_Bounds", "Greeting"]
    activity: Optional[Literal[
        "day_hiking", "backpacking", "thru_hiking", "winter_camping",
        "car_camping", "rock_climbing", "skiing", "mountaineering", "alpine_climbing"
    ]] = None
    user_environment: Optional[str] = None
    experience_level: Optional[Literal["beginner", "intermediate", "expert"]] = None


class DerivedSpecs(BaseModel):
    """All valid filterable spec keys and their allowed values.
    The Pydantic schema is compiled by Ollama into a GBNF grammar —
    the model cannot emit a key or value not listed here.
    Multiple specs = multiple non-None fields.
    """
    fill_type: Optional[Literal["down", "synthetic", "down OR synthetic"]] = None
    temp_rating_f: Optional[Literal["<=15", "<=20", "<=32"]] = None
    weight_oz: Optional[Literal["<8", "<10", "<16", "<20", "<24", "<32", "<48"]] = None
    r_value: Optional[Literal[">=2.0", ">=3.0", ">=4.5"]] = None
    water_resistance: Optional[Literal["hydrophobic_down OR synthetic"]] = None
    waterproof: Optional[Literal["true"]] = None
    seam_sealed: Optional[Literal["true"]] = None
    crampon_compatible: Optional[Literal["true"]] = None
    technology: Optional[Literal[
        "GORE-TEX OR eVent OR brand waterproof membrane",
        "GORE-TEX OR eVent",
    ]] = None
    capacity_liters: Optional[Literal["15-30", "30-55", ">=55"]] = None
    hipbelt: Optional[Literal["load-bearing"]] = None
    frame: Optional[Literal["frameless OR minimal"]] = None
    sole: Optional[Literal["lugged", "sticky rubber"]] = None
    breathability: Optional[Literal["high"]] = None
    packable: Optional[Literal["true"]] = None
    width_in: Optional[Literal[">=25"]] = None
    ankle: Optional[Literal["low OR mid"]] = None
    drop_mm: Optional[Literal["<=8"]] = None


class QueryTranslatorOutput(BaseModel):
    derived_specs: DerivedSpecs
    spec_confidence: float


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _get_router_model() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_ROUTER_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
    )


def _get_synthesizer_model() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_SYNTHESIZER_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Provider functions
# ---------------------------------------------------------------------------

def ollama_intent_router(query: str) -> dict:
    """
    Calls Ollama to classify intent and extract entities.
    Returns: intent, activity, user_environment, experience_level
    """
    prompt = (
        "You are an intent classifier for an REI outdoor gear store assistant.\n"
        "Classify the user query and extract entities.\n\n"
        "Intent values:\n"
        "  - Greeting: bare social openers or pleasantries with zero gear, product, or activity meaning\n"
        "    e.g. 'hi', 'hello', 'hey', 'good morning', 'thanks', 'bye', 'ok', 'yes', 'no'\n"
        "    NEVER use Greeting if the message contains any outdoor activity, gear type, or product word.\n"
        "    e.g. 'camping', 'hiking', 'tent', 'jacket' are Product_Search, NOT Greeting.\n"
        "  - Product_Search: looking for specific gear or product recommendations\n"
        "    e.g. 'I need a sleeping bag', 'recommend a jacket for rain'\n"
        "  - Education: how-to, what-is, explain, compare, best-for questions about gear or outdoor activities\n"
        "    e.g. 'what is the best waterproof jacket', 'how do I choose a sleeping bag', 'what is the value of GORE-TEX'\n"
        "  - Support: returns, exchanges, orders, warranty claims, repairs, account or membership issues\n"
        "    e.g. 'I want to return my jacket', 'what is the return policy', 'how do I make a warranty claim'\n"
        "    NOTE: ANY question about return policy — even phrased generally — is Support, not Product_Search.\n"
        "  - Out_of_Bounds: topics requiring licensed professionals — medical, legal, financial investment advice\n"
        "    e.g. 'what are my legal rights', 'should I invest in REI stock', 'medical treatment for altitude sickness'\n"
        "    NOTE: Questions about gear price, value, or cost-effectiveness are Education, NOT Out_of_Bounds.\n\n"
        "Activity values: day_hiking, backpacking, thru_hiking, winter_camping, car_camping, "
        "rock_climbing, skiing, mountaineering, alpine_climbing\n"
        "  - Map synonyms and related terms to the closest canonical value.\n"
        "    e.g. 'trekking' -> thru_hiking, 'trail running' -> day_hiking, 'camping' -> car_camping\n"
        "  - Use null only if the query contains NO mention or implication of any outdoor activity.\n\n"
        "user_environment examples: PNW_winter, desert_summer, alpine, coastal, humid — null if unknown.\n"
        "experience_level: beginner, intermediate, expert — null if unknown.\n\n"
        f"User query: {query}"
    )

    result: IntentRouterOutput = _get_router_model().with_structured_output(IntentRouterOutput).invoke(prompt)
    return {
        "intent": result.intent,
        "activity": result.activity,
        "user_environment": result.user_environment,
        "experience_level": result.experience_level,
    }


def ollama_query_translator(state: "GreenvestState") -> dict:
    """
    Calls Ollama to translate activity/environment context into filterable gear specs.
    Returns: derived_specs, spec_confidence
    """
    prompt = (
        "You are a gear specification translator for an REI outdoor gear store.\n"
        "Convert the customer context into filterable product specifications.\n\n"
        "Common spec keys and value formats:\n"
        "  fill_type: down | synthetic\n"
        "  temp_rating_f: <=15 | <=20 | <=32 (lower = warmer)\n"
        "  weight_oz: <32 | <48 (lighter for backpacking)\n"
        "  r_value: >=4.5 | >=2.0 (sleeping pad insulation)\n"
        "  water_resistance: hydrophobic_down | synthetic\n\n"
        "spec_confidence: 0.0-1.0. Use >0.85 when activity/environment are specific. "
        "Use <0.7 when the query is too vague.\n\n"
        f"Activity: {state.get('activity') or 'unknown'}\n"
        f"Environment: {state.get('user_environment') or 'unknown'}\n"
        f"Experience level: {state.get('experience_level') or 'unknown'}\n"
        f"Raw query: {state.get('query', '')}"
    )

    result: QueryTranslatorOutput = _get_router_model().with_structured_output(QueryTranslatorOutput).invoke(prompt)

    specs = result.derived_specs.model_dump(exclude_none=True)
    if not specs:
        specs = {"fill_type": "synthetic"}
        confidence = min(result.spec_confidence, 0.65)
    else:
        confidence = result.spec_confidence

    return {
        "derived_specs": specs,
        "spec_confidence": confidence,
    }


def ollama_synthesizer(prompt: str) -> str:
    """
    Calls Ollama to generate the final REI Greenvest recommendation.
    """
    return _get_synthesizer_model().invoke(prompt).content.strip()
