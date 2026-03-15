"""
Ollama LLM provider — calls local Ollama models, no external API keys required.
Used when USE_MOCK_LLM=false.
"""
from __future__ import annotations
import json
import re
from typing import TYPE_CHECKING

from langchain_ollama import ChatOllama

from greenvest.config import settings

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


def _get_router_model() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_ROUTER_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        format="json",
        temperature=0,
    )


def _get_synthesizer_model() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_SYNTHESIZER_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3,
    )


def _parse_json(text: str) -> dict:
    """Extract JSON from model output, tolerating markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    return json.loads(text)


def ollama_intent_router(query: str) -> dict:
    """
    Calls Ollama to classify intent and extract entities.
    Returns: intent, activity, user_environment, experience_level
    """
    prompt = (
        "You are an intent classifier for an REI outdoor gear store assistant.\n"
        "Classify the user query and extract entities.\n"
        "Return ONLY valid JSON with these exact keys:\n"
        '{"intent": "...", "activity": "...", "user_environment": "...", "experience_level": "..."}\n\n'
        "Intent must be one of: Product_Search, Education, Support, Out_of_Bounds\n"
        "  - Product_Search: looking for specific gear or product recommendations\n"
        "    e.g. 'I need a sleeping bag', 'recommend a jacket for rain'\n"
        "  - Education: how-to, what-is, explain, compare, best-for questions about gear or outdoor activities\n"
        "    e.g. 'what is the best waterproof jacket', 'how do I choose a sleeping bag', 'what is the value of GORE-TEX'\n"
        "  - Support: returns, exchanges, orders, warranty claims, repairs, account or membership issues\n"
        "    e.g. 'I want to return my jacket', 'what is the return policy', 'how do I make a warranty claim'\n"
        "    NOTE: ANY question about return policy — even phrased generally — is Support, not Product_Search.\n"
        "  - Out_of_Bounds: topics that require licensed professionals — medical diagnosis/treatment, legal advice, financial investment advice (stocks, retirement, tax planning)\n"
        "    e.g. 'what are my legal rights', 'should I invest in REI stock', 'medical treatment for altitude sickness'\n"
        "    NOTE: Questions about gear price, value, or cost-effectiveness are Education, NOT Out_of_Bounds.\n"
        "    e.g. 'is GORE-TEX worth the price?' -> Education, 'should I buy REI stock?' -> Out_of_Bounds\n\n"
        "Canonical activity values: day_hiking, backpacking, thru_hiking, winter_camping, car_camping, rock_climbing, skiing, mountaineering, alpine_climbing\n"
        "  - Map synonyms, misspellings, and related terms to the closest canonical value.\n"
        "    e.g. 'trekking' -> thru_hiking, 'hikking' -> day_hiking, 'trail running' -> day_hiking, 'camping' -> car_camping\n"
        "  - Only return null for activity if the query contains NO mention or implication of any outdoor activity.\n"
        "user_environment examples: PNW_winter, desert_summer, alpine, coastal, humid\n"
        "experience_level: beginner, intermediate, expert, or null\n"
        "Use null for user_environment and experience_level if they cannot be determined from the query.\n\n"
        f"User query: {query}"
    )

    llm = _get_router_model()
    response = llm.invoke(prompt)
    result = _parse_json(response.content)

    return {
        "intent": result.get("intent", "Product_Search"),
        "activity": result.get("activity") or None,
        "user_environment": result.get("user_environment") or None,
        "experience_level": result.get("experience_level") or None,
    }


def ollama_query_translator(state: "GreenvestState") -> dict:
    """
    Calls Ollama to translate activity/environment context into filterable gear specs.
    Returns: derived_specs, spec_confidence
    """
    prompt = (
        "You are a gear specification translator for an REI outdoor gear store.\n"
        "Convert the customer context into filterable product specifications.\n"
        "Return ONLY valid JSON with these exact keys:\n"
        '{"derived_specs": [{"spec_key": "value"}, ...], "spec_confidence": 0.0}\n\n'
        "Common spec keys and value formats:\n"
        "  fill_type: down | synthetic\n"
        "  temp_rating_f: <=15 | <=20 | <=32 (lower = warmer)\n"
        "  weight_oz: <32 | <48 (lighter for backpacking)\n"
        "  r_value: >=4.5 | >=2.0 (sleeping pad insulation)\n"
        "  water_resistance: hydrophobic_down OR synthetic\n\n"
        "spec_confidence: 0.0-1.0. Use >0.85 when activity/environment are specific.\n"
        "Use <0.7 when query is too vague to determine specs confidently.\n\n"
        f"Activity: {state.get('activity') or 'unknown'}\n"
        f"Environment: {state.get('user_environment') or 'unknown'}\n"
        f"Experience level: {state.get('experience_level') or 'unknown'}\n"
        f"Raw query: {state.get('query', '')}"
    )

    llm = _get_router_model()
    response = llm.invoke(prompt)
    result = _parse_json(response.content)

    specs = result.get("derived_specs", [])
    confidence = float(result.get("spec_confidence", 0.75))

    if not specs:
        specs = [{"fill_type": "synthetic"}]
        confidence = min(confidence, 0.65)

    return {
        "derived_specs": specs,
        "spec_confidence": confidence,
    }


def ollama_synthesizer(prompt: str) -> str:
    """
    Calls Ollama to generate the final REI Greenvest recommendation.
    """
    llm = _get_synthesizer_model()
    response = llm.invoke(prompt)
    return response.content.strip()
