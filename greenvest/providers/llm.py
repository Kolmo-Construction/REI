"""
LLM provider factory.
Returns mock callables when USE_MOCK_LLM=True.
Returns real API clients when USE_MOCK_LLM=False (wired in later phases).
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from greenvest.config import settings
from greenvest.providers.mock_llm import (
    mock_intent_router,
    mock_query_translator,
    mock_synthesizer,
)

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


def get_intent_router() -> Callable[[str], dict]:
    if settings.USE_MOCK_LLM:
        return mock_intent_router
    # Phase 10: wire Gemini Flash / GPT-4o-mini here
    raise NotImplementedError("Real LLM not configured — set USE_MOCK_LLM=true or provide API keys")


def get_query_translator() -> Callable[["GreenvestState"], dict]:
    if settings.USE_MOCK_LLM:
        return mock_query_translator
    raise NotImplementedError("Real LLM not configured — set USE_MOCK_LLM=true or provide API keys")


def get_synthesizer() -> Callable[[str], str]:
    if settings.USE_MOCK_LLM:
        return mock_synthesizer
    raise NotImplementedError("Real LLM not configured — set USE_MOCK_LLM=true or provide API keys")
