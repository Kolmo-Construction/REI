"""
LLM provider factory.
Returns mock callables when USE_MOCK_LLM=True.
Returns Ollama callables when USE_MOCK_LLM=False.
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
    from greenvest.providers.ollama_llm import ollama_intent_router
    return ollama_intent_router


def get_query_translator() -> Callable[["GreenvestState"], dict]:
    if settings.USE_MOCK_LLM:
        return mock_query_translator
    from greenvest.providers.ollama_llm import ollama_query_translator
    return ollama_query_translator


def get_synthesizer() -> Callable[[str], str]:
    if settings.USE_MOCK_LLM:
        return mock_synthesizer
    from greenvest.providers.ollama_llm import ollama_synthesizer
    return ollama_synthesizer
