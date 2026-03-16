"""
BuyerAgent — Ollama-backed persona simulator.

Plays the role of a customer interacting with the Greenvest sales agent.
Temperature > 0 ensures natural phrasing variance across Monte Carlo runs.

Conversation framing:
  SystemMessage : persona card (psychographics, constraints, instructions)
  AIMessage     : previous buyer responses  (the LLM plays the buyer)
  HumanMessage  : previous Greenvest messages (clarification questions)
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from greenvest.config import settings

if TYPE_CHECKING:
    from eval.personas import Persona

_BUYER_INSTRUCTION = (
    "You are roleplaying as a customer shopping for outdoor gear at REI. "
    "Stay strictly in character. Keep every response to 1-2 short sentences. "
    "Do NOT offer more information than asked. Do NOT ask your own questions — "
    "only answer what the associate just asked."
)


class BuyerAgent:
    """
    Simulates a human buyer with a given persona.

    temperature is slightly randomised per-run (±0.05) to ensure
    phrasing variance without losing persona coherence.
    """

    def __init__(self, persona: Persona, base_temperature: float = 0.7) -> None:
        self.persona = persona
        temperature = base_temperature + random.uniform(-0.05, 0.05)
        self._llm = ChatOllama(
            model=settings.OLLAMA_ROUTER_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    def initial_query(self) -> str:
        """Randomly sample one opening message from the persona's query pool."""
        return random.choice(self.persona.initial_queries)

    async def respond(
        self,
        agent_message: str,
        conversation_history: list[dict],
    ) -> str:
        """
        Generate a buyer response to `agent_message`, staying in persona.

        conversation_history is a list of {"role": "buyer"|"agent", "content": str}
        ordered oldest-first, NOT including the current agent_message.
        """
        messages: list = [
            SystemMessage(content=f"{_BUYER_INSTRUCTION}\n\n{self.persona.system_prompt}"),
        ]

        for turn in conversation_history:
            if turn["role"] == "buyer":
                messages.append(AIMessage(content=turn["content"]))
            else:
                messages.append(HumanMessage(content=turn["content"]))

        messages.append(HumanMessage(content=agent_message))

        response = await self._llm.ainvoke(messages)
        return response.content.strip()
