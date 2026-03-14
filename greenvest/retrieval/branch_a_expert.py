"""
Branch A: Expert Advice vector search.
Stub — returns empty list until Weaviate/pgvector is provisioned (Phase 12).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


async def search_expert_advice(state: "GreenvestState") -> list[str]:
    """Stubbed. Phase 12: replace with pgvector / Weaviate ANN search."""
    return []
