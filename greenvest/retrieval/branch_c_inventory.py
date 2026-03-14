"""
Branch C: Inventory API (parameterized SQL, store-localized).
Stub — returns empty list until PostgreSQL read-replica is provisioned (Phase 13).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from greenvest.state import GreenvestState


async def search_inventory(state: "GreenvestState") -> list[dict]:
    """Stubbed. Phase 13: replace with parameterized query against inventory_view."""
    return []
