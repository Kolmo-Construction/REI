"""
Branch B: Product Catalog retrieval.
Vertical slice uses flat JSON + simple scoring.
Production: swap for Weaviate hybrid search (RRF) — interface is identical.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from greenvest.state import GreenvestState

log = structlog.get_logger(__name__)

_CATALOG: list[dict] = []


def _load_catalog() -> list[dict]:
    path = Path(__file__).parent.parent.parent / "data" / "sample_products.json"
    with open(path) as f:
        return json.load(f)


def _get_catalog() -> list[dict]:
    global _CATALOG
    if not _CATALOG:
        _CATALOG = _load_catalog()
    return _CATALOG


def _parse_numeric_constraint(value: str) -> tuple[str, float] | None:
    """Parse '>= 4.5', '<32', '<=15' → (operator, number)."""
    m = re.match(r"(>=|<=|>|<|==?)?\s*(-?\d+(?:\.\d+)?)", str(value))
    if not m:
        return None
    op = m.group(1) or "=="
    num = float(m.group(2))
    return op, num


def _numeric_match(product_val, constraint: str) -> bool:
    if product_val is None:
        return False
    parsed = _parse_numeric_constraint(constraint)
    if parsed is None:
        return False
    op, target = parsed
    pv = float(product_val)
    return {
        ">=": pv >= target,
        "<=": pv <= target,
        ">": pv > target,
        "<": pv < target,
        "==": pv == target,
        "=": pv == target,
    }.get(op, False)


def _score_product(product: dict, derived_specs: list[dict]) -> float:
    """
    Simple relevance score: +1 per matched spec.
    Production will replace with RRF scores from Weaviate.
    """
    score = 0.0
    for spec in derived_specs:
        for key, value in spec.items():
            prod_val = product.get(key)
            if prod_val is None:
                continue
            v_str = str(value).lower()

            # Numeric constraint
            if any(op in v_str for op in [">=", "<=", ">", "<"]):
                if _numeric_match(prod_val, v_str):
                    score += 1.0
            # String / categorical match (supports "a OR b" alternatives)
            elif " or " in v_str:
                alternatives = [a.strip() for a in v_str.split(" or ")]
                if str(prod_val).lower() in alternatives:
                    score += 1.0
            else:
                if str(prod_val).lower() == v_str:
                    score += 1.0

    return score


async def search_catalog(state: "GreenvestState") -> list[dict]:
    """
    Production-compatible async interface.
    Returns top-5 products scored against derived_specs.
    Weaviate hybrid search replaces this in Phase 11.
    """
    derived_specs = state.get("derived_specs", [])
    catalog = _get_catalog()

    scored = [
        (product, _score_product(product, derived_specs))
        for product in catalog
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = [p for p, s in scored if s > 0][:5]

    # Fallback: if nothing matched, return top 5 by catalog order (broad fallback)
    if not results:
        results = catalog[:5]

    log.info(
        "branch_b_catalog",
        session_id=state["session_id"],
        total_candidates=len(catalog),
        matched=len(results),
        top_score=scored[0][1] if scored else 0,
    )
    return results
