from pathlib import Path
from typing import Optional
import yaml

_ontology: dict = {}


def _load() -> dict:
    path = Path(__file__).parent / "gear_ontology.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def lookup(term: str) -> Optional[list]:
    """
    O(n) scan of the loaded ontology.
    Keys are single strings with aliases separated by " / ".
    Returns a list of {spec_key: value} dicts if matched, None if unmatched.
    Unmatched terms fall through to the LLM.
    """
    global _ontology
    if not _ontology:
        _ontology = _load()

    term_lower = term.lower().strip()

    for category, entries in _ontology.items():
        for key, specs in entries.items():
            aliases = [a.strip().lower() for a in key.split("/")]
            if any(term_lower == alias or term_lower in alias or alias in term_lower for alias in aliases):
                return [{k: v} for k, v in specs.items() if k != "note"]

    return None


def lookup_all(terms: list[str]) -> list[dict]:
    """Run lookup for each term, accumulate unique matched specs."""
    seen_keys: set[str] = set()
    specs = []
    for term in terms:
        result = lookup(term)
        if result:
            for spec in result:
                key = list(spec.keys())[0]
                if key not in seen_keys:
                    seen_keys.add(key)
                    specs.append(spec)
    return specs
