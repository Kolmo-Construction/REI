"""
Deterministic file-editing tools for the autonomous optimization pipeline.

Public API
----------
read_node_value(filepath, node_name) -> str | list | set | None
patch_prompt(filepath, node_name, new_system_prompt) -> str   (old value)
patch_list_literal(filepath, node_name, new_items) -> list    (old items)
patch_phrase_list(filepath, new_phrases) -> list              (old phrases)
update_ontology(yaml_path, intent_key, new_specs,
                category=None) -> dict                        (old specs)

All write operations validate syntax before committing and raise EditError
on any failure so the caller can safely revert via git.
"""
from __future__ import annotations

import ast
import io
import sys
from pathlib import Path
from typing import Any

# ruamel.yaml preserves YAML comments; fall back to pyyaml
try:
    from ruamel.yaml import YAML as _RuamelYAML
    _RUAMEL_AVAILABLE = True
except ImportError:
    _RUAMEL_AVAILABLE = False
    import yaml as _yaml  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

ALLOWED_PY_FILES: frozenset[Path] = frozenset(
    (REPO_ROOT / "greenvest" / "nodes" / name).resolve()
    for name in (
        "synthesizer.py",
        "query_translator.py",
        "intent_router.py",
        "clarification_gate.py",
    )
)

ONTOLOGY_PATH: Path = (
    REPO_ROOT / "greenvest" / "ontology" / "gear_ontology.yaml"
).resolve()

ONTOLOGY_CATEGORIES = frozenset(
    {"sleeping_bags", "sleeping_pads", "footwear", "jackets", "backpacks"}
)

# Single source of truth for files the optimizer is allowed to edit.
# autonomous_optimize.py imports this instead of maintaining its own hardcoded list.
EDITABLE_FILES_REL: tuple[str, ...] = tuple(
    str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    for p in sorted(ALLOWED_PY_FILES)
) + ("greenvest/ontology/gear_ontology.yaml",)

# The query_translator file and the function name that holds the phrase list
_QUERY_TRANSLATOR = (REPO_ROOT / "greenvest" / "nodes" / "query_translator.py").resolve()
_EXTRACT_TERMS_FUNC = "_extract_terms"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class EditError(Exception):
    """Raised when a safe edit cannot be performed."""


# ---------------------------------------------------------------------------
# Internal AST helpers
# ---------------------------------------------------------------------------

def _assert_allowed_py(path: Path) -> None:
    resolved = path.resolve()
    if resolved not in ALLOWED_PY_FILES:
        raise EditError(
            f"File not in allowed edit list: {resolved}\n"
            f"Allowed: {sorted(str(p) for p in ALLOWED_PY_FILES)}"
        )


def _find_module_assign(tree: ast.Module, node_name: str) -> ast.Assign | None:
    """Return the first module-level simple assignment with the given name."""
    for stmt in ast.walk(tree):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == node_name
        ):
            return stmt
    return None


def _literal_eval_node(val_node: ast.expr) -> Any:
    try:
        return ast.literal_eval(val_node)
    except (ValueError, TypeError):
        return None


def _replace_lines(source: str, start_lineno: int, end_lineno: int, replacement: str) -> str:
    """
    Replace lines [start_lineno, end_lineno] (1-indexed, inclusive) with replacement.
    replacement should already end with a newline.
    """
    lines = source.splitlines(keepends=True)
    return "".join(lines[: start_lineno - 1] + [replacement] + lines[end_lineno:])


def _validate_python(source: str, context: str = "") -> None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise EditError(
            f"Edit produced invalid Python syntax{' in ' + context if context else ''}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public: read
# ---------------------------------------------------------------------------

def read_node_value(filepath: str | Path, node_name: str) -> str | list | set | None:
    """
    Read the current value of a module-level assignment in an allowed .py file.
    Returns the Python value (str, list, set) or None if not found / unevaluable.
    """
    path = Path(filepath).resolve()
    _assert_allowed_py(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assign = _find_module_assign(tree, node_name)
    if assign is None:
        return None
    return _literal_eval_node(assign.value)


# ---------------------------------------------------------------------------
# Public: patch_prompt
# ---------------------------------------------------------------------------

def patch_prompt(filepath: str | Path, node_name: str, new_system_prompt: str) -> str:
    """
    Replace a module-level string constant in an allowed node .py file.

    Handles parenthesised multi-line string concatenation (Python folds adjacent
    string literals into a single ast.Constant at parse time, so the AST value
    is always a plain str regardless of source layout).

    Parameters
    ----------
    filepath : path to the target .py file
    node_name : module-level variable name (e.g. '_REI_PERSONA')
    new_system_prompt : replacement string (must be non-empty)

    Returns
    -------
    The old string value that was replaced.

    Raises
    ------
    EditError on any validation failure.
    """
    path = Path(filepath).resolve()
    _assert_allowed_py(path)

    if not isinstance(new_system_prompt, str) or not new_system_prompt.strip():
        raise EditError("new_system_prompt must be a non-empty string.")

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise EditError(f"Source file has syntax errors before edit: {exc}") from exc

    assign = _find_module_assign(tree, node_name)
    if assign is None:
        raise EditError(f"Variable '{node_name}' not found as a module-level assignment in {path}.")

    val_node = assign.value
    if not (isinstance(val_node, ast.Constant) and isinstance(val_node.value, str)):
        raise EditError(
            f"'{node_name}' is not a simple string constant (found {type(val_node).__name__}). "
            "It may be an f-string or non-string — cannot safely patch."
        )

    old_value: str = val_node.value
    new_source = _replace_lines(
        source,
        assign.lineno,
        assign.end_lineno,
        f"{node_name} = {repr(new_system_prompt)}\n",
    )
    _validate_python(new_source, str(path))
    path.write_text(new_source, encoding="utf-8")
    return old_value


# ---------------------------------------------------------------------------
# Public: patch_list_literal
# ---------------------------------------------------------------------------

def patch_list_literal(
    filepath: str | Path,
    node_name: str,
    new_items: list | set | frozenset,
) -> list:
    """
    Replace a module-level list, set, or frozenset constant in an allowed node .py file.

    Parameters
    ----------
    filepath : path to the target .py file
    node_name : module-level variable name (e.g. '_ENV_SENSITIVE_ACTIVITIES')
    new_items : replacement collection (list or set of strings/ints)

    Returns
    -------
    The old items as a list.

    Raises
    ------
    EditError on any validation failure.
    """
    path = Path(filepath).resolve()
    _assert_allowed_py(path)

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise EditError(f"Source file has syntax errors before edit: {exc}") from exc

    assign = _find_module_assign(tree, node_name)
    if assign is None:
        raise EditError(f"Variable '{node_name}' not found as a module-level assignment in {path}.")

    val_node = assign.value
    old_value = _literal_eval_node(val_node)
    if old_value is None:
        raise EditError(
            f"'{node_name}' value could not be evaluated as a literal. "
            "It may reference other variables — cannot safely patch."
        )

    # Build replacement text preserving the original collection type
    if isinstance(val_node, ast.Set):
        items_repr = ",\n    ".join(repr(i) for i in sorted(new_items))
        replacement = f"{node_name} = {{\n    {items_repr},\n}}\n"
    elif isinstance(val_node, ast.List):
        items_repr = ",\n    ".join(repr(i) for i in new_items)
        replacement = f"{node_name} = [\n    {items_repr},\n]\n"
    else:
        raise EditError(
            f"'{node_name}' is a {type(val_node).__name__}, not a list or set literal."
        )

    new_source = _replace_lines(source, assign.lineno, assign.end_lineno, replacement)
    _validate_python(new_source, str(path))
    path.write_text(new_source, encoding="utf-8")
    return list(old_value)


# ---------------------------------------------------------------------------
# Public: patch_phrase_list  (query_translator._extract_terms special case)
# ---------------------------------------------------------------------------

def patch_phrase_list(filepath: str | Path, new_phrases: list[str]) -> list[str]:
    """
    Replace the phrase lookup list inside _extract_terms() in query_translator.py.

    The list has a stable, well-known structure:
        for phrase in ["PNW", "winter camping", ...]:

    Parameters
    ----------
    filepath : must point to query_translator.py (path-checked)
    new_phrases : the complete replacement list of phrase strings

    Returns
    -------
    The old phrase list.

    Raises
    ------
    EditError on any validation failure.
    """
    path = Path(filepath).resolve()
    _assert_allowed_py(path)
    if path != _QUERY_TRANSLATOR:
        raise EditError(
            "patch_phrase_list only targets query_translator.py. "
            f"Got: {path}"
        )
    if not new_phrases:
        raise EditError("new_phrases must not be empty.")

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise EditError(f"Source file has syntax errors before edit: {exc}") from exc

    # Find the list literal inside _extract_terms's for loop
    target_list: ast.List | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _EXTRACT_TERMS_FUNC:
            for child in ast.walk(node):
                if isinstance(child, ast.For) and isinstance(child.iter, ast.List):
                    target_list = child.iter
                    break
            break

    if target_list is None:
        raise EditError(
            f"Could not find a 'for phrase in [...]' list literal "
            f"in {_EXTRACT_TERMS_FUNC}() within {path}."
        )

    old_items = _literal_eval_node(target_list)
    if old_items is None:
        raise EditError("Phrase list contains non-literal elements — cannot safely patch.")

    # Superset guard: new list must contain every phrase that was already there.
    # Dropping phrases is almost certainly an LLM mistake, not an intentional fix.
    old_set = set(old_items)
    new_set = set(new_phrases)
    dropped = old_set - new_set
    if dropped:
        raise EditError(
            f"patch_phrase_list would drop {len(dropped)} existing phrase(s): "
            f"{sorted(dropped)}. "
            "new_phrases must be a superset of the current phrase list."
        )

    # Replace the list literal's exact character span using AST-reported coordinates.
    # This is purely position-based — no regex — so it survives any reformatting of
    # the surrounding for-loop as long as the list literal itself is intact.
    #
    # AST col_offset / end_col_offset are byte offsets into the source line, which
    # equals character offset for ASCII/UTF-8 source without BOM (standard Python).
    lines = source.splitlines(keepends=True)
    start_line_idx = target_list.lineno - 1      # 0-indexed line containing [
    end_line_idx = target_list.end_lineno - 1    # 0-indexed line containing ]
    col_start = target_list.col_offset           # column of [
    col_end = target_list.end_col_offset         # column of the character after ]

    items_repr = ", ".join(repr(p) for p in new_phrases)
    new_list_str = f"[{items_repr}]"

    if start_line_idx == end_line_idx:
        # Single-line list: splice within one source line
        line = lines[start_line_idx]
        new_line = line[:col_start] + new_list_str + line[col_end:]
        new_source_lines = lines[:start_line_idx] + [new_line] + lines[end_line_idx + 1:]
    else:
        # Multi-line list: take prefix from first line, suffix from last line
        first_line = lines[start_line_idx]
        last_line = lines[end_line_idx]
        new_line = first_line[:col_start] + new_list_str + last_line[col_end:]
        new_source_lines = lines[:start_line_idx] + [new_line] + lines[end_line_idx + 1:]

    new_source = "".join(new_source_lines)

    _validate_python(new_source, str(path))
    path.write_text(new_source, encoding="utf-8")
    return list(old_items)


# ---------------------------------------------------------------------------
# Public: update_ontology
# ---------------------------------------------------------------------------

def update_ontology(
    yaml_path: str | Path,
    intent_key: str,
    new_specs: dict,
    category: str | None = None,
) -> dict:
    """
    Safely parse, update, and write back gear_ontology.yaml.

    Searches all categories (or just `category` if specified) for `intent_key`
    and updates its spec dict.  If `intent_key` does not exist it is created
    under the given `category` (required when adding a new key).

    Parameters
    ----------
    yaml_path : path to gear_ontology.yaml (must match ONTOLOGY_PATH)
    intent_key : the alias key string, e.g. 'winter camping / winter camp'
    new_specs : dict of spec key→value pairs to merge (not replace) into existing specs
    category : optional category name; required when adding a new intent_key

    Returns
    -------
    The old spec dict (empty dict if key was new).

    Raises
    ------
    EditError on validation failure.
    """
    path = Path(yaml_path).resolve()
    if path != ONTOLOGY_PATH:
        raise EditError(
            f"update_ontology only targets {ONTOLOGY_PATH}. Got: {path}"
        )
    if not new_specs:
        raise EditError("new_specs must not be empty.")
    if category is not None and category not in ONTOLOGY_CATEGORIES:
        raise EditError(
            f"Unknown category '{category}'. "
            f"Valid: {sorted(ONTOLOGY_CATEGORIES)}"
        )
    _validate_ontology_specs(new_specs)

    raw = path.read_text(encoding="utf-8")

    if _RUAMEL_AVAILABLE:
        return _update_ontology_ruamel(path, raw, intent_key, new_specs, category)
    else:
        return _update_ontology_pyyaml(path, raw, intent_key, new_specs, category)


def _update_ontology_ruamel(
    path: Path,
    raw: str,
    intent_key: str,
    new_specs: dict,
    category: str | None,
) -> dict:
    from ruamel.yaml import YAML
    yml = YAML()
    yml.preserve_quotes = True
    data = yml.load(raw)

    found_cat, old_specs = _find_intent_key(data, intent_key, category)
    if found_cat is None:
        if category is None:
            raise EditError(
                f"Intent key '{intent_key}' not found in any category. "
                "Provide 'category' to create a new entry."
            )
        found_cat = category
        if data.get(found_cat) is None:
            raise EditError(f"Category '{found_cat}' not found in ontology.")
        data[found_cat][intent_key] = {}

    # Merge specs
    for k, v in new_specs.items():
        data[found_cat][intent_key][k] = v

    buf = io.StringIO()
    yml.dump(data, buf)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return old_specs


def _update_ontology_pyyaml(
    path: Path,
    raw: str,
    intent_key: str,
    new_specs: dict,
    category: str | None,
) -> dict:
    data = _yaml.safe_load(raw)
    found_cat, old_specs = _find_intent_key(data, intent_key, category)
    if found_cat is None:
        if category is None:
            raise EditError(
                f"Intent key '{intent_key}' not found in any category. "
                "Provide 'category' to create a new entry."
            )
        found_cat = category
        if data.get(found_cat) is None:
            raise EditError(f"Category '{found_cat}' not found in ontology.")
        data[found_cat][intent_key] = {}

    for k, v in new_specs.items():
        data[found_cat][intent_key][k] = v

    path.write_text(_yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(
        "[WARN] ruamel.yaml not installed — ontology saved without comment preservation. "
        "Run: uv add ruamel.yaml",
        file=sys.stderr,
    )
    return old_specs


def _validate_ontology_specs(new_specs: dict) -> None:
    """
    Raise EditError if any spec key or value in new_specs is invalid.

    Rules (derived from gear_ontology.yaml conventions):
    - Every key must be a valid Python identifier (snake_case recommended).
      Uses str.isidentifier() which rejects empty strings, strings starting
      with digits, and strings containing spaces or punctuation.
    - Every value must be a non-empty string.
      Numeric specs (temp_rating_f, weight_oz, …) are stored as quoted
      strings in the YAML (e.g. "<=15", "<32") so non-string values from
      the LLM are rejected before touching the file.
    """
    for k, v in new_specs.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise EditError(
                f"Invalid ontology spec key {k!r}: must be a non-empty identifier "
                "(letters, digits, underscores; must not start with a digit). "
                f"Got type {type(k).__name__}."
            )
        if not isinstance(v, str) or not v.strip():
            raise EditError(
                f"Ontology spec value for key {k!r} must be a non-empty string; "
                f"got {v!r} (type {type(v).__name__}). "
                "Numeric specs are stored as quoted strings, e.g. '\"<=15\"'."
            )


def _find_intent_key(
    data: dict, intent_key: str, prefer_category: str | None
) -> tuple[str | None, dict]:
    """
    Search categories for intent_key.
    Returns (category_name, existing_specs) or (None, {}) if not found.
    """
    search_order = (
        [prefer_category] if prefer_category else list(ONTOLOGY_CATEGORIES)
    )
    for cat in search_order:
        cat_data = data.get(cat)
        if cat_data and intent_key in cat_data:
            specs = dict(cat_data[intent_key]) if cat_data[intent_key] else {}
            return cat, specs
    return None, {}
