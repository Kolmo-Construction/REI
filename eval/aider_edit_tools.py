"""
Aider-backed edit engine for Python files in the autonomous optimization pipeline.

Replaces the manual AST manipulation in edit_tools.py for three edit types:
  - patch_prompt       — replace a module-level string constant
  - patch_phrase_list  — replace the phrase list in _extract_terms()
  - patch_set_literal  — replace a module-level set/list literal

update_ontology() remains in edit_tools.py — YAML editing is more reliable custom.
read_node_value() remains in edit_tools.py — used for context building only.

Aider is installed as a standalone tool (uv tool install aider-chat) and invoked
via subprocess to avoid dependency conflicts with the main project.

Public API
----------
apply_python_edit(filepath, instruction, model=None) -> bool
build_patch_prompt_instruction(node_name, new_value) -> str
build_patch_phrase_list_instruction(new_phrases) -> str
build_patch_set_literal_instruction(node_name, new_items) -> str
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from greenvest.config import settings
from eval.edit_tools import ALLOWED_PY_FILES, EditError

REPO_ROOT = Path(__file__).parent.parent


def _find_aider() -> str | None:
    """Return the path to the aider executable, or None if not found."""
    # Check uv tool bin directory first, then PATH
    uv_tool_bin = Path.home() / ".local" / "bin"
    candidate = uv_tool_bin / "aider"
    if candidate.exists():
        return str(candidate)
    return shutil.which("aider")


def apply_python_edit(
    filepath: str | Path,
    instruction: str,
    model: str | None = None,
) -> bool:
    """
    Apply a natural-language edit instruction to an allowed Python file using Aider.

    Aider validates Python syntax before writing — malformed edits are rejected
    before touching the file. No git commit is made; autonomous_optimize.py
    manages all commits.

    Parameters
    ----------
    filepath    : path to the target .py file (must be in ALLOWED_PY_FILES)
    instruction : precise description of the exact edit to make
    model       : Aider model string; defaults to ollama/{OLLAMA_CODER_MODEL}

    Returns True on success, False on failure.
    Raises EditError if the file is not in the allowed edit list.
    """
    path = Path(filepath).resolve()
    if path not in ALLOWED_PY_FILES:
        raise EditError(
            f"File not in allowed edit list: {path}\n"
            f"Allowed: {sorted(str(p) for p in ALLOWED_PY_FILES)}"
        )

    aider_bin = _find_aider()
    if not aider_bin:
        raise EditError(
            "aider executable not found. Run: uv tool install aider-chat"
        )

    if model is None:
        model = f"ollama/{settings.OLLAMA_CODER_MODEL}"

    env = os.environ.copy()
    # Aider / litellm uses OLLAMA_API_BASE, not OLLAMA_BASE_URL
    env.setdefault("OLLAMA_API_BASE", settings.OLLAMA_BASE_URL)

    cmd = [
        aider_bin,
        "--model", model,
        "--yes",              # non-interactive: auto-accept all prompts
        "--no-auto-commits",  # autonomous_optimize.py manages all commits
        "--no-git",           # skip git integration entirely; we manage git
        "--message", instruction,
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=120,
        )
        if result.returncode != 0:
            print(
                f"[WARN] Aider exited with code {result.returncode} on {path.name}:\n"
                f"{result.stderr[-500:] if result.stderr else '(no stderr)'}",
                file=sys.stderr,
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[WARN] Aider timed out editing {path.name}.", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[WARN] Aider subprocess failed on {path.name}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Instruction builders
# Produce precise, deterministic Aider instructions from the optimizer LLM's
# already-computed replacement values so Aider's job is a simple search/replace.
# ---------------------------------------------------------------------------

def build_patch_prompt_instruction(node_name: str, new_value: str) -> str:
    """Return an Aider instruction to replace a module-level string constant."""
    return (
        f"Replace the `{node_name}` string constant with exactly this assignment:\n\n"
        f"{node_name} = {repr(new_value)}\n\n"
        "Make no other changes to the file."
    )


def build_patch_phrase_list_instruction(new_phrases: list[str]) -> str:
    """Return an Aider instruction to replace the phrase list in _extract_terms()."""
    phrases_repr = ", ".join(repr(p) for p in new_phrases)
    return (
        "In the `_extract_terms` function, find the `for phrase in [...]` loop "
        "and replace the list literal with exactly:\n\n"
        f"[{phrases_repr}]\n\n"
        "Make no other changes to the file."
    )


def build_patch_set_literal_instruction(node_name: str, new_items: list) -> str:
    """Return an Aider instruction to replace a module-level set literal."""
    items_repr = ",\n    ".join(repr(i) for i in sorted(new_items))
    return (
        f"Replace the `{node_name}` set with exactly:\n\n"
        f"{node_name} = {{\n    {items_repr},\n}}\n\n"
        "Make no other changes to the file."
    )
