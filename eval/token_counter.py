"""Session-wide token counter for LLM calls in the optimization pipeline.

Handles both Anthropic and Ollama response_metadata shapes:
  - Anthropic: response_metadata["usage"] = {"input_tokens": N, "output_tokens": N}
  - Ollama:    response_metadata has "prompt_eval_count" / "eval_count" keys
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def __repr__(self) -> str:
        return f"TokenUsage(in={self.input_tokens}, out={self.output_tokens}, total={self.total})"


# Module-level session accumulators — reset between sessions via reset_session()
_session_usage: dict[str, TokenUsage] = defaultdict(TokenUsage)


def record(role: str, response_metadata: dict) -> TokenUsage:
    """Extract token counts from LangChain response_metadata and accumulate.

    Parameters
    ----------
    role : "critic" | "optimizer" (or any string key)
    response_metadata : the .response_metadata dict from a LangChain AIMessage
    """
    usage = _extract_usage(response_metadata)
    _session_usage[role] += usage
    return usage


def get_session_totals() -> dict[str, TokenUsage]:
    """Return a snapshot of accumulated token usage by role."""
    return dict(_session_usage)


def reset_session() -> None:
    """Reset all accumulators to zero."""
    _session_usage.clear()


def _extract_usage(metadata: dict) -> TokenUsage:
    """Extract token counts from LangChain response_metadata.

    Handles both Anthropic and Ollama metadata shapes.
    """
    usage_block = metadata.get("usage") or {}
    inp = int(
        usage_block.get("input_tokens")
        or metadata.get("prompt_eval_count")
        or 0
    )
    out = int(
        usage_block.get("output_tokens")
        or metadata.get("eval_count")
        or 0
    )
    return TokenUsage(input_tokens=inp, output_tokens=out)
