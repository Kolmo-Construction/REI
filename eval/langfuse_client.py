"""
Langfuse singleton client for Greenvest eval infrastructure.

Returns None gracefully when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set,
so every caller can do `lf = langfuse_client(); if lf: ...` without crashing.
"""
from __future__ import annotations

import functools
import sys
from typing import Optional

from greenvest.config import settings


@functools.lru_cache(maxsize=1)
def langfuse_client():
    """Return a cached Langfuse instance, or None if keys are not configured."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except ImportError:
        print("[WARN] langfuse package not installed.", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[WARN] Langfuse init failed: {exc}", file=sys.stderr)
        return None


def langfuse_api_client():
    """Return a Langfuse AsyncLangfuseAPI client for reading traces, or None."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse.api import AsyncLangfuseAPI
        return AsyncLangfuseAPI(
            base_url=settings.LANGFUSE_HOST,
            username=settings.LANGFUSE_PUBLIC_KEY,
            password=settings.LANGFUSE_SECRET_KEY,
        )
    except ImportError:
        return None
    except Exception as exc:
        print(f"[WARN] Langfuse API client init failed: {exc}", file=sys.stderr)
        return None
