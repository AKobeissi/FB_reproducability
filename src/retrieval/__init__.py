"""Retrieval package.

Keep imports lazy to avoid circular dependencies during module initialization.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_chroma_store", "run_bm25", "run_big2small"]


def __getattr__(name: str) -> Any:
	if name == "build_chroma_store":
		from .vectorstore import build_chroma_store
		return build_chroma_store
	if name == "run_bm25":
		from .bm25 import run_bm25
		return run_bm25
	if name == "run_big2small":
		from .big2small import run_big2small
		return run_big2small
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")