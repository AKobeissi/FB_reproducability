"""FinanceBench RAG experiment toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .experiments.orchestrator import RAGExperiment as _RAGExperiment

__all__ = ["RAGExperiment"]


def __getattr__(name: str):
    if name == "RAGExperiment":
        from .experiments.orchestrator import RAGExperiment as _RAGExperiment

        return _RAGExperiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
