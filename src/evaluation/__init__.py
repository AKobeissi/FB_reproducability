"""
Evaluation package exports.
This module intentionally avoids importing optional-heavy dependencies at import time.
Several evaluators depend on optional packages (e.g., sacrebleu, rouge-score, ragas,
bert-score). Importing them eagerly breaks lightweight workflows that only need a
subset of functionality (e.g., `Evaluator`).
"""

from __future__ import annotations

from typing import Any, Optional

RetrievalEvaluator: Optional[Any]
GenerativeEvaluator: Optional[Any]
run_scoring: Optional[Any]

try:  # pragma: no cover
    from .retrieval_evaluator import RetrievalEvaluator  # type: ignore
except Exception:  # pragma: no cover
    RetrievalEvaluator = None

try:  # pragma: no cover
    from .generative_evaluator import GenerativeEvaluator  # type: ignore
except Exception:  # pragma: no cover
    GenerativeEvaluator = None

try:  # pragma: no cover
    from .evaluate_outputs import run_scoring  # type: ignore
except Exception:  # pragma: no cover
    run_scoring = None

__all__ = ["RetrievalEvaluator", "GenerativeEvaluator", "run_scoring"]