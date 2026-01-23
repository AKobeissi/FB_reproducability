"""
Unified RAG Pipeline (legacy module).

Historically this file contained the full orchestration logic. To make the repo
easier to navigate, the canonical implementation now lives under `src/pipeline/`.

This file remains as a thin wrapper to preserve the existing import paths:
`from src.experiments.unified_pipeline import run_unified_pipeline`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.pipeline.standard_pipeline import StandardPipelineConfig, run_standard_pipeline

logger = logging.getLogger(__name__)


def run_unified_pipeline(experiment: Any, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    config = StandardPipelineConfig(
        use_hyde=bool(getattr(experiment, "unified_use_hyde", False)),
        hyde_k=int(getattr(experiment, "unified_hyde_k", 1)),
        retrieval_mode=str(getattr(experiment, "unified_retrieval", "dense")).lower(),  # type: ignore[arg-type]
        use_rerank=bool(getattr(experiment, "unified_use_rerank", False)),
    )
    return run_standard_pipeline(experiment, data, config=config)