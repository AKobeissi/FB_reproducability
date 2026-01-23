"""
Pipeline modules.

Goal: make the project easier to navigate by separating the *standard pipeline*
(ingest/index → retrieve → optionally expand/rerank → generate → evaluate)
from the growing collection of one-off experiments.
"""

from .standard_pipeline import StandardPipelineConfig, run_standard_pipeline

__all__ = ["StandardPipelineConfig", "run_standard_pipeline"]

