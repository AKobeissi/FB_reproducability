"""Compatibility shim for the historical ``rag_experiments.py`` CLI."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from financebench_rag.experiments.orchestrator import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
