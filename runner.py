"""Compatibility shim for the FinanceBench RAG runner.

This preserves the old ``python runner.py`` workflow while the real CLI now
lives under ``financebench_rag.cli.runner``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from financebench_rag.cli.runner import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
