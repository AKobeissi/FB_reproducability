#!/usr/bin/env python3
"""
Compatibility wrapper.

The main implementation lives in `scripts/aggregate_outputs_metrics.py`, but many
users run this module as:

  python -m scripts.aggregate_output_metrics ...
"""

from __future__ import annotations

from .aggregate_outputs_metrics import main


if __name__ == "__main__":
    raise SystemExit(main())

