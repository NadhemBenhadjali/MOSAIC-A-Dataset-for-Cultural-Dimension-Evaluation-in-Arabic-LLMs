#!/usr/bin/env python
"""Summarize standard MOSAIC model outputs."""

from mosaic.analysis import summarize_standard_results
from mosaic.constants import RESULTS_DIR


if __name__ == "__main__":
    summary = summarize_standard_results(
        RESULTS_DIR / "raw" / "standard_eval",
        RESULTS_DIR / "processed" / "standard_eval_summary.csv",
    )
    print(summary.to_string(index=False))
