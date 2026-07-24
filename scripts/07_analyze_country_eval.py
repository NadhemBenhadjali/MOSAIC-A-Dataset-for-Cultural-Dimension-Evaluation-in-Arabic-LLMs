#!/usr/bin/env python
"""Summarize country-conditioned MOSAIC model outputs."""

from mosaic.analysis import summarize_country_results
from mosaic.constants import RESULTS_DIR


if __name__ == "__main__":
    summary = summarize_country_results(
        RESULTS_DIR / "raw" / "country_eval",
        RESULTS_DIR / "processed" / "country_eval_summary.csv",
    )
    print(summary.to_string(index=False))
