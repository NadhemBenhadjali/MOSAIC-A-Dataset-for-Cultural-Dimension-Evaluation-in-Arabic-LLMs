#!/usr/bin/env python
"""Build normalized MOSAIC CSV files from the final dimension files."""

from mosaic.dataset import build_mosaic_dataset


if __name__ == "__main__":
    mosaic = build_mosaic_dataset()
    counts = mosaic["dimension"].value_counts().sort_index()
    print("Built data/processed/mosaic_all.csv")
    print(counts.to_string())
