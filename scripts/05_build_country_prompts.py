#!/usr/bin/env python
"""Create country-conditioned evaluation prompts."""

from __future__ import annotations

import argparse

from mosaic.constants import DATA_DIR, DIMENSIONS, RESULTS_DIR
from mosaic.io import read_csv, write_csv
from mosaic.prompts import create_country_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = RESULTS_DIR / "prompts" / "country_eval"

    for code, dimension in DIMENSIONS.items():
        df = read_csv(DATA_DIR / "processed" / dimension.source_file)
        prompts = create_country_prompts(df, dimension=code, seed=args.seed)
        output_file = output_dir / f"{code}_country_prompts.csv"
        write_csv(prompts, output_file)
        print(f"Saved {output_file}")
