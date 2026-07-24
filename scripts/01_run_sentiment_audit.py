#!/usr/bin/env python
"""Run sentiment audit for one or all MOSAIC dimension files."""

from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.constants import DATA_DIR, DIMENSIONS
from mosaic.sentiment import audit_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", choices=sorted(DIMENSIONS), default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dimensions = [DIMENSIONS[args.dimension]] if args.dimension else DIMENSIONS.values()

    for dimension in dimensions:
        input_file = DATA_DIR / "processed" / dimension.source_file
        output_file = DATA_DIR / "audits" / "sentiment" / dimension.source_file
        output_file = output_file.with_name(f"{output_file.stem}_sentiment.csv")
        audit_file(input_file, output_file, batch_size=args.batch_size)
        print(f"Saved {output_file}")
