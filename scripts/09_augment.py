#!/usr/bin/env python
"""Generate new MOSAIC-style scenarios with few-shot prompting."""

from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.augmentation import augment_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--model", default="QCRI/Fanar-1-9B-Instruct")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_csv(
        args.input,
        args.output,
        dimension=args.dimension,
        n_generations=args.n,
        model_name=args.model,
    )
    print(f"Saved {args.output}")
