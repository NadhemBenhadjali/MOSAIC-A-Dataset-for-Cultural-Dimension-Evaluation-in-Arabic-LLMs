#!/usr/bin/env python
"""Translate an English scenario CSV into culturally adapted Arabic."""

from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.translation import translate_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="QCRI/Fanar-1-9B-Instruct")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    translate_csv(args.input, args.output, model_name=args.model)
    print(f"Saved {args.output}")
