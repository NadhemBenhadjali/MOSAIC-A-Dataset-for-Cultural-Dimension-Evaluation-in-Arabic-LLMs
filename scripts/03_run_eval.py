#!/usr/bin/env python
"""Run a model on standard MOSAIC evaluation prompts."""

from __future__ import annotations

import argparse

from mosaic.constants import MODEL_REPOS, RESULTS_DIR
from mosaic.evaluation import run_prompt_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_REPOS), required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_file = RESULTS_DIR / "prompts" / "standard_eval"
    prompt_file = prompt_file / f"{args.dimension}_standard_prompts.csv"
    output_file = RESULTS_DIR / "raw" / "standard_eval"
    output_file = output_file / f"{args.dimension}_{args.model}_responses.csv"

    run_prompt_file(
        prompt_file,
        output_file,
        model_alias=args.model,
        limit=args.limit,
    )
    print(f"Saved {output_file}")
