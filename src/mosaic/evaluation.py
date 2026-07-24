"""Run MOSAIC prompts through a model and parse responses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .constants import SEEDS
from .io import read_csv, safe_literal, write_csv
from .model_runner import generate_response
from .parsing import map_to_original_option, parse_option_number


def prepare_prompt_dataframe(path: Path) -> pd.DataFrame:
    """Read prompt CSVs and parse dictionary/list columns."""

    df = read_csv(path)
    for column in ["options", "number_to_original_option"]:
        if column in df.columns:
            df[column] = df[column].apply(safe_literal)
    return df


def run_model_on_prompts(
    prompts: pd.DataFrame,
    *,
    model_alias: str,
    seeds: list[int] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run one model on a prompt dataframe."""

    seeds = seeds or SEEDS
    if limit is not None:
        prompts = prompts.head(limit).copy()

    rows = []
    for seed in seeds:
        for _, row in tqdm(
            prompts.iterrows(),
            total=len(prompts),
            desc=f"{model_alias} seed={seed}",
        ):
            response = generate_response(
                row["prompt"],
                model_alias=model_alias,
                seed=seed,
            )
            displayed = parse_option_number(response)
            original = map_to_original_option(
                displayed,
                row["number_to_original_option"],
            )

            rows.append(
                {
                    "row_id": row.get("row_id", row.name),
                    "question_idx": row.get("question_idx", row.name),
                    "dimension": row.get("dimension"),
                    "country": row.get("country"),
                    "seed": seed,
                    "response_text": response,
                    "displayed_option": displayed,
                    "selected_option": original,
                }
            )

    return pd.DataFrame(rows)


def run_prompt_file(
    prompt_file: Path,
    output_file: Path,
    *,
    model_alias: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run one model over one saved prompt CSV."""

    prompts = prepare_prompt_dataframe(prompt_file)
    results = run_model_on_prompts(
        prompts,
        model_alias=model_alias,
        limit=limit,
    )
    write_csv(results, output_file)
    return results
