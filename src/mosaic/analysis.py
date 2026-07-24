"""Analysis helpers for MOSAIC model outputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DIMENSIONS, MODEL_DISPLAY_NAMES
from .io import read_csv, write_csv


def infer_dimension_from_name(file_name: str) -> str | None:
    """Infer a dimension code from a result filename."""

    if "individualistic_vs_collectivist" in file_name:
        return "idv"
    if "long_term_vs_short_term_orientation" in file_name:
        return "lto"
    if "mas_prompts" in file_name:
        return "mas"
    if "power_distance_index" in file_name:
        return "pdi"
    if "uncertainty_avoidance" in file_name:
        return "uai"
    return None


def infer_model_from_name(file_name: str) -> str:
    """Infer the model alias from a result filename."""

    known = sorted(MODEL_DISPLAY_NAMES, key=len, reverse=True)
    for model in known:
        if re.search(rf"_{re.escape(model)}_", file_name):
            return model
    return "unknown"


def standard_score(df: pd.DataFrame, reported_option: int) -> float:
    """Return the percentage of answers selecting the paper-reported side."""

    selected = pd.to_numeric(df["selected_option"], errors="coerce")
    if selected.notna().sum() == 0:
        return np.nan
    return round((selected == reported_option).mean() * 100, 3)


def summarize_standard_results(input_dir: Path, output_file: Path) -> pd.DataFrame:
    """Summarize standard evaluation CSVs by model and dimension."""

    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        dimension_code = infer_dimension_from_name(path.name)
        if dimension_code is None:
            continue

        dimension = DIMENSIONS[dimension_code]
        df = read_csv(path)
        selected_col = "selected_option"
        if selected_col not in df.columns and "scale_position" in df.columns:
            selected_col = "scale_position"
        if selected_col != "selected_option":
            df = df.rename(columns={selected_col: "selected_option"})

        rows.append(
            {
                "model": infer_model_from_name(path.name),
                "model_name": MODEL_DISPLAY_NAMES.get(
                    infer_model_from_name(path.name),
                    infer_model_from_name(path.name),
                ),
                "dimension": dimension_code,
                "dimension_name": dimension.name,
                "reported_label": dimension.reported_label,
                "reported_option": dimension.reported_option,
                "score_percent": standard_score(df, dimension.reported_option),
                "n_rows": len(df),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["dimension", "model"])
    write_csv(summary, output_file)
    return summary


def summarize_country_results(input_dir: Path, output_file: Path) -> pd.DataFrame:
    """Summarize country-conditioned runs by model, country, and dimension."""

    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        dimension_code = infer_dimension_from_name(path.name)
        if dimension_code is None:
            continue

        df = read_csv(path)
        if "country" not in df.columns:
            continue

        dimension = DIMENSIONS[dimension_code]
        selected_col = "selected_option"
        if selected_col not in df.columns and "scale_position" in df.columns:
            selected_col = "scale_position"
        df = df.rename(columns={selected_col: "selected_option"})
        model = infer_model_from_name(path.name)

        for country, group in df.groupby("country"):
            rows.append(
                {
                    "model": model,
                    "model_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "country": country,
                    "dimension": dimension_code,
                    "dimension_name": dimension.name,
                    "reported_label": dimension.reported_label,
                    "score_percent": standard_score(
                        group,
                        dimension.reported_option,
                    ),
                    "n_rows": len(group),
                }
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["dimension", "model", "country"])
    write_csv(summary, output_file)
    return summary
