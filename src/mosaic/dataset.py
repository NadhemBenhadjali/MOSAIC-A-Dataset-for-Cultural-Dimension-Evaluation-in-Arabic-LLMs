"""Dataset loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DATA_DIR, DIMENSIONS, Dimension
from .io import read_csv, write_csv


PROMPT_COLUMN_MARKER = "Arabic_English_Prompt"
OPTION_COLUMN_MARKER = "Arabic_Expected_"


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Return the prompt column and the two Arabic option columns."""

    prompt_cols = [col for col in df.columns if PROMPT_COLUMN_MARKER in col]
    if not prompt_cols:
        raise ValueError(f"No column containing {PROMPT_COLUMN_MARKER!r} found.")

    option_cols = [
        col
        for col in df.columns
        if col.startswith(OPTION_COLUMN_MARKER) and PROMPT_COLUMN_MARKER not in col
    ]
    if len(option_cols) != 2:
        raise ValueError(f"Expected two option columns, found {option_cols}.")

    return prompt_cols[0], option_cols[0], option_cols[1]


def normalize_dimension(df: pd.DataFrame, dimension: Dimension) -> pd.DataFrame:
    """Convert one original dimension CSV into the public MOSAIC schema."""

    prompt_col, option_1_col, option_2_col = detect_columns(df)
    rows = []

    for index, row in df.iterrows():
        rows.append(
            {
                "id": f"{dimension.code}_{index + 1:04d}",
                "dimension": dimension.code,
                "dimension_name": dimension.name,
                "prompt_ar": str(row[prompt_col]),
                "option_1_ar": str(row[option_1_col]),
                "option_1_label": dimension.option_1_label,
                "option_2_ar": str(row[option_2_col]),
                "option_2_label": dimension.option_2_label,
                "source_file": dimension.source_file,
            }
        )

    return pd.DataFrame(rows)


def build_mosaic_dataset(
    source_dir: Path = DATA_DIR / "processed",
    output_dir: Path = DATA_DIR / "processed",
) -> pd.DataFrame:
    """Build normalized per-dimension files and ``mosaic_all.csv``."""

    frames = []
    for dimension in DIMENSIONS.values():
        source_path = source_dir / dimension.source_file
        df = read_csv(source_path)
        normalized = normalize_dimension(df, dimension)
        write_csv(normalized, output_dir / f"{dimension.code}.csv")
        frames.append(normalized)

    mosaic = pd.concat(frames, ignore_index=True)
    write_csv(mosaic, output_dir / "mosaic_all.csv")
    return mosaic


def load_mosaic(path: Path = DATA_DIR / "processed" / "mosaic_all.csv") -> pd.DataFrame:
    """Load the normalized MOSAIC dataset."""

    return read_csv(path)
