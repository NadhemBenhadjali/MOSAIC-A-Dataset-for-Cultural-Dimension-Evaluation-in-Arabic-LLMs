"""Input/output helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    """Read a UTF-8 CSV file."""

    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a UTF-8 CSV file, creating parent folders first."""

    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_literal(value: Any) -> Any:
    """Parse list/dict strings saved by pandas, leaving normal values intact."""

    if isinstance(value, (list, dict)):
        return value
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value
