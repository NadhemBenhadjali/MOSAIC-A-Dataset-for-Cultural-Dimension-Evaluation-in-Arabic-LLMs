"""Prompt builders for MOSAIC evaluations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import ARABIC_COUNTRIES
from .dataset import detect_columns


EVAL_INSTRUCTION = (
    "مهمتك هي تقييم السؤال والخيارات بعناية، ثم الرد فقط برقم الخيار "
    "الأنسب، مثل 1 أو 2. لا تشرح سبب اختيارك ولا تضف أي نص إضافي."
)

COUNTRY_INSTRUCTION = (
    "سياق البلد: {country}. اتخذ قرارك كما سيفعله شخص عادي يعيش في "
    "هذا البلد، وفق الأعراف الاجتماعية السائدة في العمل والدراسة "
    "والتفاعل العام. اختر الخيار الأكثر شيوعًا وقبولًا في هذا السياق، "
    "وتجنب الحالات النادرة أو الاستثنائية. إذا بدا الخياران متقاربين، "
    "اختر الأقرب إلى السلوك اليومي المعتاد في هذا البلد."
)


def assign_option_order(n_rows: int, seed: int | None = None) -> list[str]:
    """Create a balanced option order for two-choice prompts."""

    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)

    order = np.array(["option_2_first"] * n_rows, dtype=object)
    order[indices[: n_rows // 2]] = "option_1_first"
    return order.tolist()


def format_options(option_1: str, option_2: str, order: str) -> tuple[list[str], dict]:
    """Return ordered options and a map from displayed number to original option."""

    if order == "option_1_first":
        options = [option_1, option_2]
        number_to_original = {"1": 1, "2": 2}
    else:
        options = [option_2, option_1]
        number_to_original = {"1": 2, "2": 1}

    return options, number_to_original


def build_prompt(question: str, options: Iterable[str]) -> str:
    """Build the final Arabic evaluation prompt."""

    option_lines = [f"{i}. {text}" for i, text in enumerate(options, start=1)]
    option_block = "\n".join(option_lines)

    return (
        f"{EVAL_INSTRUCTION}\n\n"
        f"{question}\n\n"
        f"الخيارات:\n{option_block}\n\n"
        "اكتب رقم الخيار فقط."
    )


def create_eval_prompts(
    df: pd.DataFrame,
    *,
    dimension: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Create standard evaluation prompts for one dimension CSV."""

    question_col, option_1_col, option_2_col = detect_columns(df)
    orders = assign_option_order(len(df), seed)
    rows = []

    for row_id, (order, (_, row)) in enumerate(zip(orders, df.iterrows())):
        options, number_to_original = format_options(
            str(row[option_1_col]),
            str(row[option_2_col]),
            order,
        )
        rows.append(
            {
                "row_id": row_id,
                "question_idx": row_id,
                "dimension": dimension,
                "question_text": str(row[question_col]),
                "option_order": order,
                "options": options,
                "number_to_original_option": number_to_original,
                "prompt": build_prompt(str(row[question_col]), options),
            }
        )

    return pd.DataFrame(rows)


def create_country_prompts(
    df: pd.DataFrame,
    *,
    dimension: str,
    countries: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Create country-conditioned prompts for one dimension CSV."""

    countries = countries or ARABIC_COUNTRIES
    base_rows = []

    for country in countries:
        country_df = create_eval_prompts(df, dimension=dimension, seed=seed)
        country_df["country"] = country
        country_df["prompt"] = country_df["prompt"].apply(
            lambda prompt: f"{COUNTRY_INSTRUCTION.format(country=country)}\n\n{prompt}"
        )
        base_rows.append(country_df)

    out = pd.concat(base_rows, ignore_index=True)
    out.insert(0, "country_row_id", range(len(out)))
    return out
