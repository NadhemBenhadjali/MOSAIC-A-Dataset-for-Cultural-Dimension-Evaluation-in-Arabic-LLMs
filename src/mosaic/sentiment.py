"""Sentiment audit utilities for MOSAIC option neutrality checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from .dataset import detect_columns
from .io import read_csv, write_csv


DEFAULT_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"


def build_sentiment_pipeline(model_name: str = DEFAULT_SENTIMENT_MODEL):
    """Build an Arabic sentiment pipeline."""

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=256,
        device=device,
    )


def audit_dataframe(
    df: pd.DataFrame,
    sentiment_pipe,
    *,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Append sentiment labels and a divergence flag for the two options."""

    _, option_1_col, option_2_col = detect_columns(df)
    texts = df[option_1_col].astype(str).tolist()
    texts += df[option_2_col].astype(str).tolist()

    predictions = []
    for start in tqdm(range(0, len(texts), batch_size), desc="sentiment"):
        predictions.extend(sentiment_pipe(texts[start : start + batch_size]))

    n_rows = len(df)
    option_1 = predictions[:n_rows]
    option_2 = predictions[n_rows:]
    audited = df.copy()

    audited["sentiment_1"] = [item["label"] for item in option_1]
    audited["score_1"] = [item["score"] for item in option_1]
    audited["sentiment_2"] = [item["label"] for item in option_2]
    audited["score_2"] = [item["score"] for item in option_2]
    audited["divergence_flag"] = audited["sentiment_1"] != audited["sentiment_2"]
    return audited


def audit_file(
    input_file: Path,
    output_file: Path,
    *,
    model_name: str = DEFAULT_SENTIMENT_MODEL,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Audit one CSV file and save the result."""

    sentiment_pipe = build_sentiment_pipeline(model_name)
    df = read_csv(input_file)
    audited = audit_dataframe(df, sentiment_pipe, batch_size=batch_size)
    write_csv(audited, output_file)
    return audited
