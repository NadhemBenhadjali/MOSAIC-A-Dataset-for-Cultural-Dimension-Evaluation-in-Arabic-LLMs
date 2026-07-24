from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_mosaic_counts():
    df = pd.read_csv(ROOT / "data" / "processed" / "mosaic_all.csv")
    counts = df["dimension"].value_counts().to_dict()

    assert len(df) == 1483
    assert counts == {
        "lto": 300,
        "idv": 299,
        "uai": 298,
        "pdi": 294,
        "mas": 292,
    }
