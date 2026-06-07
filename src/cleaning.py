"""Cleaning: dtype coercion, datetime conversion, defensive filters, time sort.

Decisions (from the approved plan):
  - Drop rating == 0 and self-loops. None exist in this dataset but the filter
    documents the assumption and protects against corrupt reloads.
  - sign = +1 if rating > 0 else -1 (binary sign separate from magnitude).
  - Sort by ts then (source, target) with a stable sort so repeated reads give
    identical row orders — important for "keep last" aggregations downstream.
"""
import pandas as pd

from . import io_utils


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Unix seconds -> tz-aware UTC Timestamps.
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    mask = (df["rating"] != 0) & (df["source"] != df["target"])
    df = df.loc[mask].copy()

    df["sign"] = df["rating"].apply(lambda r: 1 if r > 0 else -1).astype("int8")

    df = df.sort_values(
        ["ts", "source", "target"], kind="mergesort"
    ).reset_index(drop=True)

    return df[["source", "target", "rating", "sign", "timestamp", "ts"]]


def run() -> pd.DataFrame:
    raw = io_utils.load_raw_edges()
    clean_df = clean(raw)
    io_utils.save_clean_edges(clean_df)
    return clean_df
