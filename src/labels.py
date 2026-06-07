"""Label generation for the "new negative edge emergence" task.

For snapshot t with boundary end_ts_t and next snapshot end_ts_next:
  y(u, v) = 1 iff there exists at least one event u->v with rating < 0 in the
  window (end_ts_t, end_ts_next].

Notes:
  - We use *raw events*, not the collapsed snapshot graph, so temporal/behavioral
    features can later use the full history and labels reflect the real
    occurrence of a new negative rating (per the user's clarification that
    last-rating-wins only applies to snapshot graph state).
  - Pairs with a pre-existing positive u->v are still candidates; if the pair
    receives a new negative rating in the window, y=1 (rating flip = distrust
    emergence, as the user confirmed).
"""
from __future__ import annotations

import pandas as pd


def next_window_negatives(
    df: pd.DataFrame, start_ts_exclusive: pd.Timestamp, end_ts_inclusive: pd.Timestamp
) -> set[tuple[int, int]]:
    """Set of directed (u, v) pairs that received at least one negative rating
    strictly after `start_ts_exclusive` and at or before `end_ts_inclusive`."""
    mask = (df["ts"] > start_ts_exclusive) & (df["ts"] <= end_ts_inclusive) & (df["sign"] == -1)
    win = df.loc[mask, ["source", "target"]]
    return set(map(tuple, win.itertuples(index=False, name=None)))


def label_candidates(
    candidates: list[tuple[int, int]],
    next_neg_pairs: set[tuple[int, int]],
) -> list[int]:
    return [1 if pair in next_neg_pairs else 0 for pair in candidates]
