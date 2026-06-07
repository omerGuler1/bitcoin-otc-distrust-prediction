"""Stage 3 - Simple interpretable trust-summary features (Kumar-style light).

For each node n we summarize its historical rating behavior using *all raw
events with ts <= end_of_month(t)* (not the collapsed snapshot graph). This
respects the user's clarification: "Use 'last rating wins' only for snapshot
graph state. History features may use all past events before time t."

Features (per candidate pair (u, v)):
  mean_rating_given_u       mean of ratings u has given to anyone
  mean_rating_received_v    mean of ratings v has received from anyone
  neg_given_ratio_u         fraction of u's given ratings that were negative
  neg_received_ratio_v      fraction of v's received ratings that were negative

Missing-value policy: if u has never given a rating (or v never received one),
we fill with 0 and set a companion "has_*" column so the model can learn that
this indicates a lack of history. We keep things simple — no fairness/goodness
iteration in Stage 3; those come later.

Leakage boundary: callers must pass `df_up_to_t`, i.e. rows with
ts <= end_of_month(t). A defensive assertion is included.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


TRUST_COLUMNS = [
    "mean_rating_given_u",
    "mean_rating_received_v",
    "neg_given_ratio_u",
    "neg_received_ratio_v",
]
# Dropped in Stage 4 per user decision B: has_given_history_u, has_received_history_v
# (near-constant 0.98+/1.00 — uninformative).


def precompute(df_up_to_t: pd.DataFrame, end_ts: pd.Timestamp) -> dict[str, Mapping]:
    """Aggregate per-source and per-target statistics once per snapshot."""
    assert df_up_to_t["ts"].max() <= end_ts, (
        "Leakage: trust-summary precompute received events beyond end_ts"
    )

    src = df_up_to_t.groupby("source", sort=False)["rating"]
    mean_given = src.mean()
    neg_given = src.apply(lambda s: float((s < 0).mean()))
    count_given = src.size()

    tgt = df_up_to_t.groupby("target", sort=False)["rating"]
    mean_received = tgt.mean()
    neg_received = tgt.apply(lambda s: float((s < 0).mean()))
    count_received = tgt.size()

    return {
        "mean_given": mean_given.to_dict(),
        "neg_given": neg_given.to_dict(),
        "count_given": count_given.to_dict(),
        "mean_received": mean_received.to_dict(),
        "neg_received": neg_received.to_dict(),
        "count_received": count_received.to_dict(),
    }


def pair_features(u: int, v: int, T: dict[str, Mapping]) -> list[float]:
    return [
        float(T["mean_given"].get(u, 0.0)),
        float(T["mean_received"].get(v, 0.0)),
        float(T["neg_given"].get(u, 0.0)),
        float(T["neg_received"].get(v, 0.0)),
    ]


def compute_matrix(
    pairs: list[tuple[int, int]], T: dict[str, Mapping]
) -> np.ndarray:
    n = len(pairs)
    X = np.empty((n, len(TRUST_COLUMNS)), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        X[i] = pair_features(u, v, T)
    return X
