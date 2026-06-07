"""Stage 4 - Temporal / recency / behavioral features.

All features use raw events with ts <= end_of_month(t) (the snapshot boundary).
A defensive assertion is included in `precompute`. Nothing here looks past `t`.

Feature list (per candidate ordered pair (u, v)):

  Node activity recency
    days_since_last_activity_u    end_ts - max(ts of any event involving u)
    days_since_last_activity_v    same for v (in or out direction)
                                  missing -> DAYS_NEVER (9999)

  Pair recency / interaction history (directed u -> v only)
    days_since_last_uv_interaction   end_ts - max(ts of events u->v)
                                     missing -> DAYS_NEVER (sentinel, not zero,
                                     so the feature doesn't conflate "just
                                     happened" with "never happened")
    uv_interaction_count_past        count of past u->v events (any rating)

  Recent negative behavior (inside RECENT_WINDOW_DAYS = 90 days before end_ts)
    recent_neg_given_u               events with source=u and rating<0
    recent_neg_received_v            events with target=v and rating<0

  Recent total activity (same window)
    recent_total_given_u             events with source=u
    recent_total_received_v          events with target=v

  Exponentially decayed activity  (tau = DECAY_TAU_DAYS = 60 days,
     weight = exp(-(end_ts - ts).days / tau))
    decayed_neg_given_u
    decayed_neg_received_v
    decayed_total_activity_u
    decayed_total_activity_v

Choice of tau and recent window: 60d half-life / 90d window are Bitcoin-OTC
-appropriate — activity bursts happen over weeks; a user labelled "recently
active" in this window will usually still be relevant to the next-month task.
These are knobs in config.py and easy to sensitivity-test later.

Design note: `days_since_last_uv_interaction` uses a SENTINEL (DAYS_NEVER)
when there is no prior u->v event. Using 0 would make "never interacted"
indistinguishable from "interacted today", which flips the signal. Tree
models handle the sentinel naturally; LR benefits from it being large and
positive (tree-friendly).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from . import config


TEMPORAL_COLUMNS = [
    "days_since_last_activity_u",
    "days_since_last_activity_v",
    "days_since_last_uv_interaction",
    "uv_interaction_count_past",
    "recent_neg_given_u",
    "recent_neg_received_v",
    "recent_total_given_u",
    "recent_total_received_v",
    "decayed_neg_given_u",
    "decayed_neg_received_v",
    "decayed_total_activity_u",
    "decayed_total_activity_v",
]


def precompute(df_up_to_t: pd.DataFrame, end_ts: pd.Timestamp) -> dict[str, Mapping]:
    assert df_up_to_t["ts"].max() <= end_ts, (
        "Leakage: temporal precompute received events beyond end_ts"
    )

    # --- Per-node last activity (any direction) ----------------------------
    last_src = df_up_to_t.groupby("source", sort=False)["ts"].max()
    last_tgt = df_up_to_t.groupby("target", sort=False)["ts"].max()
    last_any = pd.concat([last_src, last_tgt]).groupby(level=0).max()
    # days_since -> store in a plain dict for O(1) lookup
    days_since = ((end_ts - last_any).dt.total_seconds() / 86400.0).to_dict()

    # --- Per-(source, target) last ts and count ----------------------------
    pair_grp = df_up_to_t.groupby(["source", "target"], sort=False)
    pair_last_ts = pair_grp["ts"].max().to_dict()
    pair_count = pair_grp.size().to_dict()

    # --- Recent window counts ----------------------------------------------
    cutoff = end_ts - pd.Timedelta(days=config.RECENT_WINDOW_DAYS)
    recent = df_up_to_t[df_up_to_t["ts"] > cutoff]
    r_total_given = recent.groupby("source", sort=False).size().to_dict()
    r_total_received = recent.groupby("target", sort=False).size().to_dict()
    r_neg = recent[recent["sign"] == -1]
    r_neg_given = r_neg.groupby("source", sort=False).size().to_dict()
    r_neg_received = r_neg.groupby("target", sort=False).size().to_dict()

    # --- Exponentially decayed aggregates ----------------------------------
    delta_days = (end_ts - df_up_to_t["ts"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-delta_days / config.DECAY_TAU_DAYS)
    dec = df_up_to_t[["source", "target", "sign"]].copy()
    dec["w"] = w.astype(np.float32)
    d_total_given = dec.groupby("source", sort=False)["w"].sum().to_dict()
    d_total_received = dec.groupby("target", sort=False)["w"].sum().to_dict()
    d_neg = dec[dec["sign"] == -1]
    d_neg_given = d_neg.groupby("source", sort=False)["w"].sum().to_dict()
    d_neg_received = d_neg.groupby("target", sort=False)["w"].sum().to_dict()

    return {
        "end_ts": end_ts,
        "days_since": days_since,
        "pair_last_ts": pair_last_ts,
        "pair_count": pair_count,
        "r_neg_given": r_neg_given,
        "r_neg_received": r_neg_received,
        "r_total_given": r_total_given,
        "r_total_received": r_total_received,
        "d_neg_given": d_neg_given,
        "d_neg_received": d_neg_received,
        "d_total_given": d_total_given,
        "d_total_received": d_total_received,
    }


def pair_features(u: int, v: int, T: dict) -> list[float]:
    end_ts = T["end_ts"]
    days_u = T["days_since"].get(u, config.DAYS_NEVER)
    days_v = T["days_since"].get(v, config.DAYS_NEVER)

    last_uv = T["pair_last_ts"].get((u, v))
    if last_uv is None:
        pair_days = config.DAYS_NEVER
    else:
        pair_days = (end_ts - last_uv).total_seconds() / 86400.0
    pair_cnt = T["pair_count"].get((u, v), 0)

    return [
        float(days_u),
        float(days_v),
        float(pair_days),
        float(pair_cnt),
        float(T["r_neg_given"].get(u, 0)),
        float(T["r_neg_received"].get(v, 0)),
        float(T["r_total_given"].get(u, 0)),
        float(T["r_total_received"].get(v, 0)),
        float(T["d_neg_given"].get(u, 0.0)),
        float(T["d_neg_received"].get(v, 0.0)),
        float(T["d_total_given"].get(u, 0.0)),
        float(T["d_total_received"].get(v, 0.0)),
    ]


def compute_matrix(pairs: list[tuple[int, int]], T: dict) -> np.ndarray:
    n = len(pairs)
    X = np.empty((n, len(TEMPORAL_COLUMNS)), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        X[i] = pair_features(u, v, T)
    return X
