"""Cumulative monthly snapshot builder.

For each month t we save a dict with:
  t, end_ts, G_dir (directed), G_pos (undirected positive projection),
  G_neg (undirected negative projection), activity (per-node counts + last ts),
  n_edges_cum, n_nodes_cum.

Leakage boundary: each snapshot only sees edges with ts <= end_of_month(t).
Labels (next-window new-negative edges) are a separate step and live in
src/labels.py — they will be added in Stage 2 part 2.

Multi-edge handling: on Bitcoin-OTC a pair (u,v) can be re-rated. We collapse
multi-edges using "last rating wins" (most recent rating before end_of_month(t))
because it represents u's current trust judgment of v. Alternatives (mean, sum)
would blur signal; document this choice in the report.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import pandas as pd

from . import io_utils


def month_end_boundaries(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Month-end UTC timestamps covering [min_ts, max_ts]."""
    # Period arithmetic is tz-naive; convert off UTC just for the boundary calc.
    start_period = df["ts"].min().tz_convert(None).to_period("M")
    end_period = df["ts"].max().tz_convert(None).to_period("M")
    # Month-end in UTC. Use "ME" (pandas >= 2.2); falls back gracefully on 2.0/2.1.
    try:
        idx = pd.date_range(
            start=start_period.to_timestamp(how="end").tz_localize("UTC"),
            end=end_period.to_timestamp(how="end").tz_localize("UTC"),
            freq="ME",
        )
    except ValueError:
        idx = pd.date_range(
            start=start_period.to_timestamp(how="end").tz_localize("UTC"),
            end=end_period.to_timestamp(how="end").tz_localize("UTC"),
            freq="M",
        )
    return idx


def _last_rating_per_pair(df_up_to_t: pd.DataFrame) -> pd.DataFrame:
    """Collapse (u,v) multi-edges to the latest rating before end_of_month(t).

    Relies on df being time-sorted — drop_duplicates with keep='last' then picks
    the most recent row for each (source, target).
    """
    return df_up_to_t.drop_duplicates(subset=["source", "target"], keep="last")


def _undirected_weighted_count(sub: pd.DataFrame) -> nx.Graph:
    """Undirected projection where edge weight = number of directed rows
    contributing to that unordered pair (after collapsing). E.g., if both
    u->v and v->u were positive, the undirected edge {u,v} has weight 2.
    """
    g = nx.Graph()
    w = defaultdict(int)
    for u, v in sub[["source", "target"]].itertuples(index=False, name=None):
        a, b = (u, v) if u < v else (v, u)
        w[(a, b)] += 1
    g.add_weighted_edges_from([(a, b, c) for (a, b), c in w.items()])
    return g


def build_snapshot(df: pd.DataFrame, t: int, end_ts: pd.Timestamp) -> dict:
    df_t = df[df["ts"] <= end_ts]
    if df_t.empty:
        # First month may be empty if end_of_month is before first rating;
        # return a minimal snapshot.
        return {
            "t": t, "end_ts": end_ts,
            "G_dir": nx.DiGraph(), "G_pos": nx.Graph(), "G_neg": nx.Graph(),
            "activity": {}, "n_edges_cum": 0, "n_nodes_cum": 0,
        }

    collapsed = _last_rating_per_pair(df_t)

    G_dir = nx.DiGraph()
    G_dir.add_weighted_edges_from(
        collapsed[["source", "target", "rating"]].itertuples(index=False, name=None)
    )

    pos = collapsed[collapsed["sign"] == 1]
    neg = collapsed[collapsed["sign"] == -1]

    G_pos = _undirected_weighted_count(pos)
    G_neg = _undirected_weighted_count(neg)

    # Per-node activity over the FULL df_t (not just collapsed): we care about
    # total interaction count for core filtering, not unique-pair count.
    activity: dict[int, dict] = defaultdict(
        lambda: {"in": 0, "out": 0, "last_ts": None}
    )
    for u, v, ts_ in df_t[["source", "target", "ts"]].itertuples(
        index=False, name=None
    ):
        au = activity[u]
        av = activity[v]
        au["out"] += 1
        av["in"] += 1
        if au["last_ts"] is None or ts_ > au["last_ts"]:
            au["last_ts"] = ts_
        if av["last_ts"] is None or ts_ > av["last_ts"]:
            av["last_ts"] = ts_

    return {
        "t": t,
        "end_ts": end_ts,
        "G_dir": G_dir,
        "G_pos": G_pos,
        "G_neg": G_neg,
        "activity": dict(activity),
        "n_edges_cum": len(collapsed),
        "n_nodes_cum": G_dir.number_of_nodes(),
    }


def build_all(df: pd.DataFrame) -> list[dict]:
    """Build snapshots for every month in the dataset span. Returns a summary
    list (lightweight metadata only) and persists each full snapshot to disk.
    """
    ends = month_end_boundaries(df)
    summary = []
    for t, end_ts in enumerate(ends):
        snap = build_snapshot(df, t, end_ts)
        io_utils.save_snapshot(t, snap)
        summary.append(
            {
                "t": t,
                "end_ts": end_ts,
                "n_edges_cum": snap["n_edges_cum"],
                "n_nodes_cum": snap["n_nodes_cum"],
                "n_pos_edges_undir": snap["G_pos"].number_of_edges(),
                "n_neg_edges_undir": snap["G_neg"].number_of_edges(),
            }
        )
    return summary
