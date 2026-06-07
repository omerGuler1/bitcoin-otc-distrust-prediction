"""Basic exploratory data analysis: global stats and three plots.

Kept intentionally small: the goal of Stage 1 EDA is a "data sanity" report, not
a final figure set. More detailed plots live in notebooks/01_eda.ipynb.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from . import config


def global_stats(df: pd.DataFrame) -> dict:
    nodes = set(df["source"]).union(df["target"])
    # tz-naive for Period arithmetic (UTC timestamps -> naive for month labels).
    t_max = df["ts"].max().tz_convert(None)
    t_min = df["ts"].min().tz_convert(None)
    months_span = (t_max.to_period("M") - t_min.to_period("M")).n + 1
    return {
        "rows": len(df),
        "nodes": len(nodes),
        "pos_edges": int((df["sign"] == 1).sum()),
        "neg_edges": int((df["sign"] == -1).sum()),
        "min_rating": int(df["rating"].min()),
        "max_rating": int(df["rating"].max()),
        "ts_start": df["ts"].min(),
        "ts_end": df["ts"].max(),
        "months_span": months_span,
    }


def plot_rating_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["rating"], bins=range(-10, 12), edgecolor="black")
    ax.set_xlabel("rating")
    ax.set_ylabel("count")
    ax.set_title("Rating distribution")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "rating_histogram.png", dpi=150)
    plt.close(fig)


def plot_edges_per_month(df: pd.DataFrame) -> None:
    # ME = month-end; "M" is deprecated in newer pandas.
    per_month = df.set_index("ts").resample("ME").size()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    per_month.plot(ax=ax)
    ax.set_ylabel("edges / month")
    ax.set_title("Interaction volume over time")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "edges_per_month.png", dpi=150)
    plt.close(fig)


def plot_degree_distribution(df: pd.DataFrame) -> None:
    # Whole-period static directed graph; only used for EDA, not modeling.
    G = nx.DiGraph()
    G.add_edges_from(df[["source", "target"]].itertuples(index=False, name=None))
    in_deg = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, deg, title in zip(axes, [in_deg, out_deg], ["in-degree", "out-degree"]):
        ax.hist(deg, bins=60, log=True)
        ax.set_xlabel(title)
        ax.set_ylabel("count (log)")
    fig.suptitle("Degree distributions (whole-period static)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "degree_distributions.png", dpi=150)
    plt.close(fig)


def reciprocity_and_giant(df: pd.DataFrame) -> dict:
    G = nx.DiGraph()
    G.add_edges_from(df[["source", "target"]].itertuples(index=False, name=None))
    undirected = G.to_undirected()
    wcc_sizes = [len(c) for c in nx.connected_components(undirected)]
    scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
    return {
        "reciprocity": nx.reciprocity(G),
        "giant_wcc": max(wcc_sizes),
        "giant_scc": max(scc_sizes),
        "n_wcc": len(wcc_sizes),
        "n_scc": len(scc_sizes),
    }
