"""Stage 5 (part 1) - rebuild feature parquets with Louvain community features
and print per-snapshot community metadata + feature distributions.
"""
from __future__ import annotations

import pandas as pd

from src import build_features, communities as comm, features_community as fc, io_utils

TRAIN_TS = list(range(13, 46))
VAL_TS = [46, 47]
TEST_TS = [48, 49]
ALL_TS = TRAIN_TS + VAL_TS + TEST_TS


def main() -> None:
    df = io_utils.load_clean_edges()
    summary = build_features.build_all(ALL_TS, df)

    leakage_ok = all(r["max_ts_used"] <= r["end_ts"] for r in summary)
    print(f"[{'OK' if leakage_ok else 'FAIL'}] feature computation respects "
          f"ts <= end_of_month(t) for all {len(summary)} snapshots")
    print(f"Feature columns ({len(build_features.FEATURE_COLUMNS)}): "
          f"{build_features.FEATURE_COLUMNS}")

    # --- Per-snapshot Louvain metadata -------------------------------------
    rows = []
    for t in ALL_TS:
        m = comm.load(t)
        rows.append({
            "t": t,
            "n_nodes_in_Gpos": m["n_nodes_in_graph"],
            "n_communities": m["n_communities"],
            "modularity": round(m["modularity"], 4),
            "top5_sizes": m["top5_sizes"],
        })
    meta_df = pd.DataFrame(rows)
    print("\n=== Louvain community metadata per snapshot ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(meta_df.to_string(index=False))

    # --- Community feature distributions on TRAIN --------------------------
    train_df = build_features.load_split(TRAIN_TS)
    ccols = fc.COMMUNITY_COLUMNS
    pos = train_df[train_df["y"] == 1]
    neg = train_df[train_df["y"] == 0]

    print(f"\nTRAIN positives: {len(pos)}   negatives: {len(neg):,}")

    print("\n=== Community feature summary on TRAIN (all) ===")
    d = train_df[ccols].describe(percentiles=[0.5, 0.99]).T[
        ["count", "mean", "std", "min", "50%", "99%", "max"]
    ]
    d["n_missing"] = train_df[ccols].isna().sum().values
    print(d.round(4).to_string())

    print("\n=== Community feature means: positives vs negatives (TRAIN) ===")
    means = pd.DataFrame(
        {"pos_mean": pos[ccols].mean(), "neg_mean": neg[ccols].mean()}
    )
    means["abs_gap"] = (means["pos_mean"] - means["neg_mean"]).abs()
    print(means.sort_values("abs_gap", ascending=False).round(4).to_string())

    # --- Positive rates: same vs cross community --------------------------
    same_mask = train_df["same_community"] == 1
    p_same = train_df.loc[same_mask, "y"].mean() if same_mask.sum() else float("nan")
    p_cross = train_df.loc[~same_mask, "y"].mean()
    n_same = int(same_mask.sum()); n_cross = int((~same_mask).sum())
    print("\n=== Positive rate by same/cross community (TRAIN) ===")
    print(f"  same_community=1: pairs={n_same:,}  pos_rate={p_same:.6%}  "
          f"positives={int(train_df.loc[same_mask, 'y'].sum())}")
    print(f"  same_community=0: pairs={n_cross:,}  pos_rate={p_cross:.6%}  "
          f"positives={int(train_df.loc[~same_mask, 'y'].sum())}")


if __name__ == "__main__":
    main()
