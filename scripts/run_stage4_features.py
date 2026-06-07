"""Stage 4 (part 1) - rebuild feature parquets with temporal columns added.

Overwrites data/processed/features/feat_t=NNN.parquet for t in [13..49].
Confirms the leakage boundary and prints a summary of the temporal columns
(distribution on TRAIN, means for positives vs negatives).
"""
from __future__ import annotations

import pandas as pd

from src import build_features, features_temporal as ftmp, io_utils

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

    train_df = build_features.load_split(TRAIN_TS)
    tcols = ftmp.TEMPORAL_COLUMNS
    pos = train_df[train_df["y"] == 1]
    neg = train_df[train_df["y"] == 0]

    print(f"\nTRAIN positives: {len(pos)}   negatives: {len(neg):,}")

    print("\n=== Temporal feature summary on TRAIN (all) ===")
    d = train_df[tcols].describe(percentiles=[0.5, 0.99]).T[
        ["count", "mean", "std", "min", "50%", "99%", "max"]
    ]
    d["n_missing"] = train_df[tcols].isna().sum().values
    print(d.round(4).to_string())

    print("\n=== Temporal feature means: positives vs negatives (TRAIN) ===")
    means = pd.DataFrame(
        {"pos_mean": pos[tcols].mean(), "neg_mean": neg[tcols].mean()}
    )
    means["abs_gap"] = (means["pos_mean"] - means["neg_mean"]).abs()
    print(means.sort_values("abs_gap", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
