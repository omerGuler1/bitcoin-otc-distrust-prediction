"""Stage 3 (part 1) - build feature matrices for every split snapshot.

Produces data/processed/features/feat_t=NNN.parquet for t in [13..49],
then prints:
  - per-feature summary stats (count, #missing, min, mean, std, max)
    separately for positives and negatives on the training split, to check
    whether any feature trivially separates the classes.
  - leakage-boundary sanity check: for every snapshot, the maximum ts used
    during feature computation must be <= the snapshot's end_ts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import build_features, config, io_utils

# Splits are fixed per the user's final Stage-2 decision.
TRAIN_TS = list(range(13, 46))
VAL_TS = [46, 47]
TEST_TS = [48, 49]
ALL_TS = TRAIN_TS + VAL_TS + TEST_TS


def main() -> None:
    df = io_utils.load_clean_edges()
    summary = build_features.build_all(ALL_TS, df)

    # --- Leakage sanity check ------------------------------------------------
    leakage_ok = True
    for r in summary:
        if r["max_ts_used"] > r["end_ts"]:
            leakage_ok = False
            print(f"[FAIL] leakage at t={r['t']}: "
                  f"max_ts_used={r['max_ts_used']} > end_ts={r['end_ts']}")
    print(f"\n[{'OK' if leakage_ok else 'FAIL'}] feature computation respects "
          f"ts <= end_of_month(t) for all {len(summary)} snapshots")

    # --- Per-feature distribution on the training set -----------------------
    train_df = build_features.load_split(TRAIN_TS)
    feats = build_features.FEATURE_COLUMNS

    def _desc(block: pd.DataFrame) -> pd.DataFrame:
        d = block[feats].describe(percentiles=[0.5, 0.99]).T[
            ["count", "mean", "std", "min", "50%", "99%", "max"]
        ]
        d["n_missing"] = block[feats].isna().sum().values
        d["n_zero"] = (block[feats] == 0).sum().values
        return d

    pos = train_df[train_df["y"] == 1]
    neg = train_df[train_df["y"] == 0]
    print(f"\nTRAIN positives: {len(pos)}   negatives: {len(neg):,}")
    print("\n=== Feature summary on TRAIN (all) ===")
    print(_desc(train_df).round(4).to_string())
    print("\n=== Feature means: positives vs negatives (TRAIN) ===")
    means = pd.DataFrame(
        {"pos_mean": pos[feats].mean(), "neg_mean": neg[feats].mean()}
    )
    means["abs_gap"] = (means["pos_mean"] - means["neg_mean"]).abs()
    means = means.sort_values("abs_gap", ascending=False)
    print(means.round(4).to_string())


if __name__ == "__main__":
    main()
