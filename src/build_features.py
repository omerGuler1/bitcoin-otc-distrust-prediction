"""Stage 3 - Orchestrate per-snapshot feature construction and assemble
train / val / test matrices.

Per-snapshot flow:
  1. Load candidate+label parquet (from Stage 2).
  2. Load snapshot pickle (G_dir, G_pos, G_neg, activity).
  3. Slice raw cleaned edges to ts <= end_of_month(t)  <-- leakage boundary.
  4. Precompute structural and trust lookups once.
  5. Compute feature matrix for the snapshot's candidate pairs.
  6. Persist data/processed/features/feat_t=NNN.parquet with u, v, y, snap_t,
     and all feature columns.

Assembly: given a list of snapshot ids, concatenates per-snapshot parquets
into a single DataFrame and splits into X, y.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import (
    communities as comm,
    config,
    features_community as fc,
    features_structural as fs,
    features_temporal as ftmp,
    features_trust as ft,
    io_utils,
)

FEAT_DIR = config.DATA_PROCESSED / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# Master ordered feature list (Stage 5+).
FEATURE_COLUMNS = (
    fs.STRUCTURAL_COLUMNS
    + ft.TRUST_COLUMNS
    + ftmp.TEMPORAL_COLUMNS
    + fc.COMMUNITY_COLUMNS
)

# Ablation subsets.
STRUCTURAL_ONLY = fs.STRUCTURAL_COLUMNS + ft.TRUST_COLUMNS
STAGE3_FEATURES = STRUCTURAL_ONLY           # kept for back-compat
STRUCTURAL_PLUS_TEMPORAL = STRUCTURAL_ONLY + ftmp.TEMPORAL_COLUMNS
TEMPORAL_FEATURES = ftmp.TEMPORAL_COLUMNS
COMMUNITY_FEATURES = fc.COMMUNITY_COLUMNS


def feat_path(t: int) -> Path:
    return FEAT_DIR / f"feat_t={t:03d}.parquet"


def _pairs_path(t: int) -> Path:
    return config.DATA_PROCESSED / "labeled_pairs" / f"snap_t={t:03d}.parquet"


def build_snapshot_features(t: int, df_all: pd.DataFrame) -> dict:
    """Compute and persist feature matrix for snapshot t. Returns a small summary."""
    snap = io_utils.load_snapshot(t)
    end_ts = snap["end_ts"]
    pairs_df = pd.read_parquet(_pairs_path(t))
    pairs = list(zip(pairs_df["u"].tolist(), pairs_df["v"].tolist()))

    # Leakage boundary: strict ts <= end_of_month(t).
    df_up_to_t = df_all[df_all["ts"] <= end_ts]

    P = fs.precompute(snap)
    Trust = ft.precompute(df_up_to_t, end_ts)
    Tmp = ftmp.precompute(df_up_to_t, end_ts)

    # Community detection on G_pos (cached to disk per snapshot).
    comm_meta = comm.build_or_load(t, snap["G_pos"])
    Com = fc.precompute(
        comm_meta["partition"], comm_meta["sizes"], P["adj_any"], snap["G_pos"]
    )

    X_struct = fs.compute_matrix(pairs, P)
    X_trust = ft.compute_matrix(pairs, Trust)
    X_tmp = ftmp.compute_matrix(pairs, Tmp)
    X_com = fc.compute_matrix(pairs, Com)
    X = np.concatenate([X_struct, X_trust, X_tmp, X_com], axis=1)

    out = pairs_df[["u", "v", "y", "snap_t"]].copy()
    for j, col in enumerate(FEATURE_COLUMNS):
        out[col] = X[:, j]
    out.to_parquet(feat_path(t), index=False)

    return {
        "t": t,
        "n_pairs": len(pairs),
        "n_pos": int(pairs_df["y"].sum()),
        "max_ts_used": df_up_to_t["ts"].max(),
        "end_ts": end_ts,
    }


def build_all(snapshot_ids: list[int], df_all: pd.DataFrame) -> list[dict]:
    rows = []
    for t in tqdm(snapshot_ids, desc="features"):
        rows.append(build_snapshot_features(t, df_all))
    return rows


def load_split(ts_list: list[int]) -> pd.DataFrame:
    frames = [pd.read_parquet(feat_path(t)) for t in ts_list]
    return pd.concat(frames, ignore_index=True)
