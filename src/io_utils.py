"""I/O helpers: raw CSV load, clean-edges parquet, snapshot pickles.

Why parquet for edges and pickle for snapshots:
 - parquet: fast, typed, tabular — perfect for the edges dataframe.
 - pickle: networkx graph objects don't round-trip through parquet; pickle is the
   path of least resistance for this course project.
"""
import pickle

import pandas as pd

from . import config

COLUMNS = ["source", "target", "rating", "timestamp"]


def load_raw_edges() -> pd.DataFrame:
    """Load the Bitcoin-OTC CSV.

    The file has NO header row (verified by inspecting the file on disk).
    Dtypes are declared explicitly so we don't rely on pandas inference.
    """
    df = pd.read_csv(
        config.DATA_RAW,
        header=None,
        names=COLUMNS,
        dtype={
            "source": "int64",
            "target": "int64",
            "rating": "int64",
            "timestamp": "float64",
        },
    )
    return df


def save_clean_edges(df: pd.DataFrame) -> None:
    df.to_parquet(config.DATA_INTERIM / "edges_clean.parquet", index=False)


def load_clean_edges() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_INTERIM / "edges_clean.parquet")


def snapshot_path(t: int):
    return config.DATA_INTERIM / "snapshots" / f"snap_{t:03d}.pkl"


def save_snapshot(t: int, snap: dict) -> None:
    with open(snapshot_path(t), "wb") as f:
        pickle.dump(snap, f)


def load_snapshot(t: int) -> dict:
    with open(snapshot_path(t), "rb") as f:
        return pickle.load(f)
