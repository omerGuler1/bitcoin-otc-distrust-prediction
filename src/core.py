"""Core-node filter.

A node is "core at snapshot t" iff:
  (a) it has at least MIN_ACTIVITY total interactions (in + out) up to end_of_month(t),
  (b) its most recent interaction is within RECENT_DAYS of end_of_month(t).

Both conditions use data from the snapshot's precomputed `activity` dict, which
was built from events with ts <= end_ts. This keeps the leakage boundary tight.
"""
from __future__ import annotations

import pandas as pd

from . import config


def core_nodes(
    snap: dict,
    min_activity: int = config.MIN_ACTIVITY,
    recent_days: int = config.RECENT_DAYS,
) -> set[int]:
    end_ts: pd.Timestamp = snap["end_ts"]
    cutoff = end_ts - pd.Timedelta(days=recent_days)
    core: set[int] = set()
    for n, a in snap["activity"].items():
        total = a["in"] + a["out"]
        last_ts = a["last_ts"]
        if total >= min_activity and last_ts is not None and last_ts >= cutoff:
            core.add(n)
    return core
