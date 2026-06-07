"""Stage 5 - Community-aware pair features.

All features are derived from:
  - the Louvain partition of the snapshot's positive-edge graph G_pos, and
  - the undirected any-sign adjacency (already precomputed by features_structural).

Feature list (per candidate ordered pair (u, v)):
  same_community              1 iff comm(u) == comm(v)    (0 otherwise)
  cross_community_pair        complement of same_community (kept for
                              transparency and so tree models can split on
                              either directly)
  community_size_u            size of u's community (singleton fallback = 1)
  community_size_v            size of v's community
  community_size_diff_log     |log(size_u) - log(size_v)| — scale-robust gap
  neighborhood_overlap        |N(u) ∩ N(v) \\ {u,v}| / |N(u) ∪ N(v) \\ {u,v}|
                              (Easley-Kleinberg edge-overlap; classical weak-
                              tie diagnostic. Note this is related to but
                              distinct from `jaccard` already present: jaccard
                              keeps u and v in the union where applicable.)
  boundary_frac_u             fraction of u's positive neighbors that sit
                              in a DIFFERENT community than u (on G_pos)
  boundary_frac_v             same for v

Weak-tie interpretation: pairs with low neighborhood_overlap and high
boundary_frac tend to be "weak ties" — cross-cluster bridges. Under
Granovetter-style structural theory, distrust (new negative rating)
between such pairs is expected to be more visible / more likely than
inside a tight trusted community.

Leakage: all inputs are snapshot-t-only (G_pos and adj_any are built from
edges with ts <= end_of_month(t)).
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np


COMMUNITY_COLUMNS = [
    "same_community",
    "cross_community_pair",
    "community_size_u",
    "community_size_v",
    "community_size_diff_log",
    "neighborhood_overlap",
    "boundary_frac_u",
    "boundary_frac_v",
]


def precompute(
    partition: dict[int, int],
    sizes: dict[int, int],
    adj_any: dict[int, set[int]],
    G_pos: nx.Graph,
) -> dict:
    """Boundary fraction per node (on G_pos), using the Louvain partition.

    For a node n:
      boundary_frac(n) = |{m in N_pos(n) : comm(m) != comm(n)}| / |N_pos(n)|
    Nodes with no positive neighbors get 0.0 (no boundary to speak of).
    """
    boundary_frac: dict[int, float] = {}
    for n in G_pos.nodes():
        neigh = list(G_pos.neighbors(n))
        if not neigh:
            boundary_frac[n] = 0.0
            continue
        cn = partition.get(n)
        out = sum(1 for m in neigh if partition.get(m) != cn)
        boundary_frac[n] = out / len(neigh)
    return {
        "partition": partition,
        "sizes": sizes,
        "adj_any": adj_any,
        "boundary_frac": boundary_frac,
    }


def _lookup_community(n: int, C: dict) -> tuple[int, int]:
    """Return (community_id, size). Falls back to synthetic singleton for
    nodes that never appeared in G_pos (no positive edges)."""
    cid = C["partition"].get(n)
    if cid is None:
        # Synthetic singleton — unique per node, size 1.
        return (-n - 1, 1)
    return (cid, C["sizes"].get(cid, 1))


def pair_features(u: int, v: int, C: dict) -> list[float]:
    cu, su = _lookup_community(u, C)
    cv, sv = _lookup_community(v, C)
    same = 1.0 if cu == cv else 0.0
    cross = 1.0 - same
    size_diff_log = abs(math.log(max(su, 1)) - math.log(max(sv, 1)))

    # Neighborhood overlap on undirected any-sign adj (precomputed elsewhere).
    Nu = C["adj_any"].get(u, frozenset()) - {v}
    Nv = C["adj_any"].get(v, frozenset()) - {u}
    inter = len(Nu & Nv)
    union = len(Nu) + len(Nv) - inter
    nh_overlap = (inter / union) if union > 0 else 0.0

    bf_u = C["boundary_frac"].get(u, 0.0)
    bf_v = C["boundary_frac"].get(v, 0.0)

    return [
        same, cross,
        float(su), float(sv),
        float(size_diff_log),
        float(nh_overlap),
        float(bf_u), float(bf_v),
    ]


def compute_matrix(pairs: list[tuple[int, int]], C: dict) -> np.ndarray:
    n = len(pairs)
    X = np.empty((n, len(COMMUNITY_COLUMNS)), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        X[i] = pair_features(u, v, C)
    return X
