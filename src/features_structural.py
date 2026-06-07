"""Stage 3 - Classical link-prediction + signed structural features.

All features are computed on the snapshot at time t, i.e. edges with
ts <= end_of_month(t). The snapshot graph has already been collapsed to
"last rating wins" for pair state — this is intentional: classical LP and
signed structural features describe the *current* trust topology, not history.

Feature list (per candidate ordered pair (u, v)):
  Classical (undirected any-sign projection of G_dir):
    cn         Common Neighbors |N(u) & N(v)|
    jaccard    |N(u) & N(v)| / |N(u) | N(v)|
    aa         Adamic-Adar sum_{w in CN} 1 / log(|N(w)|)
    pa         Preferential Attachment |N(u)| * |N(v)|

  Signed degrees from G_dir (weight = rating, +/- sign determines bucket):
    pos_out_u, neg_out_u, pos_in_v, neg_in_v

  Signed common neighbors (on undirected projections G_pos, G_neg):
    pos_cn     |N_pos(u) & N_pos(v)|
    neg_cn     |N_neg(u) & N_neg(v)|

  Pair-level flags:
    reciprocity_flag     1 iff directed edge v->u exists in G_dir
    prior_interaction    1 iff directed edge u->v exists in G_dir
                         (since candidates exclude existing *negative* u->v,
                         this flag is effectively "prior positive u->v").

Adamic-Adar uses natural log; isolated neighbors with degree 1 contribute 0
(we skip them explicitly to avoid division by log(1) = 0).
"""
from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
import numpy as np


def _undirected_any_adjacency(G_dir: nx.DiGraph) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in G_dir.edges():
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return dict(adj)


def _neighbor_map(G: nx.Graph) -> dict[int, set[int]]:
    return {n: set(G.neighbors(n)) for n in G.nodes()}


def precompute(snap: dict) -> dict:
    """Precompute per-snapshot lookups that are reused across all candidate pairs.

    This is the main speedup: we avoid re-computing neighborhoods and degrees
    inside the per-pair loop.
    """
    G_dir: nx.DiGraph = snap["G_dir"]
    G_pos: nx.Graph = snap["G_pos"]
    G_neg: nx.Graph = snap["G_neg"]

    adj_any = _undirected_any_adjacency(G_dir)
    deg_any = {n: len(ns) for n, ns in adj_any.items()}

    adj_pos = _neighbor_map(G_pos)
    adj_neg = _neighbor_map(G_neg)

    # Signed directed degrees from G_dir (weight is the rating).
    pos_out = defaultdict(int)
    neg_out = defaultdict(int)
    pos_in = defaultdict(int)
    neg_in = defaultdict(int)
    for u, v, data in G_dir.edges(data=True):
        w = data.get("weight", 0)
        if w > 0:
            pos_out[u] += 1
            pos_in[v] += 1
        elif w < 0:
            neg_out[u] += 1
            neg_in[v] += 1

    existing_dir = set(G_dir.edges())

    return {
        "adj_any": adj_any,
        "deg_any": deg_any,
        "adj_pos": adj_pos,
        "adj_neg": adj_neg,
        "pos_out": dict(pos_out),
        "neg_out": dict(neg_out),
        "pos_in": dict(pos_in),
        "neg_in": dict(neg_in),
        "existing_dir": existing_dir,
    }


# Column order — kept stable so downstream matrices can rely on it.
STRUCTURAL_COLUMNS = [
    "cn", "jaccard", "aa", "pa",
    "pos_out_u", "neg_out_u", "pos_in_v", "neg_in_v",
    "pos_cn", "neg_cn",
    "reciprocity_flag", "prior_interaction",
]


def pair_features(u: int, v: int, P: dict) -> list[float]:
    Nu = P["adj_any"].get(u, frozenset())
    Nv = P["adj_any"].get(v, frozenset())

    inter = Nu & Nv
    cn = len(inter)
    union = len(Nu) + len(Nv) - cn
    jacc = (cn / union) if union else 0.0

    # Adamic-Adar — skip degree-1 neighbors (log(1)=0).
    aa = 0.0
    deg_any = P["deg_any"]
    for w in inter:
        d = deg_any.get(w, 0)
        if d > 1:
            aa += 1.0 / math.log(d)

    pa = float(len(Nu) * len(Nv))

    pos_Nu = P["adj_pos"].get(u, frozenset())
    pos_Nv = P["adj_pos"].get(v, frozenset())
    neg_Nu = P["adj_neg"].get(u, frozenset())
    neg_Nv = P["adj_neg"].get(v, frozenset())
    pos_cn = len(pos_Nu & pos_Nv)
    neg_cn = len(neg_Nu & neg_Nv)

    pos_out_u = P["pos_out"].get(u, 0)
    neg_out_u = P["neg_out"].get(u, 0)
    pos_in_v = P["pos_in"].get(v, 0)
    neg_in_v = P["neg_in"].get(v, 0)

    existing = P["existing_dir"]
    reciprocity_flag = 1 if (v, u) in existing else 0
    prior_interaction = 1 if (u, v) in existing else 0

    return [
        float(cn), float(jacc), float(aa), pa,
        float(pos_out_u), float(neg_out_u), float(pos_in_v), float(neg_in_v),
        float(pos_cn), float(neg_cn),
        float(reciprocity_flag), float(prior_interaction),
    ]


def compute_matrix(
    pairs: list[tuple[int, int]], P: dict
) -> np.ndarray:
    """Vectorized-ish per-pair loop. Returns an (N, len(STRUCTURAL_COLUMNS)) array."""
    n = len(pairs)
    X = np.empty((n, len(STRUCTURAL_COLUMNS)), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        X[i] = pair_features(u, v, P)
    return X
