"""Candidate pair generation.

Rules (per user's Stage 2 confirmation):
  - Ordered pair (u, v) with u != v
  - u, v both in core at snapshot t
  - v is within K_HOP hops of u on the undirected projection of G_dir
    (reachability uses any-sign edges — the 2-hop restriction is about graph
    proximity, not about trust polarity).
  - Exclude pairs with an existing NEGATIVE directed edge u->v at time t.
    (A prior *positive* u->v is kept as a candidate because a rating flip to
    negative in the next window is itself "distrust emergence".)
  - Cap at MAX_CANDIDATES via a fixed-seed random sub-sample.

k-hop BFS is implemented on an adjacency dict (not networkx) for speed — it is
called once per core node per snapshot and dominates Stage 2 runtime.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from . import config


def _undirected_adjacency(G_dir: nx.DiGraph) -> dict[int, set[int]]:
    """Build an undirected adjacency dict from a directed graph (collapsing
    u->v and v->u into one neighbor relation)."""
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in G_dir.edges():
        if u == v:  # safety — should not happen, we filtered self-loops
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _khop_reach(adj: dict[int, set[int]], source: int, k: int) -> set[int]:
    """Nodes within exactly 1..k hops of source (excludes source itself)."""
    visited = {source}
    frontier = {source}
    for _ in range(k):
        next_frontier: set[int] = set()
        for n in frontier:
            for nb in adj.get(n, ()):
                if nb not in visited:
                    next_frontier.add(nb)
                    visited.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    visited.discard(source)
    return visited


def existing_negative_directed_edges(G_dir: nx.DiGraph) -> set[tuple[int, int]]:
    """(u, v) pairs where the current (collapsed) directed edge weight is < 0.

    G_dir was built with weight=rating, so weight<0 means the current trust
    judgment u->v is negative.
    """
    out: set[tuple[int, int]] = set()
    for u, v, data in G_dir.edges(data=True):
        if data.get("weight", 0) < 0:
            out.add((u, v))
    return out


def build_candidates(
    snap: dict,
    core: set[int],
    khop: int = config.KHOP,
    max_candidates: int = config.MAX_CANDIDATES,
    seed: int = config.RANDOM_SEED,
) -> list[tuple[int, int]]:
    G_dir: nx.DiGraph = snap["G_dir"]
    adj = _undirected_adjacency(G_dir)
    existing_neg = existing_negative_directed_edges(G_dir)

    candidates: set[tuple[int, int]] = set()
    for u in core:
        reach = _khop_reach(adj, u, khop)
        for v in reach:
            if v == u:
                continue
            if v not in core:
                continue
            if (u, v) in existing_neg:
                continue
            candidates.add((u, v))

    cand_list = list(candidates)
    if len(cand_list) > max_candidates:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cand_list), size=max_candidates, replace=False)
        cand_list = [cand_list[i] for i in idx]

    return cand_list
