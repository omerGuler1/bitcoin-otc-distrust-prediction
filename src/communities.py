"""Stage 5 - Louvain community detection on the per-snapshot positive-edge
undirected weighted graph G_pos.

Design choices:
  - We use python-louvain (`community`). Weights come from G_pos (count of
    distinct positive directed edges collapsed into each undirected pair).
  - We run Louvain *only on nodes with at least one positive edge*. Isolated
    nodes (no positive edges but present in G_dir) get assigned synthetic
    singleton community ids so feature lookups don't fail.
  - `random_state` is fixed for reproducibility (Louvain is stochastic).
  - Metadata per snapshot is cached to disk so re-running feature extraction
    doesn't repeat Louvain.

Note on directionality: Louvain requires an undirected graph. Using G_pos
(positive-only, undirected) is the standard choice for "trust-community"
detection — negative edges would mean enemies-of-enemies pretending to be
friends under modularity, which is the wrong assumption for community
structure.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import community as community_louvain
import networkx as nx

from . import config, io_utils

COMM_DIR = config.DATA_INTERIM / "communities"
COMM_DIR.mkdir(parents=True, exist_ok=True)


def comm_path(t: int) -> Path:
    return COMM_DIR / f"comm_{t:03d}.pkl"


def run_louvain(G_pos: nx.Graph, seed: int = config.RANDOM_SEED) -> dict:
    """Run Louvain on the positive-edge undirected weighted graph.

    Returns a dict with:
      partition:  {node -> community_id}
      sizes:      {community_id -> size}
      modularity: float (weighted modularity of the partition)
      n_communities, top5_sizes (sorted descending), n_nodes_in_graph
    """
    if G_pos.number_of_edges() == 0:
        # Degenerate — every node its own singleton.
        part = {n: n for n in G_pos.nodes()}
        return {
            "partition": part,
            "sizes": {c: 1 for c in part.values()},
            "modularity": 0.0,
            "n_communities": len(part),
            "top5_sizes": [1] * min(5, len(part)),
            "n_nodes_in_graph": G_pos.number_of_nodes(),
        }

    partition = community_louvain.best_partition(
        G_pos, weight="weight", random_state=seed
    )
    modularity = community_louvain.modularity(partition, G_pos, weight="weight")
    sizes: dict[int, int] = {}
    for _, c in partition.items():
        sizes[c] = sizes.get(c, 0) + 1
    top5 = sorted(sizes.values(), reverse=True)[:5]
    return {
        "partition": partition,
        "sizes": sizes,
        "modularity": modularity,
        "n_communities": len(sizes),
        "top5_sizes": top5,
        "n_nodes_in_graph": G_pos.number_of_nodes(),
    }


def build_or_load(t: int, G_pos: nx.Graph, force: bool = False) -> dict:
    """Cache-aware: compute Louvain once per snapshot, reuse on subsequent calls."""
    p = comm_path(t)
    if p.exists() and not force:
        with open(p, "rb") as f:
            return pickle.load(f)
    meta = run_louvain(G_pos)
    with open(p, "wb") as f:
        pickle.dump(meta, f)
    return meta


def load(t: int) -> dict:
    with open(comm_path(t), "rb") as f:
        return pickle.load(f)
