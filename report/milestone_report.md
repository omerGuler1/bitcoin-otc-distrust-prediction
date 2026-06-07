# Community-Aware Early Warning of Distrust Emergence in Temporal Bitcoin Trust Networks

**Project Milestone Report**

---

## 1. Introduction

Online trust networks let users assign signed ratings to each other: a positive rating expresses trust, a negative rating expresses distrust. Bitcoin-OTC, an over-the-counter cryptocurrency marketplace, maintained such a web-of-trust so that users could assess the reputation of potential trading counterparts before transacting off-exchange. The resulting dataset (Kumar *et al.*, SNAP) is one of the canonical benchmarks for **signed, weighted, directed, timestamped** social network analysis.

The practical question behind this work is **early warning**: given the state of the network at the end of a time window, can we predict where *new negative edges* will emerge in the next window? Being able to rank-order pairs of users by their risk of entering a distrust relationship has direct applications in marketplace moderation, fraud prevention, and reputation governance. The problem is substantially harder than classical link prediction because the target event is (i) *rare* (≈ 10% of edges are negative overall, and only a tiny fraction of possible pairs acquire a new negative edge per window), (ii) *sign-dependent* (predicting any new edge and predicting a new negative edge are very different tasks), and (iii) *temporally causal* (the evaluation must respect the arrow of time).

### 1.1 Research question

> At snapshot *t*, can we predict whether an unconnected or positively-connected ordered pair `(u, v)` will receive a **new negative** directed edge in the next monthly window `(t, t+1]`, using only information available at time *t*?

### 1.2 Contribution

This project addresses the question through an **interpretable, feature-based pipeline** aligned with the course topics — not with heavy deep learning. The deliverables are: (a) a temporal link-prediction setup with leak-free cumulative monthly snapshots and a contiguous-block train / validation / test split; (b) four compact feature families (classical link prediction, signed structural, interpretable trust summaries, temporal / recency, community-aware); (c) Louvain community detection on the positive-edge trust graph with modularity diagnostics; (d) a three-way ablation comparing structural / structural+temporal / structural+temporal+community feature sets under Random Forest and Logistic Regression; and (e) an interpretation of the weak-tie hypothesis in a signed-network setting.

We deliberately position this work as **lighter and more interpretable** than deep signed graph approaches such as those surveyed in Chen *et al.* (2025), following course constraints against GCN / GraphSAGE / GGNN.

---

## 2. Related Work

**Kumar *et al.* (2016) — Edge weight prediction in weighted signed networks.** Introduced two node-level trust statistics computed iteratively on the signed graph: **fairness** (how reliable the ratings a node gives are) and **goodness** (how trustworthy the node itself is in the eyes of its raters). We borrow the *spirit* of these summaries — a small set of interpretable per-node rating descriptors — rather than reimplementing the full mutually-recursive definition. Specifically, we use **mean-rating-given**, **mean-rating-received**, and **negative-given** / **negative-received** ratios as lightweight "fairness-like / goodness-like" features.

**Bertazzi *et al.* (2018) — Temporal and behavioral perspective on Bitcoin-OTC.** Characterized user-level dynamics in Bitcoin-OTC, including how rating behavior evolves over time, how reciprocity plays out, and how the negative-edge community grows. We use this as motivation for the **cumulative monthly snapshot** formulation and for including temporal features such as `days_since_last_activity` and exponentially decayed activity aggregates.

**Choudhury (2024) — Community-aware temporal information for prediction.** Proposed that community structure, once extracted from the temporal graph, carries signal for link-prediction tasks beyond what pair-level topological heuristics provide. We adopt this as the rationale for running Louvain on every snapshot and engineering community-aware pair features (`same_community`, `community_size_u/v`, `boundary_frac`, `neighborhood_overlap`).

**Chen *et al.* (2025) — Deep signed graph learning survey (positioning reference).** Characterizes the landscape of GNN-based signed link-prediction methods. We use this strictly for positioning: our work is intentionally interpretable and shallow, and we do not compare against these systems directly. Where a deep model would learn embeddings end-to-end, we hand-engineer an analogous but human-readable feature set.

---

## 3. Dataset and Data-Collection Process

### 3.1 Source

The dataset is **`soc-sign-bitcoinotc.csv`** from SNAP (Kumar *et al.*, 2016). It was placed in `data/raw/` and is **not modified**; all cleaning is idempotent and produces a typed parquet file at `data/interim/edges_clean.parquet`.

### 3.2 Structure (verified on disk)

Each row is a signed rating event: `(source, target, rating, timestamp)`. The file has **no header**, integer IDs, integer rating in `[-10, +10]`, and a Unix-seconds float timestamp. Summary statistics:

| Quantity | Value |
|---|---|
| Rows | **35,592** |
| Unique nodes | **5,881** |
| Positive edges (rating > 0) | 32,029 (≈ 90%) |
| Negative edges (rating < 0) | 3,563 (≈ 10%) |
| Zero ratings | 0 |
| Rating range | `[-10, +10]` |
| Time span | 2010-11-08 → 2016-01-25 (≈ 63 months) |
| Reciprocity | 0.79 |
| Giant weakly connected component | 5,875 nodes |
| Giant strongly connected component | 4,709 nodes |

### 3.3 Cleaning decisions

- Parse with `header=None`; declare dtypes explicitly.
- Convert `timestamp` to tz-aware UTC `pd.Timestamp`.
- Defensive filters (`rating != 0`, `source != target`); none fire on this dataset but document the assumption.
- Sort by (`ts`, `source`, `target`) with a stable sort so `drop_duplicates(keep="last")` produces deterministic "latest rating wins".
- Derive binary `sign = +1 / -1`.

### 3.4 Cumulative monthly snapshots

Month-end boundaries are calendar months in UTC (63 snapshots, `t = 0 … 62`). At each `t` we build:

- `G_dir^t` — directed graph of `(u, v)` pairs with edge weight equal to the **most recent rating** before `end_of_month(t)` (last-rating-wins collapse, for snapshot graph state only).
- `G_pos^t`, `G_neg^t` — undirected weighted projections keeping only positively-signed / negatively-signed collapsed edges.
- `activity[t]` — per-node `{in, out, last_ts}` counters derived from the *full* event history `ts ≤ end_of_month(t)` (not collapsed).

Snapshot pickles are persisted under `data/interim/snapshots/snap_NNN.pkl`. Figure 1 (in `results/figures/`) shows monthly interaction volume; Figure 2 shows the rating distribution (+1 and +10 dominate, with a long left tail).

---

## 4. Problem Formulation

### 4.1 Core filtering

A node `n` is **core at snapshot *t*** iff (i) its total interactions (in + out) up to `end_of_month(t)` is at least `MIN_ACTIVITY = 3`, and (ii) its most recent activity is within `RECENT_DAYS = 180` of `end_of_month(t)`. Core sizes grow from ≈ 200 (early) to ≈ 1,200 (mid) and then contract in the dead tail of the dataset.

### 4.2 Candidate pair generation

For each snapshot *t*, candidate ordered pairs `(u, v)` satisfy:

1. `u, v ∈ Core_t`, `u ≠ v`;
2. `v` is within `K_HOP = 2` hops of `u` on the undirected any-sign projection of `G_dir^t`;
3. no existing **negative** directed edge `u → v` in `G_dir^t`. Pairs with prior positive `u → v` are kept, because a rating flip from positive to negative in the next window legitimately counts as *distrust emergence*;
4. total candidate count capped at `MAX_CANDIDATES = 200,000` via fixed-seed uniform down-sampling of negatives when exceeded.

### 4.3 Label definition

```
y(u, v) = 1  iff  ∃ event (u, v, r, ts) with r < 0 and
                  end_of_month(t) < ts ≤ end_of_month(t+1)
y(u, v) = 0  otherwise.
```

Both "no new edge" and "new positive edge" map to `y = 0`. Labels are computed from raw events, not the collapsed snapshot.

### 4.4 Train / validation / test split

A snapshot is **eligible for the split** iff (i) it has at least `MIN_POS_FOR_SPLIT = 10` raw new-negative events in its next window and (ii) it sits in the **longest contiguous block** of eligible snapshots (this drops a sporadic late-dataset island at `t = 53`). The resulting pool is `t ∈ [13, 49]`. Splits:

| split | snapshots | pairs | positives | pos rate |
|---|---|---|---|---|
| train | `t = 13 … 45` | 5,013,346 | 545 | 0.0109% |
| val   | `t = 46, 47`  |   184,648 |  16 | 0.0087% |
| test  | `t = 48, 49`  |   148,639 |  34 | 0.0229% |

The split is **locked** from Stage 2 onward; no tuning or feature design ever sees test.

### 4.5 Leakage controls

- All features at snapshot *t* are computed from `df_up_to_t = df[df["ts"] ≤ end_of_month(t)]`. A defensive `assert df_up_to_t["ts"].max() <= end_ts` is embedded in both the temporal and trust-summary precompute functions and is re-checked at every snapshot by the Stage 3–5 drivers.
- Labels are derived strictly from events in `(end_of_month(t), end_of_month(t+1)]`.
- Splits are block-temporal: no overlap by construction. An end-to-end sanity check prints `[OK] no train/val/test snapshot overlap`.

### 4.6 Class-imbalance handling

Training negatives are subsampled to a **1:100 positive:negative ratio** (keeping all 545 positives), shrinking the training matrix from ~5M to ~55k rows; validation and test remain full. Ranking metrics (ROC-AUC, PR-AUC, Precision@k) are preferred over threshold-dependent precision/recall.

---

## 5. Methodology

### 5.1 Feature families

| family | # features | source | status |
|---|---|---|---|
| Classical link prediction | 4 | undirected any-sign projection | implemented |
| Signed structural | 8 | directed graph + `G_pos` / `G_neg` | implemented |
| Interpretable trust summary | 4 | raw events (lightweight fairness/goodness-like) | implemented |
| Temporal / recency | 12 | raw events, recency windows, decay | implemented |
| Community-aware | 8 | Louvain partition of `G_pos` | implemented |
| *Iterated fairness / goodness (optional)* | 2 | mutually-recursive update | **not implemented — deferred** |

Full feature list (36 columns) is canonicalized in `src/build_features.py :: FEATURE_COLUMNS`.

### 5.2 Models

Three models are trained per feature-set ablation:

- **Random baseline** (uniform probabilities, seeded) — calibrates metric floors.
- **Logistic Regression** — `StandardScaler` + `class_weight='balanced'`, `liblinear` solver, 2000 iters.
- **Random Forest** — 200 trees, `min_samples_leaf = 20`, `class_weight = 'balanced_subsample'`.

No hyperparameter tuning beyond these defaults. This is intentional: the point of this project is feature engineering and ablation, not model selection.

### 5.3 Evaluation protocol

Primary metrics: **ROC-AUC**, **PR-AUC** (= average precision), **Precision@k** for `k ∈ {50, 100}`. Threshold-0.5 precision/recall/F1 are reported for completeness only; they are not informative at 0.01% base rate. The primary operating mode is ranking.

---

## 6. Mathematical Background

### 6.1 Classical link-prediction heuristics

Let `N(x)` denote the neighborhood of `x` in the undirected projection at snapshot *t*.

- **Common Neighbors**: `CN(u, v) = |N(u) ∩ N(v)|`
- **Jaccard**: `J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|`
- **Adamic-Adar**: `AA(u, v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / log(|N(w)|)` — weighted CN that discounts hub neighbors.
- **Preferential Attachment**: `PA(u, v) = |N(u)| · |N(v)|`

### 6.2 Louvain modularity maximization

Louvain greedily optimizes weighted modularity

```
Q = (1 / 2m) Σ_{i,j} [ A_ij - (k_i k_j / 2m) ] δ(c_i, c_j)
```

where `A_ij` is the edge weight between nodes `i, j`; `k_i` is the weighted degree of `i`; `m = (1/2) Σ_{ij} A_ij`; and `c_i` is the community of `i`. We apply Louvain on the per-snapshot **positive-edge undirected weighted graph** `G_pos^t` (weights = count of directed positive edges collapsed into the undirected pair). Random seed is fixed; outputs are cached per snapshot in `data/interim/communities/`.

Across the 37 split snapshots, modularity is consistently in **0.47 – 0.52**, with 15 – 47 communities and top-5 sizes ranging from ~300 to ~1,200 nodes — confirming that `G_pos` has well-defined community structure throughout the study period.

### 6.3 Evaluation metrics

- **ROC-AUC**: area under the ROC curve; probability that a random positive is ranked above a random negative.
- **PR-AUC** (= Average Precision): area under the precision-recall curve; at extreme imbalance, this is a stricter summary than ROC-AUC.
- **Precision@k**: fraction of true positives in the top `k` scored candidates. A ranking-oriented metric that maps directly onto the "watchlist of k risky pairs" early-warning use case.

### 6.4 Exponential decay weighting

For each event at time `ts` and snapshot boundary `end_ts`, define `Δ = (end_ts − ts).days` and weight `w = exp(−Δ / τ)`. Per-node decayed aggregates (total activity, negative count) are then `Σ_e w_e` grouped by source or target. We use `τ = 60 days` (a Bitcoin-OTC-appropriate half-life given the burstiness of user activity).

### 6.5 Neighborhood overlap (weak-tie diagnostic)

`NO(u, v) = |N(u) ∩ N(v) \ {u, v}| / |N(u) ∪ N(v) \ {u, v}|`. Granovetter-style: a weak tie bridges otherwise weakly-connected neighborhoods and therefore has *low* overlap. We expect this feature to rank highly in a community-aware feature set if the weak-tie hypothesis applies to distrust emergence.

---

## 7. Preliminary Findings and Summary Statistics

### 7.1 Network-level statistics (whole period)

Reported in §3.2. The graph is large, mostly-connected (giant WCC = 99.9% of nodes), and strongly reciprocal (r = 0.79). The positive-edge subgraph is dense; the negative-edge subgraph is sparse.

### 7.2 Per-snapshot summary (Stage 2)

Persisted at `results/stage2_summary.csv`. Key observations:

- **Core size** ramps from dozens (t = 0) to ~1,200 (t ≈ 28) and then contracts as the dataset ages out.
- **Candidate count** is dominated by 2-hop reachability on the core and hits the 200,000 cap for roughly half the mid-period snapshots.
- **Raw next-window negative count** (our eligibility proxy) fluctuates between 10 – 250 during the active period and collapses to 0 – 9 past September 2015 — the reason the split pool is truncated to `t ∈ [13, 49]`.
- **Core filtering** is the dominant positive-loss step (e.g., at `t = 21`, 244 raw new-negatives → 54 in-core → 41 in-candidates). The 2-hop restriction loses only 0 – 10 positives per snapshot and is *not* the bottleneck.

### 7.3 Louvain community snapshot (Stage 5)

Selected snapshots:

| t | `|V(G_pos)|` | # communities | modularity | top-5 sizes |
|---|---|---|---|---|
| 13 | 1,631 | 19 | 0.488 | 321 / 212 / 202 / 176 / 125 |
| 28 | 3,652 | 19 | 0.484 | 665 / 601 / 510 / 488 / 343 |
| 44 | 5,317 | 39 | 0.513 | 994 / 956 / 870 / 763 / 654 |
| 49 | 5,455 | 47 | 0.515 | 1,107 / 1,030 / 962 / 679 / 572 |

Modularity is stable in `[0.47, 0.52]` and community count grows gracefully with the network.

### 7.4 Positive rate: same-community vs cross-community (TRAIN)

| partition | pairs | positives | pos rate |
|---|---|---|---|
| same_community = 1 | 1,540,969 | 215 | **0.0140%** |
| same_community = 0 | 3,472,377 | 330 | 0.0095% |

**Same-community pairs show a ≈ 47 % higher positive rate** than cross-community pairs — the *opposite* of the naive Granovetter prediction at community granularity. We elaborate in Section 7.5.

### 7.5 Early ablation results (validation; **not the final test numbers**)

These are reported to show the feature families are behaving as expected. The final numbers will be reported on the held-out test split.

| model | structural | + temporal | + community | Δ (T − S) | Δ (C − T) |
|---|---|---|---|---|---|
| LogReg (ROC-AUC) | 0.897 | **0.944** | 0.942 | +0.047 | −0.002 |
| LogReg (PR-AUC) | 0.0032 | 0.0022 | 0.0021 | — | — |
| Random Forest (ROC-AUC) | 0.868 | 0.955 | **0.958** | +0.087 | +0.003 |
| Random Forest (PR-AUC) | 0.0017 | 0.0029 | 0.0034 | +71% rel. | +17% rel. |

**Temporal features drive the main lift**, as expected from Bertazzi-style behavioral dynamics. **Community features deliver a smaller incremental gain**, concentrated in PR-AUC — the metric that matters under extreme imbalance. The most useful *community-aware* feature is `neighborhood_overlap`, whose LR coefficient is the strongest inside the community family. This provides **partial support** for the weak-tie interpretation: at the pair level, Granovetter's pattern holds (lower shared-neighborhood = higher risk of new negative edge), but at the community level the intuitive direction reverses (distrust surfaces *inside* a cluster, not at its boundary). The synthesis we adopt for the final narrative is **"intra-community weak ties"** — pairs with few shared neighbors nested inside the same trusted cluster.

### 7.6 Illustrative feature separation (TRAIN, positives vs negatives)

Top positive-vs-negative mean gaps across families:

- Structural: `pa` (4,185 vs 915), `pos_out_u` (51.9 vs 25.5), `neg_out_u` (14.1 vs 2.8), `reciprocity_flag` (0.30 vs 0.04).
- Trust summary: `mean_rating_received_v` (0.17 vs 1.05 — victim-side signal is the strongest interpretable trust descriptor).
- Temporal: `days_since_last_activity_u` (15 vs 62 — recency is the single strongest temporal signal), `recent_total_given_u` (17 vs 4).
- Community: `neighborhood_overlap` (0.044 vs 0.059), `boundary_frac_u` (0.41 vs 0.36).

---

## 8. General Difficulties

1. **Class imbalance.** Base rate is ≈ 0.01 % on the candidate set. Most standard metrics (accuracy, threshold-based precision / recall / F1) are uninformative. Mitigation: training-negative subsampling (1:100) plus ranking metrics (ROC-AUC, PR-AUC, Precision@k).
2. **Tail of the dataset is dead.** Bitcoin-OTC activity collapses after mid-2015, leaving the last ~12 months with 0 – 9 new negatives per window. A naïve "last two snapshots" test fold yields 0 positives in validation. We address this with an eligibility rule (`≥ 10 raw new-negatives in next window`) and a **longest-contiguous-block** split-pool constraint. The eligible pool ends at `t = 49` (April 2015).
3. **Candidate explosion.** All-pairs predictions are O(n²) ≈ 3.5 × 10⁷ per snapshot. Mitigation: core filtering (active nodes only) + 2-hop restriction + 200 k cap with fixed-seed downsampling.
4. **Leakage risk.** Any feature touching post-*t* events would corrupt the evaluation. Mitigation: a single-choke-point `df_up_to_t` argument passed into every feature precompute, defensive asserts, and a per-snapshot post-hoc `max_ts_used ≤ end_ts` check.
5. **Multi-edges and rating flips.** On Bitcoin-OTC, a pair can be re-rated. For the **snapshot graph state** we apply *last-rating-wins* (current trust judgment). For **temporal / trust history features** we use the full event history. These two conventions must be kept consistent across features — they are, by design separation of the `snap` pickle from the raw `df_up_to_t` slice.
6. **Interpretability budget.** We committed to avoiding GCN / GraphSAGE / deep signed models. This keeps the story coherent and the features legible at the cost of absolute predictive power. Results so far suggest the interpretable setup is competitive enough for the early-warning use case.
7. **Louvain determinism.** Louvain is stochastic. We fix `random_state` and cache partitions per snapshot; all runs from Stage 5 onward reference the cached partition.

---

## 9. What Remains and Plan to Completion

### 9.1 Done so far (core pipeline)

- Stage 1 — data loading, cleaning, EDA, snapshot generation.
- Stage 2 — core filtering, candidate generation, label generation, split selection, sanity checks.
- Stage 3 — classical link-prediction + signed structural + trust-summary features; first baselines.
- Stage 4 — temporal features; feature correlation / redundancy analysis; 1:100 negative-subsampling training protocol.
- Stage 5 — Louvain community detection with cached metadata; community-aware features; 3-way validation ablation.
- Engineering: deterministic seeds, cached intermediate artifacts, leakage sanity checks at every stage.

### 9.2 Remaining for the final report

1. **Held-out test evaluation.** Final ablation (structural / +temporal / +temporal+community) on the test split (`t = 48, 49`). Early indicative numbers exist but will be finalized into the report-ready comparison table.
2. **Error analysis.** Top-ranked true-positive and false-positive pairs, with a short discussion of the patterns (recurring "victim" nodes, recurring "issuer" nodes, cross- vs same-community, overlap distributions).
3. **Feature importance summary**, grouped by family (structural / temporal / community), for both LR (standardized coefficients) and RF (Gini importance). Used both for interpretation and for identifying redundant features (e.g., `recent_*` vs `decayed_*` pairs are ≥ 0.9 correlated — we will note this openly rather than prune after the fact).
4. **Report figures.** Finalize: (i) degree distribution on log axes, (ii) modularity-over-time curve, (iii) core-size / candidate-count / positives-count per-snapshot panel, (iv) ROC and PR curves for the three feature sets on test, (v) a same-community vs cross-community positive-rate bar chart.
5. **Report writing.** Introduction, related work, methodology, results, discussion, limitations, conclusion. Target length: 8 – 10 pages (IEEE style).
6. **Slide deck** for the presentation, drawn from the final report.

### 9.3 Optional (only if time permits)

- **node2vec + Logistic Regression** as a learned-embedding baseline, strictly as an optional extension gated behind a working main pipeline.
- **Motif / triad counts** (signed triangles in the vicinity of the candidate pair).
- **Bitcoin-Alpha transfer test**: fit on Bitcoin-OTC and score on the sibling Bitcoin-Alpha dataset as a zero-shot generalization check.
- **Iterated Kumar-style fairness/goodness** (2 – 3 iterations, explicitly not to convergence). Deferred: the four simple trust-summary features already cover the interpretable slot; a richer iterative version would be a 2-feature add-on if the final numbers motivate it.

### 9.4 Explicitly **out of scope**

GNN / GraphSAGE / GCN / GGNN / other heavy deep learning methods. Null-model comparisons (considered at the outset but not planned for the final report).

---

## 10. Repository and Reproducibility

Project tree (abbreviated):

```
462 project/
├── data/ {raw, interim, processed}/
├── src/
│   ├── config.py, io_utils.py, cleaning.py, eda.py
│   ├── snapshots.py, core.py, candidates.py, labels.py
│   ├── features_structural.py, features_trust.py
│   ├── features_temporal.py
│   ├── communities.py, features_community.py
│   ├── build_features.py, models.py
├── scripts/ run_stage{1..7}_*.py
├── results/ {figures, stage2_summary.csv, …}
├── report/ milestone_report.md, (final_report.*)
├── requirements.txt, README.md, .gitignore
```

All intermediate artifacts are regenerable from the raw CSV via the `run_stage{1..7}_*.py` scripts; random seeds are fixed in `src/config.py :: RANDOM_SEED = 42`.

---

## 11. References

- Kumar, S., Spezzano, F., Subrahmanian, V. S., and Faloutsos, C. *Edge weight prediction in weighted signed networks*. IEEE ICDM, 2016.
- Bertazzi, I., *et al*. *Temporal and behavioral perspective on Bitcoin-OTC*. 2018.
- Choudhury, N. *Community-aware temporal network prediction*. 2024.
- Chen, Z., *et al*. *Deep signed graph learning: A survey and benchmark*. 2025.
- Granovetter, M. *The strength of weak ties*. American Journal of Sociology, 1973.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., and Lefebvre, E. *Fast unfolding of communities in large networks*. J. Stat. Mech., 2008.
- Adamic, L. A., and Adar, E. *Friends and neighbors on the web*. Social Networks, 2003.
- Easley, D., and Kleinberg, J. *Networks, Crowds, and Markets*. Cambridge University Press, 2010 (Chs. 3, 5).

*Bibliographic formatting will be standardized in the final report.*
