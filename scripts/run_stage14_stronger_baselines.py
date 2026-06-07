"""Stage 14 - Stronger baselines under the same rolling-origin protocol.

Adds, on top of the existing LR and RF in Stage 10:

  Heuristic baselines (parameter-free, transparent within-fold rank sums):
    - recency_only
    - victim_risk_only
    - issuer_risk_only
    - structural_heuristic

  Strong tabular learner:
    - LightGBM on S, S+T, S+T+C  (subsampled training, same seeds)

Protocol is identical to Stage 10:
  - 25 disjoint single-month test folds, k in {25, ..., 49}
  - TRAIN = [13 .. k-2]  (negatives subsampled 100:1)
  - VAL   = {k-1}        (held out; not used for tuning)
  - TEST  = {k}
  - Same candidate set, labels, seeds.

We also report paired statistical tests in the same Phase-D style on a
small pre-declared primary family of four LightGBM comparisons.

Outputs (new files; nothing existing is overwritten):
  results/stronger_baselines_per_fold.csv
  results/stronger_baselines_summary.csv
  results/stronger_baselines_paired_deltas.csv
  results/stronger_baselines_tests.csv
  results/stronger_baselines_primary_tests.csv
  results/stronger_baselines_summary.txt
  results/figures/stronger_baselines_pr_auc.png
  results/figures/stronger_baselines_roc_auc.png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

import lightgbm as lgb

from src import build_features, config, models

# ----------------------------------------------------------------------
# Protocol configuration -- must match Stage 10 exactly.
# ----------------------------------------------------------------------
START_T = 13
END_T = 49
MIN_TRAIN_SNAPS = 12
FOLD_KS = list(range(START_T + MIN_TRAIN_SNAPS, END_T + 1))

FEATURE_SETS = {
    "S":     build_features.STRUCTURAL_ONLY,
    "S+T":   build_features.STRUCTURAL_PLUS_TEMPORAL,
    "S+T+C": build_features.FEATURE_COLUMNS,
}
RESULTS_DIR = config.PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


# ----------------------------------------------------------------------
# Heuristic baselines: within-fold rank-sums with hand-chosen signs.
# Direction is chosen for intuitive readability; we do not tune the
# composition or the signs.
# ----------------------------------------------------------------------

# (column_name, sign) tuples. The score is the SIGN-weighted sum of
# within-fold rank-percentiles for the listed columns.
HEURISTIC_COMPONENTS = {
    "recency_only": [
        ("days_since_last_activity_u",        -1.0),  # active u = risky
        ("days_since_last_activity_v",        -1.0),  # active v = risky
        ("recent_neg_given_u",                +1.0),  # u currently negative
        ("recent_neg_received_v",             +1.0),  # v currently distrusted
        ("decayed_neg_given_u",               +1.0),
        ("decayed_neg_received_v",            +1.0),
    ],
    "victim_risk_only": [
        ("neg_in_v",                          +1.0),
        ("neg_received_ratio_v",              +1.0),
        ("recent_neg_received_v",             +1.0),
        ("decayed_neg_received_v",            +1.0),
        ("mean_rating_received_v",            -1.0),  # low mean = risky
    ],
    "issuer_risk_only": [
        ("neg_out_u",                         +1.0),
        ("neg_given_ratio_u",                 +1.0),
        ("recent_neg_given_u",                +1.0),
        ("decayed_neg_given_u",               +1.0),
        ("mean_rating_given_u",               -1.0),  # tough rater = risky
    ],
    "structural_heuristic": [
        ("cn",                                +1.0),
        ("jaccard",                           +1.0),
        ("aa",                                +1.0),
        ("pa",                                +1.0),
        ("reciprocity_flag",                  +1.0),
        ("prior_interaction",                 -1.0),  # prior pos = unlikely flip
    ],
}


def heuristic_score(test_df: pd.DataFrame, components) -> np.ndarray:
    """Within-fold rank-sum score with intuitive signs.

    Each component is converted to a within-fold rank in [0, 1] via
    `scipy.stats.rankdata(method='average') / n`. Signs are hand-chosen
    so that "higher score = more risk." No label information enters
    this score; the ranks depend only on the test-fold feature values.
    """
    n = len(test_df)
    score = np.zeros(n, dtype=np.float64)
    for col, sign in components:
        x = test_df[col].to_numpy(np.float64)
        ranks = stats.rankdata(x, method="average") / n
        score += sign * ranks
    return score


# ----------------------------------------------------------------------
# LightGBM (single configuration; no hyperparameter search).
# ----------------------------------------------------------------------

def lgbm_params(n_pos: int, n_neg: int) -> dict:
    scale = n_neg / max(n_pos, 1)
    return dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        objective="binary",
        scale_pos_weight=scale,
        random_state=config.RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
    )


def fit_lgbm(X_tr: np.ndarray, y_tr: np.ndarray,
             X_te: np.ndarray, feature_names: list[str]) -> np.ndarray:
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    model = lgb.LGBMClassifier(**lgbm_params(n_pos, n_neg))
    model.fit(X_tr, y_tr, feature_name=list(feature_names))
    return model.predict_proba(X_te)[:, 1]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def precision_at_k(y, proba, k):
    if k > len(proba):
        return float("nan")
    top = np.argpartition(-proba, k - 1)[:k]
    return float(y[top].mean())


def eval_one(y, proba):
    if y.sum() == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "p@50": float("nan"), "p@100": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "p@50": precision_at_k(y, proba, 50),
        "p@100": precision_at_k(y, proba, 100),
    }


# ----------------------------------------------------------------------
# Per-fold evaluation
# ----------------------------------------------------------------------

def run_fold(k: int, log) -> list[dict]:
    train_ts = list(range(START_T, k - 1))
    test_ts = [k]
    train_df = build_features.load_split(train_ts)
    test_df = build_features.load_split(test_ts)

    n_train, n_train_pos = len(train_df), int(train_df["y"].sum())
    n_test, n_test_pos = len(test_df), int(test_df["y"].sum())
    log(f"   k={k}: train [{train_ts[0]}..{train_ts[-1]}] "
        f"({n_train:,} pairs, {n_train_pos} pos), "
        f"test {{{test_ts[0]}}} ({n_test:,}, {n_test_pos})")
    y_te = test_df["y"].to_numpy(np.int32)

    rows: list[dict] = []

    # --- 1. Heuristics (parameter-free; no training needed) ---
    for name, components in HEURISTIC_COMPONENTS.items():
        t0 = time.time()
        score = heuristic_score(test_df, components)
        metrics = eval_one(y_te, score)
        rows.append({
            "fold_k": k, "model": name, "feature_set": "heuristic",
            "n_train_pairs": n_train, "n_train_pos": n_train_pos,
            "n_test_pairs": n_test, "n_test_pos": n_test_pos,
            "fit_seconds": time.time() - t0, **metrics,
        })
        log(f"     heuristic | {name:24s} | "
            f"AUC={metrics['roc_auc']:.4f} "
            f"PR={metrics['pr_auc']:.4f} "
            f"P@50={metrics['p@50']:.4f}")

    # --- 2. LightGBM on three feature sets ---
    for fs_name, cols in FEATURE_SETS.items():
        X_tr_full = train_df[cols].to_numpy(np.float32)
        y_tr_full = train_df["y"].to_numpy(np.int32)
        X_tr, y_tr = models.subsample_negatives(X_tr_full, y_tr_full)
        X_te = test_df[cols].to_numpy(np.float32)
        t0 = time.time()
        proba = fit_lgbm(X_tr, y_tr, X_te, cols)
        elapsed = time.time() - t0
        metrics = eval_one(y_te, proba)
        rows.append({
            "fold_k": k, "model": "lgbm", "feature_set": fs_name,
            "n_train_pairs": n_train, "n_train_pos": n_train_pos,
            "n_test_pairs": n_test, "n_test_pos": n_test_pos,
            "fit_seconds": elapsed, **metrics,
        })
        log(f"     lgbm      | {fs_name:6s}                  | "
            f"AUC={metrics['roc_auc']:.4f} "
            f"PR={metrics['pr_auc']:.4f} "
            f"P@50={metrics['p@50']:.4f} "
            f"({elapsed:.1f}s)")

    return rows


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    aggs = []
    for (m, fs), g in df.groupby(["model", "feature_set"]):
        aggs.append({
            "model": m, "feature_set": fs,
            "n_folds": len(g),
            "total_test_pos": int(g["n_test_pos"].sum()),
            "mean_roc_auc": g["roc_auc"].mean(),
            "std_roc_auc": g["roc_auc"].std(ddof=1),
            "median_roc_auc": g["roc_auc"].median(),
            "mean_pr_auc": g["pr_auc"].mean(),
            "std_pr_auc": g["pr_auc"].std(ddof=1),
            "median_pr_auc": g["pr_auc"].median(),
            "mean_p50": g["p@50"].mean(),
            "mean_p100": g["p@100"].mean(),
        })
    return pd.DataFrame(aggs)


# ----------------------------------------------------------------------
# Paired statistical tests
# ----------------------------------------------------------------------

def paired_tests(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n_zero = int((d == 0).sum())
    out = {
        "n": n, "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
        "mean_delta": float(d.mean()) if n else float("nan"),
        "median_delta": float(np.median(d)) if n else float("nan"),
        "std_delta": float(d.std(ddof=1)) if n > 1 else float("nan"),
    }
    n_nonzero = n_pos + n_neg
    out["wilcoxon_n_nonzero"] = n_nonzero
    if n_nonzero < 5:
        out["wilcoxon_p"] = float("nan")
        out["rank_biserial"] = float("nan")
        out["wilcoxon_unreliable"] = True
    else:
        r = stats.wilcoxon(d, zero_method="wilcox",
                           alternative="two-sided", method="auto")
        out["wilcoxon_p"] = float(r.pvalue)
        out["wilcoxon_unreliable"] = False
        mask = d != 0
        dnz = d[mask]
        ranks = stats.rankdata(np.abs(dnz))
        W_plus = float(ranks[dnz > 0].sum())
        W_minus = float(ranks[dnz < 0].sum())
        W_tot = W_plus + W_minus
        out["rank_biserial"] = (
            (W_plus - W_minus) / W_tot if W_tot > 0 else float("nan")
        )
    if n_nonzero == 0:
        out["sign_p"] = float("nan")
    else:
        r = stats.binomtest(min(n_pos, n_neg), n=n_nonzero, p=0.5,
                            alternative="two-sided")
        out["sign_p"] = float(r.pvalue)
    return out


def holm(p, alpha=0.05):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sp = p[order]
    thr = alpha / (n - np.arange(n))
    rej = np.zeros(n, dtype=bool)
    for i in range(n):
        if sp[i] <= thr[i]:
            rej[i] = True
        else:
            break
    out = np.zeros(n, dtype=bool)
    out[order] = rej
    return out


def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sp = p[order]
    thr = q * (np.arange(1, n + 1) / n)
    passing = sp <= thr
    if not passing.any():
        return np.zeros(n, dtype=bool)
    k_max = np.where(passing)[0].max()
    rej_s = np.zeros(n, dtype=bool)
    rej_s[: k_max + 1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = rej_s
    return out


# ----------------------------------------------------------------------
# Paired-delta construction: this stage's models vs Stage-10 LR/RF.
# ----------------------------------------------------------------------

def build_deltas(stage14_df: pd.DataFrame,
                 stage10_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-fold deltas for every comparison of interest."""

    # Pivot to (fold_k -> metric) maps per model/fset for both sources.
    metrics = ["roc_auc", "pr_auc", "p@50", "p@100"]

    def lookup(df, m, fs, metric):
        sub = df[(df["model"] == m) & (df["feature_set"] == fs)]
        return sub.set_index("fold_k")[metric]

    rows = []

    # Comparisons defined as (label, A, B) where A = lhs, B = rhs.
    A = {  # this stage's models
        "lgbm S":     ("lgbm", "S"),
        "lgbm S+T":   ("lgbm", "S+T"),
        "lgbm S+T+C": ("lgbm", "S+T+C"),
        "recency":      ("recency_only",        "heuristic"),
        "victim_risk":  ("victim_risk_only",    "heuristic"),
        "issuer_risk":  ("issuer_risk_only",    "heuristic"),
        "structural_h": ("structural_heuristic","heuristic"),
    }
    B_st10 = {
        "lr S":     ("logreg", "S"),
        "lr S+T":   ("logreg", "S+T"),
        "lr S+T+C": ("logreg", "S+T+C"),
        "rf S":     ("rf",     "S"),
        "rf S+T":   ("rf",     "S+T"),
        "rf S+T+C": ("rf",     "S+T+C"),
    }

    pairs = [
        # LightGBM ablations within itself.
        ("lgbm S+T - lgbm S",       "lgbm S+T",   None, "lgbm S",      None),
        ("lgbm S+T+C - lgbm S+T",   "lgbm S+T+C", None, "lgbm S+T",    None),
        ("lgbm S+T+C - lgbm S",     "lgbm S+T+C", None, "lgbm S",      None),
        # LightGBM vs RF / LR head-to-head per feature set.
        ("lgbm S     - rf S",       "lgbm S",     None, None, "rf S"),
        ("lgbm S+T   - rf S+T",     "lgbm S+T",   None, None, "rf S+T"),
        ("lgbm S+T+C - rf S+T+C",   "lgbm S+T+C", None, None, "rf S+T+C"),
        ("lgbm S     - lr S",       "lgbm S",     None, None, "lr S"),
        ("lgbm S+T   - lr S+T",     "lgbm S+T",   None, None, "lr S+T"),
        ("lgbm S+T+C - lr S+T+C",   "lgbm S+T+C", None, None, "lr S+T+C"),
        # Heuristics vs S+T LR and S+T RF.
        ("recency     - rf S+T",   "recency",     None, None, "rf S+T"),
        ("recency     - lr S+T",   "recency",     None, None, "lr S+T"),
        ("victim_risk - rf S+T",   "victim_risk", None, None, "rf S+T"),
        ("issuer_risk - rf S+T",   "issuer_risk", None, None, "rf S+T"),
        ("structural_h - rf S",    "structural_h",None, None, "rf S"),
    ]

    for label, A_key, _a_unused, B_key_A, B_key_B in pairs:
        m_a, fs_a = A[A_key]
        if B_key_A is not None:
            m_b, fs_b = A[B_key_A]
            src_b = stage14_df
        else:
            m_b, fs_b = B_st10[B_key_B]
            src_b = stage10_df

        for metric in metrics:
            sa = lookup(stage14_df, m_a, fs_a, metric)
            sb = lookup(src_b, m_b, fs_b, metric)
            common = sorted(set(sa.index) & set(sb.index))
            for k in common:
                rows.append({
                    "comparison": label,
                    "metric": metric,
                    "fold_k": k,
                    "value_A": float(sa.loc[k]),
                    "value_B": float(sb.loc[k]),
                    "delta": float(sa.loc[k] - sb.loc[k]),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = RESULTS_DIR / "stronger_baselines_summary.txt"
    out_per_fold = RESULTS_DIR / "stronger_baselines_per_fold.csv"
    out_summary = RESULTS_DIR / "stronger_baselines_summary.csv"
    out_deltas = RESULTS_DIR / "stronger_baselines_paired_deltas.csv"
    out_tests = RESULTS_DIR / "stronger_baselines_tests.csv"
    out_primary = RESULTS_DIR / "stronger_baselines_primary_tests.csv"

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)
        out_log.write_text("\n".join(log_lines))

    log("Stage 14 - Stronger baselines under rolling-origin")
    log(f"  folds: k = {FOLD_KS[0]}..{FOLD_KS[-1]} (n = {len(FOLD_KS)})")
    log(f"  heuristics: {list(HEURISTIC_COMPONENTS)}")
    log(f"  lgbm feature sets: {list(FEATURE_SETS)}")
    log(f"  protocol identical to Stage 10")
    log("")

    all_rows: list[dict] = []
    t_start = time.time()
    for i, k in enumerate(FOLD_KS, 1):
        t0 = time.time()
        log(f"-- fold {i}/{len(FOLD_KS)}  k={k}  "
            f"(cumulative {time.time() - t_start:.0f}s)")
        rows = run_fold(k, log)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_per_fold, index=False)
        log(f"   fold k={k} done in {time.time() - t0:.0f}s")
        log("")

    df = pd.DataFrame(all_rows)
    summary = aggregate(df)
    summary.to_csv(out_summary, index=False)

    # Paired deltas vs Stage 10 LR/RF (and within-stage14 LightGBM
    # ablations).
    stage10_df = pd.read_csv(RESULTS_DIR / "rolling_eval_per_fold.csv")
    deltas = build_deltas(df, stage10_df)
    deltas.to_csv(out_deltas, index=False)

    # Tests on each (comparison, metric).
    test_rows = []
    for (label, metric), g in deltas.groupby(["comparison", "metric"]):
        d = g["delta"].to_numpy()
        t = paired_tests(d)
        t["comparison"] = label
        t["metric"] = metric
        test_rows.append(t)
    tests = pd.DataFrame(test_rows)

    PRIMARY = {
        ("lgbm S+T - lgbm S",     "roc_auc"),
        ("lgbm S+T - lgbm S",     "pr_auc"),
        ("lgbm S+T+C - lgbm S+T", "pr_auc"),
        ("lgbm S+T   - rf S+T",   "pr_auc"),
    }
    tests["is_primary"] = tests.apply(
        lambda r: (r["comparison"], r["metric"]) in PRIMARY, axis=1
    )

    tests["holm_reject"] = False
    tests["bh_reject"] = False
    pri_mask = tests["is_primary"].to_numpy()
    p_pri = tests.loc[pri_mask, "wilcoxon_p"].fillna(1.0).to_numpy()
    tests.loc[pri_mask, "holm_reject"] = holm(p_pri, alpha=0.05)
    p_sec = tests.loc[~pri_mask, "wilcoxon_p"].fillna(1.0).to_numpy()
    tests.loc[~pri_mask, "bh_reject"] = benjamini_hochberg(p_sec, q=0.05)

    tests = tests[[
        "is_primary", "comparison", "metric", "n", "n_pos", "n_neg", "n_zero",
        "mean_delta", "median_delta", "std_delta",
        "wilcoxon_n_nonzero", "wilcoxon_unreliable",
        "wilcoxon_p", "sign_p", "rank_biserial",
        "holm_reject", "bh_reject",
    ]]
    tests.to_csv(out_tests, index=False)
    tests[tests["is_primary"]].to_csv(out_primary, index=False)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("AGGREGATE ACROSS-FOLD SUMMARY (mean +- std; n_folds = "
        f"{len(FOLD_KS)})")
    log("=" * 78)
    log(summary.round(4).to_string(index=False))
    log("")
    log("=" * 78)
    log("PRIMARY tests (LightGBM-centric, Holm-Bonferroni alpha=0.05)")
    log("=" * 78)
    for _, r in tests[tests["is_primary"]].iterrows():
        verdict = "REJECT_H0" if r["holm_reject"] else "ns"
        log(
            f"  {r['comparison']:30s} | {r['metric']:8s} | "
            f"median={r['median_delta']:+.4f} mean={r['mean_delta']:+.4f}  "
            f"+/-/0={r['n_pos']}/{r['n_neg']}/{r['n_zero']}  "
            f"Wilcoxon p={r['wilcoxon_p']:.4g}  "
            f"sign p={r['sign_p']:.4g}  "
            f"r_rb={r['rank_biserial']:+.3f}  [{verdict}]"
        )
    log("")
    log("=" * 78)
    log("SECONDARY tests (BH-FDR q=0.05, abbreviated)")
    log("=" * 78)
    for _, r in tests[~tests["is_primary"]].iterrows():
        verdict = "*" if r["bh_reject"] else "ns"
        flag = "  [unreliable]" if r["wilcoxon_unreliable"] else ""
        log(
            f"  {r['comparison']:30s} | {r['metric']:8s} | "
            f"med={r['median_delta']:+.4f} mn={r['mean_delta']:+.4f}  "
            f"+/-/0={r['n_pos']}/{r['n_neg']}/{r['n_zero']}  "
            f"W p={r['wilcoxon_p']:.4g}  "
            f"r={r['rank_biserial']:+.3f}  [{verdict}]{flag}"
        )

    # ------------------------------------------------------------------
    # Figures: bar chart of across-fold mean ROC-AUC and PR-AUC
    # ------------------------------------------------------------------
    # Order: heuristics, LR, RF, LightGBM
    label_map = [
        ("recency_only",         "heuristic", "recency-only"),
        ("victim_risk_only",     "heuristic", "victim-risk"),
        ("issuer_risk_only",     "heuristic", "issuer-risk"),
        ("structural_heuristic", "heuristic", "structural-only"),
        ("logreg",               "S",         "LR S"),
        ("logreg",               "S+T",       "LR S+T"),
        ("logreg",               "S+T+C",     "LR S+T+C"),
        ("rf",                   "S",         "RF S"),
        ("rf",                   "S+T",       "RF S+T"),
        ("rf",                   "S+T+C",     "RF S+T+C"),
        ("lgbm",                 "S",         "LGBM S"),
        ("lgbm",                 "S+T",       "LGBM S+T"),
        ("lgbm",                 "S+T+C",     "LGBM S+T+C"),
    ]

    combined = pd.concat([df, stage10_df], ignore_index=True)
    means_roc = []
    stds_roc = []
    means_pr = []
    stds_pr = []
    names = []
    for m, fs, name in label_map:
        sub = combined[(combined["model"] == m) & (combined["feature_set"] == fs)]
        if sub.empty:
            continue
        means_roc.append(sub["roc_auc"].mean())
        stds_roc.append(sub["roc_auc"].std(ddof=1))
        means_pr.append(sub["pr_auc"].mean())
        stds_pr.append(sub["pr_auc"].std(ddof=1))
        names.append(name)

    family_color = lambda n: ("#777777" if n in ("recency-only", "victim-risk",
                                                 "issuer-risk", "structural-only")
                              else "#3b7dd8" if n.startswith("LR ")
                              else "#d83b3b" if n.startswith("RF ")
                              else "#1a8a3a")

    for metric_name, means, stds, fname in [
        ("ROC-AUC", means_roc, stds_roc, "stronger_baselines_roc_auc.png"),
        ("PR-AUC",  means_pr,  stds_pr,  "stronger_baselines_pr_auc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        colors = [family_color(n) for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=3,
               color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(f"across-fold mean {metric_name}")
        ax.set_title(f"Rolling-origin {metric_name} across baselines "
                     f"(n = {len(FOLD_KS)} folds; error bars = std)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=140)
        plt.close(fig)
        log(f"Saved {FIG_DIR / fname}")

    log("")
    log(f"Saved {out_per_fold}")
    log(f"Saved {out_summary}")
    log(f"Saved {out_deltas}")
    log(f"Saved {out_tests}")
    log(f"Saved {out_primary}")


if __name__ == "__main__":
    main()
