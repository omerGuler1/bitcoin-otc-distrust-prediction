"""Stage 13 - Paired statistical tests on rolling-origin per-fold deltas.

For each (model, comparison, metric) we compute on the n = 25 fold deltas:

  - Paired Wilcoxon signed-rank test (two-sided, zero_method='wilcox').
  - Exact binomial sign test (two-sided), insensitive to ties at zero.
  - Matched-pairs rank-biserial correlation r_rb as a Wilcoxon effect size,
    on [-1, +1].

We split the test set into a small PRIMARY family (Holm-Bonferroni at
alpha = 0.05) and a SECONDARY family (Benjamini-Hochberg FDR at q = 0.05).
The split is fixed in code below.

For Precision@k, deltas have point masses at zero (many folds yield
identical top-k under both feature sets), so the Wilcoxon is unreliable
with very few non-zero pairs. We flag those rows explicitly.

Outputs (new files; nothing in results/ is overwritten):
  results/rolling_stats_tests.csv          full per-test table
  results/rolling_stats_primary_tests.csv  primary subset with Holm decision
  results/rolling_stats_summary.txt        human-readable narrative
  results/figures/rolling_paired_deltas.png   primary-family fold deltas
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config

RESULTS_DIR = config.PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DELTAS_PATH = RESULTS_DIR / "rolling_eval_paired_deltas.csv"

COMPARISONS = [
    ("(S+T) - S",        "d_ST_minus_S"),
    ("(S+T+C) - (S+T)",  "d_STC_minus_ST"),
    ("(S+T+C) - S",      "d_STC_minus_S"),
]
METRICS = ["roc_auc", "pr_auc", "p@50", "p@100"]
MODELS = ["logreg", "rf"]

PRIMARY = {
    ("rf",     "(S+T) - S",         "roc_auc"),
    ("rf",     "(S+T) - S",         "pr_auc"),
    ("rf",     "(S+T+C) - (S+T)",   "pr_auc"),
    ("logreg", "(S+T) - S",         "roc_auc"),
}


# ----------------------------------------------------------------------
# Statistics
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

    # Wilcoxon (drop zeros). Mark unreliable when too few non-zero pairs.
    n_nonzero = n_pos + n_neg
    out["wilcoxon_n_nonzero"] = n_nonzero
    if n_nonzero < 5:
        out["wilcoxon_stat"] = float("nan")
        out["wilcoxon_p"] = float("nan")
        out["wilcoxon_unreliable"] = True
        out["rank_biserial"] = float("nan")
    else:
        r = stats.wilcoxon(d, zero_method="wilcox",
                           alternative="two-sided", method="auto")
        out["wilcoxon_stat"] = float(r.statistic)
        out["wilcoxon_p"] = float(r.pvalue)
        out["wilcoxon_unreliable"] = False
        # Rank-biserial effect size r in [-1, +1].
        mask = d != 0
        dnz = d[mask]
        ranks = stats.rankdata(np.abs(dnz))
        W_plus = float(ranks[dnz > 0].sum())
        W_minus = float(ranks[dnz < 0].sum())
        W_tot = W_plus + W_minus
        out["rank_biserial"] = (
            (W_plus - W_minus) / W_tot if W_tot > 0 else float("nan")
        )

    # Sign test (exact two-sided binomial p = 0.5 on non-zero counts).
    if n_nonzero == 0:
        out["sign_p"] = float("nan")
    else:
        r = stats.binomtest(min(n_pos, n_neg), n=n_nonzero, p=0.5,
                            alternative="two-sided")
        out["sign_p"] = float(r.pvalue)
    return out


def holm(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down. Returns boolean reject array."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thr = alpha / (n - np.arange(n))
    reject_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        if sorted_p[i] <= thr[i]:
            reject_sorted[i] = True
        else:
            break
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """BH step-up FDR control. Returns boolean reject array."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thr = q * (np.arange(1, n + 1) / n)
    passing = sorted_p <= thr
    if not passing.any():
        return np.zeros(n, dtype=bool)
    k_max = np.where(passing)[0].max()
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[: k_max + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(DELTAS_PATH)
    n_folds = int(df["fold_k"].nunique())

    rows = []
    for model in MODELS:
        for metric in METRICS:
            sub = df[(df["model"] == model) & (df["metric"] == metric)]
            for cmp_name, cmp_col in COMPARISONS:
                d = sub[cmp_col].to_numpy()
                t = paired_tests(d)
                t["model"] = model
                t["metric"] = metric
                t["comparison"] = cmp_name
                t["is_primary"] = (model, cmp_name, metric) in PRIMARY
                rows.append(t)

    res = pd.DataFrame(rows)

    primary_mask = res["is_primary"].to_numpy()
    res["holm_reject"] = False
    res["bh_reject"] = False

    # Holm on the primary family
    p_primary = res.loc[primary_mask, "wilcoxon_p"].fillna(1.0).to_numpy()
    res.loc[primary_mask, "holm_reject"] = holm(p_primary, alpha=0.05)

    # BH on the secondary family
    p_secondary = res.loc[~primary_mask, "wilcoxon_p"].fillna(1.0).to_numpy()
    res.loc[~primary_mask, "bh_reject"] = benjamini_hochberg(p_secondary, q=0.05)

    # Reorder columns
    col_order = [
        "is_primary", "model", "comparison", "metric",
        "n", "n_pos", "n_neg", "n_zero",
        "mean_delta", "median_delta", "std_delta",
        "wilcoxon_n_nonzero", "wilcoxon_unreliable",
        "wilcoxon_stat", "wilcoxon_p",
        "sign_p", "rank_biserial",
        "holm_reject", "bh_reject",
    ]
    res = res[col_order]

    # Save
    out_all = RESULTS_DIR / "rolling_stats_tests.csv"
    out_primary = RESULTS_DIR / "rolling_stats_primary_tests.csv"
    res.to_csv(out_all, index=False)
    res[res["is_primary"]].to_csv(out_primary, index=False)
    print(f"Saved {out_all}")
    print(f"Saved {out_primary}")

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append("Stage 13 - Paired statistical tests on rolling-origin per-fold deltas")
    lines.append(f"  source     : {DELTAS_PATH.name}   (n_folds = {n_folds})")
    lines.append( "  primary    : Holm-Bonferroni, alpha = 0.05  (n_tests = 4)")
    lines.append( "  secondary  : Benjamini-Hochberg FDR, q = 0.05  (n_tests = 20)")
    lines.append( "  tests      : two-sided paired Wilcoxon (zero_method='wilcox')")
    lines.append( "               plus two-sided exact binomial sign test")
    lines.append( "  effect size: matched-pairs rank-biserial r_rb in [-1, +1]")
    lines.append("")
    lines.append("=" * 78)
    lines.append("PRIMARY FAMILY (Holm-Bonferroni)")
    lines.append("=" * 78)
    for _, r in res[res["is_primary"]].iterrows():
        verdict = "REJECT_H0" if r["holm_reject"] else "ns"
        flag = "  [Wilcoxon unreliable: too many zeros]" if r["wilcoxon_unreliable"] else ""
        lines.append(
            f"  {r['model']:6s} | {r['metric']:8s} | {r['comparison']:18s} | "
            f"median={r['median_delta']:+.4f}  mean={r['mean_delta']:+.4f}  "
            f"+/-/0={r['n_pos']}/{r['n_neg']}/{r['n_zero']}\n"
            f"            Wilcoxon p={r['wilcoxon_p']:.4g}   "
            f"sign p={r['sign_p']:.4g}   "
            f"r_rb={r['rank_biserial']:+.3f}   [{verdict}]{flag}"
        )
    lines.append("")
    lines.append("=" * 78)
    lines.append("SECONDARY FAMILY (Benjamini-Hochberg)")
    lines.append("=" * 78)
    for _, r in res[~res["is_primary"]].iterrows():
        verdict = "REJECT_H0" if r["bh_reject"] else "ns"
        flag = "  [Wilcoxon unreliable: too many zeros]" if r["wilcoxon_unreliable"] else ""
        lines.append(
            f"  {r['model']:6s} | {r['metric']:8s} | {r['comparison']:18s} | "
            f"median={r['median_delta']:+.4f}  mean={r['mean_delta']:+.4f}  "
            f"+/-/0={r['n_pos']}/{r['n_neg']}/{r['n_zero']}\n"
            f"            Wilcoxon p={r['wilcoxon_p']:.4g}   "
            f"sign p={r['sign_p']:.4g}   "
            f"r_rb={r['rank_biserial']:+.3f}   [{verdict}]{flag}"
        )

    out_summary = RESULTS_DIR / "rolling_stats_summary.txt"
    out_summary.write_text("\n".join(lines))
    print(f"Saved {out_summary}")
    print("\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Figure: paired fold deltas for the 4 primary comparisons
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    deltas = pd.read_csv(DELTAS_PATH)
    primary_list = [
        ("rf",     "(S+T) - S",       "roc_auc",  "d_ST_minus_S",   "RF: ROC-AUC, (S+T) - S"),
        ("rf",     "(S+T) - S",       "pr_auc",   "d_ST_minus_S",   "RF: PR-AUC, (S+T) - S"),
        ("rf",     "(S+T+C) - (S+T)", "pr_auc",   "d_STC_minus_ST", "RF: PR-AUC, (S+T+C) - (S+T)"),
        ("logreg", "(S+T) - S",       "roc_auc",  "d_ST_minus_S",   "LR: ROC-AUC, (S+T) - S"),
    ]
    for ax, (m, _cn, met, col, title) in zip(axes.ravel(), primary_list):
        sub = deltas[(deltas["model"] == m) & (deltas["metric"] == met)].sort_values("fold_k")
        d = sub[col].to_numpy()
        colors = np.where(d > 0, "#3b7dd8",
                          np.where(d < 0, "#d83b3b", "#999999"))
        ax.bar(sub["fold_k"], d, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.7)
        # Annotate with the matching test row
        row = res[(res["model"] == m) & (res["metric"] == met)
                  & (res["comparison"] == _cn)].iloc[0]
        wp = f"Wilcoxon p={row['wilcoxon_p']:.3g}"
        sp = f"sign p={row['sign_p']:.3g}"
        verdict = "REJECT" if row["holm_reject"] else "ns (after Holm)"
        ax.set_title(f"{title}\n{wp}  {sp}  r_rb={row['rank_biserial']:+.3f}  [{verdict}]",
                     fontsize=9)
        ax.set_xlabel("fold k (test snapshot)")
        ax.set_ylabel("paired delta")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Rolling-origin paired fold deltas — primary family (n = {n_folds} folds)",
                 fontsize=11)
    fig.tight_layout()
    fig_out = FIG_DIR / "rolling_paired_deltas.png"
    fig.savefig(fig_out, dpi=140)
    plt.close(fig)
    print(f"Saved {fig_out}")


if __name__ == "__main__":
    main()
