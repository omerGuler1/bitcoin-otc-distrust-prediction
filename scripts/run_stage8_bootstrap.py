"""Stage 8 - Save test predictions + bootstrap CIs on the locked test results.

This script does not change the task, the split, the features, or the model
configuration in any way. It re-fits each of the six (model, feature_set)
combinations on the same training data with the same seeds and saves the
resulting test-set predicted probabilities to disk. It then runs a paired
stratified bootstrap to attach uncertainty estimates to:

    - ROC-AUC, PR-AUC, Precision@50, Precision@100
    - the metric deltas (S+T) - S and (S+T+C) - (S+T)

Outputs (under results/):

    test_predictions.parquet     all six probability columns + y, snap_t, u, v
    bootstrap_metrics.csv        per (model, feature_set, metric): mean/std/CI
    bootstrap_deltas.csv         per (model, delta_name, metric): mean/std/CI + P(>0)
    bootstrap_summary.txt        plain-text narrative summary
    figures/bootstrap_rf_*.png   histograms for the three RF feature sets

Re-running: seeds are fixed (config.RANDOM_SEED = 42 for models and the
bootstrap RNG). The script will overwrite outputs on each run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import build_features, bootstrap as bs, config, models

TRAIN_TS = list(range(13, 46))
TEST_TS = [48, 49]

FEATURE_SETS = {
    "S":     build_features.STRUCTURAL_ONLY,
    "S+T":   build_features.STRUCTURAL_PLUS_TEMPORAL,
    "S+T+C": build_features.FEATURE_COLUMNS,
}
MODEL_NAMES = ("logreg", "rf")
RESULTS_DIR = config.PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


# ----------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------

def fit_and_predict() -> pd.DataFrame:
    """Re-fit all (model, fset) combinations and return a wide DataFrame of
    test-set probabilities plus identifier columns."""
    train_df = build_features.load_split(TRAIN_TS)
    test_df = build_features.load_split(TEST_TS)
    print(f"TRAIN n={len(train_df):,} pos={int(train_df['y'].sum())}")
    print(f"TEST  n={len(test_df):,}  pos={int(test_df['y'].sum())}")

    out = test_df[["u", "v", "snap_t", "y"]].copy().reset_index(drop=True)

    for fs_name, cols in FEATURE_SETS.items():
        X_tr_full = train_df[cols].to_numpy(np.float32)
        y_tr_full = train_df["y"].to_numpy(np.int32)
        X_tr, y_tr = models.subsample_negatives(X_tr_full, y_tr_full)
        X_te = test_df[cols].to_numpy(np.float32)
        y_te = test_df["y"].to_numpy(np.int32)
        for m_name in MODEL_NAMES:
            print(f"  fitting {m_name} on {fs_name} (features={len(cols)})")
            res = models.FITTERS[m_name](X_tr, y_tr, X_te, cols)
            assert len(res.proba_val) == len(out), "alignment mismatch"
            out[f"{m_name}__{fs_name}"] = res.proba_val.astype(np.float32)
    return out


# ----------------------------------------------------------------------
# Bootstrap analysis
# ----------------------------------------------------------------------

def run_bootstrap(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run paired stratified bootstrap per model. Returns metric and delta tables
    plus the raw per-iteration arrays (used for figures)."""
    cfg = bs.BootstrapConfig(B=1000, seed=config.RANDOM_SEED, ks=(50, 100))
    y = preds["y"].to_numpy(np.int32)

    metric_rows = []
    delta_rows = []
    raw_samples_by_model = {}

    for m_name in MODEL_NAMES:
        proba_dict = {
            fs_name: preds[f"{m_name}__{fs_name}"].to_numpy(np.float64)
            for fs_name in FEATURE_SETS
        }
        print(f"\n[bootstrap] model={m_name}  B={cfg.B}  fsets={list(proba_dict)}")
        samples = bs.stratified_bootstrap(y, proba_dict, cfg=cfg)
        raw_samples_by_model[m_name] = samples

        # Per-feature-set summaries.
        for fs_name in FEATURE_SETS:
            for metric in bs.metric_names(cfg.ks):
                s = bs.summarize(samples[fs_name][metric])
                metric_rows.append(
                    {"model": m_name, "feature_set": fs_name, "metric": metric,
                     **s}
                )

        # Paired deltas.
        for d_name, (fs2, fs1) in [
            ("(S+T) - S", ("S+T", "S")),
            ("(S+T+C) - (S+T)", ("S+T+C", "S+T")),
            ("(S+T+C) - S", ("S+T+C", "S")),
        ]:
            for metric in bs.metric_names(cfg.ks):
                d_vals = samples[fs2][metric] - samples[fs1][metric]
                s = bs.delta_summary(d_vals)
                delta_rows.append(
                    {"model": m_name, "delta": d_name, "metric": metric, **s}
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows), raw_samples_by_model


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def make_rf_histograms(raw: dict) -> None:
    if "rf" not in raw:
        return
    samples = raw["rf"]
    for metric in ("roc_auc", "pr_auc", "p@50", "p@100"):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        bins = 40
        # Use step histograms so overlap is readable.
        for fs_name, color, lw in [("S", "#888888", 1.6),
                                   ("S+T", "#3b7dd8", 1.8),
                                   ("S+T+C", "#d83b3b", 2.0)]:
            v = samples[fs_name][metric]
            ax.hist(v, bins=bins, histtype="step", linewidth=lw,
                    color=color, label=fs_name)
        ax.set_xlabel(metric)
        ax.set_ylabel("bootstrap iterations")
        ax.set_title(f"RF — bootstrap distribution of {metric} (B=1000, test set)")
        ax.legend()
        fig.tight_layout()
        fname = f"bootstrap_rf_{metric.replace('@', '_at_').replace('/', '_')}.png"
        fig.savefig(FIG_DIR / fname, dpi=140)
        plt.close(fig)


# ----------------------------------------------------------------------
# Plain-text summary
# ----------------------------------------------------------------------

def write_text_summary(
    metrics: pd.DataFrame, deltas: pd.DataFrame, path: Path
) -> None:
    lines: list[str] = []
    lines.append("Stage 8 - Bootstrap confidence intervals on the locked test set\n")
    lines.append(f"Test set: 34 positives / 148,639 candidate pairs (t = 48, 49)\n")
    lines.append("Method: paired stratified bootstrap, B = 1000, seed = 42.\n")
    lines.append("CIs are 95% percentile intervals.\n\n")

    lines.append("== Per-(model, feature_set) bootstrap summaries ==\n")
    for m in MODEL_NAMES:
        for fs in FEATURE_SETS:
            sub = metrics[(metrics["model"] == m) & (metrics["feature_set"] == fs)]
            if sub.empty:
                continue
            lines.append(f"-- {m}, {fs}")
            for _, r in sub.iterrows():
                lines.append(
                    f"   {r['metric']:>8s}: mean={r['mean']:.4f} "
                    f"std={r['std']:.4f} 95% CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                )
        lines.append("")

    lines.append(
        "== Paired-bootstrap deltas (95% percentile CI; frac_boot is the "
        "share of bootstrap iterations with delta > 0; this is NOT a "
        "frequentist p-value) =="
    )
    lines.append("")
    for m in MODEL_NAMES:
        for d in ("(S+T) - S", "(S+T+C) - (S+T)", "(S+T+C) - S"):
            sub = deltas[(deltas["model"] == m) & (deltas["delta"] == d)]
            if sub.empty:
                continue
            lines.append(f"-- {m}, delta={d}")
            for _, r in sub.iterrows():
                lines.append(
                    f"   {r['metric']:>8s}: mean={r['mean']:+.4f} "
                    f"95% CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                    f"frac_boot(>0)={r['frac_positive']:.3f}"
                )
        lines.append("")
    lines.append(
        "Caveats: percentile CIs can under-cover for skewed bootstrap "
        "distributions; P@k is bounded by min(k, n_pos) / k, so P@100 has "
        "a hard ceiling of 34/100 = 0.34 by construction. Interpret "
        "interval *overlap with zero* as 'not distinguishable from noise "
        "on this test set' rather than 'no effect'."
    )

    path.write_text("\n".join(lines))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    preds = fit_and_predict()
    preds_path = RESULTS_DIR / "test_predictions.parquet"
    preds.to_parquet(preds_path, index=False)
    print(f"\nSaved {preds_path} (shape={preds.shape})")

    metrics, deltas, raw = run_bootstrap(preds)
    metrics_out = RESULTS_DIR / "bootstrap_metrics.csv"
    deltas_out = RESULTS_DIR / "bootstrap_deltas.csv"
    metrics.to_csv(metrics_out, index=False)
    deltas.to_csv(deltas_out, index=False)

    write_text_summary(metrics, deltas, RESULTS_DIR / "bootstrap_summary.txt")
    make_rf_histograms(raw)

    print(f"\nSaved {metrics_out}")
    print(f"Saved {deltas_out}")
    print(f"Saved {RESULTS_DIR / 'bootstrap_summary.txt'}")
    print(f"Saved RF histograms under {FIG_DIR}")

    # Concise console table.
    print("\n=== Bootstrap CIs — per (model, fset, metric) ===")
    show = ["model", "feature_set", "metric", "mean", "std", "ci_lo", "ci_hi"]
    print(metrics[show].round(4).to_string(index=False))
    print("\n=== Bootstrap deltas (95% CI, P(>0)) ===")
    show_d = ["model", "delta", "metric", "mean", "ci_lo", "ci_hi", "frac_positive"]
    print(deltas[show_d].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
