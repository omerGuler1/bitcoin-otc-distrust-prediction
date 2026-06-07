"""Stage 9 - Paper artifacts (figures + supplementary error slices).

This script does not retrain models, regenerate features, or rerun
experiments. It only reads existing saved outputs and produces:

  Required figures:
    results/figures/modularity_over_time.png
    results/figures/same_vs_cross_positive_rate.png
    results/figures/test_rf_roc.png
    results/figures/test_rf_pr.png
    results/figures/test_lr_roc.png
    results/figures/test_lr_pr.png

  Optional supplementary error slices:
    results/stage7_error_top200.csv
    results/stage7_error_top500.csv

Run:
    python -m scripts.run_stage9_paper_artifacts
"""
from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config

FIG_DIR = config.PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = config.PROJECT_ROOT / "results"
COMM_DIR = config.DATA_INTERIM / "communities"
FEAT_DIR = config.DATA_PROCESSED / "features"

# Locked split — must match every prior stage.
TRAIN_TS = list(range(13, 46))
SPLIT_TS = list(range(13, 50))   # train + val + test (the 37 contiguous snapshots)

FSETS = ("S", "S+T", "S+T+C")
COLORS = {"S": "#888888", "S+T": "#3b7dd8", "S+T+C": "#d83b3b"}


# ------------------------------------------------------------------
# 1. Modularity over time
# ------------------------------------------------------------------

def _load_comm_meta(t: int) -> dict:
    with open(COMM_DIR / f"comm_{t:03d}.pkl", "rb") as f:
        return pickle.load(f)


def fig_modularity_over_time() -> Path:
    rows = []
    for t in SPLIT_TS:
        m = _load_comm_meta(t)
        rows.append({
            "t": t,
            "modularity": float(m["modularity"]),
            "n_communities": int(m["n_communities"]),
            "n_nodes_in_Gpos": int(m["n_nodes_in_graph"]),
        })
    df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(7.5, 3.8))

    ax1.plot(df["t"], df["modularity"], color="#1f4e8a",
             marker="o", linewidth=2, label="modularity")
    ax1.set_xlabel("snapshot index t (month)")
    ax1.set_ylabel("Louvain modularity", color="#1f4e8a")
    ax1.tick_params(axis="y", labelcolor="#1f4e8a")
    ax1.set_ylim(0.40, 0.55)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["n_communities"], color="#c44e00",
             marker="s", linewidth=1.2, linestyle="--", alpha=0.85,
             label="# communities")
    ax2.set_ylabel("# communities", color="#c44e00")
    ax2.tick_params(axis="y", labelcolor="#c44e00")

    # Annotate split boundaries.
    for t_bound, label in [(45.5, "train|val"), (47.5, "val|test")]:
        ax1.axvline(t_bound, color="grey", linewidth=0.8, linestyle=":")
        ax1.text(t_bound + 0.1, 0.405, label, fontsize=8, color="grey")

    fig.suptitle(
        "Louvain modularity and number of communities on G_pos per snapshot"
    )
    fig.tight_layout()
    out = FIG_DIR / "modularity_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(df)} snapshots, "
          f"modularity range {df['modularity'].min():.3f}..{df['modularity'].max():.3f})")
    return out


# ------------------------------------------------------------------
# 2. Same- vs cross-community positive rate (TRAIN)
# ------------------------------------------------------------------

def fig_same_vs_cross_positive_rate() -> Path:
    frames = [pd.read_parquet(FEAT_DIR / f"feat_t={t:03d}.parquet")
              [["y", "same_community"]] for t in TRAIN_TS]
    df = pd.concat(frames, ignore_index=True)
    df["same_community"] = (df["same_community"] >= 0.5).astype(int)

    same = df[df["same_community"] == 1]
    cross = df[df["same_community"] == 0]
    n_same, p_same = len(same), int(same["y"].sum())
    n_cross, p_cross = len(cross), int(cross["y"].sum())
    rate_same = p_same / n_same if n_same else 0.0
    rate_cross = p_cross / n_cross if n_cross else 0.0

    # Plot as percentage so the differences are visible.
    cats = ["same_community", "cross_community"]
    rates_pct = [rate_same * 100, rate_cross * 100]
    counts = [(n_same, p_same), (n_cross, p_cross)]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(cats, rates_pct,
                  color=["#3b7dd8", "#d83b3b"], alpha=0.85,
                  edgecolor="black")
    ax.set_ylabel("positive rate on TRAIN (%)")
    ax.set_ylim(0, max(rates_pct) * 1.45)
    ax.set_title("Positive rate by community membership (TRAIN, t = 13..45)")
    ax.grid(axis="y", alpha=0.3)

    for bar, rate, (n, p) in zip(bars, rates_pct, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + max(rates_pct) * 0.04,
                f"{rate:.4f}%\n({p:,} / {n:,})",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "same_vs_cross_positive_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (same: {p_same}/{n_same:,} = {rate_same:.6%}; "
          f"cross: {p_cross}/{n_cross:,} = {rate_cross:.6%})")
    return out


# ------------------------------------------------------------------
# 3. Test-set ROC and PR overlays
# ------------------------------------------------------------------

def _curves_for_model(preds: pd.DataFrame, model: str):
    """Return dict[fset] -> (fpr, tpr, auc, precision, recall, ap)."""
    y = preds["y"].to_numpy(np.int32)
    out = {}
    for fs in FSETS:
        p = preds[f"{model}__{fs}"].to_numpy(np.float64)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        out[fs] = (fpr, tpr, auc, prec, rec, ap)
    return out


def fig_roc_overlay(preds: pd.DataFrame, model: str) -> Path:
    curves = _curves_for_model(preds, model)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for fs in FSETS:
        fpr, tpr, auc, *_ = curves[fs]
        ax.plot(fpr, tpr, color=COLORS[fs], linewidth=1.8,
                label=f"{fs}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"TEST ROC curves — {model.upper()}  "
                 f"(34 positives / 148,639 pairs)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"test_{model}_roc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    return out


def fig_pr_overlay(preds: pd.DataFrame, model: str) -> Path:
    curves = _curves_for_model(preds, model)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    base_rate = preds["y"].mean()
    for fs in FSETS:
        *_, prec, rec, ap = curves[fs]
        ax.plot(rec, prec, color=COLORS[fs], linewidth=1.8,
                label=f"{fs}  (AP = {ap:.4f})")
    ax.axhline(base_rate, color="grey", linestyle="--", linewidth=1,
               label=f"base rate = {base_rate:.4%}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_yscale("log")
    ax.set_ylim(max(base_rate * 0.3, 1e-5), 1.0)
    ax.set_title(f"TEST precision-recall — {model.upper()}  "
                 f"(34 positives / 148,639 pairs)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / f"test_{model}_pr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    return out


# ------------------------------------------------------------------
# 4. Canonical headline metrics derived from the saved test parquet.
# Stage 7 metrics were computed in-memory from a separate model fit; with
# RF's n_jobs=-1 the predict_proba output can differ at the float-tie
# boundary between fits, producing small PR-AUC drift for the lowest-AP
# rows (S in particular). This recomputes headline metrics directly from
# the same probabilities that drive every figure and the bootstrap, so
# the paper has one canonical source of truth.
# ------------------------------------------------------------------

def recompute_headline_metrics(preds: pd.DataFrame) -> Path:
    y = preds["y"].to_numpy(np.int32)
    rows = []
    for m in ("logreg", "rf"):
        for fs in FSETS:
            p = preds[f"{m}__{fs}"].to_numpy(np.float64)
            row = {
                "feature_set": fs, "model": m,
                "roc_auc": float(roc_auc_score(y, p)),
                "pr_auc": float(average_precision_score(y, p)),
            }
            for k in (50, 100):
                top_idx = np.argpartition(-p, k - 1)[:k]
                row[f"precision@{k}"] = float(y[top_idx].mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "test_metrics_from_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}\n{df.round(4).to_string(index=False)}")
    return out


# ------------------------------------------------------------------
# 5. Top-200 / Top-500 error slices (supplementary)
# ------------------------------------------------------------------

def make_topk_slice(preds: pd.DataFrame, k: int) -> Path:
    """Top-k by rf__S+T+C probability with all six (model, fset) probas plus y.
    Kept slim — full feature joins are deferred to the existing top-50 CSV."""
    score = preds["rf__S+T+C"].to_numpy(np.float64)
    top_idx = np.argsort(-score)[:k]
    sub = preds.iloc[top_idx].copy()
    sub["rank"] = np.arange(1, len(sub) + 1)
    cols = (["rank", "u", "v", "snap_t", "y"]
            + [f"{m}__{fs}" for m in ("logreg", "rf") for fs in FSETS])
    out = RESULTS_DIR / f"stage7_error_top{k}.csv"
    sub[cols].to_csv(out, index=False)
    tp = int(sub["y"].sum())
    print(f"saved {out}  (top-{k}: TPs={tp}, FPs={k - tp}, "
          f"precision@{k}={tp / k:.4f})")
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(RESULTS_DIR / "test_predictions.parquet")

    print("=== Required figures ===")
    fig_modularity_over_time()
    fig_same_vs_cross_positive_rate()
    for model in ("rf", "logreg"):
        fig_roc_overlay(preds, model)
        fig_pr_overlay(preds, model)

    print("\n=== Canonical headline metrics (from saved predictions) ===")
    recompute_headline_metrics(preds)

    print("\n=== Supplementary error slices ===")
    for k in (200, 500):
        make_topk_slice(preds, k)


if __name__ == "__main__":
    main()
