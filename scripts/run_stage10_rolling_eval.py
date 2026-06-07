"""Stage 10 - Rolling-origin evaluation.

Replaces the single-test evaluation with an expanding-window rolling-origin
protocol across every eligible test snapshot. The original Stage 7 / Stage 8
results on the locked t=48,49 fold are NOT touched; the rolling protocol is
additive.

PROTOCOL
========
For each fold k in FOLD_KS (single-month test windows):

    TRAIN = candidate pairs from snapshots [START_T, k-2]
    VAL   = candidate pairs from snapshot {k-1}    (held out; not tuned on)
    TEST  = candidate pairs from snapshot {k}

  - TRAIN negatives are subsampled to TRAIN_NEG_PER_POS (100:1), exactly as
    in every prior stage. Validation and test stay full.
  - Each (model, feature_set) is fit fresh per fold: 25 folds * 6 (model,fs)
    combinations = 150 model fits.
  - Seeds (subsample, LR, RF, Louvain partitions) are all fixed; the same
    training-negative subsample is used for every (model, fs) within a fold,
    so per-fold metric *differences* between feature sets are paired.

WHY EXPANDING + SINGLE-MONTH TEST
==================================
  - Expanding (not sliding) train: never throws away earlier positives. Early
    distrust patterns are rare and we cannot afford to drop them.
  - Single-month test windows: maximize the number of disjoint test folds
    (25 vs ~12 for two-month windows), match operational deployment cadence
    ("next-month watchlist"), and produce fold-level metric vectors that are
    directly usable for paired non-parametric tests in Stage 13.

OUTPUTS (new files, no overwrites)
==================================
    results/rolling_eval_per_fold.csv     one row per (fold, model, fset)
    results/rolling_eval_summary.csv      per (model, fset): mean/std/total pos
    results/rolling_eval_log.txt          progress + run timings + summary
    results/rolling_predictions/          per-fold proba parquet for RF S+T+C
                                            (downstream Phase D / E need it)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import build_features, config, models

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
START_T = 13                            # first eligible snapshot
END_T = 49                              # last eligible snapshot
MIN_TRAIN_SNAPS = 12                    # require >= 12 snapshots in TRAIN
                                        # smallest fold has train [13..23], val 24, test 25
FOLD_KS = list(range(START_T + MIN_TRAIN_SNAPS, END_T + 1))   # k = 25..49 (25 folds)

FEATURE_SETS = {
    "S":     build_features.STRUCTURAL_ONLY,
    "S+T":   build_features.STRUCTURAL_PLUS_TEMPORAL,
    "S+T+C": build_features.FEATURE_COLUMNS,
}
MODEL_NAMES = ("logreg", "rf")
HEADLINE_KEY = ("rf", "S+T+C")          # save per-fold probas for this one

RESULTS_DIR = config.PROJECT_ROOT / "results"
PRED_DIR = RESULTS_DIR / "rolling_predictions"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def precision_at_k(y: np.ndarray, proba: np.ndarray, k: int) -> float:
    if k > len(proba):
        return float("nan")
    top = np.argpartition(-proba, k - 1)[:k]
    return float(y[top].mean())


def eval_one(y: np.ndarray, proba: np.ndarray) -> dict:
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
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    train_ts = list(range(START_T, k - 1))    # inclusive: [START, k-2]
    val_ts = [k - 1]                          # not used for tuning here
    test_ts = [k]
    train_df = build_features.load_split(train_ts)
    val_df = build_features.load_split(val_ts)
    test_df = build_features.load_split(test_ts)

    n_train, n_train_pos = len(train_df), int(train_df["y"].sum())
    n_val, n_val_pos = len(val_df), int(val_df["y"].sum())
    n_test, n_test_pos = len(test_df), int(test_df["y"].sum())
    log(f"   k={k}: train [{train_ts[0]}..{train_ts[-1]}] "
        f"({n_train:,} pairs, {n_train_pos} pos), "
        f"val {{{val_ts[0]}}} ({n_val:,}, {n_val_pos}), "
        f"test {{{test_ts[0]}}} ({n_test:,}, {n_test_pos})")

    rows: list[dict] = []
    y_te = test_df["y"].to_numpy(np.int32)

    # cached per-fold test probas for HEADLINE_KEY only
    headline_proba_to_save: np.ndarray | None = None

    for fs_name, cols in FEATURE_SETS.items():
        X_tr_full = train_df[cols].to_numpy(np.float32)
        y_tr_full = train_df["y"].to_numpy(np.int32)
        X_tr, y_tr = models.subsample_negatives(X_tr_full, y_tr_full)
        X_te = test_df[cols].to_numpy(np.float32)
        for m_name in MODEL_NAMES:
            t0 = time.time()
            res = models.FITTERS[m_name](X_tr, y_tr, X_te, cols)
            elapsed = time.time() - t0
            metrics = eval_one(y_te, res.proba_val)
            rows.append({
                "fold_k": k,
                "model": m_name,
                "feature_set": fs_name,
                "n_train_snaps": len(train_ts),
                "n_train_pairs": n_train,
                "n_train_pos": n_train_pos,
                "n_test_pairs": n_test,
                "n_test_pos": n_test_pos,
                "fit_seconds": elapsed,
                **metrics,
            })
            log(f"     {m_name:6s} | {fs_name:6s} | "
                f"AUC={metrics['roc_auc']:.4f} "
                f"PR={metrics['pr_auc']:.4f} "
                f"P@50={metrics['p@50']:.4f} "
                f"({elapsed:.1f}s)")
            if (m_name, fs_name) == HEADLINE_KEY:
                headline_proba_to_save = res.proba_val.astype(np.float32)

    if headline_proba_to_save is not None:
        out = test_df[["u", "v", "snap_t", "y"]].copy()
        out["proba_rf_stc"] = headline_proba_to_save
        out.to_parquet(PRED_DIR / f"fold_k={k:03d}.parquet", index=False)

    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_per_fold = RESULTS_DIR / "rolling_eval_per_fold.csv"
    out_summary = RESULTS_DIR / "rolling_eval_summary.csv"
    out_log = RESULTS_DIR / "rolling_eval_log.txt"

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)
        out_log.write_text("\n".join(log_lines))

    log(f"Stage 10 rolling-origin evaluation")
    log(f"  folds: k = {FOLD_KS[0]}..{FOLD_KS[-1]} (n = {len(FOLD_KS)})")
    log(f"  models: {MODEL_NAMES}")
    log(f"  feature sets: {list(FEATURE_SETS)}")
    log(f"  protocol: TRAIN [START..k-2], VAL {{k-1}}, TEST {{k}}")
    log(f"  TRAIN neg:pos subsample = {config.TRAIN_NEG_PER_POS}:1")
    log("")

    all_rows: list[dict] = []
    t_start = time.time()
    for i, k in enumerate(FOLD_KS, 1):
        t0 = time.time()
        log(f"-- fold {i}/{len(FOLD_KS)}  k={k}  "
            f"(cumulative {time.time() - t_start:.0f}s)")
        rows = run_fold(k, log)
        all_rows.extend(rows)
        # Incremental save so we can recover if interrupted.
        pd.DataFrame(all_rows).to_csv(out_per_fold, index=False)
        log(f"   fold k={k} done in {time.time() - t0:.0f}s; "
            f"per-fold CSV updated.")
        log("")

    df = pd.DataFrame(all_rows)

    # Across-fold aggregation. mean / std / median for ROC/PR/P@k; total pos.
    aggs = []
    for (m, fs), g in df.groupby(["model", "feature_set"]):
        aggs.append({
            "model": m,
            "feature_set": fs,
            "n_folds": len(g),
            "total_test_pos": int(g["n_test_pos"].sum()),
            "mean_roc_auc": g["roc_auc"].mean(),
            "std_roc_auc": g["roc_auc"].std(ddof=1),
            "median_roc_auc": g["roc_auc"].median(),
            "mean_pr_auc": g["pr_auc"].mean(),
            "std_pr_auc": g["pr_auc"].std(ddof=1),
            "median_pr_auc": g["pr_auc"].median(),
            "mean_p50": g["p@50"].mean(),
            "std_p50": g["p@50"].std(ddof=1),
            "median_p50": g["p@50"].median(),
            "mean_p100": g["p@100"].mean(),
            "std_p100": g["p@100"].std(ddof=1),
            "median_p100": g["p@100"].median(),
        })
    summary = pd.DataFrame(aggs)
    summary.to_csv(out_summary, index=False)

    log("")
    log(f"Total elapsed: {time.time() - t_start:.0f}s")
    log(f"Saved {out_per_fold}")
    log(f"Saved {out_summary}")
    log(f"Saved per-fold RF S+T+C probas under {PRED_DIR}")
    log("")
    log("Across-fold summary (mean +- std):")
    log(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
