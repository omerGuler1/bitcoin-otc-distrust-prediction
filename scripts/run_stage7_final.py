"""Stage 7 - Final evaluation on the held-out TEST split (t=48..49).

Locks in the project's final headline results.

  - Trains Logistic Regression and Random Forest on TRAIN (negatives
    subsampled 1:100; positives kept in full) for each of the three feature
    sets: S (structural+trust), S+T (+temporal), S+T+C (+community).
  - Evaluates on TEST (34 positives / 148,639 pairs; full).
  - Produces:
      results/stage7_test_metrics.csv    headline comparison
      results/stage7_importances.csv     all features with family grouping
      results/stage7_error_analysis.csv  top-50 true / false positives + features

We do NOT retune thresholds or hyperparameters here — all choices were locked
before the test split was ever looked at. What we print is the final number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import build_features, config, models

TRAIN_TS = list(range(13, 46))
VAL_TS = [46, 47]
TEST_TS = [48, 49]
K_LIST = [50, 100]

FEATURE_SETS = {
    "S":     build_features.STRUCTURAL_ONLY,
    "S+T":   build_features.STRUCTURAL_PLUS_TEMPORAL,
    "S+T+C": build_features.FEATURE_COLUMNS,
}

STRUCT_SET = set(build_features.STRUCTURAL_ONLY)
TEMPORAL_SET = set(build_features.TEMPORAL_FEATURES)
COMMUNITY_SET = set(build_features.COMMUNITY_FEATURES)


def _family(col: str) -> str:
    if col in COMMUNITY_SET:
        return "community"
    if col in TEMPORAL_SET:
        return "temporal"
    if col in STRUCT_SET:
        return "structural"
    return "other"


def _prep(df: pd.DataFrame, cols: list[str]):
    return df[cols].to_numpy(np.float32), df["y"].to_numpy(np.int32)


def _evaluate_all_ks(y, proba, ks):
    out = {}
    for k in ks:
        if len(proba) >= k:
            top_idx = np.argsort(-proba)[:k]
            out[f"precision@{k}"] = float(y[top_idx].mean())
        else:
            out[f"precision@{k}"] = float("nan")
    return out


def main() -> None:
    train_df = build_features.load_split(TRAIN_TS)
    test_df = build_features.load_split(TEST_TS)
    print(f"TRAIN: n={len(train_df):,}  pos={int(train_df['y'].sum())}")
    print(f"TEST : n={len(test_df):,}  pos={int(test_df['y'].sum())}")

    metric_rows = []
    importance_rows = []
    # Cache the final (S+T+C) RF proba for error analysis.
    best_proba = None
    best_name = "rf @ S+T+C"

    for fs_name, cols in FEATURE_SETS.items():
        X_tr_full, y_tr_full = _prep(train_df, cols)
        X_tr, y_tr = models.subsample_negatives(X_tr_full, y_tr_full)
        X_te, y_te = _prep(test_df, cols)

        for m_name in ("logreg", "rf"):
            print(f"\n[fit] {m_name} on {fs_name} (features={len(cols)}, "
                  f"train n={len(y_tr):,})")
            res = models.FITTERS[m_name](X_tr, y_tr, X_te, cols)
            core = models.evaluate(y_te, res.proba_val, k=50)
            pks = _evaluate_all_ks(y_te, res.proba_val, K_LIST)
            row = {
                "feature_set": fs_name, "model": m_name,
                "roc_auc": core["roc_auc"], "pr_auc": core["pr_auc"],
                "precision_at_0.5": core["precision"],
                "recall_at_0.5": core["recall"],
                "f1_at_0.5": core["f1"],
                **pks,
                "n_features": len(cols), "n_pos_test": int(y_te.sum()),
                "n_test": int(len(y_te)),
            }
            metric_rows.append(row)
            print(f"  {row}")

            if fs_name == "S+T+C":
                for feat, score in res.importance.items():
                    importance_rows.append({
                        "model": m_name, "feature": feat,
                        "family": _family(feat), "score": float(score),
                    })

            if fs_name == "S+T+C" and m_name == "rf":
                best_proba = res.proba_val

    # --- Final comparison table --------------------------------------------
    metrics = pd.DataFrame(metric_rows)
    metrics_out = config.PROJECT_ROOT / "results" / "stage7_test_metrics.csv"
    metrics.to_csv(metrics_out, index=False)
    print("\n=== Final test-set ablation (headline table) ===")
    show = ["feature_set", "model", "roc_auc", "pr_auc",
            "precision@50", "precision@100",
            "precision_at_0.5", "recall_at_0.5", "f1_at_0.5"]
    print(metrics[show].round(4).to_string(index=False))

    # --- Feature importances by family ------------------------------------
    imp = pd.DataFrame(importance_rows)
    imp_out = config.PROJECT_ROOT / "results" / "stage7_importances.csv"
    imp.to_csv(imp_out, index=False)

    def _top_by_family(imp_df: pd.DataFrame, model: str, fam: str, k: int = 5):
        sub = imp_df[(imp_df["model"] == model) & (imp_df["family"] == fam)].copy()
        sub["abs"] = sub["score"].abs()
        return sub.sort_values("abs", ascending=False).head(k)[["feature", "score"]]

    print("\n=== Top features by family (LR, standardized coefficients) ===")
    for fam in ("structural", "temporal", "community"):
        print(f"\n-- {fam} --")
        print(_top_by_family(imp, "logreg", fam).round(4).to_string(index=False))

    print("\n=== Top features by family (RF, Gini importance) ===")
    for fam in ("structural", "temporal", "community"):
        print(f"\n-- {fam} --")
        print(_top_by_family(imp, "rf", fam).round(4).to_string(index=False))

    # --- Error analysis on best model (RF @ S+T+C) -------------------------
    if best_proba is not None:
        err = test_df[["u", "v", "y", "snap_t"] + list(FEATURE_SETS["S+T+C"])].copy()
        err["proba"] = best_proba
        # Top 50 scored pairs overall.
        top50 = err.sort_values("proba", ascending=False).head(50).copy()
        top50["is_true_positive"] = top50["y"] == 1
        top_tp = top50[top50["is_true_positive"]].head(10).copy()
        top_fp = top50[~top50["is_true_positive"]].head(10).copy()

        # Compact views for the console (key features only).
        focus = [
            "u", "v", "snap_t", "y", "proba",
            # structural
            "cn", "aa", "pa", "neg_out_u", "neg_in_v", "reciprocity_flag",
            # temporal
            "days_since_last_activity_u", "days_since_last_activity_v",
            "decayed_neg_received_v", "decayed_total_activity_u",
            "uv_interaction_count_past",
            # community
            "same_community", "neighborhood_overlap", "boundary_frac_u",
        ]
        print("\n=== Error analysis — top-10 TRUE POSITIVES (highest-scored & correct) ===")
        print(top_tp[focus].round(3).to_string(index=False))
        print("\n=== Error analysis — top-10 FALSE POSITIVES (highest-scored & wrong) ===")
        print(top_fp[focus].round(3).to_string(index=False))

        # Persist full top-50 with all features for the report.
        err_out = config.PROJECT_ROOT / "results" / "stage7_error_analysis.csv"
        top50[["u", "v", "snap_t", "y", "proba"] + list(FEATURE_SETS["S+T+C"])].to_csv(
            err_out, index=False
        )

        # Aggregate pattern summary on top-50.
        print("\n=== Aggregate patterns on top-50 scored pairs "
              "(by prediction outcome) ===")
        agg_cols = [
            "proba",
            "neg_out_u", "neg_in_v", "decayed_neg_received_v",
            "decayed_total_activity_u", "days_since_last_activity_u",
            "same_community", "neighborhood_overlap", "boundary_frac_u",
            "uv_interaction_count_past",
        ]
        agg = top50.groupby("is_true_positive")[agg_cols].mean()
        print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
