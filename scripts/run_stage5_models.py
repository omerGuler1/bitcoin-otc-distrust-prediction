"""Stage 5 (part 2) - 3-way feature ablation on validation.

Feature sets:
  STRUCTURAL_ONLY         = structural + trust-summary            (16 cols)
  +TEMPORAL               = STRUCTURAL_ONLY + temporal            (28 cols)
  +TEMPORAL +COMMUNITY    = + community-aware                     (36 cols)

Models: Random / Logistic Regression / Random Forest.
Train negatives subsampled 1:100; val full.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import build_features, models

TRAIN_TS = list(range(13, 46))
VAL_TS = [46, 47]
K = 50


def _prep(df, cols):
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int32)
    return X, y


def _evaluate_set(name, train_df, val_df, cols):
    X_tr_full, y_tr_full = _prep(train_df, cols)
    X_tr, y_tr = models.subsample_negatives(X_tr_full, y_tr_full)
    X_va, y_va = _prep(val_df, cols)
    print(f"\n[{name}] features={len(cols)}  "
          f"train after subsample: n={len(y_tr):,} pos={int(y_tr.sum())}")
    out, imps = {}, {}
    for m_name, fit_fn in models.FITTERS.items():
        res = fit_fn(X_tr, y_tr, X_va, cols)
        out[m_name] = models.evaluate(y_va, res.proba_val, k=K)
        imps[m_name] = res.importance
    return {"metrics": out, "imp": imps}


def main() -> None:
    train_df = build_features.load_split(TRAIN_TS)
    val_df = build_features.load_split(VAL_TS)

    cols_s = build_features.STRUCTURAL_ONLY
    cols_st = build_features.STRUCTURAL_PLUS_TEMPORAL
    cols_stc = build_features.FEATURE_COLUMNS

    r_s = _evaluate_set("structural only", train_df, val_df, cols_s)
    r_st = _evaluate_set("+temporal", train_df, val_df, cols_st)
    r_stc = _evaluate_set("+temporal +community", train_df, val_df, cols_stc)

    # --- 3-way ablation table ----------------------------------------------
    print("\n=== 3-way feature ablation on VAL ===")
    rows = []
    for m in ["random", "logreg", "rf"]:
        rows.append({
            "model": m,
            "S_roc_auc": r_s["metrics"][m]["roc_auc"],
            "ST_roc_auc": r_st["metrics"][m]["roc_auc"],
            "STC_roc_auc": r_stc["metrics"][m]["roc_auc"],
            "delta_ST_over_S": r_st["metrics"][m]["roc_auc"] - r_s["metrics"][m]["roc_auc"],
            "delta_STC_over_ST": r_stc["metrics"][m]["roc_auc"] - r_st["metrics"][m]["roc_auc"],
            "S_pr_auc": r_s["metrics"][m]["pr_auc"],
            "ST_pr_auc": r_st["metrics"][m]["pr_auc"],
            "STC_pr_auc": r_stc["metrics"][m]["pr_auc"],
            f"S_p@{K}": r_s["metrics"][m][f"precision@{K}"],
            f"ST_p@{K}": r_st["metrics"][m][f"precision@{K}"],
            f"STC_p@{K}": r_stc["metrics"][m][f"precision@{K}"],
        })
    comp = pd.DataFrame(rows)
    print(comp.round(4).to_string(index=False))

    # --- Community-feature importances (in the full model only) -----------
    ccols = set(build_features.COMMUNITY_FEATURES)

    def _only_community(imp, k=10):
        ser = pd.Series(imp)
        cc = ser[ser.index.isin(ccols)]
        order = cc.abs().sort_values(ascending=False).head(k).index
        return pd.DataFrame({"feature": order, "score": cc.loc[order].values})

    print("\n=== Community features - LR coef (standardized) ===")
    print(_only_community(r_stc["imp"]["logreg"]).round(4).to_string(index=False))
    print("\n=== Community features - RF importance (Gini) ===")
    print(_only_community(r_stc["imp"]["rf"]).round(4).to_string(index=False))

    # --- Full Stage 5 top-10 for context ----------------------------------
    def _topk(imp, k=10):
        ser = pd.Series(imp)
        order = ser.abs().sort_values(ascending=False).head(k).index
        return pd.DataFrame({"feature": order, "score": ser.loc[order].values})

    print("\n=== Stage 5 top-10 LR coefficients (any feature) ===")
    print(_topk(r_stc["imp"]["logreg"]).round(4).to_string(index=False))
    print("\n=== Stage 5 top-10 RF importances (any feature) ===")
    print(_topk(r_stc["imp"]["rf"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
