"""Stage 4 (part 2) - train baselines with and without temporal features, compare.

Two feature sets:
  STAGE3  = structural + trust-summary (18 cols, has_* columns already dropped)
  STAGE4  = STAGE3 + temporal (30 cols)

Models: Random, Logistic Regression, Random Forest.

Training-set negatives are subsampled to TRAIN_NEG_PER_POS:1 per decision (A).
Validation and test remain full.

Reports:
  - validation metrics (ROC-AUC, PR-AUC, Precision, Recall, F1, P@50) for each
    (model, feature-set) combination
  - delta_AUC and delta_PR-AUC from Stage3 to Stage4 per model
  - top temporal features by LR coef and RF importance
  - comment on redundancy / instability among temporal features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import build_features, config, models

TRAIN_TS = list(range(13, 46))
VAL_TS = [46, 47]
K = 50


def _prep(df: pd.DataFrame, cols: list[str]):
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int32)
    return X, y


def _evaluate_set(name: str, train_df, val_df, cols) -> dict:
    X_tr_full, y_tr_full = _prep(train_df, cols)
    X_tr, y_tr = models.subsample_negatives(X_tr_full, y_tr_full)
    X_va, y_va = _prep(val_df, cols)

    print(f"\n[{name}] features={len(cols)}  "
          f"train after subsample: n={len(y_tr):,} pos={int(y_tr.sum())} "
          f"neg={int((y_tr == 0).sum()):,}")

    out = {}
    importances = {}
    for m_name, fit_fn in models.FITTERS.items():
        res = fit_fn(X_tr, y_tr, X_va, cols)
        out[m_name] = models.evaluate(y_va, res.proba_val, k=K)
        importances[m_name] = res.importance
    return {"metrics": out, "importances": importances}


def main() -> None:
    train_df = build_features.load_split(TRAIN_TS)
    val_df = build_features.load_split(VAL_TS)

    stage3_cols = build_features.STAGE3_FEATURES
    stage4_cols = build_features.FEATURE_COLUMNS
    assert len(stage4_cols) == len(stage3_cols) + len(
        build_features.TEMPORAL_FEATURES
    )

    r3 = _evaluate_set("STAGE3 (structural + trust)", train_df, val_df, stage3_cols)
    r4 = _evaluate_set("STAGE4 (+temporal)", train_df, val_df, stage4_cols)

    # --- Side-by-side metric comparison -------------------------------------
    print("\n=== Stage 3 vs Stage 4 validation metrics ===")
    rows = []
    for m in ["random", "logreg", "rf"]:
        a = r3["metrics"][m]; b = r4["metrics"][m]
        rows.append({
            "model": m,
            "stage3_roc_auc": a["roc_auc"], "stage4_roc_auc": b["roc_auc"],
            "delta_roc_auc": b["roc_auc"] - a["roc_auc"],
            "stage3_pr_auc": a["pr_auc"], "stage4_pr_auc": b["pr_auc"],
            "delta_pr_auc": b["pr_auc"] - a["pr_auc"],
            f"stage3_p@{K}": a[f"precision@{K}"], f"stage4_p@{K}": b[f"precision@{K}"],
        })
    comp = pd.DataFrame(rows)
    print(comp.round(4).to_string(index=False))

    # --- Temporal importances (Stage 4 models only) -------------------------
    tcols = set(build_features.TEMPORAL_FEATURES)

    def _top_temporal(imp: dict, k: int = 10) -> pd.DataFrame:
        if not imp:
            return pd.DataFrame()
        ser = pd.Series(imp)
        t_only = ser[ser.index.isin(tcols)]
        order = t_only.abs().sort_values(ascending=False).head(k).index
        return pd.DataFrame({"feature": order, "score": t_only.loc[order].values})

    print("\n=== Top temporal features - LR coefficient (standardized) ===")
    print(_top_temporal(r4["importances"]["logreg"]).round(4).to_string(index=False))
    print("\n=== Top temporal features - RF importance (Gini) ===")
    print(_top_temporal(r4["importances"]["rf"]).round(4).to_string(index=False))

    # --- Full top-10 Stage 4 (any feature) for context ---------------------
    def _topk(imp, k=10):
        ser = pd.Series(imp)
        order = ser.abs().sort_values(ascending=False).head(k).index
        return pd.DataFrame({"feature": order, "score": ser.loc[order].values})

    print("\n=== Stage 4 top-10 LR coefficients (any feature) ===")
    print(_topk(r4["importances"]["logreg"]).round(4).to_string(index=False))
    print("\n=== Stage 4 top-10 RF importances (any feature) ===")
    print(_topk(r4["importances"]["rf"]).round(4).to_string(index=False))

    # --- Redundancy diagnostic: correlation among temporal features -------
    tfeats = build_features.TEMPORAL_FEATURES
    corr = train_df[tfeats].corr().round(2)
    print("\n=== Pairwise correlation of temporal features (TRAIN) ===")
    print(corr.to_string())


if __name__ == "__main__":
    main()
