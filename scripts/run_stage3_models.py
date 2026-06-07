"""Stage 3 (part 2) - train baselines on training split, evaluate on validation.

Models: Random baseline, Logistic Regression, Random Forest.
Reports ROC-AUC, PR-AUC, Precision, Recall, F1, Precision@k.

k is chosen as min(50, 3 * n_pos_val) rounded to a clean number. With 16
validation positives, 3 * 16 = 48, so we use k=50 — a reasonable top-of-list
size that is (a) comparable to the number of positives we actually have and
(b) round enough to be defensible in the report. Larger k values (e.g. 100)
would push P@k below a meaningful ceiling because there aren't 100 positives
in val to recover. With more positives later we can revisit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import build_features, models

TRAIN_TS = list(range(13, 46))
VAL_TS = [46, 47]
K = 50


def main() -> None:
    train_df = build_features.load_split(TRAIN_TS)
    val_df = build_features.load_split(VAL_TS)

    feats = build_features.FEATURE_COLUMNS
    X_train = train_df[feats].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy(dtype=np.int32)
    X_val = val_df[feats].to_numpy(dtype=np.float32)
    y_val = val_df["y"].to_numpy(dtype=np.int32)

    print(f"TRAIN n={len(y_train):,}  pos={int(y_train.sum())}  "
          f"neg={int((y_train == 0).sum()):,}")
    print(f"VAL   n={len(y_val):,}  pos={int(y_val.sum())}  "
          f"neg={int((y_val == 0).sum()):,}")
    print(f"Evaluation k for Precision@k: {K}")

    results = {}
    importances = {}
    for name, fit_fn in models.FITTERS.items():
        print(f"\n[fit] {name} ...")
        res = fit_fn(X_train, y_train, X_val, feats)
        metrics = models.evaluate(y_val, res.proba_val, k=K)
        results[name] = metrics
        importances[name] = res.importance
        print(f"  metrics: {metrics}")

    # --- Metrics table ------------------------------------------------------
    print("\n=== Validation metrics ===")
    mdf = pd.DataFrame(results).T
    print(mdf.round(4).to_string())

    # --- Top-10 influential features ---------------------------------------
    def _topk(imp: dict, k: int = 10) -> pd.DataFrame:
        if not imp:
            return pd.DataFrame()
        ser = pd.Series(imp)
        order = ser.abs().sort_values(ascending=False).head(k).index
        return pd.DataFrame({"feature": order, "score": ser.loc[order].values})

    print("\n=== Top 10 Logistic Regression coefficients "
          "(on standardized features; + drives positive class) ===")
    print(_topk(importances["logreg"], 10).round(4).to_string(index=False))

    print("\n=== Top 10 Random Forest feature importances (Gini) ===")
    print(_topk(importances["rf"], 10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
