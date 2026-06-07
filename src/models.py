"""Stage 3 - Simple, interpretable baselines.

Three models:
  - Random baseline: uniform probabilities (seeded).
  - Logistic Regression: with StandardScaler + class_weight='balanced'.
  - Random Forest: class_weight='balanced_subsample', n_estimators=200.

Metrics on the validation set:
  ROC-AUC, PR-AUC (average precision), Precision, Recall, F1 (at default 0.5),
  and Precision@k for a pre-chosen k.

We also compute feature importances:
  - LR: standardized-coefficient magnitude (sign + |value|).
  - RF: Gini feature_importances_.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def subsample_negatives(
    X: np.ndarray,
    y: np.ndarray,
    neg_per_pos: int = config.TRAIN_NEG_PER_POS,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all positives, subsample negatives to neg_per_pos:1 ratio.

    Used for TRAINING ONLY — validation and test must remain full.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    target_neg = len(pos_idx) * neg_per_pos
    if len(neg_idx) > target_neg:
        neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return X[idx], y[idx]


@dataclass
class FitResult:
    name: str
    proba_val: np.ndarray
    # feature -> signed score (coefficient for LR, importance for RF)
    importance: dict = field(default_factory=dict)


def _random_probs(n: int, seed: int = config.RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n)


def fit_random(X_train, y_train, X_val, feature_names) -> FitResult:
    return FitResult(
        name="random",
        proba_val=_random_probs(len(X_val)),
        importance={},
    )


def fit_logreg(X_train, y_train, X_val, feature_names) -> FitResult:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="liblinear",          # stable for sparse, wide-margin-ish data
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    # Standardized coefficients = coef on scaled features
    coef = pipe.named_steps["lr"].coef_[0]
    imp = dict(zip(feature_names, coef))
    return FitResult(name="logreg", proba_val=proba, importance=imp)


def fit_rf(X_train, y_train, X_val, feature_names) -> FitResult:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_val)[:, 1]
    imp = dict(zip(feature_names, rf.feature_importances_))
    return FitResult(name="rf", proba_val=proba, importance=imp)


def evaluate(y_val: np.ndarray, proba_val: np.ndarray, k: int) -> dict:
    pred = (proba_val >= 0.5).astype(int)
    p_at_k = np.nan
    if k > 0 and len(proba_val) >= k:
        top_idx = np.argsort(-proba_val)[:k]
        p_at_k = float(y_val[top_idx].mean())
    return {
        "roc_auc": float(roc_auc_score(y_val, proba_val)) if y_val.sum() > 0 else float("nan"),
        "pr_auc": float(average_precision_score(y_val, proba_val)) if y_val.sum() > 0 else float("nan"),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        f"precision@{k}": p_at_k,
        "n_pos": int(y_val.sum()),
        "n_total": int(len(y_val)),
    }


FITTERS: dict[str, Callable] = {
    "random": fit_random,
    "logreg": fit_logreg,
    "rf": fit_rf,
}
