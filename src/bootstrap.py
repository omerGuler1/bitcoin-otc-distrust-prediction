"""Stage 8 - Bootstrap confidence intervals for the locked test-set results.

Methodology choices (documented for the paper):

  * Stratified bootstrap. The test set has 34 positives and 148,605 negatives.
    A naive bootstrap can produce a resample with 0 positives, leaving ROC-AUC
    and PR-AUC undefined. We therefore resample positives and negatives
    independently with replacement, each of size equal to its original count.
    This is the standard recommendation for ROC analysis under heavy class
    imbalance (Carpenter & Bithell 2000; Robin et al. 2011, pROC docs).

  * Paired bootstrap for deltas. To estimate uncertainty on a metric
    *difference* between two feature sets, we draw a single resample index
    vector per iteration and evaluate ALL feature sets on the same indices.
    This preserves correlation between feature sets (they score the same
    candidate pairs) and gives much tighter — and correct — CIs on the
    differences than an unpaired comparison.

  * Number of resamples B = 1000. This is a defensible default for 95%
    percentile CIs; with B = 1000 the 2.5th and 97.5th percentile estimates
    have low simulation noise relative to the underlying sampling variability
    that 34 positives can ever pin down.

  * Reporting. For each (model, feature_set, metric) we report the bootstrap
    mean, std, and 2.5 / 97.5 percentiles. For each delta we additionally
    report the fraction of bootstrap iterations in which delta > 0 — a
    Bayesian-flavoured tail probability, NOT a frequentist p-value. We are
    explicit about this distinction in the writeup.

  * What this does NOT do. We do not run a formal hypothesis test (e.g.,
    DeLong test for AUC). Bootstrap CIs estimate sampling uncertainty; we
    interpret an interval that overlaps zero as "improvement is not
    distinguishable from noise on this test set" rather than "no effect".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BootstrapConfig:
    B: int = 1000
    seed: int = 42
    ks: tuple[int, ...] = (50, 100)


def _precision_at_k(
    y: np.ndarray, proba: np.ndarray, k: int, rng: np.random.Generator
) -> float:
    """Precision at k with deterministic random tie-breaking.

    Random-Forest probabilities live on a discrete grid (one per leaf vote
    boundary), so a bootstrap resample with duplicates produces many ties
    at the top of the score list. `np.argpartition` resolves ties in an
    implementation-defined order; combined with the way we concatenate
    [positives, negatives] in the resample, this would systematically
    bias P@k. We break ties by adding a uniform [0, 1e-12] jitter per
    iteration — too small to perturb non-tied rankings, large enough to
    randomize ties.
    """
    if k > len(proba):
        raise ValueError(f"k={k} exceeds proba length {len(proba)}")
    jitter = rng.uniform(0.0, 1e-12, size=proba.shape)
    score = -(proba + jitter)
    top_idx = np.argpartition(score, k - 1)[:k]
    return float(y[top_idx].mean())


def metric_names(ks: tuple[int, ...]) -> list[str]:
    return ["roc_auc", "pr_auc"] + [f"p@{k}" for k in ks]


def _eval_all(
    y: np.ndarray,
    proba: np.ndarray,
    ks: tuple[int, ...],
    rng: np.random.Generator,
) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        **{f"p@{k}": _precision_at_k(y, proba, k, rng) for k in ks},
    }


def stratified_bootstrap(
    y: np.ndarray,
    proba_dict: Mapping[str, np.ndarray],
    cfg: BootstrapConfig = BootstrapConfig(),
) -> dict[str, dict[str, np.ndarray]]:
    """Paired stratified bootstrap.

    Returns
    -------
    samples : dict
        samples[feature_set][metric] is a length-B numpy array of bootstrap
        estimates. Index `b` is consistent across feature_sets because we
        evaluate all of them on the same resample indices each iteration.
    """
    y = np.asarray(y).astype(np.int32, copy=False)
    n_total = len(y)
    for fs_name, p in proba_dict.items():
        if len(p) != n_total:
            raise ValueError(f"proba[{fs_name}] length {len(p)} != y length {n_total}")
        if not np.isfinite(p).all():
            raise ValueError(f"proba[{fs_name}] contains non-finite values")

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0:
        raise ValueError("Stratified bootstrap requires at least one positive")

    rng = np.random.default_rng(cfg.seed)
    mnames = metric_names(cfg.ks)
    samples = {
        fs: {m: np.zeros(cfg.B, dtype=np.float64) for m in mnames}
        for fs in proba_dict
    }

    for b in range(cfg.B):
        pos_resample = rng.choice(pos_idx, size=n_pos, replace=True)
        neg_resample = rng.choice(neg_idx, size=n_neg, replace=True)
        idx_b = np.concatenate([pos_resample, neg_resample])
        y_b = y[idx_b]
        # Stratified resampling guarantees both labels are present, but
        # be defensive — sklearn AUC raises on single-class input.
        assert y_b.min() != y_b.max(), "resample collapsed to a single class"
        for fs, proba in proba_dict.items():
            p_b = proba[idx_b]
            res = _eval_all(y_b, p_b, cfg.ks, rng)
            for m in mnames:
                samples[fs][m][b] = res[m]

    return samples


def summarize(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Bootstrap point summary: mean, std, percentile CI."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lo": float(np.percentile(values, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(values, 100 * (1 - alpha / 2))),
    }


def delta_summary(
    values: np.ndarray, alpha: float = 0.05
) -> dict[str, float]:
    """Summary for a paired-bootstrap delta. Adds P(delta > 0)."""
    out = summarize(values, alpha=alpha)
    out["frac_positive"] = float(np.mean(values > 0))
    return out
