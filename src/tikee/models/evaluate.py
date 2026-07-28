"""AUC, KS, PR-AUC, Brier, P/R/F1, matriz de confusión, IC bootstrap.
ARCHITECTURE.md §8.3, §8.6."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def find_threshold_max_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """D19: umbral fijado por máximo KS en TRAIN, congelado y aplicado a test."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "gini": float(2 * roc_auc_score(y_true, y_score) - 1),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "approval_rate": float(1 - y_pred.mean()),
        "threshold": float(threshold),
    }


def bootstrap_ci_auc(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int | None = None, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(yt, ys)
    aucs = aucs[~np.isnan(aucs)]
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi
