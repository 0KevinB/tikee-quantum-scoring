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
    """Estadístico de Kolmogorov-Smirnov: max(TPR - FPR) sobre todos los umbrales."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def find_threshold_max_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """D19: umbral fijado por máximo KS en TRAIN, congelado y aplicado a test."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Batería completa de métricas de discriminación, calibración y negocio
    sobre un umbral ya fijado (ARCHITECTURE.md §8.3).

    Args:
        y_true: etiqueta binaria real.
        y_score: probabilidad predicha de la clase positiva.
        threshold: umbral de decisión (D19: fijado por máximo KS en train, nunca
            en test — usar `find_threshold_max_ks` sobre el score de train).

    Returns:
        dict con `auc`, `ks`, `pr_auc`, `gini`, `brier`, `precision`, `recall`,
        `f1`, `confusion_matrix` (tn/fp/fn/tp), `approval_rate` y `threshold`.
    """
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
    """Intervalo de confianza del AUC por bootstrap sobre el test (§8.6): mide
    incertidumbre DENTRO de una semilla, distinta de la variabilidad ENTRE
    semillas que capturan `experiments.stats`. No se deben sumar ambas fuentes.

    Args:
        y_true: etiqueta binaria real del conjunto de prueba.
        y_score: probabilidad predicha, mismo largo que `y_true`.
        n_boot: número de remuestreos.
        seed: semilla del remuestreo.
        alpha: nivel de significancia (0.05 -> IC del 95%).

    Returns:
        Tupla `(lo, hi)` con los percentiles `alpha/2` y `1-alpha/2` del AUC
        remuestreado. Los remuestreos sin ambas clases se descartan.
    """
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
