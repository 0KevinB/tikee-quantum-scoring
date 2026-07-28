"""Métricas por grupo correctas; AUC_proxy ~ 0.5 con variables aleatorias.
Ver ARCHITECTURE.md §9, §12."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tikee.models.fairness import group_metrics, proxy_detection_auc


def test_group_metrics_demographic_parity_detects_gap():
    rng = np.random.default_rng(0)
    n = 2000
    group = pd.Series(rng.choice(["A", "B"], size=n))
    y_true = rng.integers(0, 2, size=n)
    y_score = np.where(group == "A", rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    report = group_metrics(y_true, y_score, threshold=0.5, group=group)

    assert report["approval_rate"]["A"] < report["approval_rate"]["B"]
    assert report["demographic_parity_diff"] > 0.3
    assert report["disparate_impact_flag_below_80pct"]


def test_group_metrics_no_gap_when_scores_independent_of_group():
    rng = np.random.default_rng(1)
    n = 3000
    group = pd.Series(rng.choice(["A", "B"], size=n))
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.uniform(0, 1, size=n)
    report = group_metrics(y_true, y_score, threshold=0.5, group=group)
    assert report["demographic_parity_diff"] < 0.1


def test_proxy_auc_near_half_with_random_features():
    rng = np.random.default_rng(2)
    n = 1000
    X = rng.normal(0, 1, size=(n, 5))
    protected = pd.Series(rng.choice(["M", "F"], size=n))
    auc = proxy_detection_auc(X, protected, seed=2)
    assert 0.4 <= auc <= 0.6


def test_proxy_auc_high_when_feature_encodes_group():
    rng = np.random.default_rng(3)
    n = 1000
    protected = pd.Series(rng.choice(["M", "F"], size=n))
    signal = np.where(protected == "F", rng.normal(2, 0.3, n), rng.normal(-2, 0.3, n))
    X = np.column_stack([signal, rng.normal(0, 1, n)])
    auc = proxy_detection_auc(X, protected, seed=3)
    assert auc > 0.85
