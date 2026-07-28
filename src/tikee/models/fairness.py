"""Métricas de equidad por grupo + detección de proxies (AUC_proxy).
ARCHITECTURE.md §9."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from tikee.models.evaluate import ks_statistic


def group_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, group: pd.Series
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    group = pd.Series(group).reset_index(drop=True)

    per_group = {}
    approval_rate = {}
    tpr = {}
    fpr = {}
    brier = {}
    for g in group.unique():
        mask = (group == g).to_numpy()
        yt, ys, yp = y_true[mask], y_score[mask], y_pred[mask]
        auc_g = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan")
        per_group[g] = {"auc": auc_g, "n": int(mask.sum()), "ks": ks_statistic(yt, ys) if len(np.unique(yt)) > 1 else float("nan")}
        approval_rate[g] = float(1 - yp.mean())
        pos = yt == 1
        neg = yt == 0
        tpr[g] = float(yp[pos].mean()) if pos.sum() > 0 else float("nan")
        fpr[g] = float(yp[neg].mean()) if neg.sum() > 0 else float("nan")
        brier[g] = float(np.mean((ys - yt) ** 2))

    approval_values = list(approval_rate.values())
    tpr_values = [v for v in tpr.values() if not np.isnan(v)]
    fpr_values = [v for v in fpr.values() if not np.isnan(v)]

    demographic_parity_diff = max(approval_values) - min(approval_values)
    disparate_impact_ratio = (min(approval_values) / max(approval_values)) if max(approval_values) > 0 else float("nan")
    equal_opportunity_diff = (max(tpr_values) - min(tpr_values)) if tpr_values else float("nan")
    equalized_odds_diff = max(
        (max(tpr_values) - min(tpr_values)) if tpr_values else 0.0,
        (max(fpr_values) - min(fpr_values)) if fpr_values else 0.0,
    )

    return {
        "per_group": per_group,
        "approval_rate": approval_rate,
        "demographic_parity_diff": demographic_parity_diff,
        "disparate_impact_ratio": disparate_impact_ratio,
        "disparate_impact_flag_below_80pct": bool(disparate_impact_ratio < 0.80),
        "equal_opportunity_diff": equal_opportunity_diff,
        "equalized_odds_diff": equalized_odds_diff,
        "brier_by_group": brier,
    }


def proxy_detection_auc(X_selected: np.ndarray, protected_attr: pd.Series, seed: int, cv: int = 5) -> float:
    """AUC de un clasificador auxiliar que predice el atributo protegido a partir de
    las variables SELECCIONADAS por un brazo. ~0.5 = sin proxy; >0.70 = codificación
    sustancial (ARCHITECTURE.md §9.2)."""
    y = pd.factorize(protected_attr)[0]
    if len(np.unique(y)) != 2:
        raise ValueError("proxy_detection_auc espera un atributo protegido binario")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X_scaled, y, cv=cv_splitter, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))
