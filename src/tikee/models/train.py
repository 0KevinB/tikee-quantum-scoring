"""LogReg y XGBoost; CV anidada para XGBoost (E13). ARCHITECTURE.md §6.1, §8.2."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

XGB_PARAM_DISTRIBUTIONS = {
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.01, 0.29),
    "n_estimators": randint(100, 601),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 21),
    "reg_lambda": uniform(0.1, 9.9),
}


def train_logreg(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def train_xgboost_fixed(X: np.ndarray, y: np.ndarray, seed: int, params: dict[str, Any]) -> XGBClassifier:
    clf = XGBClassifier(
        **params, random_state=seed, eval_metric="auc", n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def nested_cv_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_iter: int = 30,
) -> dict[str, Any]:
    """D13: CV anidada solo para validar que XGBoost está afinado (3 semillas). No es
    el modelo final; los hiperparámetros medianos encontrados aquí se reutilizan fijos
    en las 10 semillas (protocolo de holdout, §8.2)."""
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    best_params_per_fold = []
    outer_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        base = XGBClassifier(random_state=seed, eval_metric="auc", n_jobs=-1)
        search = RandomizedSearchCV(
            base, XGB_PARAM_DISTRIBUTIONS, n_iter=n_iter, cv=inner_cv,
            scoring="roc_auc", random_state=seed, n_jobs=-1,
        )
        search.fit(X[train_idx], y[train_idx])
        best_params_per_fold.append(search.best_params_)

        from sklearn.metrics import roc_auc_score
        proba = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
        outer_scores.append(roc_auc_score(y[test_idx], proba))

    return {
        "outer_auc_scores": outer_scores,
        "outer_auc_mean": float(np.mean(outer_scores)),
        "outer_auc_std": float(np.std(outer_scores)),
        "best_params_per_fold": best_params_per_fold,
    }


def median_hyperparams(list_of_best_params: list[dict[str, Any]]) -> dict[str, Any]:
    keys = list_of_best_params[0].keys()
    out = {}
    for k in keys:
        values = [p[k] for p in list_of_best_params]
        med = float(np.median(values))
        out[k] = int(round(med)) if k in ("max_depth", "n_estimators", "min_child_weight") else med
    return out
