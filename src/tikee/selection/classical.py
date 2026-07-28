"""LASSO (L1) y stepwise forward por AIC. Brazos B0, B1 (ARCHITECTURE.md §8.1).

Operan sobre la matriz ya transformada por preprocess.py; `variable_groups` (de
`features.preprocess.get_variable_groups`) traduce columnas transformadas de vuelta
a nombres de variable declarados, para poder reportar "qué variables eligió" en el
mismo lenguaje que el brazo QUBO."""

from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV


def lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    variable_groups: dict[str, list[int]],
    seed: int,
    cv: int = 5,
) -> dict[str, Any]:
    Cs = np.logspace(-3, 1, 20)
    clf = LogisticRegressionCV(
        Cs=Cs, cv=cv, penalty="l1", solver="liblinear", scoring="roc_auc",
        random_state=seed, max_iter=5000,
    )
    clf.fit(X, y)
    coefs = clf.coef_[0]
    selected_variables = [v for v, idxs in variable_groups.items() if np.any(np.abs(coefs[idxs]) > 1e-6)]
    selected_cols = sorted(i for v in selected_variables for i in variable_groups[v])
    return {
        "selected_variables": selected_variables,
        "selected_columns": selected_cols,
        "coef": coefs.tolist(),
        "C": float(clf.C_[0]),
    }


def stepwise_forward_aic(
    X: np.ndarray,
    y: np.ndarray,
    variable_groups: dict[str, list[int]],
    candidate_vars: list[str] | None = None,
) -> dict[str, Any]:
    candidate_vars = list(candidate_vars) if candidate_vars is not None else list(variable_groups.keys())
    selected: list[str] = []
    remaining = list(candidate_vars)
    current_aic = np.inf
    improved = True

    while remaining and improved:
        improved = False
        best_var, best_aic = None, current_aic
        for v in remaining:
            cols = sorted(i for t in selected + [v] for i in variable_groups[t])
            Xs = sm.add_constant(X[:, cols], has_constant="add")
            try:
                model = sm.Logit(y, Xs).fit(disp=0, maxiter=200)
            except Exception:  # noqa: BLE001
                continue
            if model.aic < best_aic:
                best_aic = model.aic
                best_var = v
        if best_var is not None:
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            improved = True

    selected_cols = sorted(i for v in selected for i in variable_groups[v])
    return {"selected_variables": selected, "selected_columns": selected_cols, "final_aic": float(current_aic)}
