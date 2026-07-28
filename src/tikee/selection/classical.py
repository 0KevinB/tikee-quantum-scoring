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
    """Selección embebida por LASSO (L1): ajusta `LogisticRegressionCV` con C
    elegido por CV en train, y reporta como "seleccionada" toda variable con al
    menos una columna transformada de coeficiente no nulo. Brazo B0/B0b.

    Args:
        X: matriz de features ya transformada (salida de `preprocess.py`).
        y: etiqueta binaria.
        variable_groups: mapeo nombre de variable -> índices de columna en X
            (de `features.preprocess.get_variable_groups`), para poder reportar
            variables en vez de columnas dummy sueltas.
        seed: semilla de `LogisticRegressionCV`.
        cv: número de pliegues para elegir C.

    Returns:
        dict con `selected_variables`, `selected_columns`, `coef` (todos los
        coeficientes) y `C` (el hiperparámetro elegido).
    """
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
    """Selección stepwise hacia adelante por AIC (statsmodels Logit): en cada paso
    agrega la variable candidata que más reduce el AIC del modelo, hasta que
    ninguna lo mejora. Opera a nivel de variable completa (todas sus columnas
    transformadas entran o salen juntas), no por columna dummy. Brazo B1/B1b.

    Args:
        X: matriz de features ya transformada.
        y: etiqueta binaria.
        variable_groups: mapeo nombre de variable -> índices de columna en X.
        candidate_vars: subconjunto de variables a considerar; por defecto todas
            las claves de `variable_groups`.

    Returns:
        dict con `selected_variables` (en orden de entrada), `selected_columns`
        y `final_aic`.
    """
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
