"""Redundancia par a par: |Spearman| (continua-continua), V de Cramér
(categórica-categórica), razón de correlación eta (categórica-continua).
Las tres viven en [0,1] (ARCHITECTURE.md §6.2). SOLO sobre train."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér entre dos variables categóricas, en [0,1] (0 = independientes)."""
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = min(r - 1, k - 1)
    if denom == 0 or n == 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Razón de correlación eta entre una variable categórica y una continua, en
    [0,1]: raíz de la fracción de varianza de `values` explicada por `categories`."""
    values = values.to_numpy(dtype=float)
    cats = categories.to_numpy()
    overall_mean = values.mean()
    ss_between = 0.0
    for c in np.unique(cats):
        mask = cats == c
        n_c = mask.sum()
        ss_between += n_c * (values[mask].mean() - overall_mean) ** 2
    ss_total = ((values - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    eta_sq = ss_between / ss_total
    return float(np.sqrt(max(eta_sq, 0.0)))


def compute_redundancy(
    df_train: pd.DataFrame, variable_order: list[str], categorical_vars: set[str]
) -> np.ndarray:
    """Matriz de redundancia par a par en [0,1], diagonal en cero
    (ARCHITECTURE.md §6.2). Elige la medida según el tipo de cada par:
    |Spearman| (continua-continua), V de Cramér (categórica-categórica) o razón
    de correlación eta (mixto). SOLO debe llamarse con datos de entrenamiento.

    Args:
        df_train: DataFrame de train con las columnas en `variable_order`.
        variable_order: orden de variables; fija los índices i,j de la matriz.
        categorical_vars: subconjunto de `variable_order` a tratar como
            categóricas.

    Returns:
        Matriz simétrica `C` de forma (n, n) con `C[i][i] == 0`.
    """
    n = len(variable_order)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = variable_order[i], variable_order[j]
            i_cat, j_cat = vi in categorical_vars, vj in categorical_vars
            if i_cat and j_cat:
                c = cramers_v(df_train[vi], df_train[vj])
            elif i_cat and not j_cat:
                c = correlation_ratio(df_train[vi], df_train[vj])
            elif j_cat and not i_cat:
                c = correlation_ratio(df_train[vj], df_train[vi])
            else:
                c, _ = spearmanr(df_train[vi], df_train[vj])
                c = abs(c)
            C[i, j] = c
            C[j, i] = c
    return C
