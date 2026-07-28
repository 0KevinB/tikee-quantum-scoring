"""Relevancia: información mutua variable-target, normalizada a [0,1]. SOLO sobre
train (ARCHITECTURE.md §6.2, regla de higiene §3.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def compute_relevance(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    variable_order: list[str],
    categorical_vars: set[str],
    seed: int,
    n_neighbors: int = 3,
) -> np.ndarray:
    """Información mutua de cada variable con el target, normalizada al máximo
    (ARCHITECTURE.md §6.2). Usa `mutual_info_classif` de sklearn, que captura
    relaciones no lineales (importante para variables como `edad`, cuyo riesgo
    es en U). SOLO debe llamarse con datos de entrenamiento.

    Args:
        df_train: DataFrame de train con las columnas en `variable_order`.
        y_train: etiqueta binaria (default), misma longitud que `df_train`.
        variable_order: orden de variables a evaluar; fija el índice i de cada
            variable para el resto del pipeline QUBO.
        categorical_vars: subconjunto de `variable_order` a tratar como
            discretas/categóricas (se factorizan antes de estimar la MI).
        seed: semilla para `mutual_info_classif` (afecta el término de ruido del
            estimador de vecinos).
        n_neighbors: vecinos del estimador KSG de información mutua.

    Returns:
        Vector R en [0,1], `R[i]` = MI de `variable_order[i]` normalizada al
        máximo. Si toda la MI es cero, devuelve un vector de ceros.
    """
    encoded = np.empty((len(df_train), len(variable_order)))
    discrete_mask = []
    for j, v in enumerate(variable_order):
        col = df_train[v]
        if v in categorical_vars:
            encoded[:, j] = pd.factorize(col)[0]
            discrete_mask.append(True)
        else:
            encoded[:, j] = col.to_numpy(dtype=float)
            discrete_mask.append(False)

    mi = mutual_info_classif(
        encoded, y_train, discrete_features=discrete_mask, n_neighbors=n_neighbors, random_state=seed
    )
    max_mi = mi.max()
    if max_mi <= 0:
        return np.zeros_like(mi)
    return mi / max_mi
