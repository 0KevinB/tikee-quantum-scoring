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
