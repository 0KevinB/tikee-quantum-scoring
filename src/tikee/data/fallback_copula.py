"""Respaldo sin SDV: cópula gaussiana propia con scipy.stats.

Usado como `sdv_synthesizer.py` de reemplazo si `sdv` no instala (ARCHITECTURE.md §1).
Implementa la misma interfaz mínima que se espera de un sintetizador: `fit(df)` y
`sample(n)`. Preserva la matriz de correlación (Spearman/Pearson sobre rangos) por
construcción, igual que `GaussianCopulaSynthesizer`: ajusta una normal multivariante
sobre los rangos normalizados de cada columna y, al muestrear, invierte cada columna
por su función de cuantiles empírica (continuas) o su distribución de categorías
(categóricas/discretas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class FallbackGaussianCopula:
    def __init__(self, categorical_columns: list[str] | None = None):
        self.categorical_columns = set(categorical_columns or [])
        self.columns_: list[str] = []
        self.corr_: np.ndarray | None = None
        self._empirical_values: dict[str, np.ndarray] = {}
        self._category_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._rng = np.random.default_rng()

    def fit(self, df: pd.DataFrame, seed: int | None = None) -> "FallbackGaussianCopula":
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.columns_ = list(df.columns)
        n = len(df)
        normal_scores = np.empty((n, len(self.columns_)))

        for j, col in enumerate(self.columns_):
            series = df[col]
            if col in self.categorical_columns or series.dtype == object:
                categories, counts = np.unique(series.astype(str), return_counts=True)
                probs = counts / counts.sum()
                self._category_probs[col] = (categories, np.cumsum(probs))
                ranks = series.astype(str).map(
                    {c: p for c, p in zip(categories, np.cumsum(probs) - probs / 2)}
                ).to_numpy(dtype=float)
            else:
                values = series.to_numpy(dtype=float)
                self._empirical_values[col] = np.sort(values)
                ranks = (pd.Series(values).rank(method="average").to_numpy() ) / (n + 1)

            ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
            normal_scores[:, j] = norm.ppf(ranks)

        self.corr_ = np.corrcoef(normal_scores, rowvar=False)
        # asegurar SPD numérica (ridge mínimo)
        self.corr_ += np.eye(len(self.columns_)) * 1e-8
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self.corr_ is None:
            raise RuntimeError("fit() debe llamarse antes de sample()")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        d = len(self.columns_)
        z = rng.multivariate_normal(mean=np.zeros(d), cov=self.corr_, size=n)
        u = norm.cdf(z)

        out = {}
        for j, col in enumerate(self.columns_):
            if col in self._category_probs:
                categories, cum_probs = self._category_probs[col]
                idx = np.searchsorted(cum_probs, u[:, j], side="right")
                idx = np.clip(idx, 0, len(categories) - 1)
                out[col] = categories[idx]
            else:
                values = self._empirical_values[col]
                out[col] = np.quantile(values, u[:, j])
        return pd.DataFrame(out, columns=self.columns_)
