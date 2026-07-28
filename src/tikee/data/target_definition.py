"""Modelo estructural del default (ground truth, D3) y calibración de beta0 y sigma.

ARCHITECTURE.md §4.4. `z` se separa en tres piezas para poder calibrar por
bisección sin recomputar todo:
  z = beta0 + z_partial(df) + eps,   eps ~ N(0, sigma)
`z_partial` no depende de beta0 ni de sigma, así que se calcula una sola vez por
tabla. `eps` se fija con una semilla explícita para que la bisección de beta0 sea
determinista (si se resampleara eps en cada iteración, el objetivo de la bisección
tendría ruido y no convergería limpiamente).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def compute_structural_z_partial(df: pd.DataFrame) -> pd.Series:
    z = pd.Series(0.0, index=df.index)
    z += 2.20 * _z(df["ratio_cuota_ingreso"])
    z += 0.55 * _z(df["ratio_deuda_ingreso"])
    z += -1.80 * _z(df["score_buro"])
    z += 1.40 * _z(df["dias_mora_max_12m"])
    z += 0.30 * _z(df["peor_calificacion_12m"])
    z += 0.45 * _z(df["num_operaciones_vigentes"])
    z += -0.60 * _z(df["antiguedad_socio_meses"])
    z += -0.35 * _z(df["antiguedad_laboral_meses"])
    z += 0.25 * _z(df["carga_familiar"])
    z += 0.40 * (df["tipo_empleo"] == "independiente_informal").astype(float)
    z += 0.20 * (df["tipo_empleo"] == "agricultor").astype(float)
    z += -0.15 * _z((df["edad"] - 40) ** 2)
    # coeficiente cero explícito (D3, §4.4): nivel_educacion, zona_residencia, sexo
    return z


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_beta0(
    z_partial: pd.Series,
    sigma: float,
    seed: int,
    target_rate: float = 0.08,
    tol: float = 0.002,
    lo: float = -12.0,
    hi: float = 12.0,
    max_iter: int = 60,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=len(z_partial))
    z_arr = z_partial.to_numpy()

    beta0 = 0.0
    for _ in range(max_iter):
        beta0 = (lo + hi) / 2
        p = _sigmoid(z_arr + beta0 + eps)
        rate = p.mean()
        if abs(rate - target_rate) < tol:
            break
        if rate < target_rate:
            lo = beta0
        else:
            hi = beta0
    p_final = _sigmoid(z_arr + beta0 + eps)
    return beta0, p_final


def add_target(
    df: pd.DataFrame,
    sigma: float,
    seed: int,
    target_rate: float = 0.08,
    tol: float = 0.002,
) -> pd.DataFrame:
    df = df.copy()
    z_partial = compute_structural_z_partial(df)
    beta0, p_default = calibrate_beta0(z_partial, sigma, seed, target_rate, tol)

    rng = np.random.default_rng(seed + 1)
    default = rng.binomial(1, p_default)

    df["p_default_true"] = p_default
    df["default"] = default
    df.attrs["beta0"] = beta0
    df.attrs["sigma"] = sigma
    df.attrs["default_rate"] = float(default.mean())
    return df
