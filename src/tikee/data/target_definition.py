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
    """Logit estructural del ground truth (D3, ARCHITECTURE.md §4.4) sin beta0 ni
    ruido: la combinación lineal fija de coeficientes de negocio sobre las
    variables relevantes. `nivel_educacion`, `zona_residencia` y `sexo` tienen
    coeficiente cero explícito — no entran a esta suma."""
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
    """Bisección sobre beta0 hasta que `mean(sigmoid(z_partial + beta0 + eps))`
    caiga dentro de `tol` de `target_rate` (D4: tasa base 8%). `eps` se fija con
    `seed` una sola vez fuera del bucle para que la bisección sea determinista.

    Args:
        z_partial: salida de `compute_structural_z_partial`.
        sigma: desvío del ruido estructural (controla la separabilidad del
            target; se calibra por fuera, ver `scripts/f1_calibrate_sigma.py`).
        seed: semilla del ruido `eps`.
        target_rate: tasa de default objetivo.
        tol: tolerancia de la bisección sobre la tasa esperada (no confundir con
            la tolerancia más amplia de la tasa REALIZADA que exige
            `validate.check_default_rate`).
        lo, hi: cota inicial de búsqueda de beta0.
        max_iter: iteraciones máximas de bisección.

    Returns:
        Tupla `(beta0, p_default)`: el beta0 encontrado y el vector de
        probabilidades de default resultante (antes de muestrear Bernoulli).
    """
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
    """Agrega el target sintético a `df`: calibra beta0 y muestrea `default` ~
    Bernoulli(p_default). Punto de entrada principal de este módulo.

    Args:
        df: DataFrame con las 18 variables predictoras (de `seed_generator` o
            `sdv_synthesizer.synthesize`).
        sigma: desvío del ruido estructural (ya calibrado, ver
            `config/config.yaml: data.sigma_ruido`).
        seed: semilla para la calibración de beta0 y el muestreo de `default`.
        target_rate: tasa de default objetivo (D4).
        tol: tolerancia de la bisección de beta0.

    Returns:
        `df` con dos columnas nuevas, `p_default_true` y `default`, y
        `df.attrs` con `beta0`, `sigma` y `default_rate` (tasa realizada).
    """
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
