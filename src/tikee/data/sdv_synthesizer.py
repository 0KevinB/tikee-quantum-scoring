"""GaussianCopula / CTGAN / TVAE de SDV tras una interfaz común. Respaldo automático
a `fallback_copula.FallbackGaussianCopula` si `sdv` no está disponible (solo para
`method="gaussian_copula"`; CTGAN/TVAE no tienen respaldo porque son el experimento
de comparación de F2, no la fuente del dataset principal, D14)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

DERIVED_COLUMNS = ["cuota_estimada", "ratio_cuota_ingreso", "ratio_deuda_ingreso"]
PLAZO_CATEGORIES = np.array([6, 12, 18, 24, 36, 48, 60])
TASA_CUOTA_ANUAL = 0.165

# Columnas ordinales/discretas de baja cardinalidad que SDV auto-detecta como
# 'categorical' (rompe la correlación de rango con el resto, ver F1). Deben forzarse
# a 'numerical' para que GaussianCopulaSynthesizer las trate dentro de la cópula.
FORCE_NUMERICAL_COLUMNS = [
    "edad", "carga_familiar", "antiguedad_laboral_meses", "antiguedad_socio_meses",
    "ingreso_mensual", "gastos_mensuales", "score_buro", "num_operaciones_vigentes",
    "deuda_total_sistema", "peor_calificacion_12m", "dias_mora_max_12m",
    "monto_solicitado", "nivel_educacion",
]


def _build_metadata(df: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    for col in FORCE_NUMERICAL_COLUMNS:
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype="numerical")
    return metadata


def _recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula cuota_estimada, ratio_cuota_ingreso y ratio_deuda_ingreso a partir de
    las columnas base ya sintetizadas. No se dejan a cargo del sintetizador porque son
    determinísticamente dependientes (ARCHITECTURE.md §4.1, filas 17-18); dejar que un
    modelo las trate como libres rompería esa relación exacta."""
    df = df.copy()
    idx = np.abs(df["plazo_meses"].to_numpy()[:, None] - PLAZO_CATEGORIES[None, :]).argmin(axis=1)
    df["plazo_meses"] = PLAZO_CATEGORIES[idx]

    cuota_i = TASA_CUOTA_ANUAL / 12
    factor = cuota_i / (1 - (1 + cuota_i) ** (-df["plazo_meses"]))
    df["cuota_estimada"] = df["monto_solicitado"] * factor
    df["ratio_cuota_ingreso"] = (df["cuota_estimada"] / df["ingreso_mensual"]).clip(0.02, 0.90)
    df["ratio_deuda_ingreso"] = (df["deuda_total_sistema"] / df["ingreso_mensual"]).clip(0, 20)
    return df


def _reapply_hard_constraints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edad"] = df["edad"].round().clip(18, 75)
    cap = (df["edad"] - 18) * 12
    df["antiguedad_laboral_meses"] = np.minimum(df["antiguedad_laboral_meses"], cap).clip(lower=0)
    for col in ["carga_familiar", "num_operaciones_vigentes", "peor_calificacion_12m",
                "dias_mora_max_12m", "antiguedad_socio_meses", "score_buro", "nivel_educacion"]:
        df[col] = df[col].round()
    df["score_buro"] = df["score_buro"].clip(1, 999)
    df["carga_familiar"] = df["carga_familiar"].clip(0, 6)
    df["num_operaciones_vigentes"] = df["num_operaciones_vigentes"].clip(0, 8)
    df["peor_calificacion_12m"] = df["peor_calificacion_12m"].clip(1, 9)
    df["dias_mora_max_12m"] = df["dias_mora_max_12m"].clip(0, 180)
    df["antiguedad_socio_meses"] = df["antiguedad_socio_meses"].clip(1, 300)
    df["nivel_educacion"] = df["nivel_educacion"].clip(1, 5)
    df["ingreso_mensual"] = df["ingreso_mensual"].clip(460, 3500)
    df["gastos_mensuales"] = df["gastos_mensuales"].clip(200, 3000)
    df["deuda_total_sistema"] = df["deuda_total_sistema"].clip(0, 40000)
    df["monto_solicitado"] = df["monto_solicitado"].clip(500, 25000)
    return df


def synthesize(
    df: pd.DataFrame,
    method: str = "gaussian_copula",
    n_sample: int = 8000,
    seed: int | None = None,
    ctgan_epochs: int = 300,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajusta un sintetizador sobre `df` (Etapa 2, ARCHITECTURE.md §4.4-4.5) y
    muestrea `n_sample` filas. Excluye columnas derivadas y el id antes de
    ajustar, y las recalcula/regenera después de muestrear para no romper su
    relación exacta con las columnas base.

    Args:
        df: tabla semilla (de `seed_generator.generate_seed_table`).
        method: `"gaussian_copula"` (producción, D14), `"ctgan"` o `"tvae"`
            (solo para la comparación de F2, sin respaldo si SDV falta).
        n_sample: número de filas a muestrear.
        seed: semilla del sintetizador y del respaldo sin SDV.
        ctgan_epochs: épocas de entrenamiento para CTGAN/TVAE.

    Returns:
        Tupla `(sample, info)`: DataFrame sintético ya con derivadas
        recalculadas, e `info` con `engine` (motor real usado, incluye
        `"fallback_copula"` si SDV no estaba disponible), `method`,
        `wall_time_s` y `n_sample`.
    """
    t0 = time.perf_counter()

    fit_df = df.drop(columns=[c for c in DERIVED_COLUMNS + ["id_solicitud"] if c in df.columns])
    categorical_cols = [c for c in fit_df.columns if fit_df[c].dtype == object]

    try:
        if method == "gaussian_copula":
            from sdv.single_table import GaussianCopulaSynthesizer

            metadata = _build_metadata(fit_df)
            # 'gaussian_kde' en vez del ajuste paramétrico automático (beta/gamma/...):
            # con marginales discretas/zero-infladas (score_buro, dias_mora_max_12m,
            # peor_calificacion_12m) el ajuste paramétrico por defecto atenuaba la
            # correlación de rango objetivo en 15-40 puntos; KDE la preserva casi intacta
            # porque no fuerza una familia parametrica que no encaja.
            numerical_distributions = {
                c: "gaussian_kde" for c in FORCE_NUMERICAL_COLUMNS if c in fit_df.columns
            }
            synth = GaussianCopulaSynthesizer(metadata, numerical_distributions=numerical_distributions)
        elif method == "ctgan":
            from sdv.single_table import CTGANSynthesizer

            metadata = _build_metadata(fit_df)
            synth = CTGANSynthesizer(metadata, epochs=ctgan_epochs, verbose=False, cuda=False)
        elif method == "tvae":
            from sdv.single_table import TVAESynthesizer

            metadata = _build_metadata(fit_df)
            synth = TVAESynthesizer(metadata, epochs=ctgan_epochs)
        else:
            raise ValueError(f"método de síntesis desconocido: {method}")

        synth.fit(fit_df)
        sample = synth.sample(n_sample)
        engine = f"sdv:{method}"
    except ImportError:
        if method != "gaussian_copula":
            raise
        from tikee.data.fallback_copula import FallbackGaussianCopula

        fc = FallbackGaussianCopula(categorical_columns=categorical_cols)
        fc.fit(fit_df, seed=seed)
        sample = fc.sample(n_sample, seed=seed)
        engine = "fallback_copula"

    sample = _reapply_hard_constraints(sample)
    sample = _recompute_derived(sample)
    sample["id_solicitud"] = [f"SYN-{seed}-{i:06d}" for i in range(len(sample))]

    wall = time.perf_counter() - t0
    info = {"engine": engine, "method": method, "wall_time_s": wall, "n_sample": n_sample}
    return sample, info
