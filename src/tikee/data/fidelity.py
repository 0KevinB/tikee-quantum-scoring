"""SDMetrics: QualityReport, DiagnosticReport, privacidad; comparación de 3
sintetizadores (E11, E12). ARCHITECTURE.md §5."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tikee.data.sdv_synthesizer import DERIVED_COLUMNS, _build_metadata, synthesize
from tikee.data.validate import TARGET_CORRELATIONS


def run_sdmetrics_reports(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict[str, Any]:
    from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

    fit_real = real_df.drop(columns=[c for c in DERIVED_COLUMNS + ["id_solicitud"] if c in real_df.columns])
    fit_synth = synth_df[fit_real.columns]
    metadata = _build_metadata(fit_real)
    metadata_dict = metadata.to_dict()

    quality = QualityReport()
    quality.generate(fit_real, fit_synth, metadata_dict, verbose=False)
    quality_scores = quality.get_properties()

    diagnostic = DiagnosticReport()
    diagnostic.generate(fit_real, fit_synth, metadata_dict, verbose=False)
    diagnostic_scores = diagnostic.get_properties()

    return {
        "quality_overall": float(quality.get_score()),
        "quality_column_shapes": float(quality_scores.loc[quality_scores["Property"] == "Column Shapes", "Score"].iloc[0]),
        "quality_column_pair_trends": float(quality_scores.loc[quality_scores["Property"] == "Column Pair Trends", "Score"].iloc[0]),
        "diagnostic_overall": float(diagnostic.get_score()),
        "diagnostic_properties": diagnostic_scores.to_dict(orient="records"),
    }


def new_row_synthesis(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    from sdmetrics.single_table import NewRowSynthesis

    fit_real = real_df.drop(columns=[c for c in DERIVED_COLUMNS + ["id_solicitud"] if c in real_df.columns])
    fit_synth = synth_df[fit_real.columns]
    metadata = _build_metadata(fit_real)
    score = NewRowSynthesis.compute(
        real_data=fit_real, synthetic_data=fit_synth, metadata=metadata.to_dict(), numerical_match_tolerance=0.01
    )
    return float(score)


def nearest_record_distance(real_df: pd.DataFrame, synth_df: pd.DataFrame, n_sample: int = 500) -> dict[str, float]:
    """Distancia normalizada (L2 sobre columnas numéricas estandarizadas por la escala
    de la real) de cada fila sintética a su vecino real más cercano. No hay umbral fijo
    (ARCHITECTURE.md §5): se reporta la distribución como evidencia de riesgo de
    reidentificación, no un umbral pasa/no-pasa."""
    numeric_cols = [c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c])]
    real = real_df[numeric_cols].to_numpy(dtype=float)
    synth = synth_df[numeric_cols].sample(min(n_sample, len(synth_df)), random_state=0).to_numpy(dtype=float)

    scale = real.std(axis=0)
    scale[scale == 0] = 1.0
    real_n = real / scale
    synth_n = synth / scale

    dists = np.empty(len(synth_n))
    for i, row in enumerate(synth_n):
        dists[i] = np.sqrt(((real_n - row) ** 2).sum(axis=1)).min()

    return {
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "p05": float(np.percentile(dists, 5)),
        "min": float(dists.min()),
    }


def target_correlation_mae(df: pd.DataFrame, exclude_known_exceptions: bool = True) -> float:
    from tikee.data.validate import KNOWN_CORRELATION_EXCEPTIONS

    errors = []
    for a, b, target in TARGET_CORRELATIONS:
        if exclude_known_exceptions and ((a, b) in KNOWN_CORRELATION_EXCEPTIONS or (b, a) in KNOWN_CORRELATION_EXCEPTIONS):
            continue
        rho, _ = spearmanr(df[a], df[b])
        errors.append(abs(rho - target))
    return float(np.mean(errors))


def compare_synthesizers(
    seed_df: pd.DataFrame,
    full_reference_df: pd.DataFrame,
    methods: tuple[str, ...] = ("gaussian_copula", "ctgan", "tvae"),
    n_sample: int = 8000,
    seed: int = 42,
    ctgan_epochs: int = 300,
    time_limit_s: float = 40 * 60,
) -> dict[str, Any]:
    """E12. `full_reference_df` es el dataset ya con target (para calcular MAE de
    correlaciones objetivo con las mismas 13 parejas de validate.py)."""
    rows = []
    for method in methods:
        t0 = time.perf_counter()
        try:
            sample, info = synthesize(seed_df, method=method, n_sample=n_sample, seed=seed, ctgan_epochs=ctgan_epochs)
            wall = time.perf_counter() - t0
            if wall > time_limit_s:
                rows.append({
                    "method": method, "converged": False, "wall_time_s": wall,
                    "note": f"excedió el límite de {time_limit_s/60:.0f} min; resultado negativo de viabilidad, no se depura",
                })
                continue
            corr_mae = target_correlation_mae(sample)
            sdm = run_sdmetrics_reports(seed_df, sample)
            rows.append({
                "method": method,
                "converged": True,
                "wall_time_s": wall,
                "engine": info["engine"],
                "target_correlation_mae": corr_mae,
                **sdm,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": method, "converged": False, "error": str(exc), "wall_time_s": time.perf_counter() - t0})
    return {"rows": rows}
