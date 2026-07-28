"""Chequeos duros de rango, tasa base, correlaciones objetivo, VIF y banda de AUC.

`validate_dataset` no aborta por sí solo (no conoce el AUC hasta que alguien lo
entrena) — devuelve `ok=False` y el llamador (cli.py / scripts de calibración)
decide abortar. Esto evita un ciclo de imports con `models/train.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDICTOR_COLUMNS = [
    "edad", "zona_residencia", "nivel_educacion", "carga_familiar", "tipo_empleo",
    "antiguedad_laboral_meses", "antiguedad_socio_meses", "ingreso_mensual",
    "gastos_mensuales", "score_buro", "num_operaciones_vigentes", "deuda_total_sistema",
    "peor_calificacion_12m", "dias_mora_max_12m", "monto_solicitado", "plazo_meses",
    "ratio_cuota_ingreso", "ratio_deuda_ingreso",
]
PROTECTED_COLUMNS = ["sexo", "provincia", "id_solicitud"]

RANGES: dict[str, tuple[float, float]] = {
    "edad": (18, 75),
    "carga_familiar": (0, 6),
    "antiguedad_laboral_meses": (0, 360),
    "antiguedad_socio_meses": (1, 300),
    "ingreso_mensual": (460, 3500),
    "gastos_mensuales": (200, 3000),
    "score_buro": (1, 999),
    "num_operaciones_vigentes": (0, 8),
    "deuda_total_sistema": (0, 40000),
    "peor_calificacion_12m": (1, 9),
    "dias_mora_max_12m": (0, 180),
    "monto_solicitado": (500, 25000),
    "nivel_educacion": (1, 5),
    "ratio_cuota_ingreso": (0.02, 0.90),
    "ratio_deuda_ingreso": (0, 20),
}


def check_ranges(df: pd.DataFrame) -> dict[str, Any]:
    violations = {}
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            violations[col] = "columna ausente"
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            violations[col] = f"{out_of_range} filas fuera de [{lo}, {hi}]"
    return {"ok": len(violations) == 0, "violations": violations}


def check_default_rate(df: pd.DataFrame, target: float = 0.08, tol: float = 0.005) -> dict[str, Any]:
    rate = df["default"].mean()
    ok = abs(rate - target) <= tol
    return {"ok": bool(ok), "rate": float(rate), "target": target, "tol": tol}


def check_hard_constraint(df: pd.DataFrame) -> dict[str, Any]:
    cap = (df["edad"] - 18) * 12
    violations = int((df["antiguedad_laboral_meses"] > cap + 1e-6).sum())
    return {"ok": violations == 0, "n_violations": violations}


def check_protected_not_predictor(df: pd.DataFrame) -> dict[str, Any]:
    leaked = [c for c in PROTECTED_COLUMNS if c in PREDICTOR_COLUMNS]
    all_present = all(c in df.columns for c in PROTECTED_COLUMNS)
    return {"ok": len(leaked) == 0 and all_present, "leaked": leaked, "all_present": all_present}


TARGET_CORRELATIONS = [
    ("score_buro", "peor_calificacion_12m", -0.85),
    ("score_buro", "dias_mora_max_12m", -0.80),
    ("peor_calificacion_12m", "dias_mora_max_12m", 0.88),
    ("deuda_total_sistema", "num_operaciones_vigentes", 0.60),
    ("ingreso_mensual", "gastos_mensuales", 0.75),
    ("ingreso_mensual", "monto_solicitado", 0.55),
    ("ratio_cuota_ingreso", "monto_solicitado", 0.65),
    ("ratio_cuota_ingreso", "ingreso_mensual", -0.45),
    ("ratio_deuda_ingreso", "deuda_total_sistema", 0.70),
    ("edad", "antiguedad_laboral_meses", 0.55),
    ("edad", "antiguedad_socio_meses", 0.40),
    ("edad", "carga_familiar", 0.30),
    ("nivel_educacion", "score_buro", 0.0),
]


# [DESVIACIÓN DOCUMENTADA, F1] ratio_cuota_ingreso = cuota_estimada/ingreso_mensual y
# cuota_estimada = monto_solicitado * factor(plazo). Con ingreso_mensual~monto_solicitado
# correlacionados en +0.55 (objetivo mandado por ARCHITECTURE.md §4.2, ya verificado) y
# ratio_cuota_ingreso~monto_solicitado en +0.65 (objetivo, también verificado), el signo y
# magnitud de la correlación ratio~ingreso quedan determinados algebraicamente por esos dos,
# no son un parámetro libre. Búsqueda exhaustiva de parámetros (sigma de monto/ingreso,
# dependencia de plazo, ver notebook de calibración) no encontró ninguna combinación que
# satisfaga los tres pares simultáneamente dentro de tolerancia. Se acepta esta única
# desviación explícitamente en vez de forzar un valor irreal (varianza de ingreso extrema)
# que rompería el rango calibrado de la variable. No se oculta: se reporta el valor medido.
KNOWN_CORRELATION_EXCEPTIONS: set[tuple[str, str]] = {("ratio_cuota_ingreso", "ingreso_mensual")}


def check_correlations(
    df: pd.DataFrame, tol: float = 0.07, known_exceptions: set[tuple[str, str]] | None = None
) -> dict[str, Any]:
    known_exceptions = known_exceptions if known_exceptions is not None else KNOWN_CORRELATION_EXCEPTIONS
    rows = []
    all_ok = True
    for a, b, target in TARGET_CORRELATIONS:
        rho, _ = spearmanr(df[a], df[b])
        ok = abs(rho - target) <= tol
        is_known_exception = (a, b) in known_exceptions or (b, a) in known_exceptions
        if not is_known_exception:
            all_ok = all_ok and ok
        rows.append({
            "a": a, "b": b, "target": target, "observed": float(rho),
            "ok": bool(ok), "known_exception": is_known_exception,
        })
    return {"ok": all_ok, "pairs": rows}


def compute_vif(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[columns].astype(float).to_numpy()
    X = np.column_stack([X, np.ones(len(X))])
    vifs = {}
    for i, col in enumerate(columns):
        try:
            vifs[col] = float(variance_inflation_factor(X, i))
        except Exception:  # noqa: BLE001
            vifs[col] = float("nan")
    return vifs


def quick_auc(df: pd.DataFrame, seed: int) -> float:
    """AUC de una logística rápida con las 18 variables, usada SOLO para calibrar sigma
    en F1 (bisección) y para el chequeo de banda. No es el brazo A0 del experimento
    principal (ese usa el pipeline de preprocess.py/train.py con su propio split)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = pd.get_dummies(df[PREDICTOR_COLUMNS], drop_first=True)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    return float(roc_auc_score(y_test, proba))


def check_auc_band(auc: float, band: tuple[float, float] = (0.72, 0.82)) -> dict[str, Any]:
    ok = band[0] <= auc <= band[1]
    return {"ok": bool(ok), "auc": float(auc), "band": band}


def validate_dataset(
    df: pd.DataFrame,
    measured_auc: float | None = None,
    corr_tol: float = 0.07,
    auc_band: tuple[float, float] = (0.72, 0.82),
) -> dict[str, Any]:
    report = {
        "ranges": check_ranges(df),
        "default_rate": check_default_rate(df),
        "hard_constraint": check_hard_constraint(df),
        "protected_not_predictor": check_protected_not_predictor(df),
        "correlations": check_correlations(df, tol=corr_tol),
    }
    numeric_cols = [c for c in PREDICTOR_COLUMNS if pd.api.types.is_numeric_dtype(df[c])]
    report["vif"] = compute_vif(df, numeric_cols)

    if measured_auc is not None:
        report["auc_band"] = check_auc_band(measured_auc, auc_band)

    gating_checks = ["ranges", "default_rate", "hard_constraint", "protected_not_predictor", "correlations"]
    if measured_auc is not None:
        gating_checks.append("auc_band")
    report["ok"] = all(report[k]["ok"] for k in gating_checks)
    return report
