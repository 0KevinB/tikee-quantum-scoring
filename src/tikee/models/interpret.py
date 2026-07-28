"""Coeficientes, signos esperados, VIF, SHAP. Argumento regulatorio SEPS.
PLAN.md §5."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Signo esperado por el negocio (PLAN.md §5, punto 2). None = sin expectativa clara
# (p.ej. edad, cuyo riesgo es en U).
EXPECTED_SIGNS: dict[str, int] = {
    "ratio_cuota_ingreso": 1,
    "ratio_deuda_ingreso": 1,
    "score_buro": -1,
    "dias_mora_max_12m": 1,
    "peor_calificacion_12m": 1,
    "num_operaciones_vigentes": 1,
    "antiguedad_socio_meses": -1,
    "antiguedad_laboral_meses": -1,
    "carga_familiar": 1,
    "deuda_total_sistema": 1,
    "ingreso_mensual": -1,
    "gastos_mensuales": 1,
}


def get_coefficients(model, feature_names: list[str]) -> dict[str, float]:
    """Empareja los coeficientes de una regresión logística binaria (`model.coef_[0]`)
    con sus nombres de columna transformada, en el mismo orden."""
    coefs = model.coef_[0]
    return dict(zip(feature_names, coefs.tolist()))


def check_sign_coherence(coef_by_variable: dict[str, float]) -> dict[str, Any]:
    """Un signo invertido por colinealidad es un hallazgo regulatorio grave (PLAN.md
    §5, punto 2): el modelo diría al socio algo falso sobre por qué se le negó."""
    rows = []
    for var, expected in EXPECTED_SIGNS.items():
        if var not in coef_by_variable:
            continue
        observed = coef_by_variable[var]
        observed_sign = 1 if observed > 0 else (-1 if observed < 0 else 0)
        rows.append({
            "variable": var, "expected_sign": expected, "observed_coef": observed,
            "coherent": observed_sign == expected,
        })
    n_incoherent = sum(1 for r in rows if not r["coherent"])
    return {"rows": rows, "n_incoherent": n_incoherent, "all_coherent": n_incoherent == 0}


def compute_vif_final(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """VIF de las columnas del modelo FINAL (ya seleccionadas), a diferencia de
    `data.validate.compute_vif` que mide sobre las 18 variables completas. Un
    VIF más bajo aquí es un argumento de interpretabilidad medible (PLAN.md §5)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_with_const = np.column_stack([X, np.ones(len(X))])
    vifs = {}
    for i, name in enumerate(feature_names):
        try:
            vifs[name] = float(variance_inflation_factor(X_with_const, i))
        except Exception:  # noqa: BLE001
            vifs[name] = float("nan")
    return vifs


def shap_summary(model, X_sample: np.ndarray, feature_names: list[str], model_type: str = "logreg") -> dict[str, float]:
    """Importancia media |SHAP| por variable. Usa `LinearExplainer` para
    logística (exacto, rápido) o `TreeExplainer` para XGBoost.

    Args:
        model: modelo ya entrenado (LogisticRegression o XGBClassifier).
        X_sample: matriz de features sobre la que calcular SHAP.
        feature_names: nombres de columna, mismo orden que `X_sample`.
        model_type: `"logreg"` o cualquier otro valor para el explicador de árboles.

    Returns:
        dict `{nombre_variable: importancia_media_absoluta}`.
    """
    import shap

    if model_type == "logreg":
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(zip(feature_names, mean_abs.tolist()))


def top_reasons(model, x_row: np.ndarray, feature_names: list[str], n: int = 3) -> list[dict[str, Any]]:
    """Para el simulador de la app (§11, pestaña 4): las N contribuciones (coef*valor)
    de mayor magnitud, en orden. Es el argumento de interpretabilidad SEPS hecho visible."""
    coefs = model.coef_[0]
    contributions = coefs * x_row
    order = np.argsort(-np.abs(contributions))[:n]
    return [
        {"variable": feature_names[i], "contribution": float(contributions[i]), "value": float(x_row[i])}
        for i in order
    ]
