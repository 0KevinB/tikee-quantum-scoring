"""Formulario de solicitante -> probabilidad + 3 razones principales.
ARCHITECTURE.md §11. Carga un modelo ya serializado (brazo C0, Nivel A) — no entrena
nada en vivo; la reconstrucción del preprocesador es un `fit` determinista de medias/
categorías sobre datos ya en disco, no un entrenamiento de modelo."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tikee.features.preprocess import LEVEL_A_VARS, build_preprocessor, get_variable_groups  # noqa: E402
from tikee.models.interpret import top_reasons  # noqa: E402

st.title("4 · Simulador")
st.caption("El argumento de interpretabilidad SEPS hecho visible: probabilidad + las 3 razones principales.")

artifacts_path = ROOT / "reports" / "cache" / "app_artifacts.joblib"
if not artifacts_path.exists():
    st.error("Falta reports/cache/app_artifacts.joblib. Corre scripts/f7_fairness_audit.py.")
    st.stop()

# joblib.load: seguro, el único archivo leído es el caché que este mismo repo generó.
artifacts = joblib.load(artifacts_path)
bundle = artifacts["level_a"]
train_df = bundle["train_df_raw"] if "train_df_raw" in bundle else artifacts["test_df_raw"]

arm = st.selectbox("Modelo (brazo)", list(bundle["models"].keys()), index=list(bundle["models"].keys()).index("C0") if "C0" in bundle["models"] else 0)
info = bundle["models"][arm]

pre = build_preprocessor(LEVEL_A_VARS)
pre.fit(train_df[LEVEL_A_VARS])
groups = get_variable_groups(pre)
feature_names = [None] * sum(len(idxs) for idxs in groups.values())
for v, idxs in groups.items():
    if len(idxs) == 1:
        feature_names[idxs[0]] = v
    else:
        for i, idx in enumerate(idxs):
            feature_names[idx] = f"{v}_{i}"

st.header("Datos del solicitante")
c1, c2, c3 = st.columns(3)
with c1:
    edad = st.slider("Edad", 18, 75, 35)
    carga_familiar = st.slider("Carga familiar", 0, 6, 1)
    ingreso_mensual = st.number_input("Ingreso mensual (USD)", 460, 3500, 700)
    gastos_mensuales = st.number_input("Gastos mensuales (USD)", 200, 3000, 500)
    antiguedad_laboral_meses = st.slider("Antigüedad laboral (meses)", 0, 360, 24)
with c2:
    score_buro = st.slider("Score de buró", 1, 999, 650)
    dias_mora_max_12m = st.slider("Días de mora máx. (12m)", 0, 180, 0)
    peor_calificacion_12m = st.slider("Peor calificación (1-9)", 1, 9, 1)
    num_operaciones_vigentes = st.slider("Operaciones vigentes", 0, 8, 1)
    antiguedad_socio_meses = st.slider("Antigüedad como socio (meses)", 1, 300, 24)
with c3:
    monto_solicitado = st.number_input("Monto solicitado (USD)", 500, 25000, 3000)
    plazo_meses = st.selectbox("Plazo (meses)", [6, 12, 18, 24, 36, 48, 60], index=2)
    deuda_total_sistema = st.number_input("Deuda total en el sistema (USD)", 0, 40000, 1000)
    tipo_empleo = st.selectbox("Tipo de empleo", ["dependiente_formal", "independiente_informal", "agricultor", "comerciante"])
    zona_residencia = st.selectbox("Zona de residencia", ["urbana", "rural"])
nivel_educacion = st.slider("Nivel educativo (1=primaria, 5=posgrado)", 1, 5, 3)

cuota_i = 0.165 / 12
cuota_estimada = monto_solicitado * cuota_i / (1 - (1 + cuota_i) ** (-plazo_meses))
ratio_cuota_ingreso = min(max(cuota_estimada / ingreso_mensual, 0.02), 0.90)
ratio_deuda_ingreso = min(deuda_total_sistema / ingreso_mensual, 20)

import pandas as pd  # noqa: E402

row = pd.DataFrame([{
    "edad": edad, "carga_familiar": carga_familiar, "antiguedad_laboral_meses": antiguedad_laboral_meses,
    "antiguedad_socio_meses": antiguedad_socio_meses, "ingreso_mensual": ingreso_mensual,
    "gastos_mensuales": gastos_mensuales, "score_buro": score_buro,
    "num_operaciones_vigentes": num_operaciones_vigentes, "deuda_total_sistema": deuda_total_sistema,
    "peor_calificacion_12m": peor_calificacion_12m, "dias_mora_max_12m": dias_mora_max_12m,
    "monto_solicitado": monto_solicitado, "plazo_meses": plazo_meses, "nivel_educacion": nivel_educacion,
    "ratio_cuota_ingreso": ratio_cuota_ingreso, "ratio_deuda_ingreso": ratio_deuda_ingreso,
    "zona_residencia": zona_residencia, "tipo_empleo": tipo_empleo,
}])

x_full = pre.transform(row[LEVEL_A_VARS])[0]
x_selected = x_full[info["cols"]]
selected_feature_names = [feature_names[i] for i in info["cols"]]

if st.button("Calcular probabilidad de default", type="primary"):
    proba = info["model"].predict_proba(x_selected.reshape(1, -1))[0, 1]
    decision = "RECHAZAR" if proba >= info["threshold"] else "APROBAR"
    st.metric("Probabilidad de default", f"{proba:.2%}")
    st.metric("Decisión (umbral congelado)", decision)

    reasons = top_reasons(info["model"], x_selected, selected_feature_names, n=3)
    st.subheader("Las 3 razones principales")
    for r in reasons:
        direccion = "aumenta" if r["contribution"] > 0 else "reduce"
        st.write(f"- **{r['variable']}** {direccion} el riesgo (contribución = {r['contribution']:.3f})")
