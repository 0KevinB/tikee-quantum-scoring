"""Esquema, distribuciones, mapa de calor de correlación. ARCHITECTURE.md §11."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tikee.viz.plots import correlation_heatmap  # noqa: E402

st.title("1 · Datos")

dataset_path = ROOT / "data" / "processed" / "dataset_seed42.parquet"
if not dataset_path.exists():
    st.error(f"Falta {dataset_path}. Corre `python -m tikee.cli generate --seed 42` primero.")
    st.stop()

df = pd.read_parquet(dataset_path)

st.header("Esquema")
st.dataframe(df.dtypes.rename("tipo").to_frame())

st.header("Resumen")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", len(df))
col2.metric("Tasa de default", f"{df['default'].mean():.2%}")
col3.metric("Atributos protegidos", "sexo, provincia, zona_residencia")

st.header("Distribuciones")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
selected_col = st.selectbox("Variable", numeric_cols, index=numeric_cols.index("score_buro") if "score_buro" in numeric_cols else 0)
st.bar_chart(df[selected_col].value_counts(bins=20).sort_index())

st.header("Tasa de default por decil de score_buro")
df_decile = df.copy()
df_decile["decil"] = pd.qcut(df_decile["score_buro"], 10, labels=False, duplicates="drop")
st.line_chart(df_decile.groupby("decil")["default"].mean())

st.header("Mapa de calor de correlación (bloque continuo)")
corr_cols = [
    "edad", "carga_familiar", "antiguedad_laboral_meses", "antiguedad_socio_meses",
    "ingreso_mensual", "gastos_mensuales", "score_buro", "num_operaciones_vigentes",
    "deuda_total_sistema", "peor_calificacion_12m", "dias_mora_max_12m", "monto_solicitado",
]
st.plotly_chart(correlation_heatmap(df, corr_cols), use_container_width=True)

st.header("Fidelidad sintética y comparación de sintetizadores (E11, E12)")
fidelity_path = ROOT / "reports" / "fidelity" / "f2_comparison.json"
if fidelity_path.exists():
    fidelity = json.loads(fidelity_path.read_text())
    prod = fidelity["production_gaussian_copula"]
    c1, c2, c3 = st.columns(3)
    c1.metric("QualityReport", f"{prod['sdmetrics']['quality_overall']:.3f}")
    c2.metric("DiagnosticReport", f"{prod['sdmetrics']['diagnostic_overall']:.3f}")
    c3.metric("NewRowSynthesis", f"{prod['new_row_synthesis']:.3f}")
    st.subheader("Comparación de sintetizadores")
    comp_df = pd.DataFrame(fidelity["synthesizer_comparison"])
    st.dataframe(comp_df[["method", "converged", "wall_time_s", "target_correlation_mae", "quality_overall"]])
    st.caption(
        "GaussianCopula se elige como sintetizador de producción (D14): preserva la estructura "
        "de correlación inyectada por construcción, que es la premisa del proyecto."
    )
else:
    st.info("Corre `scripts/f2_fidelity_comparison.py` para ver el detalle de fidelidad.")
