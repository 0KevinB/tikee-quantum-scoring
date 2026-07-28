"""Esquema, distribuciones, mapa de calor de correlación. ARCHITECTURE.md §11."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from tikee.viz.plots import correlation_heatmap  # noqa: E402
from glossary import TERMS, VARIABLE_DESCRIPTIONS  # noqa: E402

st.title("1 · Datos")
st.caption(
    "Todo lo que hay debajo es sintético: generado por software, calibrado con rangos públicos "
    "(SEPS, INEC) para que sea creíble, sin describir a ninguna persona ni cooperativa real."
)

dataset_path = ROOT / "data" / "processed" / "dataset_seed42.parquet"
if not dataset_path.exists():
    st.error(f"Falta {dataset_path}. Corre `python -m tikee.cli generate --seed 42` primero.")
    st.stop()

df = pd.read_parquet(dataset_path)

st.download_button(
    "📥 Descargar dataset sintético completo (CSV, 8.000 filas)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="tikee_dataset_sintetico.csv",
    mime="text/csv",
    help="Datos 100% sintéticos. Libre de descargar y explorar — no contiene información real de nadie.",
)

st.header("📖 Diccionario de datos")
st.caption("Qué significa cada columna, en una frase, sin jerga.")
protected = {"sexo", "provincia", "zona_residencia"}
dict_rows = []
for col in df.columns:
    dict_rows.append({
        "Variable": col,
        "Descripción": VARIABLE_DESCRIPTIONS.get(col, "—"),
        "Tipo": str(df[col].dtype),
        "Atributo protegido": "Sí" if col in protected else "",
    })
st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True)

st.header("Resumen")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", len(df))
col2.metric("Tasa de default", f"{df['default'].mean():.2%}")
col3.metric("Atributos protegidos", "sexo, provincia, zona_residencia")
with st.expander("¿Qué es un \"atributo protegido\"?"):
    st.write(TERMS["Proxy / sesgo indirecto"])

st.header("Distribuciones")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
selected_col = st.selectbox("Variable", numeric_cols, index=numeric_cols.index("score_buro") if "score_buro" in numeric_cols else 0)
st.caption(VARIABLE_DESCRIPTIONS.get(selected_col, ""))
st.bar_chart(df[selected_col].value_counts(bins=20).sort_index())

st.header("Tasa de default por decil de score_buro")
st.caption(
    "Se ordenan los 8.000 solicitantes por score_buro y se parten en 10 grupos iguales (deciles). "
    "Si el score sirve para algo, el decil más bajo debería tener mucha más mora que el más alto."
)
df_decile = df.copy()
df_decile["decil"] = pd.qcut(df_decile["score_buro"], 10, labels=False, duplicates="drop")
st.line_chart(df_decile.groupby("decil")["default"].mean())

st.header("Mapa de calor de correlación (bloque continuo)")
st.caption(
    "Cada celda compara dos variables: rojo intenso = suben y bajan juntas, azul intenso = se mueven "
    "en direcciones opuestas, blanco = casi no hay relación. Estas relaciones fuertes (por ejemplo, "
    "historial de pagos con puntaje de buró) son justo el tipo de redundancia que el proyecto pone "
    "a prueba: ¿el método sabe elegir solo una de varias variables que dicen casi lo mismo?"
)
corr_cols = [
    "edad", "carga_familiar", "antiguedad_laboral_meses", "antiguedad_socio_meses",
    "ingreso_mensual", "gastos_mensuales", "score_buro", "num_operaciones_vigentes",
    "deuda_total_sistema", "peor_calificacion_12m", "dias_mora_max_12m", "monto_solicitado",
]
st.plotly_chart(correlation_heatmap(df, corr_cols), use_container_width=True)

st.header("🔬 Cómo se generaron los datos")
st.markdown(
    "Los datos no vienen de ninguna cooperativa: se generan con una **cópula gaussiana** (un "
    "método estadístico que primero fija cómo se relacionan las variables entre sí — la matriz de "
    "correlación de arriba — y después genera filas nuevas que respetan esas relaciones). Cada "
    "variable y cada relación se ancla a un rango público, nunca a un dato individual."
)
refs_path = ROOT / "data" / "external" / "referencias_publicas.md"
if refs_path.exists():
    lines = refs_path.read_text().splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) > 2:
        header = [c.strip() for c in table_lines[0].strip("|").split("|")]
        rows = []
        for ln in table_lines[2:]:
            cells = [re.sub(r"`", "", c.strip()) for c in ln.strip("|").split("|")]
            if len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
        refs_df = pd.DataFrame(rows)
        st.dataframe(refs_df, use_container_width=True, hide_index=True)
        st.caption(
            "**\\[SUPUESTO]** = valor plausible elegido por el equipo para dar forma realista a los "
            "datos, no una cifra citada literal de una fuente. Ninguna fila describe una cooperativa "
            "específica — es la regla de honestidad metodológica del proyecto (ver pestaña "
            "**6 · Documentación** para el detalle completo)."
        )
else:
    st.info("Falta data/external/referencias_publicas.md.")

st.header("Fidelidad sintética y comparación de sintetizadores (E11, E12)")
st.caption(
    "¿Qué tan parecidos son los datos sintéticos a datos reales creíbles, sin ser una copia? Estas "
    "tres métricas lo miden: **QualityReport** (0-1, qué tan bien se preservan las distribuciones y "
    "correlaciones originales), **DiagnosticReport** (0-1, que no haya errores estructurales obvios) "
    "y **NewRowSynthesis** (0-1, qué fracción de filas son genuinamente nuevas y no copias exactas "
    "de ningún dato de referencia)."
)
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
