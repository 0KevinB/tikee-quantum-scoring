"""Portada y navegación de la app de demostración."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from glossary import TERMS, apply_base_style  # noqa: E402

st.set_page_config(page_title="Tikee — Scoring cuántico-inspirado", page_icon="🏦", layout="wide")
apply_base_style()

st.title("Tikee — Selección de variables cuántico-inspirada para scoring crediticio")
st.caption("Proyecto de innovación universitaria · UTPL Ecuador. Inspirado en el pitch de la fintech Tikee.")

st.warning(
    "**Datos 100% sintéticos.** Ninguna cifra de esta app describe a ninguna cooperativa real. "
    "Los datos se generan con una cópula gaussiana propia + SDV, calibrados con rangos públicos "
    "(SEPS, burós de crédito, INEC) como referencia orientativa, no como fuente de datos individuales."
)

st.info(
    "**\"Cuántico-inspirado\" = recocido simulado clásico.** El brazo QAOA es un circuito cuántico "
    "**simulado clásicamente** (Qiskit Aer / primitivas de referencia). **No se usó ningún procesador "
    "cuántico.** En N=18 el óptimo global ya se conoce por enumeración exhaustiva, y QAOA es varios "
    "órdenes de magnitud más caro que enumerar — se incluye para medir esa brecha, no para proponerlo "
    "como método práctico. En N=45 QAOA es físicamente imposible de simular (563 TB de RAM)."
)

st.header("¿Qué problema resuelve este proyecto? (en una frase)")
st.markdown(
    "Un modelo de crédito necesita decidir **qué variables del solicitante usar** (edad, ingresos, "
    "historial de pagos...). Este proyecto compara un método \"inteligente\" de optimización (QUBO) "
    "contra los métodos clásicos que ya usa la industria (LASSO, selección paso a paso), para "
    "responder con honestidad: ¿el método nuevo realmente ayuda, o solo suena más sofisticado?"
)
st.success(
    "**Resultado corto:** el método nuevo no gana en precisión. Donde sí gana es en algo más "
    "importante para un regulador financiero: **usa menos variables y esconde menos información "
    "sensible** (como la zona donde vive alguien) que sobrevive de forma indirecta en el modelo. "
    "Tabla completa de resultados en la pestaña **6 · Documentación**."
)

st.header("Pregunta de investigación (versión técnica)")
st.markdown(
    "¿Formular la selección de variables como un problema **QUBO** mejora un modelo de scoring "
    "crediticio frente a LASSO o stepwise, cuando las variables candidatas están fuertemente "
    "correlacionadas entre sí?\n\n"
    "El mismo QUBO se resuelve con **cinco solucionadores** para separar tres preguntas:\n\n"
    "1. ¿Es buena **la formulación** QUBO relevancia-redundancia? → recocido vs. LASSO\n"
    "2. ¿Aporta algo **el heurístico**? → recocido vs. óptimo certificado (enumeración / MILP)\n"
    "3. ¿Aporta algo **el paradigma de circuitos**? → QAOA simulado vs. recocido\n\n"
    "El paso a paso completo del algoritmo (cómo se mide relevancia, redundancia y cómo se "
    "resuelve) está explicado en la pestaña **2 · Selección**."
)

st.header("❓ Glosario rápido")
st.caption("Términos que vas a ver repetidos en toda la app, explicados sin jerga.")
term_cols = st.columns(2)
term_items = list(TERMS.items())
half = (len(term_items) + 1) // 2
for col, items in zip(term_cols, [term_items[:half], term_items[half:]]):
    with col:
        for term, definition in items:
            with st.expander(term):
                st.write(definition)

st.header("Cómo recorrer las 6 pestañas")
tabs_guide = [
    ("1 · Datos", "Cuántas solicitudes hay, cuántas caen en mora, cómo se relacionan las variables entre sí, "
                  "el diccionario de cada columna y un botón para descargar el dataset sintético completo."),
    ("2 · Selección", "Qué variables eligió cada método (\"brazo\") y el algoritmo explicado paso a paso, en un "
                       "escenario simple (18 variables) y uno difícil (45, con \"trampas\" a propósito)."),
    ("3 · Comparación", "Qué tan bien predice cada método (curvas ROC, KS) y sus aciertos/errores lado a lado."),
    ("4 · Simulador", "La parte interactiva: arma un solicitante hipotético y el modelo calcula al instante su "
                       "probabilidad de default y las 3 razones que más pesaron en esa decisión."),
    ("5 · Estabilidad", "La evidencia estadística seria: si los resultados se repiten en 10 corridas distintas, "
                         "y el panel de equidad que revisa si el modelo esconde sesgo indirecto."),
    ("6 · Documentación", "La guía completa en prosa, el informe con todas las cifras, la tabla de resultados, "
                           "las referencias de calibración y la documentación técnica — todo dentro de la app."),
]
for name, desc in tabs_guide:
    st.markdown(f"- **{name}** — {desc}")

st.page_link("pages/6_Documentacion.py", label="📚 Ir a la documentación completa", icon="➡️")
