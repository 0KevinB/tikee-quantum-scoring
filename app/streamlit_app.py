"""Portada y navegación de la app de demostración. ARCHITECTURE.md §11."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Tikee — Scoring cuántico-inspirado", page_icon="🏦", layout="wide")

st.title("Tikee — Selección de variables cuántico-inspirada para scoring crediticio")

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

st.header("Pregunta de investigación")
st.markdown(
    "¿Formular la selección de variables como un problema **QUBO** mejora un modelo de scoring "
    "crediticio frente a LASSO o stepwise, cuando las variables candidatas están fuertemente "
    "correlacionadas entre sí?\n\n"
    "El mismo QUBO se resuelve con **cinco solucionadores** para separar tres preguntas:\n\n"
    "1. ¿Es buena **la formulación** QUBO relevancia-redundancia? → recocido vs. LASSO\n"
    "2. ¿Aporta algo **el heurístico**? → recocido vs. óptimo certificado (enumeración / MILP)\n"
    "3. ¿Aporta algo **el paradigma de circuitos**? → QAOA simulado vs. recocido"
)

st.header("Navegación")
st.markdown(
    "- **1 · Datos** — esquema, correlaciones, fidelidad sintética\n"
    "- **2 · Selección** — qué eligió cada método, matriz Q, curva QAOA\n"
    "- **3 · Comparación** — métricas, ROC, KS, matrices de confusión\n"
    "- **4 · Simulador** — probabilidad de default para un solicitante hipotético\n"
    "- **5 · Estabilidad** — resultados a través de 10 semillas, equidad y proxies"
)

st.caption(
    "Proyecto de innovación universitaria · UTPL Ecuador. Inspirado en el pitch de la fintech Tikee."
)
