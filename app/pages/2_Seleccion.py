"""Variables elegidas por método, mapa de calor de Q, energías del recocido, curva QAOA.
ARCHITECTURE.md §11."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from tikee.features.expand import ground_truth_labels  # noqa: E402
from tikee.features.preprocess import LEVEL_A_VARS  # noqa: E402
from tikee.selection.qubo_builder import build_qubo, compute_lambda  # noqa: E402
from tikee.viz.plots import qaoa_gap_curve, qubo_matrix_heatmap  # noqa: E402
from glossary import TERMS, arm_legend_rows  # noqa: E402

st.title("2 · Selección de variables")
st.markdown(
    "Cada fila de abajo es un **brazo**: un método distinto que compitió para elegir qué "
    "variables usar. Todos parten del mismo QUBO (la fórmula de \"relevancia menos redundancia\"), "
    "pero cada uno lo resuelve con un algoritmo distinto — algunos rápidos pero aproximados, otros "
    "lentos pero garantizados como óptimos."
)
with st.expander("¿Qué es el QUBO y el recocido simulado?"):
    st.write(TERMS["QUBO"])
    st.write(TERMS["Recocido simulado"])

f4_path = ROOT / "reports" / "cache" / "f4_level_a_qubo.json"
f5_path = ROOT / "reports" / "cache" / "f5_level_b.json"

if not f4_path.exists():
    st.error("Falta reports/cache/f4_level_a_qubo.json. Corre scripts/f4_run_level_a_qubo.py.")
    st.stop()

f4 = json.loads(f4_path.read_text())
gt = ground_truth_labels()

st.header("Nivel A (18 variables candidatas)")
st.caption(
    "El escenario simple: 18 variables candidatas, con el óptimo global ya conocido de antemano "
    "(por fuerza bruta), para poder verificar si cada método lo encuentra o no."
)
st.write(f"beta* = {f4['beta_star']}, k* = {f4['k_star']}, lambda* = {f4['lambda_star']:.3f}")
st.dataframe(pd.DataFrame(arm_legend_rows(["C0", "C1", "C2"])), use_container_width=True, hide_index=True)

for arm in ("C0", "C1", "C2"):
    sample = f4["solutions"][arm]["sample"]
    selected = sorted(v for i, v in enumerate(LEVEL_A_VARS) if str(sample.get(str(i), sample.get(i, 0))) == "1")
    st.write(f"**{arm}** (E={f4['solutions'][arm]['energy']:.3f}): {selected}")
st.caption(
    "\"E\" es la energía del QUBO: mientras más baja (más negativa), mejor cumple la combinación "
    "de variables el objetivo de \"relevante pero no redundante\". Fíjate que C1 y C2 — los dos "
    "métodos que garantizan el óptimo — eligieron exactamente la misma lista: es la prueba de que "
    "el problema está bien planteado e implementado."
)

st.subheader("Marcas de trampa (Nivel A)")
st.caption("nivel_educacion y zona_residencia tienen coeficiente cero en el target — "
           "un selector guiado solo por relevancia marginal puede recogerlas por su asociación con el bloque C.")

st.subheader("Curva de brecha QAOA vs. óptimo conocido")
if "qaoa" in f4 and f4["qaoa"]:
    st.plotly_chart(qaoa_gap_curve(f4["qaoa"]), use_container_width=True)
    st.caption(
        "QAOA con p=1,2,3 alcanza el óptimo exacto en este problema (Nivel A). No es una ventaja "
        "cuántica: el paisaje de esta instancia es fácil, y QAOA es órdenes de magnitud más caro "
        "que la enumeración exhaustiva que ya conoce el óptimo. Ver ARCHITECTURE.md §7.4."
    )

if f5_path.exists():
    st.header("Nivel B (45 variables candidatas)")
    st.caption(
        "El escenario difícil y más realista: 45 variables candidatas, con dos columnas de puro "
        "ruido metidas a propósito, para ver si algún método cae en la trampa de \"elegirlas gratis\"."
    )
    f5 = json.loads(f5_path.read_text())
    st.write(f"beta* = {f5['beta_star']}, k* = {f5['k_star']}")
    st.write(f"MILP (límite 600s): status={f5['milp_status']['status']}, "
             f"mip_gap={f5['milp_status'].get('mip_gap')}")
    st.dataframe(pd.DataFrame(arm_legend_rows(["C0b", "C2b", "C4b"])), use_container_width=True, hide_index=True)

    for arm in ("C0b", "C2b", "C4b"):
        if arm in f5["results"]:
            sel = f5["results"][arm].get("selected_variables", [])
            noise = [v for v in sel if gt.get(v) == "irrelevant"]
            st.write(f"**{arm}**: {sel}")
            if noise:
                st.warning(f"{arm} incluyó variables irrelevantes/ruido: {noise}")

    st.caption(
        "A k=20, varios solucionadores incluyen alguna variable de ruido puro (f44/f45) o "
        "irrelevante (f02/f03): el criterio relevancia-redundancia no penaliza una variable que "
        "es simultáneamente irrelevante y no-redundante, así que 'sale gratis' cuando k excede el "
        "número de variables genuinamente útiles. Es una limitación real del criterio, detectada "
        "por diseño (ARCHITECTURE.md §4.6)."
    )
else:
    st.info("Corre scripts/f5_run_level_b.py para ver el detalle de Nivel B.")
