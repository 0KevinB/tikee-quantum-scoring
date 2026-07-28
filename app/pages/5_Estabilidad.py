"""Cajas de AUC por semilla, frecuencia de selección, Nemenyi, panel de equidad.
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

from tikee.viz.plots import auc_boxplot, selection_frequency_heatmap  # noqa: E402
from glossary import LEVEL_A_ARMS, LEVEL_B_ARMS, TERMS, apply_base_style, arm_legend_rows  # noqa: E402

apply_base_style()

st.title("5 · Estabilidad, equidad y proxies")
st.caption(
    "La evidencia estadística seria: ¿los resultados se repiten si se corre el experimento varias "
    "veces, o fue suerte de una sola corrida? Y el panel más importante del proyecto: ¿el modelo "
    "esconde información sensible sin que se le haya pedido?"
)

results_path = ROOT / "reports" / "results.json"
fairness_path = ROOT / "reports" / "cache" / "f7_fairness.json"

if not results_path.exists():
    st.error("Falta reports/results.json. Corre scripts/f6_run_multiseed.py.")
    st.stop()

results = json.loads(results_path.read_text())

level = st.radio("Nivel", ["Nivel A", "Nivel B"], horizontal=True)
level_data = results[level]
arms_this_level = LEVEL_A_ARMS if level == "Nivel A" else LEVEL_B_ARMS

with st.expander(f"¿Qué es cada brazo (A0, B0, C0...) en {level}?", expanded=True):
    st.caption("Los gráficos de esta página usan estos códigos en la leyenda — así se lee cada uno.")
    st.dataframe(pd.DataFrame(arm_legend_rows(arms_this_level)), use_container_width=True, hide_index=True)

st.header("AUC por brazo a través de 10 semillas")
st.caption(
    "Cada corrida usa una \"semilla\" distinta (una forma distinta de mezclar y dividir los datos), "
    "para no confiar en un solo resultado con suerte. Cajas más cortas = resultado más consistente."
)
st.plotly_chart(auc_boxplot(level_data["score_matrix"]), use_container_width=True)

st.header("Prueba de Friedman")
with st.expander("¿Qué es la prueba de Friedman / Nemenyi?"):
    st.write(TERMS["Friedman / Nemenyi"])
friedman = level_data["friedman"]
fc1, fc2, fc3 = st.columns(3)
fc1.metric("Estadístico", f"{friedman['statistic']:.3f}")
fc2.metric("p-valor", f"{friedman['p_value']:.2e}")
fc3.metric("¿Diferencia real?", "Sí, significativa" if friedman["significant_at_0.05"] else "No concluyente")
st.caption("p-valor menor a 0.05 = las diferencias entre métodos no son casualidad del muestreo.")
if level_data.get("nemenyi"):
    st.write(f"Diferencia crítica de Nemenyi: {level_data['nemenyi']['critical_difference']:.4f}")
    st.caption("Dos brazos cuyo AUC promedio difiere más que esta cifra son estadísticamente distintos entre sí.")
    pairs_df = pd.DataFrame(level_data["nemenyi"]["pairs"])
    st.dataframe(pairs_df[pairs_df["significant"]])

st.header("Frecuencia de selección (variable x método)")
st.caption("Qué tan seguido cada método eligió cada variable a través de las 10 semillas. Más oscuro = más consistente.")
freq = {arm: s["frequency"] for arm, s in level_data["selection_stability"].items()}
st.plotly_chart(selection_frequency_heatmap(freq), use_container_width=True)

st.header("Índice de Jaccard (estabilidad de selección)")
with st.expander("¿Qué es el índice de Jaccard?"):
    st.write(TERMS["Índice de Jaccard"])
jaccard_df = pd.DataFrame({
    arm: {"jaccard": s["jaccard"], "n_vars_medio": s["mean_n_vars"]}
    for arm, s in level_data["selection_stability"].items()
}).T
st.dataframe(jaccard_df.round(3))
st.caption("Un método que elige variables distintas en cada semilla (Jaccard bajo) es inutilizable "
           "en una política de crédito escrita, aunque su AUC medio sea alto.")

st.header("⚖️ Panel de equidad y detección de proxies (semilla 42)")
st.caption(
    "La cifra más importante del proyecto. Ningún modelo usa sexo, provincia ni zona de residencia "
    "como variable de entrada — pero eso no garantiza que no pueda 'adivinarlos' combinando otras "
    "variables. Este panel lo mide directamente."
)
with st.expander("¿Qué es \"AUC de proxy\" y la regla del 80%?"):
    st.write(TERMS["AUC de proxy"])
    st.write(TERMS["Paridad demográfica / impacto dispar"])
if fairness_path.exists():
    fairness = json.loads(fairness_path.read_text())
    key = "level_a" if level == "Nivel A" else "level_b"
    for arm, res in fairness[key].items():
        with st.expander(f"{arm} — {res['n_vars']} variables"):
            st.write("**AUC de proxy** (¿el subconjunto codifica el atributo protegido? 0.5 = no, 1.0 = totalmente):")
            st.dataframe(pd.DataFrame.from_dict(res["proxy_detection_auc"], orient="index", columns=["AUC de proxy"]).round(3))
            st.write("**Diferencia de paridad demográfica por atributo** (0 = tasas de aprobación iguales entre grupos):")
            parity = {k: v["demographic_parity_diff"] for k, v in res["fairness_by_group"].items()}
            st.dataframe(pd.DataFrame.from_dict(parity, orient="index", columns=["Diferencia de paridad"]).round(3))
            st.write("**Razón de impacto dispar** (regla del 80%: por debajo de 0.80 es señal de alerta):")
            disparate = {k: v["disparate_impact_ratio"] for k, v in res["fairness_by_group"].items()}
            st.dataframe(pd.DataFrame.from_dict(disparate, orient="index", columns=["Razón de impacto"]).round(3))

    st.subheader("¿QUBO introduce más o menos sesgo indirecto que LASSO?")
    key = "level_a" if level == "Nivel A" else "level_b"
    lasso_arm, qubo_arm = ("B0", "C0") if level == "Nivel A" else ("B0b", "C0b")
    if lasso_arm in fairness[key] and qubo_arm in fairness[key]:
        lasso_proxy = fairness[key][lasso_arm]["proxy_detection_auc"]
        qubo_proxy = fairness[key][qubo_arm]["proxy_detection_auc"]
        comp = pd.DataFrame({lasso_arm: lasso_proxy, qubo_arm: qubo_proxy})
        st.dataframe(comp.round(3))
        st.caption(
            "AUC de proxy ~0.5 = sin proxy; >0.70 = codificación sustancial. Menos variables (QUBO) "
            "tiende a romper más rutas indirectas hacia el atributo protegido — mecanismo de "
            "parsimonia, no de diseño explícito contra el sesgo."
        )
else:
    st.info("Corre scripts/f7_fairness_audit.py para ver el panel de equidad.")
