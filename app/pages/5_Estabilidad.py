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

from tikee.viz.plots import auc_boxplot, selection_frequency_heatmap  # noqa: E402

st.title("5 · Estabilidad, equidad y proxies")

results_path = ROOT / "reports" / "results.json"
fairness_path = ROOT / "reports" / "cache" / "f7_fairness.json"

if not results_path.exists():
    st.error("Falta reports/results.json. Corre scripts/f6_run_multiseed.py.")
    st.stop()

results = json.loads(results_path.read_text())

level = st.radio("Nivel", ["Nivel A", "Nivel B"], horizontal=True)
level_data = results[level]

st.header("AUC por brazo a través de 10 semillas")
st.plotly_chart(auc_boxplot(level_data["score_matrix"]), use_container_width=True)

st.header("Prueba de Friedman")
st.json(level_data["friedman"])
if level_data.get("nemenyi"):
    st.write(f"Diferencia crítica de Nemenyi: {level_data['nemenyi']['critical_difference']:.4f}")
    pairs_df = pd.DataFrame(level_data["nemenyi"]["pairs"])
    st.dataframe(pairs_df[pairs_df["significant"]])

st.header("Frecuencia de selección (variable x método)")
freq = {arm: s["frequency"] for arm, s in level_data["selection_stability"].items()}
st.plotly_chart(selection_frequency_heatmap(freq), use_container_width=True)

st.header("Índice de Jaccard (estabilidad de selección)")
jaccard_df = pd.DataFrame({
    arm: {"jaccard": s["jaccard"], "n_vars_medio": s["mean_n_vars"]}
    for arm, s in level_data["selection_stability"].items()
}).T
st.dataframe(jaccard_df.round(3))
st.caption("Un método que elige variables distintas en cada semilla (Jaccard bajo) es inutilizable "
           "en una política de crédito escrita, aunque su AUC medio sea alto.")

st.header("Panel de equidad y detección de proxies (semilla 42)")
if fairness_path.exists():
    fairness = json.loads(fairness_path.read_text())
    key = "level_a" if level == "Nivel A" else "level_b"
    for arm, res in fairness[key].items():
        with st.expander(f"{arm} — {res['n_vars']} variables"):
            st.write("**AUC_proxy** (¿el subconjunto codifica el atributo protegido?):")
            st.json(res["proxy_detection_auc"])
            st.write("**Diferencia de paridad demográfica por atributo:**")
            st.json({k: v["demographic_parity_diff"] for k, v in res["fairness_by_group"].items()})
            st.write("**Razón de impacto dispar (regla del 80%):**")
            st.json({k: v["disparate_impact_ratio"] for k, v in res["fairness_by_group"].items()})

    st.subheader("¿QUBO introduce más o menos sesgo indirecto que LASSO?")
    key = "level_a" if level == "Nivel A" else "level_b"
    lasso_arm, qubo_arm = ("B0", "C0") if level == "Nivel A" else ("B0b", "C0b")
    if lasso_arm in fairness[key] and qubo_arm in fairness[key]:
        lasso_proxy = fairness[key][lasso_arm]["proxy_detection_auc"]
        qubo_proxy = fairness[key][qubo_arm]["proxy_detection_auc"]
        comp = pd.DataFrame({lasso_arm: lasso_proxy, qubo_arm: qubo_proxy})
        st.dataframe(comp.round(3))
        st.caption(
            "AUC_proxy ~0.5 = sin proxy; >0.70 = codificación sustancial. Menos variables (QUBO) "
            "tiende a romper más rutas indirectas hacia el atributo protegido — mecanismo de "
            "parsimonia, no de diseño explícito contra el sesgo."
        )
else:
    st.info("Corre scripts/f7_fairness_audit.py para ver el panel de equidad.")
