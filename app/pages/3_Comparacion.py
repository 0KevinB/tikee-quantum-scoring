"""Tabla de métricas, ROC, KS, matrices de confusión. ARCHITECTURE.md §11."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tikee.viz.plots import confusion_matrix_figure, ks_curve, roc_overlay  # noqa: E402

st.title("3 · Comparación de brazos")

artifacts_path = ROOT / "reports" / "cache" / "app_artifacts.joblib"
results_path = ROOT / "reports" / "results.json"

if not artifacts_path.exists():
    st.error("Falta reports/cache/app_artifacts.joblib. Corre scripts/f7_fairness_audit.py.")
    st.stop()

# joblib.load: seguro, el único archivo leído es el caché que este mismo repo generó.
artifacts = joblib.load(artifacts_path)

level = st.radio("Nivel", ["Nivel A", "Nivel B"], horizontal=True)
key = "level_a" if level == "Nivel A" else "level_b"
bundle = artifacts[key]

st.header("ROC superpuestas (semilla 42)")
st.plotly_chart(roc_overlay(bundle["models"], bundle["X_test"], bundle["y_test"]), use_container_width=True)

st.header("Curva KS por brazo")
arm_choice = st.selectbox("Brazo", list(bundle["models"].keys()))
info = bundle["models"][arm_choice]
st.plotly_chart(ks_curve(info["model"], info["cols"], bundle["X_test"], bundle["y_test"]), use_container_width=True)

if results_path.exists():
    st.header("Tabla de métricas (media ± dp a través de 10 semillas)")
    results = json.loads(results_path.read_text())
    summary = results[level]["summary"]
    st.dataframe(pd.DataFrame(summary).T.round(4))

    st.header("Matrices de confusión lado a lado (semilla 42)")
    cols_st = st.columns(min(4, len(bundle["models"])))
    from sklearn.metrics import confusion_matrix

    for i, (arm, info) in enumerate(bundle["models"].items()):
        score = info["model"].predict_proba(bundle["X_test"][:, info["cols"]])[:, 1]
        pred = (score >= info["threshold"]).astype(int)
        tn, fp, fn, tp = confusion_matrix(bundle["y_test"], pred, labels=[0, 1]).ravel()
        with cols_st[i % len(cols_st)]:
            st.plotly_chart(confusion_matrix_figure({"tn": tn, "fp": fp, "fn": fn, "tp": tp}), use_container_width=True)
            st.caption(arm)
else:
    st.info("Corre scripts/f6_run_multiseed.py para ver la tabla agregada de métricas.")
