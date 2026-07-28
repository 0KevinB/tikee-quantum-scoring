"""ROC superpuestas, KS, matriz de correlación, barras de métricas.
ARCHITECTURE.md §8, §11."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve


def roc_overlay(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for name, info in models.items():
        model, cols = info["model"], info["cols"]
        score = model.predict_proba(X_test[:, cols])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, score)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="azar"))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", title="ROC superpuestas")
    return fig


def ks_curve(model, cols, X_test, y_test) -> go.Figure:
    score = model.predict_proba(X_test[:, cols])[:, 1]
    fpr, tpr, thr = roc_curve(y_test, score)
    ks = tpr - fpr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thr, y=tpr, mode="lines", name="TPR"))
    fig.add_trace(go.Scatter(x=thr, y=fpr, mode="lines", name="FPR"))
    fig.add_trace(go.Scatter(x=thr, y=ks, mode="lines", name="KS", line=dict(dash="dot")))
    fig.update_layout(xaxis_title="Umbral", title=f"KS máximo = {ks.max():.3f}")
    return fig


def confusion_matrix_figure(cm: dict) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, text_auto=True, x=["pred 0", "pred 1"], y=["real 0", "real 1"], color_continuous_scale="Blues")
    fig.update_layout(title="Matriz de confusión")
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    corr = df[columns].corr(method="spearman")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Matriz de correlación (Spearman)")
    return fig


def qubo_matrix_heatmap(Q: dict, n: int, variable_names: list[str]) -> go.Figure:
    M = np.zeros((n, n))
    for (i, j), v in Q.items():
        M[i, j] = v
        M[j, i] = v
    fig = px.imshow(M, x=variable_names, y=variable_names, color_continuous_scale="RdBu_r")
    fig.update_layout(title="Matriz Q")
    return fig


def selection_frequency_heatmap(frequency_by_arm: dict[str, dict[str, float]]) -> go.Figure:
    df = pd.DataFrame(frequency_by_arm).fillna(0.0)
    fig = px.imshow(df, color_continuous_scale="Viridis", zmin=0, zmax=1, text_auto=".1f")
    fig.update_layout(title="Frecuencia de selección (variable x método)", xaxis_title="Brazo", yaxis_title="Variable")
    return fig


def auc_boxplot(score_matrix: dict[str, dict]) -> go.Figure:
    df = pd.DataFrame(score_matrix)
    fig = go.Figure()
    for arm in df.columns:
        fig.add_trace(go.Box(y=df[arm], name=arm))
    fig.update_layout(title="AUC por brazo a través de 10 semillas", yaxis_title="AUC")
    return fig


def qaoa_gap_curve(qaoa_results: dict[str, dict]) -> go.Figure:
    ps = sorted(int(p) for p in qaoa_results.keys())
    gaps = [qaoa_results[str(p)].get("gap_vs_optimal", 0.0) for p in ps]
    fig = go.Figure(go.Scatter(x=ps, y=gaps, mode="lines+markers"))
    fig.update_layout(xaxis_title="p (profundidad del circuito)", yaxis_title="Brecha vs. óptimo",
                       title="QAOA: brecha respecto al óptimo conocido vs. p")
    return fig
