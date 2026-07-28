"""Genera las figuras estáticas requeridas por la Definición de listo (PLAN.md §8):
ROC superpuestas, gráfico KS, matrices de confusión, diagrama de diferencia crítica.

Uso: .venv/bin/python scripts/f8_generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from tikee.experiments.stats import plot_critical_difference  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def roc_and_ks(level_key: str, suffix: str, artifacts: dict) -> None:
    bundle = artifacts[level_key]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, info in bundle["models"].items():
        score = info["model"].predict_proba(X_test[:, info["cols"]])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={np.trapz(tpr, fpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="azar")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC superpuestas — {suffix} (semilla 42)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"roc_overlay_{suffix}.png", dpi=150)
    plt.close(fig)

    ref_arm = "C0" if "C0" in bundle["models"] else list(bundle["models"].keys())[0]
    info = bundle["models"][ref_arm]
    score = info["model"].predict_proba(X_test[:, info["cols"]])[:, 1]
    fpr, tpr, thr = roc_curve(y_test, score)
    ks = tpr - fpr
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr, tpr, label="TPR")
    ax.plot(thr, fpr, label="FPR")
    ax.plot(thr, ks, "--", label=f"KS (máx={ks.max():.3f})")
    ax.set_xlabel("Umbral")
    ax.set_title(f"Curva KS — {ref_arm} ({suffix}, semilla 42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"ks_curve_{suffix}.png", dpi=150)
    plt.close(fig)

    n_arms = len(bundle["models"])
    fig, axes = plt.subplots(1, n_arms, figsize=(3 * n_arms, 3))
    if n_arms == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, bundle["models"].items()):
        score = info["model"].predict_proba(X_test[:, info["cols"]])[:, 1]
        pred = (score >= info["threshold"]).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["real 0", "real 1"], fontsize=7)
    fig.suptitle(f"Matrices de confusión — {suffix} (semilla 42)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_matrices_{suffix}.png", dpi=150)
    plt.close(fig)


def main() -> int:
    artifacts_path = ROOT / "reports" / "cache" / "app_artifacts.joblib"
    # joblib.load: seguro, unico archivo leido es el caché que este repo generó.
    artifacts = joblib.load(artifacts_path)
    roc_and_ks("level_a", "nivel_a", artifacts)
    roc_and_ks("level_b", "nivel_b", artifacts)
    print("ROC, KS y matrices de confusión guardadas.")

    results = json.loads((ROOT / "reports" / "results.json").read_text())
    for level_name, suffix in (("Nivel A", "nivel_a"), ("Nivel B", "nivel_b")):
        nem = results[level_name].get("nemenyi")
        if nem:
            plot_critical_difference(
                nem["average_ranks"], nem["critical_difference"],
                str(FIG_DIR / f"critical_difference_{suffix}.png"),
            )
            print(f"Diagrama de diferencia crítica ({level_name}) guardado.")

        score_matrix = pd.DataFrame(results[level_name]["score_matrix"])
        fig, ax = plt.subplots(figsize=(8, 4))
        score_matrix.boxplot(ax=ax)
        ax.set_ylabel("AUC")
        ax.set_title(f"AUC por brazo a través de 10 semillas — {level_name}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"auc_boxplot_{suffix}.png", dpi=150)
        plt.close(fig)

    f4 = json.loads((ROOT / "reports" / "cache" / "f4_level_a_qubo.json").read_text())
    if f4.get("qaoa"):
        ps = sorted(int(p) for p in f4["qaoa"].keys())
        gaps = [f4["qaoa"][str(p)].get("gap_vs_optimal", 0.0) for p in ps]
        wall_times = [f4["qaoa"][str(p)].get("wall_time_s", 0.0) for p in ps]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(ps, gaps, "o-", markersize=10)
        ax1.set_ylim(-0.05, 1.0)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax1.set_xlabel("p (profundidad del circuito)")
        ax1.set_ylabel("Brecha vs. óptimo conocido")
        ax1.set_title("Brecha (todas ≈ 0: alcanza el óptimo)")
        ax1.set_xticks(ps)
        for p, g in zip(ps, gaps):
            ax1.annotate(f"{g:.1e}", (p, g), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)

        ax2.bar([str(p) for p in ps], [t / 60 for t in wall_times], color="#B23A48")
        ax2.set_xlabel("p (profundidad del circuito)")
        ax2.set_ylabel("Tiempo de pared (minutos)")
        ax2.set_title("Costo: minutos por profundidad")

        fig.suptitle("QAOA (Nivel A, semilla 42): óptimo exacto, a un costo alto")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIG_DIR / "qaoa_gap_curve.png", dpi=150)
        plt.close(fig)
        print("Curva de brecha QAOA guardada.")

    print(f"\nTodas las figuras en {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
