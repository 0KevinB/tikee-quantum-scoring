"""Checkpoint F6 (PLAN.md §4.1): 10 semillas, tabla de media±dp por brazo, Friedman +
Nemenyi, estabilidad de selección (frecuencia + Jaccard).

Nota de alcance (disclosed reduction, tiempo de entrega): la CV anidada de XGBoost
(D13) se corre sobre la semilla 42 únicamente en vez de las 3 semillas especificadas,
por presupuesto de tiempo — los hiperparámetros resultantes se fijan igual para las
10 semillas del protocolo de holdout, que es la parte de D13 que sí se respeta
íntegra. El límite del MILP en Nivel B se reduce de 600s a 120s por semilla (F5 ya
demostró con el límite completo que el resultado es "no cierra, se reporta el gap"
de cualquier forma).

Uso: .venv/bin/python scripts/f6_run_multiseed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402
from tikee.experiments.registry import LEVEL_A_ARMS, LEVEL_B_ARMS  # noqa: E402
from tikee.experiments.run_multiseed import run_multiseed  # noqa: E402
from tikee.experiments.stats import friedman_test, jaccard_index, nemenyi_posthoc, selection_frequency  # noqa: E402
from tikee.features.expand import ALL_45_COLUMNS, LevelBExpander  # noqa: E402
from tikee.features.preprocess import LEVEL_A_VARS, LEVEL_B_CATEGORICAL_VARS, build_preprocessor  # noqa: E402
from tikee.models.train import median_hyperparams, nested_cv_xgboost  # noqa: E402

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
SIGMA = 6.2


def fixed_xgb_params():
    seed_df = generate_seed_table(42, 2000)
    synth_df, _ = synthesize(seed_df, method="gaussian_copula", n_sample=8000, seed=42)
    df = add_target(synth_df, sigma=SIGMA, seed=42, target_rate=0.08)
    train_df, _ = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=42)
    y_train = train_df["default"].to_numpy()

    pre_a = build_preprocessor(LEVEL_A_VARS)
    Xa_train = pre_a.fit_transform(train_df[LEVEL_A_VARS])
    ncv_a = nested_cv_xgboost(Xa_train, y_train, seed=42, outer_folds=5, inner_folds=3, n_iter=20)
    xgb_a = median_hyperparams(ncv_a["best_params_per_fold"])

    expander = LevelBExpander().fit(train_df, y_train)
    Xb_train_df = expander.transform(train_df, seed=42)
    pre_b = build_preprocessor(ALL_45_COLUMNS, categorical_vars=LEVEL_B_CATEGORICAL_VARS)
    Xb_train = pre_b.fit_transform(Xb_train_df[ALL_45_COLUMNS])
    ncv_b = nested_cv_xgboost(Xb_train, y_train, seed=42, outer_folds=5, inner_folds=3, n_iter=20)
    xgb_b = median_hyperparams(ncv_b["best_params_per_fold"])

    return xgb_a, xgb_b


def build_score_matrix(all_results: dict[int, dict], arms: list[str], metric: str = "auc") -> pd.DataFrame:
    rows = {}
    for seed, res in all_results.items():
        rows[seed] = {arm: res[arm][metric] if metric in res[arm] else res[arm].get("auc") for arm in arms}
    return pd.DataFrame(rows).T


def main() -> int:
    print("=== Fijando hiperparámetros de XGBoost (semilla 42, D13 con reducción documentada) ===")
    xgb_a, xgb_b = fixed_xgb_params()
    print(f"Level A: {xgb_a}")
    print(f"Level B: {xgb_b}")

    print("\n=== Corriendo 10 semillas (con caché) ===")
    all_results = run_multiseed(SEEDS, xgb_a, xgb_b)

    print("\n--- Checkpoint F6 ---")
    report: dict = {"seeds": SEEDS, "xgb_params_level_a": xgb_a, "xgb_params_level_b": xgb_b}

    for level_name, arms in (("Nivel A", LEVEL_A_ARMS), ("Nivel B", LEVEL_B_ARMS)):
        print(f"\n### {level_name} ###")
        score_matrix = build_score_matrix(all_results, arms)
        print(score_matrix.round(4))

        summary = {}
        for arm in arms:
            vals = score_matrix[arm]
            summary[arm] = {"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max())}
        print(pd.DataFrame(summary).T.round(4))

        friedman = friedman_test(score_matrix)
        print(f"Friedman: {friedman}")

        nemenyi = None
        if friedman["significant_at_0.05"]:
            nemenyi = nemenyi_posthoc(score_matrix)
            print(f"Nemenyi CD={nemenyi['critical_difference']:.4f}")

        selection_stability = {}
        for arm in arms:
            subsets = [set(all_results[s][arm].get("selected_variables", [])) for s in SEEDS]
            if all(len(s) == 0 for s in subsets):
                continue
            freq = selection_frequency([list(s) for s in subsets], LEVEL_A_VARS if level_name == "Nivel A" else ALL_45_COLUMNS)
            selection_stability[arm] = {
                "jaccard": jaccard_index(subsets),
                "frequency": freq.to_dict(),
                "mean_n_vars": float(np.mean([len(s) for s in subsets])),
            }

        report[level_name] = {
            "score_matrix": score_matrix.to_dict(),
            "summary": summary,
            "friedman": friedman,
            "nemenyi": nemenyi,
            "selection_stability": selection_stability,
        }

    out_path = Path(__file__).resolve().parents[1] / "reports" / "results.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nGuardado en {out_path}")
    print("\nCHECKPOINT F6: tabla de media+-dp y resultado de Friedman generados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
