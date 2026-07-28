"""Checkpoint F3 (PLAN.md §4.1): brazos A0, A1, B0, B0x, B1 con métricas e IC en un
results.json parcial (Nivel A, sin los brazos QUBO todavía). Semilla 42.

Uso: .venv/bin/python scripts/f3_run_level_a.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402
from tikee.features.preprocess import build_preprocessor, get_variable_groups, LEVEL_A_VARS  # noqa: E402
from tikee.models.evaluate import bootstrap_ci_auc, compute_metrics, find_threshold_max_ks  # noqa: E402
from tikee.models.train import median_hyperparams, nested_cv_xgboost, train_logreg, train_xgboost_fixed  # noqa: E402
from tikee.selection.classical import lasso_select, stepwise_forward_aic  # noqa: E402

SEED = 42
SIGMA = 6.2


def evaluate_arm(name, X_train, y_train, X_test, y_test, classifier, seed, xgb_params=None):
    t0 = time.perf_counter()
    if classifier == "logreg":
        model = train_logreg(X_train, y_train, seed)
    elif classifier == "xgboost":
        model = train_xgboost_fixed(X_train, y_train, seed, xgb_params)
    else:
        raise ValueError(classifier)
    wall = time.perf_counter() - t0

    score_train = model.predict_proba(X_train)[:, 1]
    score_test = model.predict_proba(X_test)[:, 1]
    threshold = find_threshold_max_ks(y_train, score_train)

    metrics = compute_metrics(y_test, score_test, threshold)
    ci_lo, ci_hi = bootstrap_ci_auc(y_test, score_test, n_boot=1000, seed=seed)
    metrics["auc_ci95"] = [ci_lo, ci_hi]
    metrics["train_wall_time_s"] = wall
    metrics["arm"] = name
    return metrics


def main() -> int:
    seed_df = generate_seed_table(SEED, 2000)
    synth_df, _ = synthesize(seed_df, method="gaussian_copula", n_sample=8000, seed=SEED)
    df = add_target(synth_df, sigma=SIGMA, seed=SEED, target_rate=0.08)

    train_df, test_df = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=SEED)
    y_train, y_test = train_df["default"].to_numpy(), test_df["default"].to_numpy()

    pre = build_preprocessor(LEVEL_A_VARS)
    X_train = pre.fit_transform(train_df[LEVEL_A_VARS])
    X_test = pre.transform(test_df[LEVEL_A_VARS])
    variable_groups = get_variable_groups(pre)

    print(f"n_train={len(train_df)} n_test={len(test_df)} n_features_transformed={X_train.shape[1]}")

    print("\n--- CV anidada XGBoost (D13, semilla 42) ---")
    ncv = nested_cv_xgboost(X_train, y_train, seed=SEED, outer_folds=5, inner_folds=3, n_iter=30)
    print(f"AUC outer folds: {ncv['outer_auc_scores']}")
    print(f"AUC media±dp: {ncv['outer_auc_mean']:.4f} ± {ncv['outer_auc_std']:.4f}")
    xgb_params = median_hyperparams(ncv["best_params_per_fold"])
    print(f"Hiperparámetros medianos (fijos para las 10 semillas, D13): {xgb_params}")

    results = {}

    results["A0"] = evaluate_arm("A0", X_train, y_train, X_test, y_test, "logreg", SEED)
    results["A1"] = evaluate_arm("A1", X_train, y_train, X_test, y_test, "xgboost", SEED, xgb_params)

    lasso = lasso_select(X_train, y_train, variable_groups, SEED)
    print(f"\nLASSO seleccionó ({len(lasso['selected_variables'])}): {lasso['selected_variables']}")
    X_train_b0 = X_train[:, lasso["selected_columns"]]
    X_test_b0 = X_test[:, lasso["selected_columns"]]
    results["B0"] = evaluate_arm("B0", X_train_b0, y_train, X_test_b0, y_test, "logreg", SEED)
    results["B0"]["selected_variables"] = lasso["selected_variables"]
    results["B0x"] = evaluate_arm("B0x", X_train_b0, y_train, X_test_b0, y_test, "xgboost", SEED, xgb_params)
    results["B0x"]["selected_variables"] = lasso["selected_variables"]

    stepwise = stepwise_forward_aic(X_train, y_train, variable_groups)
    print(f"Stepwise-AIC seleccionó ({len(stepwise['selected_variables'])}): {stepwise['selected_variables']}")
    X_train_b1 = X_train[:, stepwise["selected_columns"]]
    X_test_b1 = X_test[:, stepwise["selected_columns"]]
    results["B1"] = evaluate_arm("B1", X_train_b1, y_train, X_test_b1, y_test, "logreg", SEED)
    results["B1"]["selected_variables"] = stepwise["selected_variables"]

    print("\n--- Resultados Nivel A (parcial, sin brazos QUBO) ---")
    for arm, m in results.items():
        print(f"{arm:5s} AUC={m['auc']:.4f} [{m['auc_ci95'][0]:.4f},{m['auc_ci95'][1]:.4f}]  "
              f"KS={m['ks']:.4f}  #vars={len(m.get('selected_variables', LEVEL_A_VARS))}")

    out_dir = Path(__file__).resolve().parents[1] / "reports" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f3_level_a_partial.json").write_text(json.dumps({
        "nested_cv_xgboost": {k: v for k, v in ncv.items() if k != "best_params_per_fold"},
        "xgb_fixed_params": xgb_params,
        "results": results,
    }, indent=2, default=str))
    print(f"\nGuardado en {out_dir / 'f3_level_a_partial.json'}")

    print("\n--- Checkpoint F3 ---")
    print("cuatro/cinco brazos con métricas e IC: OK (A0, A1, B0, B0x, B1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
