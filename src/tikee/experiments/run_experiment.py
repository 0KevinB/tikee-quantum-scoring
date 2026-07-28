"""Orquestador: corre todos los brazos multi-semilla (Nivel A + Nivel B, sin QAOA
ni ExactSolver-Nivel-B) para UNA semilla. ARCHITECTURE.md §3, §6.5."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from tikee.data.seed_generator import generate_seed_table
from tikee.data.sdv_synthesizer import synthesize
from tikee.data.target_definition import add_target
from tikee.features.expand import ALL_45_COLUMNS, LevelBExpander
from tikee.features.preprocess import (
    CATEGORICAL_VARS,
    LEVEL_A_VARS,
    LEVEL_B_CATEGORICAL_VARS,
    build_preprocessor,
    get_variable_groups,
)
from tikee.models.evaluate import bootstrap_ci_auc, compute_metrics, find_threshold_max_ks
from tikee.models.train import train_logreg, train_xgboost_fixed
from tikee.selection.annealer import solve_qubo_sa
from tikee.selection.classical import lasso_select, stepwise_forward_aic
from tikee.selection.exact import solve_qubo_exact
from tikee.selection.milp import solve_qubo_milp
from tikee.selection.qubo_builder import build_qubo, compute_lambda, decode_selection, verify_cardinality
from tikee.selection.random_baseline import random_variable_subsets
from tikee.selection.redundancy import compute_redundancy
from tikee.selection.relevance import compute_relevance
from tikee.selection.tabu import solve_qubo_tabu

SIGMA = 6.2
ALPHA = 1.0
BETA_GRID = [0.25, 0.5, 1.0, 2.0]
K_GRID_A = [5, 6, 7, 8, 9, 10]
K_GRID_B = [6, 8, 10, 12, 15, 20]
MILP_TIME_LIMIT_B = 120  # reducido de 600s para que el bucle de 10 semillas sea viable en el plazo


def _cv_auc(X, y, cols, seed):
    if len(cols) == 0:
        return 0.5
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X[:, cols], y, cv=cv, scoring="roc_auc").mean())


def _sweep_beta_k(R, C, k_grid, n, X_train, y_train, variable_order, variable_groups, seed):
    best = None
    for beta in BETA_GRID:
        for k in k_grid:
            lam = compute_lambda(R, C, ALPHA, beta, n)
            Q = build_qubo(R, C, ALPHA, beta, k, lam=lam)
            sa = solve_qubo_sa(Q, num_reads=150, num_sweeps=300, seed=seed)
            if not verify_cardinality(sa["sample"], k):
                lam *= 2
                Q = build_qubo(R, C, ALPHA, beta, k, lam=lam)
                sa = solve_qubo_sa(Q, num_reads=150, num_sweeps=300, seed=seed)
            selected = decode_selection(sa["sample"], variable_order)
            cols = sorted(i for v in selected for i in variable_groups[v])
            auc = _cv_auc(X_train, y_train, cols, seed)
            if best is None or auc > best["cv_auc"]:
                best = {"beta": beta, "k": k, "cv_auc": auc}
    return best


def _eval_arm(name, X_train, X_test, y_train, y_test, cols, seed, classifier="logreg", xgb_params=None, selected_variables=None):
    if classifier == "logreg":
        model = train_logreg(X_train[:, cols], y_train, seed)
    else:
        model = train_xgboost_fixed(X_train[:, cols], y_train, seed, xgb_params)
    score_train = model.predict_proba(X_train[:, cols])[:, 1]
    score_test = model.predict_proba(X_test[:, cols])[:, 1]
    thr = find_threshold_max_ks(y_train, score_train)
    m = compute_metrics(y_test, score_test, thr)
    lo, hi = bootstrap_ci_auc(y_test, score_test, n_boot=500, seed=seed)
    m["auc_ci95"] = [lo, hi]
    m["arm"] = name
    m["selected_variables"] = selected_variables if selected_variables is not None else []
    return m


def run_seed(seed: int, xgb_params_a: dict[str, Any], xgb_params_b: dict[str, Any]) -> dict[str, Any]:
    seed_df = generate_seed_table(seed, 2000)
    synth_df, _ = synthesize(seed_df, method="gaussian_copula", n_sample=8000, seed=seed)
    df = add_target(synth_df, sigma=SIGMA, seed=seed, target_rate=0.08)
    train_df, test_df = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=seed)
    y_train, y_test = train_df["default"].to_numpy(), test_df["default"].to_numpy()

    results: dict[str, Any] = {"seed": seed}

    # ---- Nivel A ----
    pre = build_preprocessor(LEVEL_A_VARS)
    Xa_train = pre.fit_transform(train_df[LEVEL_A_VARS])
    Xa_test = pre.transform(test_df[LEVEL_A_VARS])
    vgroups_a = get_variable_groups(pre)

    Ra = compute_relevance(train_df, y_train, LEVEL_A_VARS, set(CATEGORICAL_VARS), seed=seed)
    Ca = compute_redundancy(train_df, LEVEL_A_VARS, set(CATEGORICAL_VARS))
    best_a = _sweep_beta_k(Ra, Ca, K_GRID_A, len(LEVEL_A_VARS), Xa_train, y_train, LEVEL_A_VARS, vgroups_a, seed)
    lam_a = compute_lambda(Ra, Ca, ALPHA, best_a["beta"], len(LEVEL_A_VARS))
    Qa = build_qubo(Ra, Ca, ALPHA, best_a["beta"], best_a["k"], lam=lam_a)

    c0 = solve_qubo_sa(Qa, num_reads=1000, num_sweeps=1000, seed=seed)
    c1 = solve_qubo_exact(Qa, n_vars=len(LEVEL_A_VARS))
    c2 = solve_qubo_milp(Qa, n=len(LEVEL_A_VARS))
    results["c1_eq_c2_level_a"] = bool(c1["sample"] == c2["sample"])

    results["A0"] = _eval_arm("A0", Xa_train, Xa_test, y_train, y_test, list(range(Xa_train.shape[1])), seed, "logreg", selected_variables=LEVEL_A_VARS)
    results["A1"] = _eval_arm("A1", Xa_train, Xa_test, y_train, y_test, list(range(Xa_train.shape[1])), seed, "xgboost", xgb_params_a, selected_variables=LEVEL_A_VARS)

    lasso_a = lasso_select(Xa_train, y_train, vgroups_a, seed)
    results["B0"] = _eval_arm("B0", Xa_train, Xa_test, y_train, y_test, lasso_a["selected_columns"], seed, "logreg", selected_variables=lasso_a["selected_variables"])
    results["B0x"] = _eval_arm("B0x", Xa_train, Xa_test, y_train, y_test, lasso_a["selected_columns"], seed, "xgboost", xgb_params_a, selected_variables=lasso_a["selected_variables"])

    step_a = stepwise_forward_aic(Xa_train, y_train, vgroups_a)
    results["B1"] = _eval_arm("B1", Xa_train, Xa_test, y_train, y_test, step_a["selected_columns"], seed, "logreg", selected_variables=step_a["selected_variables"])

    for name, sample in (("C0", c0["sample"]), ("C1", c1["sample"]), ("C2", c2["sample"])):
        sel = decode_selection(sample, LEVEL_A_VARS)
        cols = sorted(i for v in sel for i in vgroups_a[v])
        results[name] = _eval_arm(name, Xa_train, Xa_test, y_train, y_test, cols, seed, "logreg", selected_variables=sel)

    random_subsets_a = random_variable_subsets(LEVEL_A_VARS, best_a["k"], n_repetitions=100, seed=seed)
    random_aucs_a = [_cv_auc(Xa_train, y_train, sorted(i for v in s for i in vgroups_a[v]), seed) for s in random_subsets_a]
    results["R"] = {"arm": "R", "auc": float(np.mean(random_aucs_a)), "auc_std": float(np.std(random_aucs_a)), "selected_variables": []}

    # ---- Nivel B ----
    expander = LevelBExpander().fit(train_df, y_train)
    Xb_train_df = expander.transform(train_df, seed=seed)
    Xb_test_df = expander.transform(test_df, seed=seed)
    pre_b = build_preprocessor(ALL_45_COLUMNS, categorical_vars=LEVEL_B_CATEGORICAL_VARS)
    Xb_train = pre_b.fit_transform(Xb_train_df[ALL_45_COLUMNS])
    Xb_test = pre_b.transform(Xb_test_df[ALL_45_COLUMNS])
    vgroups_b = get_variable_groups(pre_b)

    categorical_b = set(LEVEL_B_CATEGORICAL_VARS)
    Rb = compute_relevance(Xb_train_df, y_train, ALL_45_COLUMNS, categorical_b | {"f36", "f37"}, seed=seed)
    Cb = compute_redundancy(Xb_train_df, ALL_45_COLUMNS, categorical_vars=categorical_b)
    best_b = _sweep_beta_k(Rb, Cb, K_GRID_B, len(ALL_45_COLUMNS), Xb_train, y_train, ALL_45_COLUMNS, vgroups_b, seed)
    lam_b = compute_lambda(Rb, Cb, ALPHA, best_b["beta"], len(ALL_45_COLUMNS))
    Qb = build_qubo(Rb, Cb, ALPHA, best_b["beta"], best_b["k"], lam=lam_b)

    c0b = solve_qubo_sa(Qb, num_reads=1000, num_sweeps=1000, seed=seed)
    c2b = solve_qubo_milp(Qb, n=len(ALL_45_COLUMNS), cardinality_k=best_b["k"], time_limit=MILP_TIME_LIMIT_B)
    c4b = solve_qubo_tabu(Qb, num_reads=200, seed=seed)

    all_cols_b = list(range(Xb_train.shape[1]))
    results["A0b"] = _eval_arm("A0b", Xb_train, Xb_test, y_train, y_test, all_cols_b, seed, "logreg", selected_variables=ALL_45_COLUMNS)
    results["A1b"] = _eval_arm("A1b", Xb_train, Xb_test, y_train, y_test, all_cols_b, seed, "xgboost", xgb_params_b, selected_variables=ALL_45_COLUMNS)

    lasso_b = lasso_select(Xb_train, y_train, vgroups_b, seed)
    results["B0b"] = _eval_arm("B0b", Xb_train, Xb_test, y_train, y_test, lasso_b["selected_columns"], seed, "logreg", selected_variables=lasso_b["selected_variables"])

    step_b = stepwise_forward_aic(Xb_train, y_train, vgroups_b)
    results["B1b"] = _eval_arm("B1b", Xb_train, Xb_test, y_train, y_test, step_b["selected_columns"], seed, "logreg", selected_variables=step_b["selected_variables"])

    for name, sample in (("C0b", c0b["sample"]), ("C2b", c2b["sample"]), ("C4b", c4b["sample"])):
        if not sample:
            continue
        sel = decode_selection(sample, ALL_45_COLUMNS)
        cols = sorted(i for v in sel for i in vgroups_b[v])
        results[name] = _eval_arm(name, Xb_train, Xb_test, y_train, y_test, cols, seed, "logreg", selected_variables=sel)

    random_subsets_b = random_variable_subsets(ALL_45_COLUMNS, best_b["k"], n_repetitions=100, seed=seed)
    random_aucs_b = [_cv_auc(Xb_train, y_train, sorted(i for v in s for i in vgroups_b[v]), seed) for s in random_subsets_b]
    results["Rb"] = {"arm": "Rb", "auc": float(np.mean(random_aucs_b)), "auc_std": float(np.std(random_aucs_b)), "selected_variables": []}

    results["beta_k_star_level_a"] = best_a
    results["beta_k_star_level_b"] = best_b
    return results
