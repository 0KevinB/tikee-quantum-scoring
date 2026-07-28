"""Checkpoint F5 (PLAN.md §4.1): Nivel B (N=45) — recocido, MILP-Glover (límite 600s),
Tabu; tiempo de solución y calidad de energía. Sin ExactSolver ni QAOA (imposibles en
N=45, ARCHITECTURE.md §7.0). QAOA de Nivel A ya corrió en F4.

Uso: .venv/bin/python scripts/f5_run_level_b.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402
from tikee.features.expand import ALL_45_COLUMNS, LevelBExpander, ground_truth_labels  # noqa: E402
from tikee.features.preprocess import LEVEL_B_CATEGORICAL_VARS, build_preprocessor, get_variable_groups  # noqa: E402
from tikee.models.evaluate import bootstrap_ci_auc, compute_metrics, find_threshold_max_ks  # noqa: E402
from tikee.models.train import median_hyperparams, nested_cv_xgboost, train_logreg, train_xgboost_fixed  # noqa: E402
from tikee.selection.annealer import solve_qubo_sa  # noqa: E402
from tikee.selection.classical import lasso_select, stepwise_forward_aic  # noqa: E402
from tikee.selection.milp import solve_qubo_milp  # noqa: E402
from tikee.selection.qubo_builder import build_qubo, compute_lambda, decode_selection, verify_cardinality  # noqa: E402
from tikee.selection.random_baseline import random_variable_subsets  # noqa: E402
from tikee.selection.redundancy import compute_redundancy  # noqa: E402
from tikee.selection.relevance import compute_relevance  # noqa: E402
from tikee.selection.tabu import solve_qubo_tabu  # noqa: E402

SEED = 42
SIGMA = 6.2
ALPHA = 1.0
BETA_GRID = [0.25, 0.5, 1.0, 2.0]
K_GRID = [6, 8, 10, 12, 15, 20]
MILP_TIME_LIMIT = 600


def cv_auc(X_train, y_train, cols, seed):
    if len(cols) == 0:
        return 0.5
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X_train[:, cols], y_train, cv=cv, scoring="roc_auc").mean())


def eval_final(name, X_train, X_test, y_train, y_test, cols, seed, classifier="logreg", xgb_params=None):
    if classifier == "logreg":
        model = train_logreg(X_train[:, cols], y_train, seed)
    else:
        model = train_xgboost_fixed(X_train[:, cols], y_train, seed, xgb_params)
    score_train = model.predict_proba(X_train[:, cols])[:, 1]
    score_test = model.predict_proba(X_test[:, cols])[:, 1]
    thr = find_threshold_max_ks(y_train, score_train)
    m = compute_metrics(y_test, score_test, thr)
    lo, hi = bootstrap_ci_auc(y_test, score_test, n_boot=1000, seed=seed)
    m["auc_ci95"] = [lo, hi]
    m["arm"] = name
    return m


def main() -> int:
    seed_df = generate_seed_table(SEED, 2000)
    synth_df, _ = synthesize(seed_df, method="gaussian_copula", n_sample=8000, seed=SEED)
    df = add_target(synth_df, sigma=SIGMA, seed=SEED, target_rate=0.08)

    train_df, test_df = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=SEED)
    y_train, y_test = train_df["default"].to_numpy(), test_df["default"].to_numpy()

    expander = LevelBExpander().fit(train_df, y_train)
    X45_train_df = expander.transform(train_df, seed=SEED)
    X45_test_df = expander.transform(test_df, seed=SEED)

    pre_b = build_preprocessor(ALL_45_COLUMNS, categorical_vars=LEVEL_B_CATEGORICAL_VARS)
    X_train = pre_b.fit_transform(X45_train_df[ALL_45_COLUMNS])
    X_test = pre_b.transform(X45_test_df[ALL_45_COLUMNS])
    variable_groups = get_variable_groups(pre_b)
    gt_labels = ground_truth_labels()

    print(f"n_train={len(train_df)} n_test={len(test_df)} n_vars_nivel_b={len(ALL_45_COLUMNS)}")

    print("\n=== Relevancia y redundancia Nivel B (SOLO train) ===")
    categorical_b = set(LEVEL_B_CATEGORICAL_VARS)
    discrete_flags_vars = categorical_b | {"f36", "f37"}
    R = compute_relevance(X45_train_df, y_train, ALL_45_COLUMNS, discrete_flags_vars, seed=SEED)
    C = compute_redundancy(X45_train_df, ALL_45_COLUMNS, categorical_vars=categorical_b)
    print(f"Top-5 relevancia: {sorted(zip(ALL_45_COLUMNS, R), key=lambda t: -t[1])[:5]}")
    print(f"R[f44] (ruido normal)={R[ALL_45_COLUMNS.index('f44')]:.4f}  "
          f"R[f45] (ruido uniforme)={R[ALL_45_COLUMNS.index('f45')]:.4f}")

    print("\n=== Barrido beta x k Nivel B (24 combinaciones, recocido + CV 5-fold, SOLO train) ===")
    sweep = []
    for beta in BETA_GRID:
        for k in K_GRID:
            lam = compute_lambda(R, C, ALPHA, beta, len(ALL_45_COLUMNS))
            Q = build_qubo(R, C, ALPHA, beta, k, lam=lam)
            sa = solve_qubo_sa(Q, num_reads=200, num_sweeps=500, seed=SEED)
            if not verify_cardinality(sa["sample"], k):
                lam *= 2
                Q = build_qubo(R, C, ALPHA, beta, k, lam=lam)
                sa = solve_qubo_sa(Q, num_reads=200, num_sweeps=500, seed=SEED)
            selected = decode_selection(sa["sample"], ALL_45_COLUMNS)
            cols = sorted(i for v in selected for i in variable_groups[v])
            auc = cv_auc(X_train, y_train, cols, SEED)
            sweep.append({"beta": beta, "k": k, "cv_auc": auc, "selected": selected})

    best = max(sweep, key=lambda r: r["cv_auc"])
    beta_star, k_star = best["beta"], best["k"]
    print(f"Mejor (beta,k) = ({beta_star},{k_star})  CV_AUC={best['cv_auc']:.4f}")

    lam_star = compute_lambda(R, C, ALPHA, beta_star, len(ALL_45_COLUMNS))
    Q_star = build_qubo(R, C, ALPHA, beta_star, k_star, lam=lam_star)

    print(f"\n=== Solucionadores Nivel B sobre Q* (beta={beta_star}, k={k_star}) ===")
    t0 = time.perf_counter()
    c0b = solve_qubo_sa(Q_star, num_reads=1000, num_sweeps=1000, seed=SEED)
    t_c0b = time.perf_counter() - t0
    print(f"C0b (recocido): E={c0b['energy']:.4f}  t={t_c0b:.3f}s  #vars={sum(c0b['sample'].values())}")

    t0 = time.perf_counter()
    c2b = solve_qubo_milp(Q_star, n=len(ALL_45_COLUMNS), cardinality_k=k_star, time_limit=MILP_TIME_LIMIT)
    t_c2b = time.perf_counter() - t0
    print(f"C2b (MILP, límite {MILP_TIME_LIMIT}s): E={c2b['energy']:.4f}  t={t_c2b:.3f}s  "
          f"status={c2b['status']} mip_gap={c2b.get('mip_gap')}")

    t0 = time.perf_counter()
    c4b = solve_qubo_tabu(Q_star, num_reads=200, seed=SEED)
    t_c4b = time.perf_counter() - t0
    print(f"C4b (Tabu): E={c4b['energy']:.4f}  t={t_c4b:.3f}s  #vars={sum(c4b['sample'].values())}")

    sa_matches_milp = c0b["sample"] == c2b["sample"] if c2b["sample"] else False
    tabu_matches_milp = c4b["sample"] == c2b["sample"] if c2b["sample"] else False
    print(f"\nC0b == C2b: {sa_matches_milp}   C4b == C2b: {tabu_matches_milp}")

    print("\n=== Baseline aleatorio Rb (k al azar x 100) ===")
    random_subsets = random_variable_subsets(ALL_45_COLUMNS, k_star, n_repetitions=100, seed=SEED)
    random_aucs = [cv_auc(X_train, y_train, sorted(i for v in s for i in variable_groups[v]), SEED) for s in random_subsets]
    print(f"AUC aleatorio: media={np.mean(random_aucs):.4f} dp={np.std(random_aucs):.4f}")

    print("\n=== Brazos sin selección / clásicos Nivel B ===")
    ncv = nested_cv_xgboost(X_train, y_train, seed=SEED, outer_folds=5, inner_folds=3, n_iter=20)
    xgb_params = median_hyperparams(ncv["best_params_per_fold"])
    all_cols = list(range(X_train.shape[1]))

    results = {}
    results["A0b"] = eval_final("A0b", X_train, X_test, y_train, y_test, all_cols, SEED, "logreg")
    results["A1b"] = eval_final("A1b", X_train, X_test, y_train, y_test, all_cols, SEED, "xgboost", xgb_params)

    lasso = lasso_select(X_train, y_train, variable_groups, SEED)
    results["B0b"] = eval_final("B0b", X_train, X_test, y_train, y_test, lasso["selected_columns"], SEED)
    results["B0b"]["selected_variables"] = lasso["selected_variables"]

    stepwise = stepwise_forward_aic(X_train, y_train, variable_groups)
    results["B1b"] = eval_final("B1b", X_train, X_test, y_train, y_test, stepwise["selected_columns"], SEED)
    results["B1b"]["selected_variables"] = stepwise["selected_variables"]

    for name, sample in (("C0b", c0b["sample"]), ("C2b", c2b["sample"]), ("C4b", c4b["sample"])):
        if not sample:
            continue
        selected = decode_selection(sample, ALL_45_COLUMNS)
        cols = sorted(i for v in selected for i in variable_groups[v])
        results[name] = eval_final(name, X_train, X_test, y_train, y_test, cols, SEED)
        results[name]["selected_variables"] = selected
        results[name]["noise_columns_selected"] = [v for v in selected if gt_labels.get(v) == "irrelevant"]

    print("\n--- Resultados Nivel B ---")
    for arm, m in results.items():
        n_v = len(m.get("selected_variables", ALL_45_COLUMNS))
        print(f"{arm:6s} AUC={m['auc']:.4f} [{m['auc_ci95'][0]:.4f},{m['auc_ci95'][1]:.4f}]  #vars={n_v}")

    for arm in ("C0b", "C2b", "C4b"):
        if arm in results and results[arm].get("noise_columns_selected"):
            print(f"ALERTA: {arm} incluyó ruido puro: {results[arm]['noise_columns_selected']}")

    out_dir = Path(__file__).resolve().parents[1] / "reports" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f5_level_b.json").write_text(json.dumps({
        "beta_star": beta_star, "k_star": k_star,
        "sweep": sweep,
        "solver_times": {"C0b": t_c0b, "C2b": t_c2b, "C4b": t_c4b},
        "milp_status": {"status": c2b["status"], "mip_gap": c2b.get("mip_gap"), "success": c2b["success"]},
        "sa_matches_milp": sa_matches_milp, "tabu_matches_milp": tabu_matches_milp,
        "random_baseline_auc": {"mean": float(np.mean(random_aucs)), "std": float(np.std(random_aucs))},
        "results": results,
    }, indent=2, default=str))
    print(f"\nGuardado en {out_dir / 'f5_level_b.json'}")
    print("\nCHECKPOINT F5: curva tiempo-vs-calidad generada (ver JSON). QAOA Nivel B: imposible (563TB), no se ejecuta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
