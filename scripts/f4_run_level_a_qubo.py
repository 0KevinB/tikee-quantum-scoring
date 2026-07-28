"""Checkpoint F4 (PLAN.md §4.1, INNEGOCIABLE): C1 (ExactSolver) y C2 (MILP) deben
devolver EXACTAMENTE la misma solución en Nivel A (N=18). Si no coinciden, la
linealización de Glover o lambda tienen un error y no se avanza.

Uso: .venv/bin/python scripts/f4_run_level_a_qubo.py
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
from tikee.features.preprocess import CATEGORICAL_VARS, LEVEL_A_VARS, build_preprocessor, get_variable_groups  # noqa: E402
from tikee.models.evaluate import bootstrap_ci_auc, compute_metrics, find_threshold_max_ks  # noqa: E402
from tikee.selection.annealer import solve_qubo_sa  # noqa: E402
from tikee.selection.exact import solve_qubo_exact  # noqa: E402
from tikee.selection.milp import solve_qubo_milp  # noqa: E402
from tikee.selection.qaoa import solve_qubo_qaoa  # noqa: E402
from tikee.selection.qubo_builder import build_qubo, compute_lambda, decode_selection, verify_cardinality  # noqa: E402
from tikee.selection.random_baseline import random_variable_subsets  # noqa: E402
from tikee.selection.redundancy import compute_redundancy  # noqa: E402
from tikee.selection.relevance import compute_relevance  # noqa: E402

SEED = 42
SIGMA = 6.2
ALPHA = 1.0
BETA_GRID = [0.25, 0.5, 1.0, 2.0]
K_GRID = [5, 6, 7, 8, 9, 10]


def cv_auc_for_selection(X_train, y_train, cols, seed):
    if len(cols) == 0:
        return 0.5
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_train[:, cols], y_train, cv=cv, scoring="roc_auc")
    return float(scores.mean())


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

    print("=== Matrices de relevancia y redundancia (SOLO train) ===")
    R = compute_relevance(train_df, y_train, LEVEL_A_VARS, set(CATEGORICAL_VARS), seed=SEED)
    C = compute_redundancy(train_df, LEVEL_A_VARS, set(CATEGORICAL_VARS))
    print(f"R: {dict(zip(LEVEL_A_VARS, np.round(R, 3)))}")

    print("\n=== Barrido beta x k (24 combinaciones, AUC en CV de 5 pliegues, SOLO train) ===")
    sweep_results = []
    for beta in BETA_GRID:
        for k in K_GRID:
            lam = compute_lambda(R, C, ALPHA, beta, len(LEVEL_A_VARS))
            Q = build_qubo(R, C, ALPHA, beta, k, lam=lam)
            sa = solve_qubo_sa(Q, num_reads=200, num_sweeps=200, seed=SEED)
            if not verify_cardinality(sa["sample"], k):
                lam *= 2
                Q = build_qubo(R, C, ALPHA, beta, k, lam=lam)
                sa = solve_qubo_sa(Q, num_reads=200, num_sweeps=200, seed=SEED)
            selected_vars = decode_selection(sa["sample"], LEVEL_A_VARS)
            cols = sorted(i for v in selected_vars for i in variable_groups[v])
            auc = cv_auc_for_selection(X_train, y_train, cols, SEED)
            sweep_results.append({"beta": beta, "k": k, "cv_auc": auc, "selected_variables": selected_vars})

    best = max(sweep_results, key=lambda r: r["cv_auc"])
    print(f"Mejor (beta, k) = ({best['beta']}, {best['k']})  CV_AUC={best['cv_auc']:.4f}")
    print(f"Variables: {best['selected_variables']}")

    beta_star, k_star = best["beta"], best["k"]
    lam_star = compute_lambda(R, C, ALPHA, beta_star, len(LEVEL_A_VARS))
    Q_star = build_qubo(R, C, ALPHA, beta_star, k_star, lam=lam_star)

    print(f"\n=== Solucionadores sobre Q* (beta={beta_star}, k={k_star}, lambda={lam_star:.3f}) ===")

    t0 = time.perf_counter()
    c0 = solve_qubo_sa(Q_star, num_reads=1000, num_sweeps=1000, seed=SEED)
    print(f"C0 (recocido):  {sorted(decode_selection(c0['sample'], LEVEL_A_VARS))}  E={c0['energy']:.4f}  "
          f"t={time.perf_counter()-t0:.3f}s")

    t0 = time.perf_counter()
    c1 = solve_qubo_exact(Q_star, n_vars=len(LEVEL_A_VARS))
    print(f"C1 (ExactSolver): {sorted(decode_selection(c1['sample'], LEVEL_A_VARS))}  E={c1['energy']:.4f}  "
          f"t={time.perf_counter()-t0:.3f}s")

    t0 = time.perf_counter()
    c2 = solve_qubo_milp(Q_star, n=len(LEVEL_A_VARS))
    print(f"C2 (MILP-Glover): {sorted(decode_selection(c2['sample'], LEVEL_A_VARS))}  E={c2['energy']:.4f}  "
          f"t={time.perf_counter()-t0:.3f}s  status={c2['status']}")

    c2_hard = solve_qubo_milp(Q_star, n=len(LEVEL_A_VARS), cardinality_k=k_star)
    print(f"C2 (MILP hard-k): {sorted(decode_selection(c2_hard['sample'], LEVEL_A_VARS))}  E={c2_hard['energy']:.4f}")

    print("\n--- CHECKPOINT F4 (innegociable) ---")
    c1_c2_match = c1["sample"] == c2["sample"]
    c2_penalized_vs_hard_match = c2["sample"] == c2_hard["sample"]
    print(f"C1 == C2 (penalizado, sin restricción dura): {c1_c2_match}")
    print(f"C2 (penalizado) == C2 (restricción dura Sigma x=k): {c2_penalized_vs_hard_match}")
    if not c1_c2_match:
        print("\nCHECKPOINT F4: FALLA. C1 y C2 no coinciden. La formulación tiene un error. NO SE AVANZA.")
        return 1

    print("\n=== QAOA (Nivel A, N=18, p in {1,2,3}) ===")
    qaoa_results = {}
    for p in (1, 2, 3):
        t0 = time.perf_counter()
        try:
            c3 = solve_qubo_qaoa(Q_star, n=len(LEVEL_A_VARS), p=p, seed=SEED, maxiter=150, n_restarts=2)
            wall = time.perf_counter() - t0
            gap = abs(c3["energy"] - c1["energy"]) / max(abs(c1["energy"]), 1e-9)
            print(f"QAOA p={p}: E={c3['energy']:.4f}  gap_vs_optimo={gap:.4f}  t={wall:.1f}s")
            qaoa_results[p] = {**c3, "gap_vs_optimal": gap, "wall_time_s": wall}
        except Exception as exc:  # noqa: BLE001
            print(f"QAOA p={p}: FALLÓ ({exc})")
            qaoa_results[p] = {"error": str(exc)}

    print("\n=== Baseline aleatorio R (k al azar x 100) ===")
    random_subsets = random_variable_subsets(LEVEL_A_VARS, k_star, n_repetitions=100, seed=SEED)
    random_aucs = []
    for subset in random_subsets:
        cols = sorted(i for v in subset for i in variable_groups[v])
        random_aucs.append(cv_auc_for_selection(X_train, y_train, cols, SEED))
    print(f"AUC aleatorio: media={np.mean(random_aucs):.4f} dp={np.std(random_aucs):.4f}")

    def eval_final(name, selected_vars):
        cols = sorted(i for v in selected_vars for i in variable_groups[v])
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(X_train[:, cols], y_train)
        score_train = clf.predict_proba(X_train[:, cols])[:, 1]
        score_test = clf.predict_proba(X_test[:, cols])[:, 1]
        thr = find_threshold_max_ks(y_train, score_train)
        m = compute_metrics(y_test, score_test, thr)
        lo, hi = bootstrap_ci_auc(y_test, score_test, n_boot=1000, seed=SEED)
        m["auc_ci95"] = [lo, hi]
        m["selected_variables"] = selected_vars
        m["arm"] = name
        return m

    final_results = {
        "C0": eval_final("C0", decode_selection(c0["sample"], LEVEL_A_VARS)),
        "C1": eval_final("C1", decode_selection(c1["sample"], LEVEL_A_VARS)),
        "C2": eval_final("C2", decode_selection(c2["sample"], LEVEL_A_VARS)),
    }
    for p, res in qaoa_results.items():
        if "sample" in res:
            final_results[f"C3_p{p}"] = eval_final(f"C3_p{p}", decode_selection(res["sample"], LEVEL_A_VARS))

    print("\n--- Resultados brazos QUBO Nivel A ---")
    for arm, m in final_results.items():
        print(f"{arm:8s} AUC={m['auc']:.4f} [{m['auc_ci95'][0]:.4f},{m['auc_ci95'][1]:.4f}]  #vars={len(m['selected_variables'])}")

    out_dir = Path(__file__).resolve().parents[1] / "reports" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f4_level_a_qubo.json").write_text(json.dumps({
        "beta_star": beta_star, "k_star": k_star, "lambda_star": lam_star,
        "sweep_results": sweep_results,
        "checkpoint_f4_c1_eq_c2": c1_c2_match,
        "c2_penalized_eq_hard": c2_penalized_vs_hard_match,
        "solutions": {
            "C0": {"sample": c0["sample"], "energy": c0["energy"]},
            "C1": {"sample": c1["sample"], "energy": c1["energy"]},
            "C2": {"sample": c2["sample"], "energy": c2["energy"]},
        },
        "qaoa": {str(p): {k: v for k, v in r.items() if k != "sample"} for p, r in qaoa_results.items()},
        "random_baseline_auc": {"mean": float(np.mean(random_aucs)), "std": float(np.std(random_aucs))},
        "final_results": final_results,
    }, indent=2, default=str))
    print(f"\nGuardado en {out_dir / 'f4_level_a_qubo.json'}")
    print("\nCHECKPOINT F4: PASA (C1 == C2 exactamente)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
