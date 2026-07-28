"""Checkpoint F7 (PLAN.md §4.1): tabla de equidad por brazo y grupo + detección de
proxies. ARCHITECTURE.md §9.

Usa las variables seleccionadas por cada brazo en la semilla 42 (cacheadas por F6 en
reports/cache/results_seed42.joblib) y re-entrena el clasificador para poder evaluar
sobre el conjunto de test con las columnas protegidas, que `run_experiment.py` no
expone (para no inflar el caché de las 10 semillas con columnas de equidad que solo
se necesitan una vez).

Uso: .venv/bin/python scripts/f7_fairness_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402
from tikee.features.expand import ALL_45_COLUMNS, LevelBExpander  # noqa: E402
from tikee.features.preprocess import LEVEL_A_VARS, LEVEL_B_CATEGORICAL_VARS, build_preprocessor, get_variable_groups  # noqa: E402
from tikee.models.evaluate import find_threshold_max_ks  # noqa: E402
from tikee.models.fairness import group_metrics, proxy_detection_auc  # noqa: E402
from tikee.models.train import train_logreg  # noqa: E402

SEED = 42
SIGMA = 6.2
PROTECTED_ATTRS = ["sexo", "zona_residencia", "provincia"]
ARMS_LEVEL_A = ["A0", "B0", "C0", "C1", "C2"]
ARMS_LEVEL_B = ["A0b", "B0b", "C0b", "C2b"]


def audit_arm(name, selected_vars, X_train, X_test, y_train, y_test, variable_groups, test_df):
    cols = sorted(i for v in selected_vars for i in variable_groups.get(v, []))
    if not cols:
        cols = list(range(X_train.shape[1]))
    model = train_logreg(X_train[:, cols], y_train, SEED)
    score_train = model.predict_proba(X_train[:, cols])[:, 1]
    score_test = model.predict_proba(X_test[:, cols])[:, 1]
    threshold = find_threshold_max_ks(y_train, score_train)

    fairness_by_group = {}
    for attr in PROTECTED_ATTRS:
        fairness_by_group[attr] = group_metrics(y_test, score_test, threshold, test_df[attr])

    proxy_aucs = {}
    for attr in PROTECTED_ATTRS:
        groups = test_df[attr]
        if groups.nunique() != 2:
            # colapsar a binario mas frecuente vs resto para provincia (5 categorias)
            top = groups.value_counts().idxmax()
            groups = (groups == top).map({True: top, False: "otras"})
        try:
            proxy_aucs[attr] = proxy_detection_auc(X_test[:, cols], groups, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            proxy_aucs[attr] = {"error": str(exc)}

    return {
        "arm": name, "n_vars": len(selected_vars) if selected_vars else X_train.shape[1],
        "fairness_by_group": fairness_by_group, "proxy_detection_auc": proxy_aucs,
        "_model": model, "_cols": cols, "_threshold": threshold, "_score_test": score_test,
    }


def main() -> int:
    cache_path = Path(__file__).resolve().parents[1] / "reports" / "cache" / "results_seed42.joblib"
    if not cache_path.exists():
        print(f"Falta {cache_path}; corre scripts/f6_run_multiseed.py primero.", file=sys.stderr)
        return 1
    # joblib.load deserializa con pickle; seguro aquí porque el único archivo leído es
    # el caché local que scripts/f6_run_multiseed.py escribió en esta misma máquina.
    cached = joblib.load(cache_path)

    seed_df = generate_seed_table(SEED, 2000)
    synth_df, _ = synthesize(seed_df, method="gaussian_copula", n_sample=8000, seed=SEED)
    df = add_target(synth_df, sigma=SIGMA, seed=SEED, target_rate=0.08)
    train_df, test_df = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=SEED)
    y_train, y_test = train_df["default"].to_numpy(), test_df["default"].to_numpy()
    test_df = test_df.reset_index(drop=True)

    pre_a = build_preprocessor(LEVEL_A_VARS)
    Xa_train = pre_a.fit_transform(train_df[LEVEL_A_VARS])
    Xa_test = pre_a.transform(test_df[LEVEL_A_VARS])
    vgroups_a = get_variable_groups(pre_a)

    expander = LevelBExpander().fit(train_df, y_train)
    Xb_train_df = expander.transform(train_df, seed=SEED)
    Xb_test_df = expander.transform(test_df, seed=SEED)
    pre_b = build_preprocessor(ALL_45_COLUMNS, categorical_vars=LEVEL_B_CATEGORICAL_VARS)
    Xb_train = pre_b.fit_transform(Xb_train_df[ALL_45_COLUMNS])
    Xb_test = pre_b.transform(Xb_test_df[ALL_45_COLUMNS])
    vgroups_b = get_variable_groups(pre_b)

    report = {"level_a": {}, "level_b": {}}

    print("=== Equidad Nivel A ===")
    for arm in ARMS_LEVEL_A:
        sel = cached[arm].get("selected_variables", [])
        res = audit_arm(arm, sel, Xa_train, Xa_test, y_train, y_test, vgroups_a, test_df)
        report["level_a"][arm] = res
        dpd = {k: v["demographic_parity_diff"] for k, v in res["fairness_by_group"].items()}
        proxy = res["proxy_detection_auc"]
        print(f"{arm:5s} #vars={res['n_vars']:2d}  paridad_demografica={dpd}  proxy_auc={proxy}")

    print("\n=== Equidad Nivel B ===")
    for arm in ARMS_LEVEL_B:
        sel = cached[arm].get("selected_variables", [])
        res = audit_arm(arm, sel, Xb_train, Xb_test, y_train, y_test, vgroups_b, test_df)
        report["level_b"][arm] = res
        dpd = {k: v["demographic_parity_diff"] for k, v in res["fairness_by_group"].items()}
        proxy = res["proxy_detection_auc"]
        print(f"{arm:5s} #vars={res['n_vars']:2d}  paridad_demografica={dpd}  proxy_auc={proxy}")

    print("\n--- Pregunta central F7: ¿QUBO introduce más o menos sesgo indirecto que LASSO? ---")
    for level, arms in (("level_a", ("B0", "C0")), ("level_b", ("B0b", "C0b"))):
        if arms[0] in report[level] and arms[1] in report[level]:
            lasso_proxy = report[level][arms[0]]["proxy_detection_auc"]
            qubo_proxy = report[level][arms[1]]["proxy_detection_auc"]
            print(f"{level}: LASSO proxy_auc={lasso_proxy}  QUBO proxy_auc={qubo_proxy}")

    # Persistir modelos + datos de test para la app (§11: "no entrena nada en vivo
    # salvo el simulador, que carga un modelo ya serializado"). El bundle de abajo
    # también alimenta las páginas 2/3 (ROC, matrices de confusión) por inferencia
    # sobre modelos ya entrenados, sin re-entrenar nada al cargar la app.
    artifacts_dir = Path(__file__).resolve().parents[1] / "reports" / "cache"
    app_artifacts = {
        "level_a": {
            "models": {arm: {"model": report["level_a"][arm]["_model"], "cols": report["level_a"][arm]["_cols"],
                              "threshold": report["level_a"][arm]["_threshold"]} for arm in ARMS_LEVEL_A},
            "X_test": Xa_test, "y_test": y_test, "vgroups": vgroups_a, "variables": LEVEL_A_VARS,
        },
        "level_b": {
            "models": {arm: {"model": report["level_b"][arm]["_model"], "cols": report["level_b"][arm]["_cols"],
                              "threshold": report["level_b"][arm]["_threshold"]} for arm in ARMS_LEVEL_B},
            "X_test": Xb_test, "y_test": y_test, "vgroups": vgroups_b, "variables": ALL_45_COLUMNS,
        },
        "test_df_raw": test_df,
        "train_df_raw": train_df,
    }
    joblib.dump(app_artifacts, artifacts_dir / "app_artifacts.joblib")
    print(f"Modelos serializados para la app en {artifacts_dir / 'app_artifacts.joblib'}")

    # limpiar objetos no serializables antes de volcar a JSON
    for level in ("level_a", "level_b"):
        for arm in report[level]:
            for k in ("_model", "_cols", "_threshold", "_score_test"):
                report[level][arm].pop(k, None)

    out_path = artifacts_dir / "f7_fairness.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nGuardado en {out_path}")
    print("\nCHECKPOINT F7: tabla de equidad + detección de proxies generada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
