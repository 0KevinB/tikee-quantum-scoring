"""Bucle sobre las 10 semillas con caché en disco (joblib). ARCHITECTURE.md §3, §8.5.

Los hiperparámetros de XGBoost se fijan UNA vez (no se re-afinan por semilla, D13) y
se pasan a `run_seed`.

Nota de seguridad: `joblib.load` deserializa con pickle. Aquí es seguro porque los
únicos archivos leídos son los que este mismo módulo escribió previamente en
`reports/cache/` (caché local propia, nunca datos de terceros ni de red)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tikee.experiments.run_experiment import run_seed  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[3] / "reports" / "cache"


def run_multiseed(
    seeds: list[int], xgb_params_a: dict, xgb_params_b: dict, force: bool = False
) -> dict[int, dict]:
    """Corre `run_experiment.run_seed` para cada semilla en `seeds`, cacheando el
    resultado de cada una en `reports/cache/results_seed{N}.joblib` para poder
    reanudar sin recomputar (riesgo R7 de PLAN.md).

    Args:
        seeds: lista de semillas a correr (D11: 10 semillas para el protocolo
            principal, `config.yaml: seeds.multiseed`).
        xgb_params_a: hiperparámetros fijos de XGBoost, Nivel A.
        xgb_params_b: hiperparámetros fijos de XGBoost, Nivel B.
        force: si True, recomputa aunque exista caché (ignora los `.joblib` ya
            guardados).

    Returns:
        dict `{semilla: resultado de run_seed}`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for seed in seeds:
        cache_path = CACHE_DIR / f"results_seed{seed}.joblib"
        if cache_path.exists() and not force:
            print(f"[seed {seed}] cache encontrado, se reutiliza: {cache_path}")
            all_results[seed] = joblib.load(cache_path)
            continue

        print(f"[seed {seed}] corriendo pipeline completo...")
        t0 = time.perf_counter()
        result = run_seed(seed, xgb_params_a, xgb_params_b)
        wall = time.perf_counter() - t0
        result["wall_time_s"] = wall
        joblib.dump(result, cache_path)
        print(f"[seed {seed}] listo en {wall:.1f}s -> {cache_path}")
        all_results[seed] = result

    return all_results
