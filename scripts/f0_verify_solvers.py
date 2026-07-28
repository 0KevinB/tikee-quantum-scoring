"""Checkpoint F0 (PLAN.md §4.1): los cuatro solucionadores coinciden en un QUBO de
juguete de 4 variables cuyo óptimo se conoce por enumeración exhaustiva de los 2^4=16
estados (equivalente a resolverlo "a mano" a esta escala).

Uso: .venv/bin/python scripts/f0_verify_solvers.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tikee.selection.annealer import solve_qubo_sa  # noqa: E402
from tikee.selection.exact import solve_qubo_exact  # noqa: E402
from tikee.selection.milp import solve_qubo_milp  # noqa: E402
from tikee.selection.qaoa import solve_qubo_qaoa  # noqa: E402

N = 4
K = 2
ALPHA = 1.0
BETA = 1.0

R = [0.9, 0.5, 0.8, 0.2]
C = {
    (0, 1): 0.9, (0, 2): 0.1, (0, 3): 0.2,
    (1, 2): 0.2, (1, 3): 0.1,
    (2, 3): 0.3,
}

LAMBDA = 2 * (ALPHA * max(R) + BETA * max(C.values()) * N)


def build_toy_qubo() -> dict[tuple[int, int], float]:
    Q: dict[tuple[int, int], float] = {}
    for i in range(N):
        Q[(i, i)] = -ALPHA * R[i] + LAMBDA * (1 - 2 * K)
    for (i, j), cij in C.items():
        Q[(i, j)] = BETA * cij + 2 * LAMBDA
    return Q


def energy(Q: dict[tuple[int, int], float], x: tuple[int, ...]) -> float:
    e = 0.0
    for (i, j), qij in Q.items():
        if i == j:
            e += qij * x[i]
        else:
            e += qij * x[i] * x[j]
    return e


def brute_force_ground_truth(Q: dict[tuple[int, int], float]) -> dict:
    best_x, best_e = None, float("inf")
    for x in itertools.product([0, 1], repeat=N):
        e = energy(Q, x)
        if e < best_e:
            best_e, best_x = e, x
    return {"sample": {i: best_x[i] for i in range(N)}, "energy": best_e}


def samples_match(a: dict[int, int], b: dict[int, int]) -> bool:
    return all(int(a[i]) == int(b[i]) for i in range(N))


def main() -> int:
    Q = build_toy_qubo()
    truth = brute_force_ground_truth(Q)
    print(f"Ground truth (enumeración 2^{N}=16 estados): {truth}")
    print(f"lambda usada: {LAMBDA:.4f}\n")

    results = {}

    sa = solve_qubo_sa(Q, num_reads=200, num_sweeps=200, seed=42)
    results["simulated_annealing"] = sa
    print(f"SA     -> sample={sa['sample']} energy={sa['energy']:.4f} "
          f"({sa['wall_time_s']*1000:.1f} ms)")

    ex = solve_qubo_exact(Q, n_vars=N)
    results["exact_solver"] = ex
    print(f"Exact  -> sample={ex['sample']} energy={ex['energy']:.4f} "
          f"({ex['wall_time_s']*1000:.1f} ms)")

    mi = solve_qubo_milp(Q, n=N)
    results["milp_highs"] = mi
    print(f"MILP   -> sample={mi['sample']} energy={mi['energy']:.4f} "
          f"({mi['wall_time_s']*1000:.1f} ms) status={mi['status']}")

    try:
        qa = solve_qubo_qaoa(Q, n=N, p=1, seed=42, maxiter=300)
        results["qaoa_p1"] = qa
        print(f"QAOA   -> sample={qa['sample']} energy={qa['energy']:.4f} "
              f"({qa['wall_time_s']:.2f} s, p=1)")
        qaoa_ok = samples_match(qa["sample"], truth["sample"]) or abs(qa["energy"] - truth["energy"]) < 1e-6
    except Exception as exc:  # noqa: BLE001
        print(f"QAOA   -> FALLÓ: {exc}")
        results["qaoa_p1"] = {"error": str(exc)}
        qaoa_ok = False

    checks = {
        "sa_matches_truth": samples_match(sa["sample"], truth["sample"]),
        "exact_matches_truth": samples_match(ex["sample"], truth["sample"]),
        "milp_matches_truth": mi["sample"] is not None and samples_match(mi["sample"], truth["sample"]),
        "qaoa_matches_truth": qaoa_ok,
    }

    print("\n--- Checkpoint F0 ---")
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'FALLA'}")

    all_ok = all(checks.values())
    print(f"\nCHECKPOINT F0: {'PASA' if all_ok else 'FALLA'}")

    out_path = Path(__file__).resolve().parents[1] / "reports" / "cache" / "f0_checkpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ground_truth": truth,
        "checks": checks,
        "all_ok": all_ok,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "sample" or True} for k, v in results.items()},
    }, indent=2, default=str))
    print(f"Detalle guardado en {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
