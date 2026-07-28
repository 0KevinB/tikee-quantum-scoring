"""QUBO -> QuadraticProgram -> circuito QAOA, expectativa exacta por vector de estado.

Solo N<=24 (memoria de amplitudes, ARCHITECTURE.md §2.4 y §7.4). Brazo C3.

Nota de implementación (D18): se usa `qiskit.primitives.Sampler` (la implementación de
referencia de Terra), que calcula probabilidades exactas por simulación de vector de
estado, sin ruido de muestreo — equivalente en exactitud a `AerSimulator(method=
"statevector")` para los tamaños de este proyecto (N<=24) y sin depender de opciones
de shots de Aer. Si `qiskit-aer` está disponible se puede pasar `sampler` propio desde
`qiskit_aer.primitives.Sampler(run_options={"shots": None})` sin cambiar esta función.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

MAX_N_QAOA = 24


def _qubo_to_quadratic_program(Q: dict[tuple[int, int], float], n: int):
    from qiskit_optimization import QuadraticProgram

    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {i: Q.get((i, i), 0.0) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q.get((i, j), 0.0)
            if qij == 0.0:
                qij = Q.get((j, i), 0.0)
            if qij != 0.0:
                quadratic[(i, j)] = qij

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def solve_qubo_qaoa(
    Q: dict[tuple[int, int], float],
    n: int,
    p: int = 1,
    seed: int | None = None,
    sampler: Any = None,
    maxiter: int = 200,
    n_restarts: int = 1,
) -> dict[str, Any]:
    """COBYLA es un optimizador local: `n_restarts` > 1 relanza desde puntos iniciales
    aleatorios distintos y se queda con la energía más baja. No cambia el paradigma
    (sigue siendo QAOA con expectativa exacta por vector de estado, D18); solo mitiga
    el riesgo de mínimos locales del bucle clásico externo."""
    if n > MAX_N_QAOA:
        raise ValueError(f"QAOA simulado solo es viable hasta N<={MAX_N_QAOA}; N={n}")

    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    if seed is not None:
        algorithm_globals.random_seed = seed
    rng = np.random.default_rng(seed)

    if sampler is None:
        from qiskit.primitives import Sampler

        sampler = Sampler()

    qp = _qubo_to_quadratic_program(Q, n)

    best_result = None
    total_wall = 0.0
    total_evals = 0
    for _ in range(max(1, n_restarts)):
        initial_point = rng.uniform(0, 2 * np.pi, size=2 * p)
        qaoa = QAOA(
            sampler=sampler,
            optimizer=COBYLA(maxiter=maxiter),
            reps=p,
            initial_point=initial_point,
        )
        optimizer = MinimumEigenOptimizer(qaoa)

        t0 = time.perf_counter()
        result = optimizer.solve(qp)
        total_wall += time.perf_counter() - t0

        raw = getattr(result, "min_eigen_solver_result", None)
        if raw is not None:
            total_evals += getattr(raw, "cost_function_evals", 0) or 0

        if best_result is None or result.fval < best_result.fval:
            best_result = result

    x = [int(round(v)) for v in best_result.x]
    return {
        "sample": {i: x[i] for i in range(n)},
        "energy": float(best_result.fval),
        "wall_time_s": total_wall,
        "p": p,
        "n_evaluations": total_evals or None,
        "n_restarts": n_restarts,
    }
