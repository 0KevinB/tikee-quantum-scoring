"""dimod.ExactSolver: techo de optimalidad por enumeración. Solo N<=20. Brazo C1."""

from __future__ import annotations

import time
from typing import Any

import dimod


def solve_qubo_exact(Q: dict[tuple[int, int], float], n_vars: int) -> dict[str, Any]:
    """Enumera todos los 2^n_vars estados y devuelve el óptimo global exacto.

    Args:
        Q: matriz QUBO en formato `{(i,i): valor, (i,j): valor}`.
        n_vars: número de variables binarias; debe ser <=20 (2^20 ≈ 1M estados,
            el límite práctico de `dimod.ExactSolver`).

    Returns:
        dict con `sample` (solución óptima), `energy`, `wall_time_s` y
        `n_states_enumerated`.

    Raises:
        ValueError: si n_vars > 20.
    """
    if n_vars > 20:
        raise ValueError(f"ExactSolver solo es viable para N<=20 (2^N estados); N={n_vars}")

    t0 = time.perf_counter()
    result = dimod.ExactSolver().sample_qubo(Q).first
    wall = time.perf_counter() - t0

    return {
        "sample": {int(k): int(v) for k, v in result.sample.items()},
        "energy": float(result.energy),
        "wall_time_s": wall,
        "n_states_enumerated": 2**n_vars,
    }
