"""Linealización de Glover -> MILP -> HiGHS (vía scipy.optimize.milp). Brazos C2 / C2b.

Dos usos (ARCHITECTURE.md §7.3), ambos soportados por `solve_qubo_milp`:
1. Con `cardinality_k` fijado -> óptimo certificado del problema con restricción DURA.
2. Sin `cardinality_k` (None) -> resuelve el QUBO penalizado tal cual, para la prueba
   cruzada contra 1: deben coincidir si lambda es suficiente.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def _offdiag_pairs(Q: dict[tuple[int, int], float], n: int) -> list[tuple[int, int, float]]:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q.get((i, j), 0.0)
            if qij == 0.0:
                qij = Q.get((j, i), 0.0)
            if qij != 0.0:
                pairs.append((i, j, qij))
    return pairs


def solve_qubo_milp(
    Q: dict[tuple[int, int], float],
    n: int,
    cardinality_k: int | None = None,
    time_limit: float | None = None,
) -> dict[str, Any]:
    """Resuelve el QUBO como MILP vía la linealización de Glover (y_ij = x_i·x_j),
    con HiGHS como backend de `scipy.optimize.milp`. Óptimo certificado, o gap
    acotado si se corta por `time_limit` (ARCHITECTURE.md §7.3).

    Args:
        Q: matriz QUBO en formato `{(i,i): valor, (i,j): valor}`.
        n: número de variables binarias originales (sin contar las auxiliares
            y_ij que introduce la linealización).
        cardinality_k: si se da, se agrega `Σx_i = k` como restricción DURA (uso
            1 de §7.3: óptimo certificado del problema real). Si es None, se
            resuelve el QUBO penalizado tal cual (uso 2: prueba cruzada de lambda
            contra el uso 1 — deben coincidir).
        time_limit: límite de tiempo en segundos para HiGHS; si se corta antes de
            cerrar, `mip_gap` en el resultado acota qué tan lejos puede estar la
            solución del óptimo verdadero.

    Returns:
        dict con `sample` (o None si no hay solución factible), `energy`,
        `wall_time_s`, `status`, `success`, `mip_gap` y `message`.
    """
    pairs = _offdiag_pairs(Q, n)
    n_y = len(pairs)
    n_vars = n + n_y

    c = np.zeros(n_vars)
    for i in range(n):
        c[i] = Q.get((i, i), 0.0)
    for idx, (_, _, qij) in enumerate(pairs):
        c[n + idx] = qij

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    for idx, (i, j, qij) in enumerate(pairs):
        y = n + idx
        if qij > 0:
            row = np.zeros(n_vars)
            row[i] = -1
            row[j] = -1
            row[y] = 1
            rows.append(row)
            lb.append(-1.0)
            ub.append(np.inf)
        else:
            row1 = np.zeros(n_vars)
            row1[i] = 1
            row1[y] = -1
            rows.append(row1)
            lb.append(0.0)
            ub.append(np.inf)

            row2 = np.zeros(n_vars)
            row2[j] = 1
            row2[y] = -1
            rows.append(row2)
            lb.append(0.0)
            ub.append(np.inf)

    constraints = []
    if rows:
        constraints.append(LinearConstraint(np.vstack(rows), lb, ub))

    if cardinality_k is not None:
        row = np.zeros(n_vars)
        row[:n] = 1
        constraints.append(LinearConstraint(row, cardinality_k, cardinality_k))

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)

    options: dict[str, Any] = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = time_limit

    t0 = time.perf_counter()
    res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)
    wall = time.perf_counter() - t0

    if res.x is None:
        return {
            "sample": None,
            "energy": None,
            "wall_time_s": wall,
            "status": res.status,
            "success": False,
            "message": res.message,
        }

    x = np.round(res.x[:n]).astype(int)
    return {
        "sample": {i: int(x[i]) for i in range(n)},
        "energy": float(res.fun),
        "wall_time_s": wall,
        "status": res.status,
        "success": bool(res.success),
        "mip_gap": getattr(res, "mip_gap", None),
        "message": res.message,
    }
