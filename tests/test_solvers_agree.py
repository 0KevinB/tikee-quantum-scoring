"""QUBO de 8 vars: recocido, ExactSolver, MILP y QAOA p=3 coinciden.

La prueba de regresión más importante del repositorio (ARCHITECTURE.md §12):
separa el mérito de la formulación QUBO del mérito de cada solucionador. Si esta
prueba falla, la linealización de Glover o el mapeo a Ising tienen un error y
nada de lo que se mida después con estos solucionadores es confiable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tikee.selection.annealer import solve_qubo_sa
from tikee.selection.exact import solve_qubo_exact
from tikee.selection.milp import solve_qubo_milp
from tikee.selection.qaoa import solve_qubo_qaoa

N = 8
K = 4
ALPHA = 1.0
BETA = 1.0


def _build_toy_qubo(seed: int = 7) -> dict[tuple[int, int], float]:
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.1, 1.0, size=N)
    C = {}
    for i in range(N):
        for j in range(i + 1, N):
            C[(i, j)] = float(rng.uniform(0.0, 1.0))

    lam = 2 * (ALPHA * R.max() + BETA * max(C.values()) * N)

    Q: dict[tuple[int, int], float] = {}
    for i in range(N):
        Q[(i, i)] = -ALPHA * float(R[i]) + lam * (1 - 2 * K)
    for (i, j), cij in C.items():
        Q[(i, j)] = BETA * cij + 2 * lam
    return Q


@pytest.fixture(scope="module")
def toy_qubo():
    return _build_toy_qubo()


def _cardinality(sample: dict[int, int]) -> int:
    return sum(sample.values())


def test_exact_solver_respects_cardinality(toy_qubo):
    result = solve_qubo_exact(toy_qubo, n_vars=N)
    assert _cardinality(result["sample"]) == K


def test_sa_matches_exact(toy_qubo):
    exact = solve_qubo_exact(toy_qubo, n_vars=N)
    sa = solve_qubo_sa(toy_qubo, num_reads=2000, num_sweeps=1000, seed=42)

    assert _cardinality(sa["sample"]) == K
    assert sa["sample"] == exact["sample"]
    assert abs(sa["energy"] - exact["energy"]) < 1e-6


def test_milp_matches_exact(toy_qubo):
    exact = solve_qubo_exact(toy_qubo, n_vars=N)
    mi = solve_qubo_milp(toy_qubo, n=N)

    assert mi["success"]
    assert _cardinality(mi["sample"]) == K
    assert mi["sample"] == exact["sample"]
    assert abs(mi["energy"] - exact["energy"]) < 1e-6


def test_milp_hard_cardinality_matches_penalized(toy_qubo):
    """Verificación de lambda (ARCHITECTURE.md §7.3): el MILP con Sigma x = k DURA
    debe coincidir con el MILP que resuelve el QUBO penalizado sin restricción dura."""
    penalized = solve_qubo_milp(toy_qubo, n=N)
    hard = solve_qubo_milp(toy_qubo, n=N, cardinality_k=K)

    assert hard["success"]
    assert hard["sample"] == penalized["sample"]
    assert abs(hard["energy"] - penalized["energy"]) < 1e-6


def test_qaoa_p3_matches_exact(toy_qubo):
    exact = solve_qubo_exact(toy_qubo, n_vars=N)
    qa = solve_qubo_qaoa(toy_qubo, n=N, p=3, seed=42, maxiter=300, n_restarts=8)

    assert _cardinality(qa["sample"]) == K
    assert abs(qa["energy"] - exact["energy"]) < 1e-4
    assert qa["sample"] == exact["sample"]
