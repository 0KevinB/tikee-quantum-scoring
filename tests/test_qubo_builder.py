"""Simetría de Q, diagonal, cardinalidad k, casos alpha=0 y beta=0.
Ver ARCHITECTURE.md §6.4, §9, §12."""

from __future__ import annotations

import numpy as np

from tikee.selection.exact import solve_qubo_exact
from tikee.selection.qubo_builder import build_qubo, compute_lambda, verify_cardinality

N = 6
K = 3
ALPHA = 1.0
BETA = 1.0

R = np.array([0.9, 0.2, 0.7, 0.5, 0.1, 0.8])
C = np.array([
    [0, 0.1, 0.2, 0.9, 0.1, 0.2],
    [0.1, 0, 0.1, 0.1, 0.05, 0.1],
    [0.2, 0.1, 0, 0.2, 0.1, 0.9],
    [0.9, 0.1, 0.2, 0, 0.1, 0.2],
    [0.1, 0.05, 0.1, 0.1, 0, 0.1],
    [0.2, 0.1, 0.9, 0.2, 0.1, 0],
])


def test_diagonal_formula():
    lam = compute_lambda(R, C, ALPHA, BETA, N)
    Q = build_qubo(R, C, ALPHA, BETA, K, lam=lam)
    for i in range(N):
        expected = -ALPHA * R[i] + lam * (1 - 2 * K)
        assert abs(Q[(i, i)] - expected) < 1e-9


def test_offdiagonal_symmetric_energy():
    """Q se guarda triangular superior; la energía debe coincidir con la fórmula
    matricial completa Q_ij*x_i*x_j (una sola vez por par, no duplicada)."""
    lam = compute_lambda(R, C, ALPHA, BETA, N)
    Q = build_qubo(R, C, ALPHA, BETA, K, lam=lam)
    x = np.array([1, 0, 1, 1, 0, 0])

    energy_from_qubo = sum(
        Q[(i, i)] * x[i] if i == j else Q.get((i, j), 0.0) * x[i] * x[j]
        for i in range(N) for j in range(N) if i <= j
    )

    Q_full = np.zeros((N, N))
    for i in range(N):
        Q_full[i, i] = -ALPHA * R[i] + lam * (1 - 2 * K)
        for j in range(i + 1, N):
            Q_full[i, j] = BETA * C[i, j] + 2 * lam
    energy_manual = float(x @ Q_full @ x - sum(Q_full[i, j] * x[i] * x[j] for i in range(N) for j in range(N) if i > j))
    # x @ Q_full @ x ya cuenta cada par (i,j) i<j una vez (triangular superior), coincide directo
    energy_manual = float(x @ Q_full @ x)
    assert abs(energy_from_qubo - energy_manual) < 1e-9


def test_beta_zero_large_lambda_selects_exactly_k():
    lam = 50.0
    Q = build_qubo(R, C, ALPHA, beta=0.0, k=K, lam=lam)
    result = solve_qubo_exact(Q, n_vars=N)
    assert verify_cardinality(result["sample"], K)


def test_alpha_zero_selects_least_mutually_correlated():
    lam = 50.0
    Q = build_qubo(R, C, alpha=0.0, beta=BETA, k=K, lam=lam)
    result = solve_qubo_exact(Q, n_vars=N)
    selected = [i for i, v in result["sample"].items() if v == 1]
    assert len(selected) == K

    from itertools import combinations
    best_redundancy = min(
        sum(C[i, j] for i in combo for j in combo if i < j) for combo in combinations(range(N), K)
    )
    selected_redundancy = sum(C[i, j] for i in selected for j in selected if i < j)
    # puede haber empates (varios subconjuntos con la misma redundancia mínima);
    # lo que importa es que el solucionador alcance ese mínimo, no un índice exacto.
    assert abs(selected_redundancy - best_redundancy) < 1e-9
