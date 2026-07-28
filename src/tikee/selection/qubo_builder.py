"""Matrices de relevancia/redundancia -> matriz Q (ARCHITECTURE.md §6.3, §6.4)."""

from __future__ import annotations

import numpy as np


def compute_lambda(R: np.ndarray, C: np.ndarray, alpha: float, beta: float, n: int) -> float:
    return 2 * (alpha * R.max() + beta * C.max() * n)


def build_qubo(
    R: np.ndarray, C: np.ndarray, alpha: float, beta: float, k: int, lam: float | None = None
) -> dict[tuple[int, int], float]:
    n = len(R)
    if lam is None:
        lam = compute_lambda(R, C, alpha, beta, n)

    Q: dict[tuple[int, int], float] = {}
    for i in range(n):
        Q[(i, i)] = -alpha * R[i] + lam * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] != 0:
                Q[(i, j)] = beta * C[i, j] + 2 * lam
            else:
                Q[(i, j)] = 2 * lam
    return Q


def decode_selection(sample: dict[int, int], variable_order: list[str]) -> list[str]:
    return [variable_order[i] for i, v in sample.items() if v == 1]


def verify_cardinality(sample: dict[int, int], k: int) -> bool:
    return sum(sample.values()) == k
