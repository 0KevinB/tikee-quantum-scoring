"""Matrices de relevancia/redundancia -> matriz Q (ARCHITECTURE.md §6.3, §6.4)."""

from __future__ import annotations

import numpy as np


def compute_lambda(R: np.ndarray, C: np.ndarray, alpha: float, beta: float, n: int) -> float:
    """Regla de ARCHITECTURE.md §6.4 para el multiplicador de la penalización de
    cardinalidad: debe dominar sobre relevancia y redundancia para que la
    restricción Σx_i = k se cumpla en el óptimo.

    Args:
        R: vector de relevancia normalizado en [0,1], una entrada por variable.
        C: matriz de redundancia simétrica en [0,1], diagonal en cero.
        alpha: peso del término de relevancia (ver `build_qubo`).
        beta: peso del término de redundancia.
        n: número de variables candidatas (len(R)).

    Returns:
        El valor de lambda a pasar a `build_qubo`.
    """
    return 2 * (alpha * R.max() + beta * C.max() * n)


def build_qubo(
    R: np.ndarray, C: np.ndarray, alpha: float, beta: float, k: int, lam: float | None = None
) -> dict[tuple[int, int], float]:
    """Construye la matriz QUBO relevancia-redundancia con penalización de
    cardinalidad (ARCHITECTURE.md §6.3-6.4): minimiza -alpha*R_i*x_i sumado a
    beta*C_ij*x_i*x_j, más lambda*(Σx_i - k)^2 expandido en la diagonal y fuera de
    ella. Se entrega en triangular superior, listo para `sample_qubo()` de dimod.

    Args:
        R: vector de relevancia normalizado en [0,1] (una entrada por variable).
        C: matriz de redundancia simétrica en [0,1], diagonal en cero.
        alpha: peso del término de relevancia.
        beta: peso del término de redundancia.
        k: cardinalidad objetivo (número de variables a seleccionar).
        lam: lambda de la penalización de cardinalidad; si es None se calcula con
            `compute_lambda`.

    Returns:
        Diccionario `{(i,i): valor}` para la diagonal y `{(i,j): valor}` con i<j
        para los términos cuadráticos — el formato que espera dimod.
    """
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
    """Traduce una solución binaria (índice -> 0/1) a la lista de nombres de
    variable seleccionados, según el orden usado al construir R/C."""
    return [variable_order[i] for i, v in sample.items() if v == 1]


def verify_cardinality(sample: dict[int, int], k: int) -> bool:
    """True si la solución tiene exactamente k variables encendidas. Si es False,
    lambda es insuficiente (ARCHITECTURE.md §6.4): hay que duplicarlo y resolver
    de nuevo."""
    return sum(sample.values()) == k
