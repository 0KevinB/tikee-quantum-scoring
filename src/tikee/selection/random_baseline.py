"""k variables al azar x N repeticiones. Control de ruido (decisión D6, brazo R/Rb)."""

from __future__ import annotations

import numpy as np


def random_variable_subsets(
    all_vars: list[str], k: int, n_repetitions: int = 100, seed: int | None = None
) -> list[list[str]]:
    """Genera `n_repetitions` subconjuntos de k variables elegidas al azar sin
    reemplazo, para el brazo de control R/Rb: si un método no supera al azar en
    AUC, no aportó nada (D6).

    Args:
        all_vars: universo de variables candidatas.
        k: tamaño de cada subconjunto.
        n_repetitions: cuántos subconjuntos independientes generar.
        seed: semilla para reproducibilidad.

    Returns:
        Lista de `n_repetitions` listas de nombres de variable.
    """
    rng = np.random.default_rng(seed)
    return [list(rng.choice(all_vars, size=k, replace=False)) for _ in range(n_repetitions)]
