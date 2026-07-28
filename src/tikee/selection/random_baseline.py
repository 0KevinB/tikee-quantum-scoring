"""k variables al azar x N repeticiones. Control de ruido (decisión D6, brazo R/Rb)."""

from __future__ import annotations

import numpy as np


def random_variable_subsets(
    all_vars: list[str], k: int, n_repetitions: int = 100, seed: int | None = None
) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [list(rng.choice(all_vars, size=k, replace=False)) for _ in range(n_repetitions)]
