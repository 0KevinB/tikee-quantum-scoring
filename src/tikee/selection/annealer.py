"""Recocido simulado sobre Q (dwave-samplers, fallback neal). Brazo C0 / C0b."""

from __future__ import annotations

import time
from typing import Any

try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler


def solve_qubo_sa(
    Q: dict[tuple[int, int], float],
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Resuelve un QUBO por recocido simulado, heurístico sin garantía de
    optimalidad (ARCHITECTURE.md §7.1).

    Args:
        Q: matriz QUBO en formato `{(i,i): valor, (i,j): valor}` (triangular
            superior), el formato que produce `qubo_builder.build_qubo`.
        num_reads: número de lecturas/intentos independientes del sampler.
        num_sweeps: número de sweeps de temperatura por lectura.
        seed: semilla para reproducibilidad.

    Returns:
        dict con `sample` (mejor solución binaria), `energy`, `wall_time_s`,
        `mean_energy` (promedio entre lecturas) y `n_reads_at_best` (cuántas
        lecturas alcanzaron la mejor energía — proxy de qué tan fácil es el
        paisaje del problema).
    """
    sampler = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    wall = time.perf_counter() - t0

    best = sampleset.first
    energies = sampleset.record.energy
    n_at_best = int((abs(energies - best.energy) < 1e-9).sum())

    return {
        "sample": {int(k): int(v) for k, v in best.sample.items()},
        "energy": float(best.energy),
        "wall_time_s": wall,
        "mean_energy": float(energies.mean()),
        "n_reads_at_best": n_at_best,
        "num_reads": num_reads,
    }


def verify_cardinality(sample: dict[int, int], k: int) -> bool:
    """True si la solución tiene exactamente k variables encendidas."""
    return sum(sample.values()) == k
