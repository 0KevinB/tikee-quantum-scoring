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
    return sum(sample.values()) == k
