"""TabuSampler: segundo heurístico, diagnostica si el paisaje es fácil. Brazo C4b."""

from __future__ import annotations

import time
from typing import Any

from dwave.samplers import TabuSampler


def solve_qubo_tabu(
    Q: dict[tuple[int, int], float],
    num_reads: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    sampler = TabuSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, seed=seed)
    wall = time.perf_counter() - t0

    best = sampleset.first
    return {
        "sample": {int(k): int(v) for k, v in best.sample.items()},
        "energy": float(best.energy),
        "wall_time_s": wall,
    }
