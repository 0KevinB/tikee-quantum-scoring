"""Friedman, post-hoc de Nemenyi, diagrama de diferencia crítica, Jaccard.
ARCHITECTURE.md §8.5 (decisión D12)."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

# q_alpha (Nemenyi, dos colas, alpha=0.05) — tabla estándar (Demsar 2006).
NEMENYI_Q_ALPHA_005: dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
}


def average_ranks(score_matrix: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """`score_matrix`: filas = semillas, columnas = brazos. Rango 1 = mejor."""
    ranked = score_matrix.rank(axis=1, ascending=not higher_is_better)
    return ranked.mean(axis=0)


def friedman_test(score_matrix: pd.DataFrame) -> dict[str, Any]:
    stat, p = friedmanchisquare(*[score_matrix[col] for col in score_matrix.columns])
    return {"statistic": float(stat), "p_value": float(p), "significant_at_0.05": bool(p < 0.05)}


def nemenyi_posthoc(score_matrix: pd.DataFrame, higher_is_better: bool = True, alpha: float = 0.05) -> dict[str, Any]:
    if alpha != 0.05:
        raise ValueError("solo se tabuló q_alpha para alpha=0.05 (D12)")
    k = score_matrix.shape[1]
    n = score_matrix.shape[0]
    if k not in NEMENYI_Q_ALPHA_005:
        raise ValueError(f"tabla de Nemenyi no cubre k={k} brazos")

    q_alpha = NEMENYI_Q_ALPHA_005[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    ranks = average_ranks(score_matrix, higher_is_better=higher_is_better)

    pairs = []
    for a, b in combinations(score_matrix.columns, 2):
        diff = abs(ranks[a] - ranks[b])
        pairs.append({"a": a, "b": b, "rank_diff": float(diff), "significant": bool(diff > cd)})

    return {"critical_difference": float(cd), "average_ranks": ranks.to_dict(), "pairs": pairs}


def jaccard_index(subsets: list[set[str]]) -> float:
    if len(subsets) < 2:
        return 1.0
    scores = []
    for a, b in combinations(subsets, 2):
        union = a | b
        if not union:
            scores.append(1.0)
            continue
        scores.append(len(a & b) / len(union))
    return float(np.mean(scores))


def selection_frequency(subsets_by_seed: list[list[str]], all_vars: list[str]) -> pd.Series:
    counts = pd.Series(0, index=all_vars, dtype=float)
    for subset in subsets_by_seed:
        for v in subset:
            counts[v] += 1
    return counts / len(subsets_by_seed)


def plot_critical_difference(average_ranks_dict: dict[str, float], cd: float, out_path: str) -> None:
    import matplotlib.pyplot as plt

    names = list(average_ranks_dict.keys())
    ranks = [average_ranks_dict[n] for n in names]
    order = np.argsort(ranks)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.5))
    y_positions = np.arange(len(names))
    ax.scatter(ranks, y_positions, zorder=3)
    for y, i in enumerate(order):
        ax.text(ranks[i], y, f"  {names[i]} ({ranks[i]:.2f})", va="center")
    best_rank = min(ranks)
    ax.plot([best_rank, best_rank + cd], [-1, -1], marker="|", color="black")
    ax.text(best_rank + cd / 2, -1.4, f"CD = {cd:.3f}", ha="center")
    ax.set_yticks([])
    ax.set_xlabel("Rango promedio (menor = mejor)")
    ax.set_title("Diagrama de diferencia crítica (Nemenyi, α=0.05)")
    ax.set_ylim(-2, len(names))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
