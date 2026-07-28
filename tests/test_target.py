"""Tasa base en banda; monotonicidad del riesgo vs score de buró; correlaciones
objetivo dentro de tolerancia. Ver ARCHITECTURE.md §9, §12."""

from __future__ import annotations

import pandas as pd

from tikee.data.validate import check_correlations, check_default_rate


def test_default_rate_in_band(full_dataset_seed42):
    report = check_default_rate(full_dataset_seed42, target=0.08, tol=0.005)
    assert report["ok"], report


def test_risk_decreases_monotonically_with_score_buro_decile(full_dataset_seed42):
    df = full_dataset_seed42.copy()
    df["decil_score"] = pd.qcut(df["score_buro"], 10, labels=False, duplicates="drop")
    rate_by_decile = df.groupby("decil_score")["default"].mean().sort_index()
    diffs = rate_by_decile.diff().dropna()
    assert (diffs <= 0.015).sum() >= len(diffs) - 1, (
        f"la tasa de default no decrece monótonamente por decil de score_buro: {rate_by_decile.to_dict()}"
    )


def test_target_correlations_within_tolerance(full_dataset_seed42):
    report = check_correlations(full_dataset_seed42, tol=0.07)
    failing = [p for p in report["pairs"] if not p["ok"]]
    assert report["ok"], f"pares fuera de tolerancia: {failing}"
