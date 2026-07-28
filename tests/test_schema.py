"""El dataset cumple tipos, rangos y restricciones duras. Ver ARCHITECTURE.md §9, §12."""

from __future__ import annotations

from tikee.data.validate import (
    PREDICTOR_COLUMNS,
    PROTECTED_COLUMNS,
    check_hard_constraint,
    check_protected_not_predictor,
    check_ranges,
)


def test_all_predictor_columns_present(full_dataset_seed42):
    for col in PREDICTOR_COLUMNS:
        assert col in full_dataset_seed42.columns


def test_ranges_within_schema(full_dataset_seed42):
    report = check_ranges(full_dataset_seed42)
    assert report["ok"], report["violations"]


def test_hard_constraint_antiguedad_laboral(full_dataset_seed42):
    report = check_hard_constraint(full_dataset_seed42)
    assert report["ok"], f"{report['n_violations']} filas violan antiguedad_laboral <= (edad-18)*12"


def test_protected_attributes_exist_and_not_predictors(full_dataset_seed42):
    report = check_protected_not_predictor(full_dataset_seed42)
    assert report["ok"]
    assert not (set(PROTECTED_COLUMNS) & set(PREDICTOR_COLUMNS))


def test_protected_attributes_present_in_dataframe(full_dataset_seed42):
    for col in PROTECTED_COLUMNS:
        assert col in full_dataset_seed42.columns
