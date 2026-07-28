"""expand.py produce 45 columnas deterministas; sin fuga fuera de train.
Ver ARCHITECTURE.md §4.6, §12."""

from __future__ import annotations

from tikee.features.expand import ALL_45_COLUMNS, LevelBExpander, ground_truth_labels


def test_produces_exactly_45_columns(full_dataset_seed42):
    df = full_dataset_seed42
    expander = LevelBExpander().fit(df, df["default"].to_numpy())
    out = expander.transform(df, seed=42)
    assert out.shape[1] == 45
    assert list(out.columns) == ALL_45_COLUMNS


def test_deterministic_given_seed(full_dataset_seed42):
    df = full_dataset_seed42
    expander = LevelBExpander().fit(df, df["default"].to_numpy())
    out1 = expander.transform(df, seed=7)
    out2 = expander.transform(df, seed=7)
    assert out1.equals(out2)


def test_noise_columns_are_pure_noise(full_dataset_seed42):
    df = full_dataset_seed42
    expander = LevelBExpander().fit(df, df["default"].to_numpy())
    out = expander.transform(df, seed=42)
    assert abs(out["f44"].mean()) < 0.1
    assert 0.0 <= out["f45"].min() and out["f45"].max() <= 1.0


def test_ground_truth_labels_cover_all_columns_and_flag_traps():
    labels = ground_truth_labels()
    assert set(labels.keys()) == set(ALL_45_COLUMNS)
    assert labels["f30"] == "irrelevant"
    assert labels["f44"] == "irrelevant"
    assert labels["f45"] == "irrelevant"
    # f10 = score_buro (posición 10 en LEVEL_A_ORIGINAL), señal fuerte del ground truth
    assert labels["f10"] == "relevant"


def test_fit_only_uses_train_statistics(full_dataset_seed42):
    """Ajustar en un subconjunto (train) y transformar otro (test) no debe fallar ni
    usar estadísticos del segundo — es la regla de higiene de ARCHITECTURE.md §3."""
    df = full_dataset_seed42
    train_df = df.iloc[: len(df) // 2]
    test_df = df.iloc[len(df) // 2:]
    expander = LevelBExpander().fit(train_df, train_df["default"].to_numpy())
    out_test = expander.transform(test_df, seed=1)
    assert out_test.shape[1] == 45
    assert not out_test.isna().any().any()
