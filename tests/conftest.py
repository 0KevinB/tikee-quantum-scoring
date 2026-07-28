"""Fixtures compartidas. Ver ARCHITECTURE.md §9."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tikee.config import load_config  # noqa: E402
from tikee.data.seed_generator import generate_seed_table  # noqa: E402
from tikee.data.sdv_synthesizer import synthesize  # noqa: E402
from tikee.data.target_definition import add_target  # noqa: E402

SEED = 42


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def full_dataset_seed42(cfg):
    sigma = cfg.sigma_ruido
    if sigma is None:
        pytest.skip("sigma_ruido aun no calibrado en config.yaml (checkpoint F1 pendiente)")
    seed_df = generate_seed_table(SEED, cfg.raw["data"]["n_seed_rows"])
    synth_df, _ = synthesize(
        seed_df, method="gaussian_copula", n_sample=cfg.raw["data"]["n_synthetic_rows"], seed=SEED
    )
    df = add_target(synth_df, sigma=sigma, seed=SEED, target_rate=cfg.raw["data"]["tasa_base_default"])
    return df
