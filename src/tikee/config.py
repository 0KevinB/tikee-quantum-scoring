"""Carga config.yaml / schema.yaml / experiments.yaml y fija semillas globales."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    raw: dict[str, Any]
    schema: dict[str, Any]
    experiments: dict[str, Any]

    @property
    def seeds_multiseed(self) -> list[int]:
        return self.raw["seeds"]["multiseed"]

    @property
    def sigma_ruido(self) -> float | None:
        return self.raw["data"]["sigma_ruido"]

    def path(self, key: str) -> Path:
        return REPO_ROOT / self.raw["paths"][key]


def load_config() -> Config:
    return Config(
        raw=_load_yaml("config.yaml"),
        schema=_load_yaml("schema.yaml"),
        experiments=_load_yaml("experiments.yaml"),
    )


def set_global_seed(seed: int) -> None:
    """Propaga la semilla a random, numpy. sklearn/xgboost/SDV toman random_state explícito por llamada."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dirs(cfg: Config) -> None:
    for key in ("raw", "interim", "processed", "reports", "cache", "figures", "fidelity"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
