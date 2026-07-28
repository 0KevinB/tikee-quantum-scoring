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
    """Contenedor de los tres YAML de `config/`. Se construye con `load_config()`,
    nunca directamente."""

    raw: dict[str, Any]
    schema: dict[str, Any]
    experiments: dict[str, Any]

    @property
    def seeds_multiseed(self) -> list[int]:
        """Las 10 semillas del protocolo principal (D11)."""
        return self.raw["seeds"]["multiseed"]

    @property
    def sigma_ruido(self) -> float | None:
        """Sigma calibrado en F1; None si todavía no se corrió la calibración
        (ver `scripts/f1_calibrate_sigma.py`)."""
        return self.raw["data"]["sigma_ruido"]

    def path(self, key: str) -> Path:
        """Resuelve una clave de `config.yaml: paths` a una ruta absoluta bajo
        la raíz del repo (p.ej. `cfg.path("cache")` -> `.../reports/cache`)."""
        return REPO_ROOT / self.raw["paths"][key]


def load_config() -> Config:
    """Carga `config.yaml`, `schema.yaml` y `experiments.yaml` de `config/` en un
    único objeto `Config`."""
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
    """Crea (si no existen) los directorios de salida declarados en
    `config.yaml: paths` — raw, interim, processed, reports, cache, figures,
    fidelity."""
    for key in ("raw", "interim", "processed", "reports", "cache", "figures", "fidelity"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
