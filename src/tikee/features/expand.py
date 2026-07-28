"""Nivel A (18) -> Nivel B (45), determinista y trazable (decisión D9,
ARCHITECTURE.md §4.6). Todos los estadísticos (deciles WOE) se ajustan en train y se
aplican a test — `LevelBExpander.fit` solo debe verse con datos de train."""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL_A_ORIGINAL = [
    "edad", "zona_residencia", "nivel_educacion", "carga_familiar", "tipo_empleo",
    "antiguedad_laboral_meses", "antiguedad_socio_meses", "ingreso_mensual",
    "gastos_mensuales", "score_buro", "num_operaciones_vigentes", "deuda_total_sistema",
    "peor_calificacion_12m", "dias_mora_max_12m", "monto_solicitado", "plazo_meses",
    "ratio_cuota_ingreso", "ratio_deuda_ingreso",
]

# f01..f18 mantienen el nombre original (trazabilidad); f19+ usan fXX.
GROUND_TRUTH_RELEVANT_ORIGINAL = {
    "ratio_cuota_ingreso", "ratio_deuda_ingreso", "score_buro", "dias_mora_max_12m",
    "peor_calificacion_12m", "num_operaciones_vigentes", "antiguedad_socio_meses",
    "antiguedad_laboral_meses", "carga_familiar", "tipo_empleo", "edad",
}
GROUND_TRUTH_IRRELEVANT_ORIGINAL = {"nivel_educacion", "zona_residencia"}

INTERACTIONS = {
    "f19": ("ratio_cuota_ingreso", "score_buro"),
    "f20": ("ratio_cuota_ingreso", "dias_mora_max_12m"),
    "f21": ("score_buro", "num_operaciones_vigentes"),
    "f22": ("ingreso_mensual", "carga_familiar"),
    "f23": ("antiguedad_socio_meses", "score_buro"),
    "f24": ("monto_solicitado", "plazo_meses"),
    "f25": ("deuda_total_sistema", "num_operaciones_vigentes"),
    "f26": ("edad", "antiguedad_laboral_meses"),
    "f27": ("ratio_deuda_ingreso", "peor_calificacion_12m"),
    "f28": ("gastos_mensuales", "carga_familiar"),
}


def _standardize(s: pd.Series, mean: float, std: float) -> pd.Series:
    return (s - mean) / (std if std > 0 else 1.0)


class LevelBExpander:
    def __init__(self) -> None:
        self._z_stats: dict[str, tuple[float, float]] = {}
        self._score_decile_edges: np.ndarray | None = None
        self._score_decile_woe: dict[int, float] | None = None
        self._fitted = False

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray) -> "LevelBExpander":
        for col in LEVEL_A_ORIGINAL:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                self._z_stats[col] = (float(df_train[col].mean()), float(df_train[col].std(ddof=0)))

        deciles, edges = pd.qcut(df_train["score_buro"], 10, retbins=True, duplicates="drop")
        self._score_decile_edges = edges
        y = pd.Series(y_train, index=df_train.index)
        overall_good = (y == 0).sum()
        overall_bad = (y == 1).sum()
        woe = {}
        for i, interval in enumerate(deciles.cat.categories):
            mask = deciles == interval
            good = (y[mask] == 0).sum() or 0.5
            bad = (y[mask] == 1).sum() or 0.5
            woe[i] = float(np.log((good / overall_good) / (bad / overall_bad)))
        self._score_decile_woe = woe
        self._fitted = True
        return self

    def _z(self, df: pd.DataFrame, col: str) -> pd.Series:
        mean, std = self._z_stats[col]
        return _standardize(df[col], mean, std)

    def _woe_score_buro(self, df: pd.DataFrame) -> pd.Series:
        edges = self._score_decile_edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        bin_idx = pd.cut(df["score_buro"], bins=edges, labels=False, include_lowest=True)
        return bin_idx.map(self._score_decile_woe).astype(float)

    def transform(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit() debe llamarse (sobre train) antes de transform()")

        out = pd.DataFrame(index=df.index)
        for i, col in enumerate(LEVEL_A_ORIGINAL, start=1):
            out[f"f{i:02d}"] = df[col]

        for fname, (a, b) in INTERACTIONS.items():
            za = self._z(df, a) if a in self._z_stats else pd.get_dummies(df[a], drop_first=True).iloc[:, 0]
            zb = self._z(df, b) if b in self._z_stats else pd.get_dummies(df[b], drop_first=True).iloc[:, 0]
            out[fname] = za * zb

        out["f29"] = (df["tipo_empleo"] == "independiente_informal").astype(float) * self._z(df, "ratio_cuota_ingreso")
        out["f30"] = (df["zona_residencia"] == "rural").astype(float) * self._z(df, "score_buro")

        out["f31"] = np.log(df["ingreso_mensual"])
        out["f32"] = np.log1p(df["deuda_total_sistema"])
        out["f33"] = (df["edad"] - 40) ** 2
        out["f34"] = np.sqrt(df["monto_solicitado"])
        out["f35"] = self._woe_score_buro(df)
        out["f36"] = (df["dias_mora_max_12m"] > 0).astype(float)
        out["f37"] = (df["dias_mora_max_12m"] > 30).astype(float)
        out["f38"] = np.log1p(df["antiguedad_socio_meses"])

        excedente = df["ingreso_mensual"] - df["gastos_mensuales"]
        out["f39"] = excedente
        out["f40"] = excedente / df["cuota_estimada"].replace(0, np.nan)
        out["f40"] = out["f40"].fillna(0.0)
        out["f41"] = df["deuda_total_sistema"] / (df["num_operaciones_vigentes"] + 1)
        out["f42"] = df["monto_solicitado"] / df["ingreso_mensual"]
        out["f43"] = df["ingreso_mensual"] / (1 + df["carga_familiar"])

        rng = np.random.default_rng(seed)
        out["f44"] = rng.normal(0, 1, size=len(df))
        out["f45"] = rng.uniform(0, 1, size=len(df))

        return out


def ground_truth_labels() -> dict[str, str]:
    labels = {}
    for i, col in enumerate(LEVEL_A_ORIGINAL, start=1):
        name = f"f{i:02d}"
        if col in GROUND_TRUTH_IRRELEVANT_ORIGINAL:
            labels[name] = "irrelevant"
        elif col in GROUND_TRUTH_RELEVANT_ORIGINAL:
            labels[name] = "relevant"
        else:
            labels[name] = "instrumental"
    for fname in INTERACTIONS:
        labels[fname] = "relevant"
    labels["f29"] = "relevant"
    labels["f30"] = "irrelevant"  # trampa: contiene zona_residencia (coef. cero)
    for fname in ["f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38"]:
        labels[fname] = "relevant"
    for fname in ["f39", "f40", "f41", "f42", "f43"]:
        labels[fname] = "relevant"
    labels["f44"] = "irrelevant"
    labels["f45"] = "irrelevant"
    return labels


ALL_45_COLUMNS = [f"f{i:02d}" for i in range(1, 46)]
