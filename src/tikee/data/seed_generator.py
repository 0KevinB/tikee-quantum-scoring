"""Tabla semilla (2.000 filas) con la estructura de correlación intencional.

Etapa 1 de la generación en dos etapas (ARCHITECTURE.md §4.4). Cópula gaussiana
propia sobre el bloque continuo/ordinal de §4.2, más el sesgo indirecto de §4.3
inyectado como desplazamientos condicionales a los atributos protegidos (D15),
aplicados DESPUÉS de la cópula para no forzar toda la estructura de sesgo dentro
de una única matriz de correlación de 18x18 (que además no está especificada por
completo: solo hay 9 pares objetivo entre las 12 variables continuas/ordinales de
este bloque; el resto de la estructura -sexo, zona, tipo_empleo, plazo_meses,
nivel_educacion- se maneja por diseño explícito, no por cópula).

`ratio_cuota_ingreso` y `ratio_deuda_ingreso` son "derivadas" (ARCHITECTURE.md §4.1,
filas 17-18): no se muestrean directamente, se calculan de las variables base. Sus
correlaciones objetivo (§4.2) son una consecuencia esperada, no un parámetro directo
de la cópula — es exactamente lo que la calibración de F1 debe verificar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm

CONTINUOUS_BLOCK = [
    "edad",
    "carga_familiar",
    "antiguedad_laboral_meses",
    "antiguedad_socio_meses",
    "ingreso_mensual",
    "gastos_mensuales",
    "score_buro",
    "num_operaciones_vigentes",
    "deuda_total_sistema",
    "peor_calificacion_12m",
    "dias_mora_max_12m",
    "monto_solicitado",
]

# ARCHITECTURE.md §4.2 — pares objetivo de rho de Spearman entre el bloque continuo.
TARGET_CORR_PAIRS: dict[tuple[str, str], float] = {
    ("score_buro", "peor_calificacion_12m"): -0.85,
    ("score_buro", "dias_mora_max_12m"): -0.80,
    ("peor_calificacion_12m", "dias_mora_max_12m"): 0.88,
    ("deuda_total_sistema", "num_operaciones_vigentes"): 0.60,
    ("ingreso_mensual", "gastos_mensuales"): 0.75,
    ("ingreso_mensual", "monto_solicitado"): 0.55,
    ("edad", "antiguedad_laboral_meses"): 0.55,
    ("edad", "antiguedad_socio_meses"): 0.40,
    ("edad", "carga_familiar"): 0.30,
}

PLAZO_CATEGORIES = [6, 12, 18, 24, 36, 48, 60]
PLAZO_PROBS = [0.10, 0.20, 0.15, 0.20, 0.20, 0.10, 0.05]

TIPO_EMPLEO_CATEGORIES = ["dependiente_formal", "independiente_informal", "agricultor", "comerciante"]
TIPO_EMPLEO_BASE_PROBS = np.array([0.45, 0.30, 0.10, 0.15])

TASA_CUOTA_ANUAL = 0.165


def _nearest_psd_correlation(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Proyección a la matriz de correlación semidefinida positiva más cercana
    (recorte de autovalores). Necesaria porque solo 9 de los 66 pares del bloque
    continuo tienen un objetivo especificado; el resto se fija en 0 y la matriz
    resultante no siempre es exactamente PSD por construcción."""
    sym = (corr + corr.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, eps, None)
    reconstructed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(reconstructed))
    reconstructed = reconstructed / np.outer(d, d)
    np.fill_diagonal(reconstructed, 1.0)
    return reconstructed


def _spearman_to_pearson(rho_s: float) -> float:
    """Fórmula estándar de la cópula gaussiana: convierte una correlación de
    Spearman objetivo a la correlación de Pearson que hay que imponer sobre las
    normales latentes para lograrla asintóticamente."""
    return 2 * np.sin(np.pi * rho_s / 6)


# F1 (calibración empírica, ver reports/cache/f1_sigma_calibration.json): dias_mora_max_12m
# tiene ~45% de masa en cero (ver DIAS_MORA_ZERO_MASS mas abajo). Con esa masa puntual, el
# Spearman máximo alcanzable entre dias_mora y una variable continua está acotado bien por
# debajo de 1 incluso con Pearson de entrada casi perfecto (verificado empíricamente: con
# Pearson=-0.995 el Spearman de salida no supera ~-0.79). Por eso estos dos pares NO usan la
# fórmula estándar spearman->pearson (que asume marginales continuas) sino un valor de entrada
# ya inflado para compensar la atenuación medida.
_MANUAL_PEARSON_OVERRIDES: dict[tuple[str, str], float] = {
    ("score_buro", "dias_mora_max_12m"): -0.90,
    ("peor_calificacion_12m", "dias_mora_max_12m"): 0.93,
    # el resto del bloque se atenua colateralmente por la proyeccion PSD del par anterior
    # (matriz completa, no aislada); se compensa con una entrada algo mas fuerte que 0.40.
    ("edad", "antiguedad_socio_meses"): 0.50,
}


def _build_target_correlation_matrix() -> pd.DataFrame:
    n = len(CONTINUOUS_BLOCK)
    idx = {name: i for i, name in enumerate(CONTINUOUS_BLOCK)}
    corr = np.eye(n)
    for (a, b), rho_s in TARGET_CORR_PAIRS.items():
        rho_p = _MANUAL_PEARSON_OVERRIDES.get((a, b)) or _MANUAL_PEARSON_OVERRIDES.get((b, a))
        if rho_p is None:
            rho_p = _spearman_to_pearson(rho_s)
        corr[idx[a], idx[b]] = rho_p
        corr[idx[b], idx[a]] = rho_p
    corr = _nearest_psd_correlation(corr)
    return pd.DataFrame(corr, index=CONTINUOUS_BLOCK, columns=CONTINUOUS_BLOCK)


def _zero_inflated_ppf(u: np.ndarray, zero_mass: float, tail_ppf) -> np.ndarray:
    out = np.zeros_like(u)
    mask = u > zero_mass
    u_tail = (u[mask] - zero_mass) / (1 - zero_mass)
    out[mask] = tail_ppf(u_tail)
    return out


def _apply_marginals(u_block: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = pd.DataFrame(index=u_block.index)

    out["edad"] = gamma_dist(a=6.0, scale=6.0, loc=18).ppf(u_block["edad"]).round()
    out["edad"] = out["edad"].clip(18, 75)

    out["carga_familiar"] = (u_block["carga_familiar"] * 7).clip(0, 6).round()

    antig_lab_years_cap = (out["edad"] - 18) * 12
    raw_antig_lab = lognorm(s=1.0, scale=40).ppf(u_block["antiguedad_laboral_meses"])
    out["antiguedad_laboral_meses"] = np.minimum(raw_antig_lab, antig_lab_years_cap).clip(0, 360).round()

    out["antiguedad_socio_meses"] = lognorm(s=0.9, scale=45).ppf(u_block["antiguedad_socio_meses"]).clip(1, 300).round()

    out["ingreso_mensual"] = lognorm(s=0.55, scale=560).ppf(u_block["ingreso_mensual"]).clip(460, 3500)

    # std=0.20 (no 0.12): con ratio determinista + poco ruido, gastos queda casi
    # perfectamente correlacionado con ingreso (~0.87, fuera de tolerancia del objetivo
    # 0.75 de ARCHITECTURE.md §4.2). Calibrado empíricamente para caer en banda.
    gastos_ratio = np.clip(rng.normal(0.75, 0.20, size=len(u_block)), 0.25, 1.15)
    out["gastos_mensuales"] = (out["ingreso_mensual"] * gastos_ratio).clip(200, 3000)

    out["score_buro"] = (beta_dist(a=4.5, b=2.0).ppf(u_block["score_buro"]) * 998 + 1).clip(1, 999).round()

    out["num_operaciones_vigentes"] = (u_block["num_operaciones_vigentes"] * 9).clip(0, 8).round()

    # [DESVIACIÓN DOCUMENTADA] ARCHITECTURE.md §4.5 sugiere ~18% de masa en cero para
    # deuda_total_sistema. Con esa masa y una dispersión realista, ratio_deuda_ingreso
    # (=deuda/ingreso) queda correlacionado con deuda_total_sistema en ~0.90-0.96,
    # fuera de la tolerancia del objetivo 0.70 (§4.2) para el par derivado. Se redujo la
    # masa en cero a ~5% y la dispersión (s=0.4 en vez de 1.1) para que la correlación del
    # ratio caiga en banda; verificado empíricamente en F1 (reports/cache/f1_sigma_calibration.json).
    out["deuda_total_sistema"] = _zero_inflated_ppf(
        u_block["deuda_total_sistema"].to_numpy(),
        zero_mass=0.05,
        tail_ppf=lambda u: lognorm(s=0.32, scale=6000).ppf(u),
    )
    out["deuda_total_sistema"] = out["deuda_total_sistema"].clip(0, 40000)

    out["peor_calificacion_12m"] = (u_block["peor_calificacion_12m"] * 9).clip(1, 9).round()

    # [DESVIACIÓN DOCUMENTADA] ARCHITECTURE.md §4.5 sugiere ~72% de masa en cero para
    # dias_mora_max_12m. Verificado empíricamente (F1): con 72% de masa puntual el
    # Spearman máximo alcanzable con score_buro/peor_calificacion_12m está acotado
    # (~-0.79 incluso con Pearson de entrada ~-1), y no sobrevive además el paso de
    # SDV. Se redujo a ~45% para poder cumplir la tolerancia de las correlaciones
    # objetivo del trío colineal (§4.2), que es la premisa central del proyecto y
    # tiene prioridad sobre el valor exacto de masa en cero (que la propia arquitectura
    # marca con "~", aproximado).
    out["dias_mora_max_12m"] = _zero_inflated_ppf(
        u_block["dias_mora_max_12m"].to_numpy(),
        zero_mass=0.45,
        tail_ppf=lambda u: (-np.log1p(-u * (1 - np.exp(-180 / 45)))) * 45,
    )
    out["dias_mora_max_12m"] = out["dias_mora_max_12m"].clip(0, 180).round()

    monto_raw = lognorm(s=0.7, scale=3500).ppf(u_block["monto_solicitado"]).clip(500, 25000)
    out["monto_solicitado"] = (monto_raw / 50).round() * 50

    return out


def _sample_protected_and_context(n: int, rng: np.random.Generator) -> pd.DataFrame:
    sexo = rng.choice(["M", "F"], size=n, p=[0.5, 0.5])
    zona = rng.choice(["urbana", "rural"], size=n, p=[0.62, 0.38])
    provincia = rng.choice(
        ["Loja", "Pichincha", "Guayas", "Azuay", "otras"],
        size=n,
        p=[0.15, 0.25, 0.25, 0.15, 0.20],
    )
    return pd.DataFrame({"sexo": sexo, "zona_residencia": zona, "provincia": provincia})


def _sample_tipo_empleo(ctx: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """D15: P(independiente_informal) +8pp si sexo=F; P(agricultor) mucho mayor si rural."""
    n = len(ctx)
    out = np.empty(n, dtype=object)
    for i in range(n):
        probs = TIPO_EMPLEO_BASE_PROBS.copy()
        if ctx["sexo"].iat[i] == "F":
            probs[1] += 0.08
            probs[0] -= 0.08
        if ctx["zona_residencia"].iat[i] == "rural":
            probs[2] += 0.30
            probs[0] -= 0.20
            probs[3] -= 0.10
        probs = np.clip(probs, 0.01, None)
        probs = probs / probs.sum()
        out[i] = rng.choice(TIPO_EMPLEO_CATEGORIES, p=probs)
    return out


def _apply_protected_bias(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """D15 — dependencias inyectadas hacia otras variables, coeficiente CERO en el
    target (eso se aplica en target_definition.py, no aquí)."""
    df = df.copy()
    n = len(df)

    is_f = (df["sexo"] == "F").to_numpy()
    df.loc[is_f, "ingreso_mensual"] = df.loc[is_f, "ingreso_mensual"] * np.exp(-0.18)
    df.loc[is_f, "carga_familiar"] = (df.loc[is_f, "carga_familiar"] + 0.4).clip(upper=6)

    is_rural = (df["zona_residencia"] == "rural").to_numpy()
    df.loc[is_rural, "score_buro"] = (df.loc[is_rural, "score_buro"] - 60).clip(lower=1)
    df.loc[is_rural, "antiguedad_socio_meses"] = (df.loc[is_rural, "antiguedad_socio_meses"] * 1.25).clip(upper=300)

    provincia_income_factor = {"Loja": 0.95, "Pichincha": 1.15, "Guayas": 1.05, "Azuay": 1.0, "otras": 0.90}
    factors = df["provincia"].map(provincia_income_factor).to_numpy()
    df["ingreso_mensual"] = (df["ingreso_mensual"] * factors).clip(460, 3500)

    df["ingreso_mensual"] = df["ingreso_mensual"].clip(460, 3500)
    df["carga_familiar"] = df["carga_familiar"].round().clip(0, 6)
    return df


def generate_seed_table(seed: int, n: int = 2000) -> pd.DataFrame:
    """Genera la tabla semilla de Etapa 1 (ARCHITECTURE.md §4.4-4.5): cópula
    gaussiana sobre el bloque continuo con la estructura de correlación de §4.2,
    atributos protegidos y sesgo indirecto inyectado (D15), categóricas
    independientes, y las dos variables derivadas (`ratio_cuota_ingreso`,
    `ratio_deuda_ingreso`) calculadas al final desde las columnas base.

    Args:
        seed: semilla para todo el muestreo aleatorio de esta tabla.
        n: número de filas a generar (2.000 por defecto, D11: la semilla se
            amplía después a 8.000 filas vía `sdv_synthesizer.synthesize`).

    Returns:
        DataFrame con las 18 variables predictoras + `sexo`, `provincia`,
        `zona_residencia`, `id_solicitud` — sin target todavía (lo agrega
        `target_definition.add_target`).
    """
    rng = np.random.default_rng(seed)

    ctx = _sample_protected_and_context(n, rng)
    tipo_empleo = _sample_tipo_empleo(ctx, rng)

    target_corr = _build_target_correlation_matrix()
    z = rng.multivariate_normal(mean=np.zeros(len(CONTINUOUS_BLOCK)), cov=target_corr.to_numpy(), size=n)
    u_block = pd.DataFrame(norm.cdf(z), columns=CONTINUOUS_BLOCK)
    u_block = u_block.clip(1e-6, 1 - 1e-6)

    continuous = _apply_marginals(u_block, rng)

    plazo = rng.choice(PLAZO_CATEGORIES, size=n, p=PLAZO_PROBS)
    nivel_educacion = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.20, 0.30, 0.25, 0.15, 0.10])

    df = pd.concat([ctx.reset_index(drop=True), continuous.reset_index(drop=True)], axis=1)
    df["tipo_empleo"] = tipo_empleo
    df["plazo_meses"] = plazo
    df["nivel_educacion"] = nivel_educacion

    df = _apply_protected_bias(df, rng)

    lab_cap = (df["edad"] - 18) * 12
    df["antiguedad_laboral_meses"] = np.minimum(df["antiguedad_laboral_meses"], lab_cap).clip(lower=0)

    cuota_i = TASA_CUOTA_ANUAL / 12
    factor = cuota_i / (1 - (1 + cuota_i) ** (-df["plazo_meses"]))
    df["cuota_estimada"] = df["monto_solicitado"] * factor

    df["ratio_cuota_ingreso"] = (df["cuota_estimada"] / df["ingreso_mensual"]).clip(0.02, 0.90)
    df["ratio_deuda_ingreso"] = (df["deuda_total_sistema"] / df["ingreso_mensual"]).clip(0, 20)

    df["id_solicitud"] = [f"SEED-{seed}-{i:06d}" for i in range(n)]

    ordered = [
        "id_solicitud", "sexo", "provincia", "zona_residencia",
        "edad", "nivel_educacion", "carga_familiar", "tipo_empleo",
        "antiguedad_laboral_meses", "antiguedad_socio_meses",
        "ingreso_mensual", "gastos_mensuales",
        "score_buro", "num_operaciones_vigentes", "deuda_total_sistema",
        "peor_calificacion_12m", "dias_mora_max_12m",
        "monto_solicitado", "plazo_meses", "cuota_estimada",
        "ratio_cuota_ingreso", "ratio_deuda_ingreso",
    ]
    return df[ordered]
