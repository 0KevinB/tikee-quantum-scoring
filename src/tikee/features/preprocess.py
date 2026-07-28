"""ColumnTransformer: escalado numérico + one-hot categórico, un transformer por
variable para que `get_variable_groups` pueda mapear cada variable declarada del
esquema a sus columnas de salida (necesario para F4: una máscara de selección por
variable, no por columna dummy). `fit` SOLO en train (ARCHITECTURE.md §3, regla 1)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_VARS = [
    "edad", "carga_familiar", "antiguedad_laboral_meses", "antiguedad_socio_meses",
    "ingreso_mensual", "gastos_mensuales", "score_buro", "num_operaciones_vigentes",
    "deuda_total_sistema", "peor_calificacion_12m", "dias_mora_max_12m",
    "monto_solicitado", "plazo_meses", "nivel_educacion",
    "ratio_cuota_ingreso", "ratio_deuda_ingreso",
]
CATEGORICAL_VARS = ["zona_residencia", "tipo_empleo"]
LEVEL_A_VARS = NUMERIC_VARS + CATEGORICAL_VARS

# expand.py Grupo 1 copia las 18 originales verbatim bajo fXX; f02=zona_residencia,
# f05=tipo_empleo conservan su naturaleza categórica.
LEVEL_B_CATEGORICAL_VARS = ["f02", "f05"]


def build_preprocessor(
    variables: list[str] | None = None, categorical_vars: list[str] | None = None
) -> ColumnTransformer:
    """`categorical_vars` por defecto son las de Nivel A (`zona_residencia`,
    `tipo_empleo`). Nivel B las conserva bajo otro nombre (f02, f05 — Grupo 1 de
    `expand.py` copia las 18 originales verbatim, incluidas las categóricas) y debe
    pasar su propio conjunto explícitamente."""
    variables = variables if variables is not None else LEVEL_A_VARS
    categorical_vars = categorical_vars if categorical_vars is not None else CATEGORICAL_VARS
    transformers = []
    for v in variables:
        if v in categorical_vars:
            transformers.append((v, OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), [v]))
        else:
            transformers.append((v, StandardScaler(), [v]))
    return ColumnTransformer(transformers, remainder="drop")


def get_variable_groups(ct: ColumnTransformer) -> dict[str, list[int]]:
    """Requiere `ct` ya ajustado. Devuelve, por nombre de variable declarada, los
    índices de columna que ocupa en la matriz transformada (1 para numéricas, >1
    para categóricas con one-hot)."""
    groups = {}
    for name, sl in ct.output_indices_.items():
        if name == "remainder":
            continue
        groups[name] = list(range(sl.start, sl.stop))
    return groups
