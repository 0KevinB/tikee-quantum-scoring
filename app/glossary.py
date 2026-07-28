"""Contenido en lenguaje simple: glosario, brazos y diccionario de datos.

Fuente de verdad técnica: ARCHITECTURE.md §8.1 (brazos), config/schema.yaml
(variables), data/external/referencias_publicas.md (calibración). Este módulo
solo traduce eso a texto para gente que ve la app por primera vez y no va a
leer ningún .md — ver docs/GUIA_USUARIO.md para la versión completa en prosa.
"""

from __future__ import annotations

# arm_code -> (selección de variables, clasificador, qué pregunta responde)
ARM_INFO: dict[str, tuple[str, str, str]] = {
    "A0": ("Ninguna (usa todas las variables)", "Regresión logística", "Base interpretable y regulatoriamente segura"),
    "A1": ("Ninguna (usa todas las variables)", "XGBoost", "Techo de desempeño; peor caso de interpretabilidad"),
    "B0": ("LASSO (clásico)", "Regresión logística", "Selección clásica embebida"),
    "B0x": ("LASSO (clásico)", "XGBoost", "Controla el efecto del clasificador"),
    "B1": ("Stepwise por AIC (clásico)", "Regresión logística", "Lo que usa hoy una cooperativa"),
    "C0": ("QUBO + recocido simulado", "Regresión logística", "La hipótesis central del proyecto"),
    "C0x": ("QUBO + recocido simulado", "XGBoost", "Misma hipótesis, otro clasificador"),
    "C1": ("QUBO + enumeración exacta", "Regresión logística", "Techo de optimalidad: ¿el heurístico se equivocó?"),
    "C2": ("QUBO + MILP certificado", "Regresión logística", "Verificación cruzada de C1"),
    "C3": ("QUBO + QAOA (circuito cuántico simulado)", "Regresión logística", "Brecha del paradigma de circuitos"),
    "C4": ("QUBO + búsqueda Tabu", "Regresión logística", "Otro heurístico clásico de comparación"),
    "R": ("k variables al azar × 100", "Regresión logística", "Control: ¿la diferencia supera al azar?"),
}


def _base_code(arm: str) -> str:
    """'C0b' -> 'C0', 'Rb' -> 'R' (Nivel B usa el mismo código + sufijo 'b')."""
    return arm[:-1] if arm.endswith("b") and arm[:-1] in ARM_INFO else arm


def describe_arm(arm: str) -> tuple[str, str, str]:
    """Devuelve (selección, clasificador, pregunta) para un código de brazo, en cualquier nivel."""
    base = _base_code(arm)
    info = ARM_INFO.get(base, ("Método no documentado", "—", "—"))
    if base != arm:
        seleccion, clasificador, pregunta = info
        return (f"{seleccion} (Nivel B, 45 variables candidatas)", clasificador, pregunta)
    return info


def arm_legend_rows(arms: list[str]) -> list[dict[str, str]]:
    """Filas listas para st.dataframe: una tabla 'qué es cada brazo' para los códigos dados."""
    rows = []
    for arm in arms:
        seleccion, clasificador, pregunta = describe_arm(arm)
        rows.append({
            "Brazo": arm,
            "Cómo elige las variables": seleccion,
            "Clasificador": clasificador,
            "Qué pregunta responde": pregunta,
        })
    return rows


# término -> explicación en una o dos frases, sin jerga
TERMS: dict[str, str] = {
    "Brazo": (
        "Cada método distinto que se puso a competir. Es como probar varias recetas para el "
        "mismo plato: todos parten de los mismos datos, pero cada 'brazo' elige un subconjunto "
        "distinto de variables (y a veces un algoritmo distinto para clasificar)."
    ),
    "QUBO": (
        "Sigla de 'Quadratic Unconstrained Binary Optimization'. Es la fórmula matemática que "
        "convierte 'elegir las mejores variables' en un problema de optimización: cada variable "
        "es 0 (no se usa) o 1 (sí se usa), y se busca la combinación que minimiza una fórmula que "
        "castiga variables irrelevantes y variables redundantes entre sí."
    ),
    "Recocido simulado": (
        "El algoritmo clásico (no cuántico) que resuelve el QUBO en la práctica. Prueba cambios al "
        "azar y a veces acepta soluciones un poco peores para no quedarse atascado — inspirado en "
        "cómo se enfría un metal despacio para que sus átomos se acomoden en la estructura más "
        "estable posible."
    ),
    "QAOA": (
        "Quantum Approximate Optimization Algorithm: un algoritmo diseñado para computadoras "
        "cuánticas. En este proyecto se ejecuta en software normal, simulando cómo se comportaría "
        "un circuito cuántico — nunca corrió en un procesador cuántico real. Se incluye solo para "
        "medir qué tan lejos está esa promesa de ser útil hoy."
    ),
    "LASSO": (
        "Método clásico de selección de variables: ajusta un modelo estadístico que automáticamente "
        "reduce a cero el peso de las variables menos útiles."
    ),
    "Stepwise": (
        "Método clásico donde se van agregando variables una por una, revisando en cada paso si "
        "el modelo mejora. Es, a grandes rasgos, lo que usa hoy una cooperativa de crédito típica."
    ),
    "MILP": (
        "Optimización lineal entera mixta: un método más lento pero que garantiza encontrar la "
        "mejor solución posible y certificarla matemáticamente. Se usa aquí como 'juez' para "
        "confirmar si el recocido simulado encontró de verdad el óptimo."
    ),
    "Probabilidad de default": (
        "'Default' = dejar de pagar (más de 90 días de atraso en 12 meses). Es el número que "
        "calcula el modelo: qué tan probable es, según el historial de gente parecida, que este "
        "solicitante caiga en mora si se le aprueba el crédito."
    ),
    "AUC": (
        "Área bajo la curva ROC. Va de 0.5 (el modelo adivina igual que tirar una moneda) a 1.0 "
        "(perfecto). Mide qué tan bien el modelo distingue entre quien va a pagar y quien no."
    ),
    "Curva ROC": (
        "Compara, probando distintos umbrales de decisión, cuántos casos de mora detecta el modelo "
        "contra cuántas falsas alarmas genera al hacerlo."
    ),
    "Curva KS": (
        "Mide la mayor distancia entre la distribución de puntajes de quienes sí caen en mora y "
        "quienes no. Mientras más separadas estén las dos curvas, mejor distingue el modelo."
    ),
    "Matriz de confusión": (
        "Tabla que cuenta, para un umbral de decisión ya fijado, los cuatro resultados posibles: "
        "aprobar a quien sí pagaba (acierto), aprobar a quien no pagó (error costoso), rechazar a "
        "quien sí pagaba (oportunidad perdida) y rechazar a quien no iba a pagar (acierto)."
    ),
    "Índice de Jaccard": (
        "Mide qué tan parecidas son las variables elegidas entre una corrida y otra, de 0 (listas "
        "totalmente distintas) a 1 (idénticas). Un método que cambia de variables cada vez que ve "
        "datos nuevos (Jaccard bajo) no sirve para escribir una política de crédito fija por escrito."
    ),
    "Friedman / Nemenyi": (
        "Pruebas estadísticas que confirman si las diferencias de desempeño entre métodos, vistas a "
        "través de varias corridas, son reales o pura casualidad del muestreo."
    ),
    "Proxy / sesgo indirecto": (
        "Ocurre cuando el modelo, sin usar directamente un dato protegido (sexo, zona de "
        "residencia...), igual puede 'adivinarlo' combinando otras variables — lo que perpetúa "
        "discriminación indirecta sin que nadie se lo haya pedido explícitamente."
    ),
    "AUC de proxy": (
        "Qué tan bien se puede predecir un atributo protegido (por ejemplo, zona de residencia) "
        "usando solo las variables que el modelo sí usa. Cerca de 0.5 = el modelo no da ninguna "
        "pista; cerca de 1.0 = esconde esa información sin decirlo."
    ),
    "Paridad demográfica / impacto dispar": (
        "Miden si el modelo aprueba créditos a tasas parecidas entre grupos (hombres/mujeres, "
        "urbano/rural). La 'regla del 80%' es un estándar legal común: la tasa de aprobación del "
        "grupo menos favorecido no debería caer bajo el 80% de la del grupo más favorecido."
    ),
}


# variable del dataset -> descripción en español simple
VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "edad": "Edad del solicitante, en años.",
    "sexo": "Sexo del solicitante (M/F). Atributo protegido: no se le da al modelo como variable de entrada; solo se vigila para detectar si el modelo la 'adivina' indirectamente.",
    "provincia": "Provincia de residencia. Atributo protegido, mismo propósito que sexo.",
    "zona_residencia": "Vive en zona urbana o rural. Atributo protegido, mismo propósito que sexo.",
    "nivel_educacion": "Nivel educativo alcanzado, de 1 (primaria) a 5 (posgrado).",
    "carga_familiar": "Número de personas que dependen económicamente del solicitante.",
    "tipo_empleo": "Tipo de trabajo: dependiente formal, independiente informal, agricultor o comerciante.",
    "antiguedad_laboral_meses": "Meses que lleva en su empleo o actividad actual.",
    "antiguedad_socio_meses": "Meses que lleva siendo socio de la cooperativa.",
    "ingreso_mensual": "Ingreso mensual declarado, en USD.",
    "gastos_mensuales": "Gastos mensuales declarados, en USD.",
    "score_buro": "Puntaje de historial crediticio del buró de crédito (1-999; más alto es mejor historial).",
    "num_operaciones_vigentes": "Cantidad de créditos u operaciones activas que tiene ahora mismo en el sistema financiero.",
    "deuda_total_sistema": "Deuda total que ya tiene en todo el sistema financiero en este momento, en USD.",
    "peor_calificacion_12m": "La peor calificación de riesgo que tuvo en cualquier operación en los últimos 12 meses (1 = al día, 9 = incobrable).",
    "dias_mora_max_12m": "El mayor número de días que se atrasó en un pago durante los últimos 12 meses.",
    "monto_solicitado": "Monto de crédito que está solicitando, en USD.",
    "plazo_meses": "Plazo en meses en que pagaría el crédito solicitado.",
    "cuota_estimada": "Cuota mensual estimada del crédito, calculada con una tasa de referencia de 16.5% anual.",
    "ratio_cuota_ingreso": "Qué fracción del ingreso mensual se iría en la cuota de este crédito. Variable calculada, no declarada directamente.",
    "ratio_deuda_ingreso": "Cuántas veces el ingreso mensual representa la deuda que ya tiene. Variable calculada.",
    "id_solicitud": "Identificador único de la solicitud. Es sintético, no identifica a ninguna persona real.",
    "default": "Variable objetivo: 1 si el solicitante entró en mora (más de 90 días de atraso) en los 12 meses siguientes, 0 si pagó a tiempo.",
    "p_default_true": "Probabilidad real usada solo para generar los datos sintéticos (no la ve ningún modelo) — el 'terreno de verdad' contra el que se compara todo.",
}
