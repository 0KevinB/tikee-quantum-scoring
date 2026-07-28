# ARCHITECTURE.md — Tikee Quantum-Inspired Scoring

Documento técnico. **Versión 2.0 — alcance ampliado.** Consistente con PLAN.md v2.0.
Detalle suficiente para implementar sin volver a decidir nada.

---

## 1. Stack tecnológico

**Python 3.11** (decisión D1). Las versiones están fijadas exactamente; ver R10 de PLAN.md sobre deriva de API.

```
# requirements.txt
# --- núcleo ---
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.1
xgboost==2.1.1
statsmodels==0.14.2          # VIF, stepwise, prueba de Friedman
pyyaml==6.0.2
pyarrow==17.0.0
joblib==1.4.2                # caché de resultados por semilla

# --- datos sintéticos ---
sdv==1.15.0
sdmetrics==0.16.0            # reportes de calidad, diagnóstico y privacidad

# --- optimización combinatoria ---
dimod==0.12.17
dwave-samplers==1.3.0        # SimulatedAnnealingSampler, TabuSampler, SteepestDescent
highspy==1.7.2               # solucionador MILP exacto para la linealización de Glover
pulp==2.9.0                  # modelado MILP (backend HiGHS/CBC)

# --- circuitos cuánticos simulados ---
qiskit==1.2.4
qiskit-aer==0.15.1
qiskit-algorithms==0.3.1     # QAOA (movido fuera de qiskit.algorithms en Qiskit 1.0)
qiskit-optimization==0.6.1   # QuadraticProgram, MinimumEigenOptimizer

# --- interpretabilidad y visualización ---
shap==0.46.0
matplotlib==3.9.1
plotly==5.23.0
streamlit==1.37.1

# --- pruebas ---
pytest==8.3.2
```

**Notas de compatibilidad que el implementador debe verificar en F0:**

- Si `dwave-samplers` falla: respaldo `dwave-neal==0.6.0`, cambiando `from dwave.samplers import SimulatedAnnealingSampler` por `from neal import SimulatedAnnealingSampler`. La API de `.sample_qubo()` es idéntica. `selection/annealer.py` debe hacerlo con `try/except ImportError`.
- Qiskit 1.0 **eliminó** `qiskit.algorithms`; QAOA vive en `qiskit_algorithms`. También se retiraron `execute()` y `QuantumInstance`. El código debe usar primitivas (`Sampler`, `Estimator`). Verificar contra la documentación vigente, no contra memoria.
- Si SDV no instala: respaldo `data/fallback_copula.py`, implementado y probado en F0, no dejado como plan teórico.
- `highspy` puede usarse directamente o vía `scipy.optimize.milp` (que ya lleva HiGHS incorporado desde SciPy 1.9). Preferir `scipy.optimize.milp` si evita una dependencia.

---

## 2. Estructura del repositorio

```
tikee-quantum-scoring/
├── README.md · PLAN.md · ARCHITECTURE.md
├── requirements.txt · Makefile · .gitignore
├── config/
│   ├── config.yaml               # semillas, n_filas, tasa base, sigma, hiperparámetros, rutas
│   ├── schema.yaml               # esquema declarativo de las 18 variables (fuente de verdad)
│   └── experiments.yaml          # definición declarativa de los brazos (Nivel A y B)
├── data/
│   ├── external/referencias_publicas.md
│   ├── raw/                      # semilla programática pre-SDV, por semilla aleatoria
│   ├── interim/                  # salida cruda de cada sintetizador
│   └── processed/                # dataset_seed{N}.parquet finales
├── src/tikee/
│   ├── config.py                 # carga YAML, fija semillas globales, gestiona el caché
│   ├── data/
│   │   ├── seed_generator.py     # cópula gaussiana + marginales objetivo + sesgo inyectado
│   │   ├── sdv_synthesizer.py    # GaussianCopula / CTGAN / TVAE tras una interfaz común
│   │   ├── fallback_copula.py    # respaldo sin SDV (scipy.stats). Se implementa en F0
│   │   ├── target_definition.py  # modelo estructural + calibración de beta0 y sigma
│   │   ├── fidelity.py           # SDMetrics: Quality, Diagnostic, privacidad
│   │   └── validate.py           # chequeos duros; aborta si el dataset no sirve
│   ├── features/
│   │   ├── preprocess.py         # ColumnTransformer. fit SOLO en train
│   │   └── expand.py             # Nivel A (18) -> Nivel B (45), determinista y trazable
│   ├── selection/
│   │   ├── classical.py          # LASSO (L1) y stepwise forward por AIC
│   │   ├── relevance.py          # información mutua variable-target
│   │   ├── redundancy.py         # Spearman / V de Cramér / razón de correlación
│   │   ├── qubo_builder.py       # matrices -> Q; expansión de la penalización
│   │   ├── annealer.py           # recocido simulado
│   │   ├── tabu.py               # TabuSampler, tercer heurístico
│   │   ├── exact.py              # dimod.ExactSolver. Solo N<=20
│   │   ├── milp.py               # linealización de Glover -> MILP -> HiGHS. Óptimo certificado
│   │   ├── qaoa.py               # QUBO -> Ising -> circuito -> Qiskit Aer. Solo N<=24
│   │   └── random_baseline.py    # k al azar x 100 repeticiones
│   ├── models/
│   │   ├── train.py              # LogReg y XGBoost; CV anidada para XGBoost
│   │   ├── evaluate.py           # métricas + IC bootstrap
│   │   ├── interpret.py          # coeficientes, signos, VIF, SHAP
│   │   └── fairness.py           # métricas por grupo + detección de proxies
│   ├── experiments/
│   │   ├── registry.py           # brazos declarativos
│   │   ├── run_experiment.py     # orquestador de una semilla
│   │   ├── run_multiseed.py      # bucle sobre semillas + caché en disco
│   │   └── stats.py              # Friedman, Nemenyi, diagrama de diferencia crítica
│   ├── viz/plots.py
│   └── cli.py
├── app/
│   ├── Inicio.py
│   ├── glossary.py
│   └── pages/
│       ├── 1_Datos.py · 2_Seleccion.py · 3_Comparacion.py
│       ├── 4_Simulador.py · 5_Estabilidad.py · 6_Documentacion.py
├── notebooks/00_exploracion.ipynb
├── tests/
│   ├── conftest.py · test_schema.py · test_target.py
│   ├── test_qubo_builder.py · test_solvers_agree.py
│   ├── test_expand.py · test_metrics.py · test_fairness.py
├── reports/
│   ├── figures/ · RESULTS.md · INFORME.md
│   └── fidelity/                 # salidas de SDMetrics
└── scripts/run_all.sh
```

---

## 3. Flujo de datos punta a punta

```
config/schema.yaml + config.yaml
        │
        ▼
  ┌─ POR CADA SEMILLA s ∈ {42, 101, ..., 909}  (10 semillas, decisión D11) ─────────┐
  │                                                                                  │
  │  [1] seed_generator.py(s)  ──► data/raw/seed_s.parquet                           │
  │        2.000 filas: cópula gaussiana con la matriz de correlación objetivo       │
  │        (§4.2) + dependencias de atributos protegidos (§4.3) + marginales (§4.5)  │
  │                          │                                                       │
  │  [2] sdv_synthesizer.py ──► data/interim/synth_s.parquet                         │
  │        GaussianCopulaSynthesizer.fit(seed).sample(8000)                           │
  │                          │                                                       │
  │  [3] target_definition.py ──► data/processed/dataset_s.parquet                   │
  │        ratios derivados -> logit estructural -> p_default                        │
  │        -> Bernoulli; beta0 calibrado a 8%; sigma calibrado a AUC∈[0,72;0,82]     │
  │                          │                                                       │
  │  [4] validate.py ──► ABORTA si rangos, tasa base, correlaciones o AUC fallan     │
  │                          │                                                       │
  │  [5] split estratificado 70/30, random_state = s                                 │
  │        X_train,y_train ──► preprocess.fit_transform                              │
  │        X_test ,y_test  ──► preprocess.transform    ← SOLO transform              │
  │                          │                                                       │
  │        ┌─────────────────┴─────────────────┐                                     │
  │        ▼                                   ▼                                     │
  │   NIVEL A (18 vars)                  expand.py ──► NIVEL B (45 vars)             │
  │        │                                   │                                     │
  │  ┌─────┼──────┬──────────┐           ┌─────┼──────┬──────────┐                   │
  │  ▼     ▼      ▼          ▼           ▼     ▼      ▼          ▼                   │
  │ sin  clásica QUBO      random      sin  clásica QUBO      random                 │
  │ sel.  LASSO  ├─ SA (C0)             sel.  LASSO  ├─ SA (C0b)                     │
  │       step   ├─ Exact (C1)                step   ├─ MILP (C2b, con límite)       │
  │              ├─ MILP (C2)                        └─ Tabu (C4b)                   │
  │              └─ QAOA (C3)                        ✗ Exact y QAOA imposibles       │
  │        └─────────────────┬─────────────────┘                                     │
  │                          ▼                                                       │
  │  [6] train.py — mismo clasificador, mismos hiperparámetros, distintas columnas   │
  │  [7] evaluate.py sobre X_test (UNA sola vez)                                     │
  │  [8] fairness.py sobre X_test por grupo protegido                                │
  │                          ▼                                                       │
  │              reports/cache/results_s.json                                        │
  └──────────────────────────┬───────────────────────────────────────────────────────┘
                             ▼
        [9] run_multiseed.py agrega  ──►  reports/results.json
        [10] stats.py: Friedman + Nemenyi + estabilidad de selección
        [11] viz/plots.py + RESULTS.md + INFORME.md  ──►  app Streamlit
```

**Reglas de higiene innegociables:**

1. Toda selección de variables — incluidas las matrices de relevancia y redundancia — se calcula **exclusivamente sobre `X_train`**. Calcularlas sobre el dataset completo es fuga de información y anula el experimento.
2. La expansión a Nivel B (`expand.py`) usa estadísticos (medias, desvíos, cortes de decil, WOE) **ajustados en train y aplicados a test**. Un decil calculado sobre el conjunto completo es fuga.
3. El test se toca **una sola vez por semilla**, al final, para todos los brazos a la vez.

---

## 4. Esquema del dataset sintético

### 4.1 Las 18 variables predictoras (Nivel A)

Bloques: **A** = capacidad de pago · **B** = historial crediticio · **C** = estabilidad/demografía · **D** = ruido.

| # | Variable | Tipo | Rango / categorías | Bloque | Rol previsto |
|---|----------|------|--------------------|--------|--------------|
| 1 | `edad` | int | 18–75 | C | Señal débil, riesgo en U |
| 2 | `zona_residencia` | cat | urbana, rural | D | **Ruido en el target**, pero proxy de `provincia` |
| 3 | `nivel_educacion` | ordinal | 1=primaria … 5=posgrado | D | Ruido controlado |
| 4 | `carga_familiar` | int | 0–6 | C | Señal débil positiva |
| 5 | `tipo_empleo` | cat | dependiente_formal, independiente_informal, agricultor, comerciante | C | Señal media |
| 6 | `antiguedad_laboral_meses` | int | 0–360 | C | Señal media negativa |
| 7 | `antiguedad_socio_meses` | int | 1–300 | C | Señal media negativa |
| 8 | `ingreso_mensual` | float USD | 460–3.500 | A | Señal vía ratios |
| 9 | `gastos_mensuales` | float USD | 200–3.000 | A | Colineal con 8 |
| 10 | `score_buro` | int | 1–999 | B | **Señal fuerte negativa** |
| 11 | `num_operaciones_vigentes` | int | 0–8 | B | Señal media positiva |
| 12 | `deuda_total_sistema` | float USD | 0–40.000 | B | Colineal con 8 y 11 |
| 13 | `peor_calificacion_12m` | ordinal | 1=A1 … 9=E | B | **Señal fuerte**, colineal con 10 y 14 |
| 14 | `dias_mora_max_12m` | int | 0–180 | B | **Señal fuerte**, colineal con 10 y 13 |
| 15 | `monto_solicitado` | float USD | 500–25.000 | A | Señal vía cuota |
| 16 | `plazo_meses` | int | {6,12,18,24,36,48,60} | A | Señal vía cuota |
| 17 | `ratio_cuota_ingreso` | float derivada | 0,02–0,90 | A | **Señal fuerte positiva** |
| 18 | `ratio_deuda_ingreso` | float derivada | 0–20 | A | Señal media positiva |

Auxiliar de cálculo, no predictora: `cuota_estimada = monto·(i/12) / (1 − (1+i/12)^(−plazo))` con `i = 0,165` anual (tasa referencial de consumo, configurable).

### 4.2 Estructura de correlación intencional

Sin colinealidad la hipótesis no se puede probar. Objetivos de |ρ| de Spearman, verificados en `validate.py` con tolerancia **±0,07**:

**Bloque B — historial (el más colineal; imita datos de buró reales):**

| Par | ρ objetivo | Justificación |
|-----|-----------|---------------|
| `score_buro` ↔ `peor_calificacion_12m` | **−0,85** | La calificación es función del historial que alimenta el score |
| `score_buro` ↔ `dias_mora_max_12m` | **−0,80** | Ídem |
| `peor_calificacion_12m` ↔ `dias_mora_max_12m` | **+0,88** | La calificación regulatoria se define **por** tramos de días de mora |
| `deuda_total_sistema` ↔ `num_operaciones_vigentes` | +0,60 | Más operaciones, más saldo |

Este trío (10, 13, 14) es el **caso de prueba central**: son casi la misma información. Un buen selector debe quedarse con una o dos, no con las tres.

**Bloque A — capacidad de pago:**

| Par | ρ objetivo |
|-----|-----------|
| `ingreso_mensual` ↔ `gastos_mensuales` | **+0,75** |
| `ingreso_mensual` ↔ `monto_solicitado` | +0,55 |
| `ratio_cuota_ingreso` ↔ `monto_solicitado` | +0,65 |
| `ratio_cuota_ingreso` ↔ `ingreso_mensual` | −0,45 |
| `ratio_deuda_ingreso` ↔ `deuda_total_sistema` | +0,70 |

**Bloque C — estabilidad:**

| Par | ρ objetivo |
|-----|-----------|
| `edad` ↔ `antiguedad_laboral_meses` | +0,55 |
| `edad` ↔ `antiguedad_socio_meses` | +0,40 |
| `edad` ↔ `carga_familiar` | +0,30 |

**Bloque D:** `nivel_educacion` con |ρ| < 0,15 contra todo. `zona_residencia` es distinta: tiene coeficiente **cero en el target** pero **sí** está correlacionada con otras variables (§4.3) — es la trampa más interesante del diseño, porque un selector guiado solo por relevancia marginal puede recogerla por su asociación con el bloque C.

**Restricción dura:** `antiguedad_laboral_meses ≤ (edad − 18)·12`, impuesta por recorte posterior en `validate.py`.

### 4.3 Atributos protegidos y sesgo indirecto inyectado (decisión D15)

Se generan pero **nunca entran al modelo como predictores**: `sexo` (M/F), `provincia` (Loja, Pichincha, Guayas, Azuay, otras), `id_solicitud`.

El punto crítico del diseño: si los atributos protegidos fueran independientes de todo, la auditoría de equidad no encontraría nada **por construcción** y sería teatro. Por eso se inyectan dependencias hacia otras variables, **manteniendo coeficiente cero en el target**:

| Atributo protegido | Dependencia inyectada | Referencia orientativa |
|--------------------|----------------------|------------------------|
| `sexo = F` | `log(ingreso_mensual)` desplazado −0,18 | Brecha de ingresos laborales, INEC |
| `sexo = F` | P(`tipo_empleo` = independiente_informal) mayor en ~8 pp | Informalidad por sexo, ENEMDU |
| `sexo = F` | `carga_familiar` +0,4 en promedio | Jefatura de hogar |
| `zona_residencia = rural` | `score_buro` desplazado −60 puntos | Menor bancarización / historial más corto |
| `zona_residencia = rural` | P(`tipo_empleo` = agricultor) mucho mayor | Estructura del empleo rural |
| `zona_residencia = rural` | `antiguedad_socio_meses` mayor | Las cooperativas rurales tienen vínculos más largos |
| `provincia` | Modula `ingreso_mensual` y la mezcla urbana/rural | Diferencias regionales |

**Coeficiente en el target: `sexo` = 0,00 y `zona_residencia` = 0,00, explícitamente.**

Consecuencia experimental: **no hay discriminación directa por construcción.** Cualquier disparidad que la auditoría encuentre es **sesgo indirecto puro**, propagado a través de variables correlacionadas. Es el caso que ocurre en la práctica y el único interesante de medir. Esto debe estar escrito en el informe: sin ello, un lector podría creer que el sesgo hallado se metió a mano en el target.

### 4.4 Definición del target

**Definición de negocio (la que va al informe):**

> `default = 1` si la operación alcanza **más de 90 días de mora** en cualquier momento de los **12 meses** posteriores al desembolso. Corresponde a las categorías de riesgo **C2, D y E** de la calificación SEPS. Los casos de 1–90 días se consideran buen pagador (`default = 0`): son mora temprana, no incumplimiento.

Definición estándar de industria ("bad ≡ 90+ dpd, ventana de 12 meses"), defendible ante un tribunal.

**Definición operativa (cómo se genera) — el ground truth del experimento (D3):**

```
z = β0
  + 2.20 · Z(ratio_cuota_ingreso)        # A — fuerte
  + 0.55 · Z(ratio_deuda_ingreso)        # A — media
  - 1.80 · Z(score_buro)                 # B — fuerte
  + 1.40 · Z(dias_mora_max_12m)          # B — fuerte
  + 0.30 · Z(peor_calificacion_12m)      # B — media, redundante a propósito con las dos anteriores
  + 0.45 · Z(num_operaciones_vigentes)   # B — media
  - 0.60 · Z(antiguedad_socio_meses)     # C — media
  - 0.35 · Z(antiguedad_laboral_meses)   # C — media
  + 0.25 · Z(carga_familiar)             # C — débil
  + 0.40 · [tipo_empleo == independiente_informal]
  + 0.20 · [tipo_empleo == agricultor]
  - 0.15 · Z((edad − 40)²)               # C — riesgo en U
  + 0.00 · Z(nivel_educacion)            # D — cero explícito
  + 0.00 · [zona_residencia == rural]    # D — cero explícito
  + 0.00 · [sexo == F]                   # PROTEGIDO — cero explícito
  + ε,   ε ~ Normal(0, σ)

p_default = sigmoide(z)
default   ~ Bernoulli(p_default)
```

`Z(·)` = estandarización sobre la población sintética.

**Calibración de β0:** bisección hasta `mean(p_default) = 0,08 ± 0,002` (D4).

**Calibración de σ (riesgo R1, el más importante):** σ controla la separabilidad. Objetivo: que el modelo base (logística con las 18 variables) obtenga **AUC de test en [0,72 · 0,82]**, banda realista de un scoring de cooperativa. Procedimiento en F1: bisección sobre σ (partir de σ=1,3), 5–8 iteraciones de generar → entrenar logística → medir AUC. `validate.py` **rechaza** el dataset fuera de banda. σ se fija una vez y se reutiliza en las 10 semillas.

**Ground truth de relevancia (Nivel A):**

- **Relevantes (11):** `ratio_cuota_ingreso`, `ratio_deuda_ingreso`, `score_buro`, `dias_mora_max_12m`, `peor_calificacion_12m`, `num_operaciones_vigentes`, `antiguedad_socio_meses`, `antiguedad_laboral_meses`, `carga_familiar`, `tipo_empleo`, `edad`.
- **Irrelevantes (2):** `nivel_educacion`, `zona_residencia`.
- **Instrumentales (5):** `ingreso_mensual`, `gastos_mensuales`, `monto_solicitado`, `plazo_meses`, `deuda_total_sistema` — no entran directamente al logit pero son los ingredientes de los ratios que sí entran. Categoría propia: seleccionarlas no es un error, pero seleccionarlas *en lugar* de los ratios sí es peor.

Esto habilita métricas que ningún trabajo con datos reales puede reportar (§8.5).

### 4.5 Método de generación en dos etapas

SDV no puede inventar una estructura de correlación que no exista en su entrada. Por eso:

**Etapa 1 — semilla programática (`seed_generator.py`).** Muestreo de una normal multivariante con la matriz de correlación objetivo de §4.2 y §4.3, y transformación de cada columna a su marginal deseada por la transformada integral de probabilidad (`norm.cdf` → `ppf` de la distribución objetivo). Marginales:

| Variable | Marginal |
|----------|----------|
| `ingreso_mensual` | lognormal, mediana ≈ 620 USD, cola hasta 3.500 |
| `gastos_mensuales` | lognormal condicionada al ingreso |
| `score_buro` | beta reescalada a 1–999, sesgada a la derecha (masa en 700–900) |
| `dias_mora_max_12m` | mixtura inflada en cero: ~72% exactamente 0, resto exponencial truncada a 180 |
| `deuda_total_sistema` | inflada en cero (~18%) + lognormal |
| `edad` | gamma desplazada, moda ≈ 38 |
| `monto_solicitado` | lognormal redondeada a múltiplos de 50 |
| `plazo_meses` | categórica sobre {6,12,18,24,36,48,60} |
| categóricas | multinomial con las proporciones de `referencias_publicas.md` |

**Etapa 2 — SDV (`sdv_synthesizer.py`).** `GaussianCopulaSynthesizer` con `SingleTableMetadata` derivada de `schema.yaml`; ajuste sobre las 2.000 filas semilla, muestreo de 8.000. Aporta regularización de marginales, tratamiento homogéneo de categóricas, y la trazabilidad de usar una herramienta estándar.

**Este orden es deliberado y debe explicarse en el informe:** la semilla impone la estructura causal que queremos estudiar; SDV la reproduce y la amplía. Presentar SDV como si hubiera "descubierto" las correlaciones sería deshonesto (anti-criterio de PLAN.md §2.7).

### 4.6 Nivel B — expansión determinista a 45 candidatas (decisión D9)

`expand.py` produce exactamente 45 columnas de forma determinista y trazable. Todos los estadísticos (medias, desvíos, cortes de decil, tablas WOE) se **ajustan en train**.

**Grupo 1 — las 18 originales** (`f01`–`f18`).

**Grupo 2 — 12 interacciones** (productos de variables estandarizadas), `f19`–`f30`:

| ID | Interacción | Motivación |
|----|-------------|------------|
| f19 | `ratio_cuota_ingreso × score_buro` | Capacidad × historial: la interacción de negocio más plausible |
| f20 | `ratio_cuota_ingreso × dias_mora_max_12m` | Ídem |
| f21 | `score_buro × num_operaciones_vigentes` | Historial × exposición |
| f22 | `ingreso_mensual × carga_familiar` | Ingreso per cápita implícito |
| f23 | `antiguedad_socio_meses × score_buro` | Relación con la cooperativa × historial externo |
| f24 | `monto_solicitado × plazo_meses` | Exposición total |
| f25 | `deuda_total_sistema × num_operaciones_vigentes` | Redundante a propósito |
| f26 | `edad × antiguedad_laboral_meses` | Redundante a propósito |
| f27 | `ratio_deuda_ingreso × peor_calificacion_12m` | Cruce de bloques A×B |
| f28 | `gastos_mensuales × carga_familiar` | Presión de gasto |
| f29 | `[tipo_empleo=informal] × ratio_cuota_ingreso` | Interacción con efecto real en el logit |
| f30 | `[zona_residencia=rural] × score_buro` | **Trampa:** contiene una variable de coeficiente cero |

**Grupo 3 — 8 transformaciones no lineales**, `f31`–`f38`:
`log(ingreso_mensual)` · `log1p(deuda_total_sistema)` · `(edad−40)²` · `sqrt(monto_solicitado)` · **WOE de deciles de `score_buro`** · `[dias_mora_max_12m > 0]` · `[dias_mora_max_12m > 30]` · `log1p(antiguedad_socio_meses)`.

**Grupo 4 — 5 recombinaciones financieras**, `f39`–`f43`:
`excedente = ingreso − gastos` · `cobertura_cuota = excedente / cuota_estimada` · `deuda_media_operacion = deuda_total / (num_operaciones+1)` · `monto_sobre_ingreso = monto_solicitado / ingreso` · `ingreso_per_capita = ingreso / (1 + carga_familiar)`.

**Grupo 5 — 2 variables de ruido puro**, `f44`–`f45`: `ruido_normal ~ N(0,1)` y `ruido_uniforme ~ U(0,1)`, independientes de todo. **Un selector que incluya cualquiera de estas dos está sobreajustando, y es un fallo detectable de forma inequívoca.**

**Ground truth de Nivel B:** una variable expandida es relevante si es una transformación monótona o un producto que involucra únicamente variables relevantes de §4.4. Las claramente irrelevantes son `f30`, `f44`, `f45` más las derivadas de `nivel_educacion`. `expand.py` debe emitir esta etiqueta junto a cada columna, no reconstruirla después a mano.

**Por qué así y no ampliando el esquema a 45 variables crudas:** conserva la trazabilidad al ground truth (sabemos exactamente de qué original deriva cada expandida), y reproduce el fenómeno real de la ingeniería de variables en scoring, donde la explosión de candidatas viene de transformaciones, no de nuevas fuentes de datos.

### 4.7 Calibración con referencias públicas

`data/external/referencias_publicas.md` registra, para cada rango: fuente, año, valor tomado y transformación. Fuentes previstas: **SEPS** (boletines del sector cooperativo: morosidad por segmento → tasa base 8%, distribución por tipo de crédito, montos promedio); **burós de crédito de Ecuador** (escala 1–999 y forma de la distribución); **INEC/ENEMDU** (distribución de ingresos, informalidad → proporciones de `tipo_empleo`, estructura de hogares → `carga_familiar`, reparto urbano/rural, brecha de ingresos por sexo → §4.3); **Junta de Política y Regulación Monetaria** (tasas máximas por segmento → tasa de la cuota).

**Regla: rango orientativo, no dato.** Toda cifra no verificable en fuente pública se marca `[SUPUESTO]` con su justificación. Un supuesto declarado es aceptable; un supuesto disfrazado de dato citado, no.

---

## 5. Validación de fidelidad sintética (E11, E12)

`fidelity.py`, ejecutado en F2 sobre los tres sintetizadores.

**Métricas de SDMetrics:**

| Reporte | Qué mide | Umbral de aceptación |
|---------|----------|---------------------|
| `QualityReport` — Column Shapes | KS complement / TV complement por columna | Puntuación global > 0,85 |
| `QualityReport` — Column Pair Trends | Similitud de correlación por pares | Puntuación global > 0,80 |
| `DiagnosticReport` — Data Validity | Rangos, categorías, tipos | 1,00 (debe ser perfecto) |
| `DiagnosticReport` — Data Structure | Columnas coincidentes | 1,00 |
| `NewRowSynthesis` | Filas sintéticas que no son copias de la semilla | > 0,95 |
| Distancia al registro más cercano | Riesgo de reidentificación | Reportar distribución, no umbral |

**Métrica propia y decisiva — preservación de la estructura objetivo:** error absoluto medio entre las 13 correlaciones objetivo de §4.2 y las observadas en el dataset sintético. **Esta es la métrica que manda**, porque es la premisa del proyecto. Umbral: EAM < 0,05.

**Comparación de sintetizadores (E12):**

| Sintetizador | Tiempo esperado | Fortaleza esperada | Debilidad esperada |
|--------------|----------------|--------------------|--------------------|
| `GaussianCopulaSynthesizer` | segundos | Preserva correlaciones **por construcción** | Marginales forzadas a familias paramétricas |
| `CTGANSynthesizer` (300 épocas) | 10–40 min CPU | Marginales multimodales más fieles | Puede no preservar correlaciones finas; inestable entre corridas |
| `TVAESynthesizer` | 5–20 min CPU | Intermedio | Ídem |

**La tensión que el informe debe explicitar (decisión D14):** es posible que CTGAN gane en puntuación de calidad de SDMetrics y aun así se elija la cópula gaussiana, porque las métricas de fidelidad general y el objetivo de investigación **no coinciden aquí**: necesitamos una estructura de correlación conocida y controlada, no la máxima fidelidad marginal. Elegir CTGAN porque puntúa mejor destruiría el objeto de estudio. Reportar esta tensión con números es material de informe de primer nivel; ocultarla sería un error metodológico.

---

## 6. Formulación QUBO

### 6.1 Variables de decisión

```
x_i ∈ {0,1},   i = 1..N       x_i = 1  ⟺  la variable i entra al modelo
```

`N = 18` (Nivel A), `N = 45` (Nivel B). Las categóricas (`tipo_empleo`, `zona_residencia`) reciben **una sola** binaria a nivel de variable, no una por categoría: se selecciona la variable completa o ninguna de sus columnas one-hot. Correcto tanto estadística como regulatoriamente — no tiene sentido justificar un crédito con "solo la categoría agricultor de tipo de empleo".

### 6.2 Matrices de entrada (calculadas SOLO sobre train)

**Relevancia** `R ∈ [0,1]^N`:

```
R_i = MI(x_i ; y)  normalizada:  R_i = MI_i / max_j(MI_j)
```

Con `sklearn.feature_selection.mutual_info_classif` (`random_state=s`, `n_neighbors=3`). Captura relaciones no lineales, a diferencia de la correlación — importante para `edad`, cuyo riesgo es en U y cuya correlación lineal es casi nula.

**Redundancia** `C ∈ [0,1]^{N×N}`, `C_ii = 0`:

| Tipo de par | Medida |
|-------------|--------|
| continua ↔ continua | \|ρ de Spearman\| |
| categórica ↔ categórica | V de Cramér |
| categórica ↔ continua | razón de correlación η |

Las tres viven en [0,1] y son comparables. Se elige Spearman en lugar de información mutua entre variables por costo y estabilidad: MI por pares sobre continuas es lenta y ruidosa; Spearman es inmediata y captura la monotonicidad, que es la forma dominante de redundancia en este dataset. **Simplificación deliberada; debe declararse en el informe.**

### 6.3 Función objetivo

Se minimiza la energía:

```
E(x) = − α · Σ_i R_i · x_i                    ← premiar relevancia
       + β · Σ_{i<j} C_ij · x_i · x_j          ← penalizar redundancia entre seleccionadas
       + λ · ( Σ_i x_i − k )²                  ← forzar cardinalidad ≈ k
```

Los dos primeros términos son **mRMR (máxima relevancia, mínima redundancia)** expresado como QUBO. El tercero mete la restricción en el objetivo, que es la forma canónica en un QUBO: "unconstrained" significa que las restricciones **deben** absorberse en la función objetivo.

### 6.4 Reducción a la forma matricial

Como `x_i ∈ {0,1}` implica `x_i² = x_i`:

```
(Σ_i x_i − k)²  =  Σ_i x_i  +  2·Σ_{i<j} x_i x_j  −  2k·Σ_i x_i  +  k²
                =  Σ_i (1 − 2k)·x_i  +  2·Σ_{i<j} x_i x_j  +  k²
```

`k²` es constante y no altera el argmin: se descarta (se guarda si se quieren reportar energías absolutas).

**Matriz QUBO final — esto es literalmente lo que se implementa:**

```
Q[i][i] = −α · R_i  +  λ · (1 − 2k)                 (lineales, diagonal)
Q[i][j] =  β · C_ij  +  2λ            para i < j     (cuadráticos, triangular superior)
```

Se entrega como diccionario `{(i,i): valor, (i,j): valor}` en triangular superior; `sample_qubo()` lo acepta directamente.

**Hiperparámetros:**

| Parámetro | Valor | Cómo se fija |
|-----------|-------|--------------|
| `α` | 1,0 | Fijo por normalización (`R` ya está en [0,1]) |
| `β` | barrido {0,25 · 0,5 · 1,0 · 2,0} | **AUC en CV de 5 pliegues sobre train.** Nunca en test |
| `k` | Nivel A: {5,6,7,8,9,10} · Nivel B: {6,8,10,12,15,20} | Ídem |
| `λ` | `2 · (α·max R_i + β·max C_ij · N)` | Regla, no barrido: debe dominar para que la restricción se cumpla |

Barrido de 24 combinaciones por nivel y semilla. Cada QUBO de N=18 se resuelve en milisegundos; N=45 en segundos.

**Verificación obligatoria de λ:** si alguna solución devuelve `Σxᵢ ≠ k`, λ es demasiado bajo — duplicarlo y volver a resolver. `annealer.py` debe comprobarlo y avisar, nunca fallar en silencio. La prueba cruzada definitiva está en §7.3: el MILP con cardinalidad como restricción **dura** debe dar la misma solución que el QUBO penalizado; si difieren, λ es insuficiente.

### 6.5 Extensiones documentadas pero no implementadas

- **Costo de adquisición:** `+ γ·Σ_i c_i·x_i` con `c_i` = costo de obtener la variable (una consulta al buró cuesta dinero real). Convierte el problema en selección consciente del costo operativo.
- **Inclusión obligatoria por política:** forzar `x_score_buro = 1` con un premio lineal grande, o fijando la variable en el sampler.
- **Restricción de equidad:** penalizar la selección conjunta de variables con alta asociación al atributo protegido. Es la vía natural de mitigación, y queda como trabajo futuro (decisión D16).
- **Embedding en topología Pegasus** para hardware D-Wave real, con la estimación de qubits físicos requeridos por un QUBO denso.

---

## 7. Solucionadores

Cinco métodos sobre **el mismo `Q`**. Esta es la contribución técnica del trabajo: separar el mérito de la formulación, del heurístico, y del paradigma de cómputo.

### 7.0 Tabla de aplicabilidad

| Solucionador | Nivel A (N=18) | Nivel B (N=45) | Garantía |
|--------------|:--------------:|:--------------:|----------|
| Recocido simulado (`SimulatedAnnealingSampler`) | ✅ | ✅ | Ninguna (heurístico) |
| Tabu (`TabuSampler`) | ✅ | ✅ | Ninguna (heurístico) |
| Enumeración exacta (`dimod.ExactSolver`) | ✅ 262.144 estados | ❌ 3,5×10¹³ estados | Óptimo global |
| **MILP Glover + HiGHS** | ✅ | ✅ (con límite de tiempo) | **Óptimo certificado o gap acotado** |
| QAOA simulado (Qiskit Aer) | ✅ | ❌ **563 TB de RAM** | Ninguna |

### 7.1 Recocido simulado — brazos C0 / C0b

```python
try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler

sampleset = SimulatedAnnealingSampler().sample_qubo(
    Q, num_reads=1000, num_sweeps=1000, seed=s
)
best = sampleset.first.sample
```

Se registran: mejor energía, energía media, número de lecturas que alcanzan la mejor energía (proxy de robustez del paisaje) y tiempo de pared.

### 7.2 Enumeración exacta — brazo C1 (solo Nivel A)

```python
import dimod
best_exact = dimod.ExactSolver().sample_qubo(Q).first
```

Enumera 2¹⁸ = 262.144 estados en segundos. **Solo para N ≤ 20.**

### 7.3 MILP con linealización de Glover — brazos C2 / C2b (decisión D10)

La adición más importante de la v2.0. Introduciendo `y_ij = x_i·x_j`:

```
min  Σ_i Q_ii·x_i  +  Σ_{i<j} Q_ij·y_ij
s.a.
     y_ij ≥ x_i + x_j − 1                    para todo (i,j) con Q_ij > 0
     y_ij ≥ 0                                 para todo (i,j) con Q_ij > 0
     y_ij ≤ x_i ,  y_ij ≤ x_j                para todo (i,j) con Q_ij < 0
     Σ_i x_i = k                              ← cardinalidad como restricción DURA
     x_i ∈ {0,1}
```

**Detalle de eficiencia que hay que implementar:** solo se necesitan las dos desigualdades "hacia abajo" cuando `Q_ij > 0` (el minimizador ya empuja `y` hacia abajo) y las dos "hacia arriba" cuando `Q_ij < 0`. Esto reduce a la mitad las restricciones frente a la linealización completa. En N=45 son 990 pares → ~1.980 restricciones en vez de ~3.960.

**Dos usos, ambos obligatorios:**

1. **Con `Σx = k` dura** — da el óptimo certificado del problema *que realmente queremos resolver*.
2. **Con la penalización λ en el objetivo y sin restricción dura** — debe dar la misma solución que 1. **Si difieren, λ es insuficiente** y el QUBO estaba resolviendo otro problema. Esta es la verificación de correctitud más valiosa del proyecto.

Nivel A: debe coincidir exactamente con `ExactSolver` (checkpoint F4 de PLAN.md). Nivel B: límite de 600 s; si no cierra, se reporta el **gap de optimalidad** de HiGHS, que sigue acotando cuán lejos puede estar el recocido.

### 7.4 QAOA simulado — brazo C3 (solo Nivel A; decisiones D17, D18)

**Mapeo QUBO → Ising.** Con `x_i = (1 − z_i)/2`, `z_i ∈ {−1,+1}` autovalor de `Z_i`:

```
h_i    = −(1/2)·Q_ii − (1/4)·Σ_{j≠i} Q_ij
J_ij   =  (1/4)·Q_ij
offset =  (1/2)·Σ_i Q_ii + (1/4)·Σ_{i<j} Q_ij

H_C = Σ_i h_i·Z_i + Σ_{i<j} J_ij·Z_i·Z_j + offset
H_B = Σ_i X_i                                        (mezclador estándar)
```

**Circuito.** Estado inicial `|+⟩^⊗N`. `p` capas, cada una:
`e^{−iγ_l H_C}` = N puertas `RZ` + N(N−1)/2 puertas `RZZ` → **171 puertas por capa en N=18** (QUBO denso).
`e^{−iβ_l H_B}` = N puertas `RX`.
Con `p=3`: ~570 puertas. Perfectamente simulable por vector de estado.

**Ejecución.** `p ∈ {1,2,3}`, optimizador clásico COBYLA sobre los `2p` parámetros, **expectativa exacta por vector de estado** (sin ruido de muestreo, D18), backend `qiskit_aer.AerSimulator(method="statevector")`. Vía `qiskit_optimization.QuadraticProgram` + `MinimumEigenOptimizer(QAOA(...))`, o construyendo el circuito a mano si la API de la versión da problemas.

**Métricas registradas — ninguna de ellas es "¿ganó?":**

| Métrica | Qué responde |
|---------|--------------|
| Brecha de energía respecto al óptimo conocido `(E_QAOA − E*)/\|E*\|` | Cuán cerca llega |
| Probabilidad de medir la cadena óptima | Calidad real de la distribución de salida |
| Número de evaluaciones del circuito | Costo del bucle clásico externo |
| Tiempo de pared vs. `ExactSolver` | El dato incómodo y honesto |
| AUC del modelo entrenado con su selección | Comparabilidad con el resto de brazos |

**Encuadre obligatorio en el informe (D17):** en N=18 el óptimo global ya se conoce por enumeración, y simular QAOA es **varios órdenes de magnitud más caro** que esa enumeración. QAOA no puede ganar aquí, y en Nivel B no puede ni ejecutarse (563 TB de RAM). Se incluye para **cuantificar la brecha entre la promesa cuántica y la práctica en un problema de tamaño real de cooperativa**, que es un resultado legítimo y más valioso que insinuar una ventaja inexistente.

### 7.5 Tabu — brazo C4b (Nivel B)

`TabuSampler` de `dwave-samplers` como segundo heurístico. Responde una pregunta que el recocido solo no puede: si el recocido iguala al óptimo del MILP, ¿es porque el recocido es bueno o porque **el paisaje del problema es fácil**? Si Tabu también lo iguala, la respuesta es la segunda — y eso desactiva buena parte del argumento a favor de los métodos cuántico-inspirados en este dominio. Es una comparación barata y muy informativa.

---

## 8. Diseño experimental

### 8.1 Brazos

**Nivel A (N=18):**

| ID | Selección | Clasificador | Qué pregunta responde |
|----|-----------|--------------|----------------------|
| **A0** | Ninguna | Regresión logística | Base interpretable y regulatoriamente segura |
| **A1** | Ninguna | XGBoost (CV anidada) | Techo de desempeño; peor caso de interpretabilidad |
| **B0** | LASSO (L1, `C` por CV) | Logística | Selección clásica embebida |
| **B0x** | LASSO | XGBoost | Controla el efecto del clasificador |
| **B1** | Stepwise forward por AIC | Logística | Lo que usa hoy una cooperativa |
| **C0** | QUBO + recocido simulado | Logística | **La hipótesis** |
| **C0x** | QUBO + recocido | XGBoost | Ídem, otro clasificador |
| **C1** | QUBO + enumeración exacta | Logística | Techo de optimalidad: aísla formulación de heurístico |
| **C2** | QUBO + MILP certificado | Logística | Verificación cruzada de C1 |
| **C3** | QUBO + QAOA simulado | Logística | Brecha del paradigma de circuitos |
| **R** | k al azar × 100 | Logística | **Control:** ¿la diferencia supera al azar? |

**Nivel B (N=45):** `A0b`, `A1b`, `B0b`, `B1b`, `C0b` (recocido), `C2b` (MILP con límite), `C4b` (Tabu), `Rb`. **Sin C1b ni C3b** — imposibles (§7.0).

**Control de comparación justa:** todos los brazos comparten semilla, split, preprocesamiento y — cuando comparten clasificador — exactamente los mismos hiperparámetros. **Lo único que cambia entre brazos es el subconjunto de columnas.** Cualquier otra diferencia invalida la comparación.

### 8.2 Protocolo

- **Split** estratificado 70/30, `random_state = s`. n_train ≈ 5.600, n_test ≈ 2.400 (≈192 defaults en test — suficiente para un AUC estable, escaso para métricas en la cola; hay que decirlo).
- **Selección de hiperparámetros:** CV estratificada de 5 pliegues **dentro de train** para `C` de LASSO y para β, k del QUBO.
- **XGBoost (E13):** CV **anidada**, externa 5 × interna 3, `RandomizedSearchCV` con 30 configuraciones sobre `max_depth ∈ [2,6]`, `learning_rate ∈ [0,01, 0,3]`, `n_estimators ∈ [100, 600]`, `subsample ∈ [0,6, 1,0]`, `colsample_bytree ∈ [0,6, 1,0]`, `min_child_weight ∈ [1, 20]`, `reg_lambda ∈ [0,1, 10]`. Solo sobre **3 semillas** (decisión D13); en las 10 semillas se usa el protocolo de holdout con los hiperparámetros medianos hallados.
- **Test:** se toca **una sola vez por semilla**, al final, para todos los brazos a la vez.
- **Sin balanceo de clases** (D20).
- **Umbral** fijado por máximo KS en train, congelado y aplicado a test (D19).

### 8.3 Métricas

**Discriminación:**

| Métrica | Definición | Lectura de referencia |
|---------|-----------|----------------------|
| **AUC-ROC** | `roc_auc_score` | Primaria (D8). Banda esperada 0,72–0,82 |
| **KS** | `max(TPR − FPR)` | Métrica de industria. > 0,30 aceptable en consumo |
| **PR-AUC** | `average_precision_score` | Más informativa con 8% de positivos |
| **Gini** | `2·AUC − 1` | Lenguaje del sector |

**Clasificación en el umbral operativo:** precisión, recall y F1 sobre `default = 1`; matriz de confusión completa; y **tasa de aprobación** resultante — métrica de negocio: un modelo que gana AUC pero aprueba al 40% de los solicitantes es inservible para la cooperativa.

**Calibración:** puntuación de Brier. Importa porque una cooperativa provisiona sobre la probabilidad, no sobre la etiqueta.

**Costo computacional:** tiempo de pared del paso de selección. Si el QUBO es 100× más lento por 0,003 de AUC, ese número **es** la conclusión.

### 8.4 Métricas de calidad de la selección (posibles solo por el ground truth)

- Número de variables seleccionadas.
- **Precisión de recuperación:** proporción de seleccionadas que son verdaderamente relevantes.
- **Cobertura:** proporción de las relevantes que fueron capturadas.
- **Falsos positivos:** ¿incluyó `nivel_educacion` o `zona_residencia`? En Nivel B: ¿incluyó `f44`/`f45` (ruido puro)? **Fallo inequívoco.**
- **Prueba del trío colineal:** ¿cuántas de `{score_buro, peor_calificacion_12m, dias_mora_max_12m}` sobreviven? **Indicador clave de la hipótesis** — el QUBO debería quedarse con 1–2, no con 3.
- **Preferencia ratio vs. instrumental:** ¿eligió `ratio_cuota_ingreso` o eligió `ingreso` y `monto` por separado?
- **VIF máximo** del modelo final.

### 8.5 Estabilidad multi-semilla (E9)

10 semillas; cada una regenera el dataset **y** re-hace el split (D11) — solo re-splitear subestima la incertidumbre.

- Por brazo: media, desviación, mínimo y máximo de AUC, KS y número de variables.
- **Frecuencia de selección:** mapa de calor variable × método, con la proporción de semillas en que cada variable fue elegida. **Un método que elige variables distintas en cada semilla es inutilizable en una cooperativa, aunque su AUC medio sea alto.** Esta métrica puede ser más decisiva que el AUC y hay que darle el mismo peso en el informe.
- **Índice de Jaccard** medio entre los subconjuntos elegidos por un mismo método a través de pares de semillas — una sola cifra que resume la estabilidad.
- **Friedman + Nemenyi (D12):** prueba de Friedman sobre los rangos de los brazos a través de las 10 semillas; si resulta significativa (`p < 0,05`), post-hoc de Nemenyi con **diagrama de diferencia crítica**. Es el procedimiento estándar para comparar múltiples algoritmos sobre múltiples conjuntos, sin supuesto de normalidad y con control de comparaciones múltiples.

### 8.6 Significancia estadística

Con ~192 defaults en test, diferencias de AUC menores a ~0,02 probablemente sean ruido. Obligatorio:

1. **IC del 95% por bootstrap** (1.000 remuestreos del test) por brazo y semilla.
2. Si los intervalos se solapan → **no hay diferencia demostrada**, y hay que escribirlo con esas palabras.
3. El baseline aleatorio **R** da la escala del ruido: si C0 cae dentro de ±1 desviación de R, el QUBO no aportó nada.
4. A través de semillas: Friedman + Nemenyi (§8.5).
5. **Las dos fuentes de incertidumbre — bootstrap dentro de una semilla y varianza entre semillas — no se suman ingenuamente.** Se reportan por separado y se explica qué mide cada una.

### 8.7 Tabla de decisión de conclusiones — PRE-REGISTRADA

**Esta tabla se fija antes de correr el experimento y no se modifica después.** Es lo que impide racionalizar el resultado. Conservada de la v1.0 y extendida con las reglas multi-semilla.

| Resultado observado | Conclusión que se escribe |
|---------------------|---------------------------|
| C0 supera a B0 con IC no solapados **y** Friedman/Nemenyi significativo a través de semillas | Hipótesis **respaldada** en este entorno sintético |
| C0 supera a B0 en una semilla pero Friedman no es significativo | Hipótesis **no respaldada**: la diferencia no sobrevive al cambio de semilla. Es el caso que más comúnmente se reporta mal en la literatura |
| Los IC se solapan pero C0 usa menos variables, menor VIF o mayor estabilidad de selección | Hipótesis **parcialmente respaldada**: sin ganancia predictiva, con ganancia de parsimonia e interpretabilidad — que tiene valor regulatorio propio |
| Los IC se solapan y no hay ventaja de parsimonia ni de estabilidad | Hipótesis **no respaldada**. Resultado válido y reportable |
| C0 por debajo de B0 | Hipótesis **rechazada**. Se diagnostica el culpable comparando contra C1/C2: si C1 también pierde, falla **la formulación**; si C1 gana y C0 no, falla **el heurístico** |
| **C0 = C1 = C2 = C4b** (todos alcanzan el óptimo) | El paisaje del problema es **fácil**. Se concluye que ningún método de optimización avanzado se justifica a esta escala, y se identifica a partir de qué N dejaría de serlo |
| QAOA con `p=3` no cierra la brecha respecto al óptimo | Se reporta tal cual, con el costo asociado. **Es el resultado esperado y no se persigue afinándolo** |

### 8.8 Formato de la tabla de resultados

`reports/RESULTS.md` — una fila por brazo y nivel:

| Brazo | Nivel | Selección | Clasif. | #Vars | AUC media ± dp | AUC [IC95%] | KS | PR-AUC | Brier | P | R | F1 | Tasa aprob. | Recuperación | Trío colineal | Jaccard | VIF máx | t_sel (s) |
|-------|-------|-----------|---------|-------|----------------|-------------|----|--------|-------|---|---|----|-------------|--------------|---------------|---------|---------|-----------|

---

## 9. Auditoría de equidad (E8)

`models/fairness.py`, fase F7. Grupos evaluados: `sexo` (M/F), `zona_residencia` (urbana/rural), `provincia` (agrupada en Loja / Pichincha / Guayas / Azuay / otras).

### 9.1 Métricas por grupo, para cada brazo

| Métrica | Definición | Lectura |
|---------|-----------|---------|
| AUC por grupo | AUC restringido a cada grupo | ¿El modelo discrimina peor para un grupo? |
| Diferencia de paridad demográfica | `máx_g P(ŷ=1\|g) − mín_g P(ŷ=1\|g)` | Brecha en tasa de negación |
| Razón de impacto dispar | `mín_g / máx_g` de la tasa de aprobación | **Regla del 80%**: por debajo de 0,80 se enciende la alarma |
| Diferencia de igualdad de oportunidad | Brecha de TPR entre grupos | ¿Se detecta el incumplimiento igual de bien en todos? |
| Diferencia de tasa de error igualada | Brecha máxima de TPR y FPR | Criterio más estricto |
| Brier por grupo | Calibración por grupo | Un modelo mal calibrado en un grupo provisiona mal |

### 9.2 Detección de proxies — el aporte metodológico

Para cada brazo, se entrena un clasificador auxiliar que predice el **atributo protegido** a partir de **las variables seleccionadas por ese brazo**:

```
AUC_proxy(brazo, atributo) = AUC de   atributo_protegido ~ variables_seleccionadas(brazo)
```

Un `AUC_proxy` alto significa que el subconjunto elegido **codifica el atributo protegido**, aunque este nunca haya entrado al modelo. Es una medida limpia, cuantitativa y fácil de comunicar a un regulador. Referencia de lectura: `AUC_proxy ≈ 0,5` no hay proxy; `> 0,70` hay codificación sustancial.

### 9.3 La pregunta abierta de la fase

> **¿La selección vía QUBO introduce más o menos sesgo indirecto que LASSO?**

**No hay respuesta esperada, y eso es lo interesante.** Dos mecanismos tiran en direcciones opuestas:

- El QUBO **penaliza la redundancia**, lo que podría eliminar variables que son proxies mutuos del atributo protegido → menos sesgo.
- Pero al eliminar redundancia, el QUBO **concentra** la señal en una sola variable del bloque; si esa variable es precisamente la más asociada al atributo protegido (por ejemplo `score_buro`, desplazado −60 puntos en zona rural por diseño §4.3), el sesgo podría **concentrarse** en lugar de diluirse → más sesgo.

Sea cual sea el resultado, es un hallazgo publicable y conecta directamente con §5 de PLAN.md. **Registrar la predicción antes de correrlo**, para poder decir después si se acertó.

### 9.4 Límite que hay que declarar

La auditoría se hace **sobre datos sintéticos con sesgo que nosotros inyectamos**. Detecta si un método de selección **propaga** sesgo indirecto; **no dice nada** sobre el sesgo real del crédito en Ecuador. Confundir ambas cosas sería el error más grave que este trabajo podría cometer.

---

## 10. Interfaz de línea de comandos

```bash
python -m tikee.cli generate   --seed 42 --n 8000
python -m tikee.cli fidelity   --synthesizer {gaussian_copula|ctgan|tvae|all}
python -m tikee.cli select     --level {A|B} --method {lasso|stepwise|qubo-sa|qubo-exact|qubo-milp|qubo-qaoa|tabu|random}
python -m tikee.cli train      --arm A0 --seed 42
python -m tikee.cli evaluate   --seed 42
python -m tikee.cli fairness   --seed 42
python -m tikee.cli multiseed  --seeds 42,101,202,303,404,505,606,707,808,909
python -m tikee.cli report                      # results.json -> RESULTS.md + figuras
python -m tikee.cli all                         # pipeline completo
```

`scripts/run_all.sh` = `python -m tikee.cli all && streamlit run app/Inicio.py`.
Los resultados intermedios se cachean por semilla en `reports/cache/` con `joblib`, para poder reanudar sin recomputar (riesgo R7).

---

## 11. App web (Streamlit) — 5 pestañas

Lee `reports/results.json` y `data/processed/`. **No entrena nada en vivo** salvo el simulador, que carga un modelo ya serializado — así la demo nunca se cuelga delante del tribunal.

| Página | Contenido |
|--------|-----------|
| **Portada** | Problema, hipótesis, advertencia de datos sintéticos, aclaración de "cuántico-inspirado" |
| **1 · Datos** | Tabla del esquema, histogramas, mapa de calor de correlación, tasa de default por decil de `score_buro`, resumen de fidelidad SDMetrics y comparación de los tres sintetizadores |
| **2 · Selección** | Qué eligió cada método (Nivel A y B), mapa de calor de la matriz `Q`, energía del recocido por lectura, **curva de brecha-vs-`p` de QAOA**, marca visual sobre las variables-trampa y de ruido |
| **3 · Comparación** | Tabla de métricas, curvas ROC superpuestas, gráfico KS, matrices de confusión lado a lado, tiempo de selección vs. AUC |
| **4 · Simulador** | Formulario de solicitante → probabilidad de default, decisión según el umbral congelado, y **las 3 razones principales** (contribución de cada coeficiente) — el argumento de interpretabilidad SEPS hecho visible |
| **5 · Estabilidad** | Cajas de AUC por brazo a través de las 10 semillas, mapa de calor de frecuencia de selección, índice de Jaccard, diagrama de diferencia crítica de Nemenyi, y panel de equidad por grupo con la detección de proxies |

---

## 12. Tests

| Archivo | Verifica |
|---------|----------|
| `test_schema.py` | Toda columna existe, tipo correcto, dentro de rango; `antiguedad_laboral ≤ (edad−18)·12`; los atributos protegidos existen y **no** están entre los predictores |
| `test_target.py` | Tasa de default en [0,075 · 0,085]; la tasa decrece monótonamente por decil de `score_buro`; las 13 correlaciones objetivo dentro de ±0,07 |
| `test_qubo_builder.py` | Lectura simétrica de `Q`; la diagonal contiene `λ(1−2k)`; con `β=0` y λ grande la solución tiene exactamente k variables; con `α=0` selecciona las k menos correlacionadas entre sí |
| `test_solvers_agree.py` | **En un QUBO de 8 variables: recocido, `ExactSolver`, MILP y QAOA `p=3` devuelven la misma solución.** Es la prueba de regresión más importante del repositorio |
| `test_expand.py` | `expand.py` produce exactamente 45 columnas, deterministas dada la semilla; las etiquetas de ground truth son coherentes; **ningún estadístico se ajusta fuera de train** |
| `test_metrics.py` | KS y AUC contra un caso pequeño calculado a mano |
| `test_fairness.py` | Las métricas por grupo se calculan sobre el grupo correcto; `AUC_proxy` ≈ 0,5 con variables aleatorias |

---

## 13. Reproducibilidad y presupuesto de cómputo

**Reproducibilidad:**

- Lista de semillas en `config.yaml`, propagada a numpy, sklearn, xgboost, SDV, el sampler de recocido y COBYLA.
- Versiones fijadas exactamente en `requirements.txt`.
- `results.json` guarda, junto a cada métrica, el hash de la configuración que la produjo.
- `dataset_seed42.parquet` versionado si pesa < 50 MB; el resto se reconstruye con `cli generate --seed N`.

**Presupuesto estimado (riesgo R7):**

| Componente | Costo unitario | Total |
|-----------|---------------|-------|
| Generación de datos | ~10 s por semilla | 2 min |
| CTGAN (una vez) | 10–40 min | 40 min |
| CV anidada de XGBoost | ~15 min por brazo y semilla | 3 semillas × 2 brazos ≈ 1,5 h |
| Barrido QUBO Nivel A | ~4 min por semilla | 40 min |
| Barrido QUBO Nivel B | ~15 min por semilla | 2,5 h |
| MILP Nivel B | hasta 600 s por instancia | hasta 2 h |
| QAOA Nivel A | ~20 min por configuración `p` | 1 h |
| Bootstrap y equidad | ~5 min por semilla | 50 min |
| **Total aproximado** | | **~9–10 h de cómputo** |

Todo en un portátil. El cómputo **no** es el cuello de botella; lo es el tiempo de desarrollo (§0 de PLAN.md). Los resultados se cachean por semilla para poder reanudar tras una interrupción.
