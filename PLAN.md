# PLAN.md — Tikee Quantum-Inspired Scoring

**Proyecto:** Selección de variables cuántico-inspirada para scoring crediticio en cooperativas SEPS (Ecuador)
**Institución:** UTPL — proyecto de innovación universitaria
**Fecha de planificación:** 27 de julio de 2026
**Versión:** 2.0 — alcance ampliado (reemplaza la v1.0 de 10 horas)
**Estado:** planificación cerrada. Implementación = fase siguiente (Sonnet).

---

## 0. Nota de calibración de plazo — LEER PRIMERO

Este documento describe un **alcance ampliado que requiere ~9 días de trabajo (≈70 horas efectivas)**. El plazo declarado es "hasta mañana" (28 de julio de 2026), es decir **≈16–20 horas efectivas**.

La aritmética no cierra. Estimación por bloque de trabajo nuevo:

| Bloque | Horas estimadas |
|--------|----------------|
| Núcleo (dataset + 3 brazos + métricas + informe) | 10 |
| Nivel B (N≈45) como parte central | 6 |
| Fidelidad SDMetrics completa + comparar 3 sintetizadores | 7 |
| Estabilidad multi-semilla (10 semillas) + Friedman/Nemenyi | 8 |
| QAOA simulado (Qiskit Aer) | 10 |
| Auditoría de equidad | 7 |
| CV anidada + búsqueda de hiperparámetros XGBoost | 9 |
| App Streamlit de 5 pestañas | 7 |
| Informe académico ampliado | 6 |
| **Total** | **70** |

**Dos caminos, hay que elegir uno:**

- **Camino COMPLETO** — cronograma F0–F8 de §4. Requiere mover la entrega a ~5 de agosto de 2026.
- **Camino ENTREGA-MAÑANA** — subconjunto definido en §4.4. Núcleo + Nivel B + equidad + estabilidad ligera (3 semillas). **Sin QAOA, sin CV anidada, sin comparación de sintetizadores.** Cabe en 18 h con esfuerzo sostenido.

El resto del documento está escrito para el camino COMPLETO, con marcas `[MAÑANA]` en todo lo que sobrevive al recorte. **No se debe empezar a implementar sin decidir esto**, porque el orden de trabajo de los dos caminos difiere desde la fase F1.

---

## 1. Resumen ejecutivo

Construimos un dataset sintético de solicitudes de crédito calibrado con referencias públicas ecuatorianas, con multicolinealidad y sesgos indirectos inyectados a propósito, y probamos si formular la **selección de variables** como un problema **QUBO** mejora un modelo de scoring frente a LASSO o stepwise. El mismo QUBO se resuelve con **tres familias de solucionadores** — recocido simulado, QAOA simulado sobre circuitos, y solucionadores exactos con certificado de optimalidad — para separar limpiamente tres preguntas que la literatura suele confundir:

1. ¿Es buena **la formulación** QUBO relevancia-redundancia como criterio de selección?
2. ¿Aporta algo **el heurístico** de recocido frente a un óptimo certificado?
3. ¿Aporta algo **el circuito cuántico simulado** frente al recocido?

Todo se reporta con intervalos de confianza, a través de múltiples semillas, con auditoría de equidad y con una discusión cuantificada del costo regulatorio de interpretabilidad ante la SEPS.

---

## 2. Alcance

### 2.1 Núcleo obligatorio `[MAÑANA]`

| # | Entregable | Por qué es imprescindible |
|---|-----------|---------------------------|
| E1 | Dataset sintético SDV, 8.000 filas, 18 variables predictoras + target + atributos protegidos | Sin datos no hay proyecto |
| E2 | Evidencia medida de multicolinealidad intencional (matriz de correlación + VIF) | Es la premisa de la hipótesis |
| E3 | Brazos de modelado: todas las variables / selección clásica / selección QUBO | Es la comparación que responde la pregunta |
| E4 | Métricas: AUC-ROC, KS, PR-AUC, Brier, precision/recall/F1, matriz de confusión, IC bootstrap | Es la evidencia |
| E5 | App web Streamlit | Es la demostración visible |
| E6 | `reports/INFORME.md`: análisis honesto + interpretabilidad SEPS | Es el valor académico |

### 2.2 Expansiones incorporadas al alcance

| # | Expansión | Estado | Justificación |
|---|-----------|--------|---------------|
| **E7** | **Nivel B: N≈45 variables expandidas** | **Central** `[MAÑANA]` | Es el único régimen donde el annealing tiene un rol genuino. Sin esto el proyecto no puede afirmar nada sobre escalabilidad |
| **E8** | **Auditoría de equidad** con atributos protegidos | **Central** `[MAÑANA]` | Conecta con la interpretabilidad regulatoria y añade una capa académica propia. Barato de implementar sobre lo ya construido |
| **E9** | **Estabilidad multi-semilla** + Friedman/Nemenyi | **Central** (10 semillas; `[MAÑANA]` con 3) | Sin esto, cualquier diferencia observada es anecdótica. Es la corrección metodológica de mayor retorno por hora |
| **E10** | **Solucionador exacto certificado en Nivel B** (linealización de Glover → MILP → HiGHS) | **Central** `[MAÑANA]` | Ver §2.3. Es la adición técnica más importante de la v2.0 |
| **E11** | **Fidelidad SDMetrics completa** (Quality + Diagnostic + privacidad) | Alta | Convierte "usamos SDV" en "validamos el dataset sintético", que es lo que un tribunal va a preguntar |
| **E12** | **Comparación GaussianCopula vs CTGAN vs TVAE** | Media | Buen material de informe. Es un experimento independiente y aislable |
| **E13** | **CV anidada + búsqueda de hiperparámetros XGBoost** | Media | Elimina la objeción "XGBoost no estaba afinado" |
| **E14** | **QAOA simulado (Qiskit Aer)** | **Condicional — ver §2.4** | Alto valor narrativo, viabilidad acotada por física, no por tiempo |
| **E15** | App Streamlit de 5 pestañas (añade Estabilidad) | Alta | La pestaña de estabilidad es lo que distingue una demo de un experimento |

### 2.3 Adición no solicitada pero necesaria: óptimo certificado en Nivel B (E10)

Un QUBO con restricción de cardinalidad **se puede linealizar exactamente** (linealización de Glover: `y_ij = x_i·x_j` con tres desigualdades por par) y resolverse como MILP con un solucionador de rama y corte (HiGHS, libre). En N=45 eso son 45 binarias + 990 auxiliares — grande, pero no intratable, sobre todo con la relajación LP que dan las matrices de redundancia dispersas.

**Por qué importa:** sin esto, el Nivel B compara recocido contra *nada*, y cualquier resultado se lee como "el annealing funciona". Con esto, la comparación real es **recocido vs. óptimo certificado**, y el resultado probable es:

> *"En N=45 el recocido simulado alcanza el óptimo certificado por MILP en el 100% de las instancias, en 2 s frente a 90 s. La ventaja del annealing es de tiempo, no de calidad, y el régimen donde el óptimo exacto deja de ser alcanzable está por encima de las escalas de este problema."*

Esa frase es un resultado defendible y probablemente **el hallazgo central del trabajo**. Es exactamente el mismo tipo de rigor que llevó a reformular la hipótesis en dos niveles en la v1.0, aplicado un nivel más arriba. Si el MILP no termina en el tiempo límite, el gap de optimalidad reportado por HiGHS **sigue siendo información**: acota cuán lejos puede estar el annealing.

### 2.4 QAOA (E14): viabilidad honesta antes de comprometerse

QAOA se simula con **un qubit por variable binaria**. La memoria del vector de estado crece como `2^N × 16 bytes`:

| N | Amplitudes | Memoria | Veredicto |
|---|-----------|---------|-----------|
| 18 | 262.144 | 4 MB | Trivial ✅ |
| 24 | 16,8 M | 268 MB | Viable ✅ |
| 28 | 268 M | 4,3 GB | Límite de un portátil ⚠️ |
| 30 | 1,07 G | 17 GB | Fuera de alcance ❌ |
| **45** | **3,5 × 10¹³** | **563 TB** | **Físicamente imposible** ❌ |

**Consecuencia dura: QAOA solo puede correr en el Nivel A (N=18). No existe forma de correrlo en el Nivel B.** Cualquier plan que prometa "QAOA en las 45 variables" es falso, y hay que decirlo en el informe.

Y en el Nivel A ya conocemos el óptimo global por enumeración. Por lo tanto:

> **QAOA en este proyecto no puede ganar. Su valor es medir cuánto se acerca a un óptimo conocido, con cuánta profundidad de circuito `p`, y a qué costo — que es varios órdenes de magnitud mayor que enumerar las 262.144 soluciones a la fuerza.**

Esto no es un defecto del plan: es el resultado. Un trabajo universitario que mide y reporta esto honestamente vale más que uno que insinúa una ventaja cuántica inexistente. **Decisión: E14 entra al alcance con este encuadre explícito, no como candidato a ganar** (decisión D17). Si el tribunal pregunta "¿es esto computación cuántica?", la respuesta preparada es: *"es un circuito cuántico simulado clásicamente, y lo usamos para cuantificar la brecha entre la promesa y la práctica en un problema de tamaño real de cooperativa"*.

### 2.5 Qué SIGUE recortado

La lista de la v1.0 casi desaparece. Lo que **no** entra, con motivo:

| Se recorta | Motivo |
|-----------|--------|
| **Hardware cuántico real (D-Wave Leap, IBM Quantum)** | Requiere cuenta, cuota y cola. Además el embedding de un QUBO denso de 45 variables en topología Pegasus consume cientos de qubits físicos y el ruido domina el resultado. Queda documentado como trabajo futuro con la estimación de recursos |
| **Despliegue en nube, autenticación, base de datos** | No aporta al argumento científico |
| **API REST / integración con core bancario** | Es una plataforma de producción, no un proyecto académico |
| **Datos reales de cualquier cooperativa** | No los tenemos, y obtenerlos exige convenio y tratamiento de datos personales. Es *la* limitación del trabajo y se declara como tal |
| **Mitigación de sesgo** (reweighing, restricciones de equidad) | La auditoría E8 **mide**; corregir es un proyecto propio. Se propone como trabajo futuro |
| **Nivel C (N > 100)** | El cuello de botella pasa a ser el cálculo de la matriz de redundancia y el sobreajuste de la selección, no el optimizador. Cambia la pregunta de investigación |

### 2.6 Correcciones estructurales que se conservan intactas

Estas cuatro decisiones de la v1.0 fueron validadas en revisión y **no se modifican, se profundizan**:

1. **Reformulación en dos niveles** (N=18 exacto vs. N≈45 intratable por enumeración). Se amplía con el óptimo certificado por MILP (E10), que traslada la frontera de "intratable" hacia arriba y obliga a ser aún más preciso sobre dónde empieza el régimen interesante.
2. **Ground truth explícito del target.** Se explota más: se añaden métricas de recuperación por bloque, análisis de cuál de las variables colineales elige cada método, y variables-trampa también en el Nivel B (§4.6 de ARCHITECTURE.md).
3. **Tabla de decisión de conclusiones fijada antes de correr** (ARCHITECTURE.md §8.7) y **prohibición de tocar α, β, k después de ver el test.** Intacta, y extendida con reglas de decisión multi-semilla.
4. **Argumento de interpretabilidad regulatoria SEPS.** Intacto y ampliado con la auditoría de equidad (§5).

### 2.7 Criterios de éxito y anti-criterios

El proyecto es exitoso si se entregan E1–E10 con conclusiones respaldadas por números. **La hipótesis puede fallar y el proyecto seguir siendo exitoso.**

**Anti-criterios — prácticas prohibidas, verificables en el repositorio:**

- Ajustar α, β, k, `p` de QAOA o cualquier hiperparámetro después de ver métricas de test.
- Reportar la mejor de varias semillas. Se reportan **todas** o ninguna.
- Calcular relevancia o redundancia sobre el dataset completo (fuga de información).
- Presentar el recocido simulado como "computación cuántica".
- Omitir el brazo de óptimo certificado cuando existe, porque desfavorece a la hipótesis.
- Presentar SDV como si hubiera "descubierto" las correlaciones que nosotros inyectamos.

---

## 3. Decisiones registradas

| ID | Decisión | Alternativa descartada | Razón |
|----|----------|------------------------|-------|
| D1 | Python 3.11 | 3.12 / 3.13 | SDV, el stack de D-Wave y Qiskit Aer tienen ruedas estables probadas en 3.11 |
| D2 | `dwave-samplers` con respaldo `dwave-neal` | Solo `dwave-neal` | `neal` está en mantenimiento; el sampler vive en `dwave-samplers`. API idéntica |
| D3 | Target por **modelo estructural explícito** | Dejar que SDV genere el target | Sin ground truth no se puede juzgar si una selección es correcta |
| D4 | Tasa base de default = 8% | 3% o 20% | Rango orientativo de morosidad de cartera, cooperativas segmento 1, boletines SEPS |
| D5 | Multicolinealidad inyectada a propósito por bloques | Correlaciones naturales de SDV | La hipótesis trata sobre variables correlacionadas; deben existir por construcción y ser medibles |
| D6 | Baseline aleatorio (k al azar × 100) | Solo comparar con LASSO | Sin este control no se distingue mejora real de ruido de muestreo |
| D7 | Streamlit | Flask/FastAPI + React | Interfaz de datos en Python; React sería el proyecto entero |
| D8 | Métrica primaria = AUC-ROC en test; KS secundaria | KS primaria | AUC más estable con n moderado; KS se reporta por ser el lenguaje del sector |
| **D9** | **Nivel B por expansión determinista de variables** (interacciones + binning + ruido), no por generar más columnas crudas | Ampliar el esquema a 45 variables originales | Mantiene el ground truth trazable: sabemos exactamente qué variable expandida deriva de cuál original y cuáles son puro ruido |
| **D10** | **Óptimo certificado por linealización de Glover + HiGHS** en ambos niveles | Solo `dimod.ExactSolver` (limitado a N≤20) | Da un punto de comparación honesto en Nivel B. Ver §2.3 |
| **D11** | **10 semillas** para el experimento principal; cada semilla regenera datos *y* re-hace el split | Semilla única; o solo re-splitear | Captura variabilidad de generación **y** de partición. Solo re-splitear subestima la incertidumbre |
| **D12** | **Friedman + Nemenyi** para comparar métodos a través de semillas | Prueba t pareada por par de métodos | Es el procedimiento estándar para comparar múltiples algoritmos sobre múltiples conjuntos, sin supuesto de normalidad y con control de comparaciones múltiples |
| **D13** | **CV anidada** (externa 5 × interna 3) solo sobre 3 semillas | CV anidada sobre las 10 semillas | Costo cuadrático. 3 semillas bastan para sostener "XGBoost estaba afinado"; las 10 se usan con el protocolo de holdout |
| **D14** | Sintetizador de producción = **GaussianCopulaSynthesizer**; CTGAN y TVAE se corren como **experimento de comparación documentado**, no como fuente del dataset principal | Elegir el de mejor puntuación de SDMetrics | La cópula gaussiana **preserva por construcción** la estructura de correlación que inyectamos, que es la premisa del proyecto. Elegir CTGAN porque puntúa mejor en fidelidad marginal destruiría el objeto de estudio. Se documenta esta tensión: **la métrica de calidad y el objetivo de investigación no coinciden aquí**, y se explica por qué gana el objetivo |
| **D15** | Atributos protegidos (`sexo`, `provincia`) **generados con dependencia real hacia otras variables** pero con **coeficiente cero en el target** | Generarlos independientes | Si fueran independientes la auditoría de equidad no encontraría nada por construcción, y sería teatro. Así, cualquier disparidad hallada es **sesgo indirecto puro vía proxies** — el caso interesante y el que ocurre en la práctica |
| **D16** | La equidad se **mide**, no se corrige | Añadir mitigación | Mitigar es un proyecto propio. Medir ya es una contribución |
| **D17** | QAOA entra con el encuadre "no puede ganar, medimos la brecha" | QAOA como candidato a superar al recocido | Ver §2.4. Es físicamente imposible en Nivel B y redundante en Nivel A |
| **D18** | QAOA con `p ∈ {1, 2, 3}`, optimizador COBYLA, expectativa exacta por vector de estado | Muestreo con shots finitos | El vector de estado da la expectativa sin ruido de muestreo, aísla el efecto de `p`, y en N=18 es más barato que muestrear |
| **D19** | Umbral de decisión fijado por **máximo KS en train**, congelado y aplicado a test | Umbral óptimo en test | Fijarlo en test es fuga de información |
| **D20** | Sin balanceo de clases (SMOTE/submuestreo) | Rebalancear | Con 8% de positivos y modelos que emiten probabilidades, rebalancear distorsiona la calibración sin mejorar el ranking. Se decide por umbral |

---

## 4. Cronograma

### 4.1 Camino COMPLETO — fases por día

Nueve fases. Cada una cierra con un **checkpoint verificable**: una condición objetiva que decide si se avanza, se itera o se recorta. **No se pasa de fase con el checkpoint en rojo.**

---

#### **F0 · Día 1 — Entorno y validación de riesgo técnico** (6 h)

El objetivo de esta fase es **descubrir hoy los problemas de instalación**, no descubrirlos el día 7.

- Entorno virtual Python 3.11, `pip install -r requirements.txt`.
- Verificación real de los tres stacks críticos, con un script mínimo que **ejecute** algo, no solo importe:
  - SDV: ajustar `GaussianCopulaSynthesizer` sobre 50 filas de juguete y muestrear.
  - D-Wave: resolver un QUBO de 4 variables con solución conocida a mano, con `SimulatedAnnealingSampler` y con `ExactSolver`.
  - Qiskit: correr QAOA `p=1` sobre ese mismo QUBO de 4 variables y comprobar que llega al óptimo.
  - HiGHS: resolver el mismo QUBO linealizado y verificar que da la misma solución.
- **Implementar el camino de respaldo de SDV** (cópula gaussiana propia con `scipy.stats`) y comprobar que produce la misma estructura de correlación. No dejarlo como plan B teórico.
- Esqueleto de `config.py`, `config.yaml`, `schema.yaml`.

> **Checkpoint F0:** los cuatro solucionadores coinciden en la solución del QUBO de juguete de 4 variables. Si Qiskit no instala → E14 sale del alcance hoy, no el día 7.

---

#### **F1 · Día 2 — Generación de datos y calibración** (8 h)

- `seed_generator.py`: semilla de 2.000 filas con la estructura de correlación de ARCHITECTURE.md §4.2, incluyendo las dependencias de los atributos protegidos (D15).
- `sdv_synthesizer.py`: ajuste y muestreo de 8.000 filas.
- `target_definition.py`: modelo estructural, calibración de `β0` a tasa base 8%.
- **Calibración iterativa de σ** — este es el trabajo real de la fase y necesita varias corridas. Objetivo: AUC del modelo base con todas las variables en la banda **[0,72 · 0,82]**. Procedimiento: búsqueda por bisección sobre σ, 5–8 iteraciones de generar → entrenar logística rápida → medir AUC.
- `validate.py`: chequeos duros de rango, tasa base, correlaciones objetivo (tolerancia ±0,07), VIF, banda de AUC, restricción `antiguedad_laboral ≤ (edad−18)·12`.
- Tests `test_schema.py` y `test_target.py` en verde.

> **Checkpoint F1:** `dataset.parquet` con AUC base en banda, `default.mean()` en [0,075 · 0,085], las 13 correlaciones objetivo dentro de tolerancia, `pytest` en verde. **Sin este checkpoint todo lo demás mide ruido.**

---

#### **F2 · Día 3 — Fidelidad sintética y comparación de sintetizadores** (7 h) `[E11, E12]`

- SDMetrics completo sobre GaussianCopula: `QualityReport` (formas de columna y tendencias de pares), `DiagnosticReport` (validez y estructura), y privacidad (`NewRowSynthesis`, distancia al registro más cercano).
- Entrenar CTGAN (300 épocas) y TVAE sobre la misma semilla. Registrar tiempo de entrenamiento.
- Tabla comparativa: puntuación de calidad, preservación de las 13 correlaciones objetivo, tiempo, estabilidad entre corridas.
- Redactar la justificación de D14: por qué se elige la cópula gaussiana **aunque otro puntúe mejor en fidelidad marginal**.

> **Checkpoint F2:** tabla de tres sintetizadores completa y decisión D14 escrita con números que la respalden.
> *Si CTGAN no converge o tarda más de 40 min: se reporta como resultado negativo de viabilidad y se sigue. No se depura CTGAN.*

---

#### **F3 · Día 4 — Modelos base, preprocesamiento y selección clásica** (8 h)

- `preprocess.py`: `ColumnTransformer` congelado, ajustado **solo en train**.
- Brazos A0 (logística) y A1 (XGBoost) con las 18 variables.
- **CV anidada** para XGBoost: externa 5-fold, interna 3-fold, `RandomizedSearchCV` con 30 configuraciones sobre `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`. `[E13]`
- Brazos B0 (LASSO) y B1 (stepwise forward por AIC).
- `evaluate.py` completo, incluido el IC bootstrap del AUC con 1.000 remuestreos.
- Tests `test_metrics.py` en verde.

> **Checkpoint F3:** cuatro brazos con métricas e IC en `results.json`. La tabla de resultados ya se puede leer, aunque le falten los brazos de QUBO.

---

#### **F4 · Día 5 — QUBO y solucionadores, Nivel A** (8 h)

- `qubo_builder.py`: matrices de relevancia (información mutua) y redundancia (Spearman / V de Cramér / η), construcción de `Q`. **Solo sobre train.**
- Barrido de β × k (24 combinaciones) por CV de 5 pliegues **en train**.
- Brazo C0: recocido simulado. Brazo C1: `dimod.ExactSolver`. Brazo C2: MILP-Glover + HiGHS (debe coincidir con C1 — es la prueba cruzada de que la linealización está bien). Brazo R: baseline aleatorio.
- Verificación de λ: si alguna solución no cumple `Σxᵢ = k`, duplicar λ y volver a resolver.
- Tests `test_qubo_builder.py` en verde, incluidos los casos límite `α=0` y `β=0`.

> **Checkpoint F4:** C1 y C2 devuelven **exactamente la misma solución**. Si no coinciden, la formulación tiene un error y no se avanza. Este es el control de calidad más importante de todo el proyecto.

---

#### **F5 · Día 6 — Nivel B (N≈45) y QAOA** (9 h) `[E7, E14]`

Mañana — Nivel B:
- `expand.py`: expansión determinista a 45 candidatas (ARCHITECTURE.md §4.6), con trazabilidad al ground truth y variables-trampa.
- Recocido simulado sobre N=45. MILP-Glover + HiGHS con límite de 600 s; si no cierra, se reporta el gap.
- `TabuSampler` como tercer punto de referencia heurístico.
- Medición de **tiempo de solución y calidad de energía**, no solo AUC.

Tarde — QAOA (Nivel A únicamente):
- Traducción QUBO → Ising → `QuadraticProgram` de Qiskit.
- QAOA con `p ∈ {1,2,3}`, COBYLA, expectativa exacta por vector de estado.
- Métricas: brecha de energía respecto al óptimo conocido, probabilidad de la cadena de bits óptima, número de evaluaciones del circuito, tiempo de pared.

> **Checkpoint F5:** curva de "brecha respecto al óptimo vs. `p`" para QAOA, y tabla de tiempo-vs-calidad para N=45. Si QAOA `p=3` no baja de una brecha del 5%, **eso es el resultado** y se reporta; no se persigue afinándolo.

---

#### **F6 · Día 7 — Estabilidad multi-semilla** (8 h) `[E9]`

- Ejecutar el pipeline completo con **10 semillas** (42, 101, 202, …). Cada semilla regenera el dataset y re-hace el split (D11).
- Por brazo: media, desviación estándar, mínimo y máximo del AUC, KS y número de variables.
- **Estabilidad de la selección:** con qué frecuencia cada variable es elegida por cada método a través de las 10 semillas (mapa de calor de frecuencia de selección). Un método que elige variables distintas en cada semilla es inutilizable en una cooperativa, aunque su AUC medio sea alto. **Esta métrica puede ser más decisiva que el AUC.**
- Prueba de Friedman sobre los rangos de los brazos a través de semillas; si es significativa, post-hoc de Nemenyi con diagrama de diferencia crítica (D12).
- Estabilidad del trío colineal: ¿cuántas y cuáles de `{score_buro, peor_calificacion_12m, dias_mora_max_12m}` sobreviven en cada semilla?

> **Checkpoint F6:** tabla de media ± desviación por brazo y resultado de Friedman. **Aquí se decide la conclusión del trabajo** según la tabla de ARCHITECTURE.md §8.7.

---

#### **F7 · Día 8 — Equidad e interpretabilidad** (8 h) `[E8]`

- Métricas por grupo (`sexo`, `zona_residencia`, `provincia` agrupada) para cada brazo: AUC por grupo, diferencia de paridad demográfica, diferencia de igualdad de oportunidad (brecha de TPR), razón de impacto dispar (regla del 80%), tasa de aprobación por grupo.
- **Detección de proxies:** para cada brazo, entrenar un clasificador auxiliar que prediga el atributo protegido a partir de **las variables seleccionadas**. Un AUC alto significa que el subconjunto elegido codifica el atributo protegido aunque este nunca entró al modelo. Es una medida limpia y cuantitativa de sesgo indirecto.
- Pregunta central de la fase: **¿la selección vía QUBO introduce más o menos sesgo indirecto que LASSO?** No hay respuesta esperada — el QUBO penaliza redundancia, lo que *podría* eliminar proxies correlacionados, o *podría* concentrar la señal en uno solo de ellos y empeorarlo.
- `interpret.py`: coeficientes, verificación de signos esperados por el negocio, VIF del modelo final, SHAP.
- Redacción de la sección de interpretabilidad regulatoria (§5).

> **Checkpoint F7:** tabla de equidad por brazo y grupo + resultado de la detección de proxies.

---

#### **F8 · Día 9 — App, informe y cierre** (8 h) `[E5, E6, E15]`

- App Streamlit, 5 pestañas (ARCHITECTURE.md §11).
- `RESULTS.md` generado desde `results.json`.
- `INFORME.md`: análisis honesto siguiendo la tabla de decisión, discusión de interpretabilidad, equidad, limitaciones.
- `README.md` con reproducción en tres comandos. `pytest` completo en verde.
- Revisión final contra la lista de anti-criterios de §2.7.

> **Checkpoint F8:** `make setup && make experiment && make app` funciona desde cero en un directorio limpio.

---

### 4.2 Ruta crítica y paralelismo

```
F0 → F1 → F3 → F4 → F5 → F6 → F8      (ruta crítica, 55 h)
       └→ F2  (independiente tras F1)
                        └→ F7  (necesita F6, pero F7 y la app de F8 son paralelizables)
```

**F1 es el cuello de botella real.** Si la calibración de σ no cierra, todo lo posterior mide un problema que no es el que se quería estudiar. Está presupuestada con margen de iteración precisamente por eso.

### 4.3 Puntos de decisión (a diferencia de "orden de sacrificio")

La v1.0 tenía una lista de emergencia. La v2.0 la reemplaza por **tres puntos de decisión programados**, cada uno con criterio objetivo:

| Cuándo | Decisión | Criterio |
|--------|----------|----------|
| Fin de F0 | ¿E14 (QAOA) entra? | Solo si el QUBO de juguete de 4 variables se resuelve con QAOA `p=1` sin incidencias de instalación |
| Fin de F2 | ¿E12 (CTGAN/TVAE) se reporta como comparación o como nota de viabilidad? | Según si CTGAN converge en menos de 40 min |
| Fin de F4 | ¿Nivel B usa 10 semillas o 5? | Según el tiempo de pared medido del MILP en N=45 |

### 4.4 Camino ENTREGA-MAÑANA (si el plazo no se mueve)

Subconjunto de 18 h sobre dos jornadas. **Solo lo marcado `[MAÑANA]`.**

| Jornada | Bloque | h | Contenido |
|---------|--------|---|-----------|
| Hoy (27 jul) | J1 | 1,5 | F0 reducido: entorno + verificación de SDV, D-Wave y HiGHS. **Sin Qiskit** |
| Hoy | J2 | 3,0 | F1 completo, incluida la calibración iterativa de σ |
| Hoy | J3 | 2,5 | F3 sin CV anidada (XGBoost con parámetros fijos y razonables) |
| Hoy | J4 | 2,0 | F4 completo, Nivel A |
| Mañana (28 jul) | J5 | 2,5 | F5 solo Nivel B. **Sin QAOA** |
| Mañana | J6 | 2,0 | F6 con **3 semillas** en vez de 10; sin Friedman (n insuficiente), solo media ± rango |
| Mañana | J7 | 2,0 | F7 reducido: métricas de equidad por grupo + detección de proxies. Sin SHAP |
| Mañana | J8 | 2,5 | F8: app de 3 pestañas (Comparación, Simulador, Estabilidad) + informe |

**Queda fuera del camino MAÑANA:** QAOA (E14), fidelidad SDMetrics completa (E11), comparación de sintetizadores (E12), CV anidada (E13), 10 semillas y Friedman/Nemenyi (E9 completo), pestañas 1 y 2 de la app.

Aun así se conservan: los dos niveles, el óptimo certificado, la equidad, la estabilidad en versión mínima y el informe honesto. **Es un proyecto defendible.** Lo que se pierde es el diferencial cuántico narrativo (QAOA) y la potencia estadística.

---

## 5. Interpretabilidad regulatoria y equidad — el argumento central

Esta sección no es decorativa: es lo que convierte un ejercicio técnico en un proyecto de innovación con criterio.

**El hecho regulatorio.** Una cooperativa regulada por la SEPS debe poder sustentar la negación de un crédito ante el socio y ante el supervisor. Un modelo cuya selección de variables proviene de un optimizador combinatorio agrega opacidad **sobre** la que ya aporta el modelo.

**La distinción que el informe debe dejar clarísima:**

- La **selección** vía QUBO es opaca en el *procedimiento*: es difícil explicar por qué se eligió ese subconjunto y no otro.
- Pero el **modelo resultante** puede ser perfectamente interpretable: si el clasificador final es una regresión logística sobre 7 variables, cada coeficiente es explicable, auditable y comunicable al socio.
- Por lo tanto: **QUBO + regresión logística es más defendible ante la SEPS que XGBoost con las 18 variables.** Probablemente la conclusión más contraintuitiva y valiosa del trabajo, y **no depende de que la hipótesis principal se confirme.**

**Lo que hay que cuantificar (no afirmar):**

1. **Parsimonia:** número de variables del modelo final por brazo. Menos variables = expediente de crédito más simple y menor costo de consulta al buró.
2. **Coherencia de signos:** ¿los coeficientes de la logística conservan el signo que el negocio espera (mayor `ratio_cuota_ingreso` ⇒ mayor riesgo)? Un signo invertido por colinealidad es un hallazgo regulatorio **grave** — significa que el modelo diría al socio algo falso sobre por qué se le negó — y hay que reportarlo como tal.
3. **VIF del modelo final.** Si el brazo QUBO baja el VIF, ese es un argumento de interpretabilidad medible.
4. **Estabilidad de la selección entre semillas** (de F6). Un método que elige variables distintas cada vez no puede sostener una política de crédito escrita.
5. **Sesgo indirecto** (de F7). Si el subconjunto elegido permite predecir el sexo del solicitante con AUC de 0,75, el modelo tiene un problema regulatorio real aunque el sexo nunca haya sido una variable de entrada.
6. **El intercambio, con números:** si el brazo QUBO gana 0,004 de AUC y a cambio hay que justificar un heurístico de optimización ante el regulador, **la recomendación honesta es no adoptarlo.** El informe debe estar dispuesto a escribir esa frase.

---

## 6. Riesgos

| # | Riesgo | Prob. | Impacto | Mitigación |
|---|--------|-------|---------|------------|
| R1 | **El target sintético queda demasiado fácil** (AUC > 0,90 en todos los brazos) y todos los brazos empatan arriba | Alta | Alto | Riesgo principal. F1 dedica horas a calibrar σ por bisección; `validate.py` **rechaza** el dataset fuera de la banda [0,72 · 0,82] |
| R2 | El plazo real es mañana y el alcance es de 9 días | **Muy alta** | **Muy alto** | §0 y §4.4. Requiere una decisión del responsable del proyecto, no una mitigación técnica |
| R3 | SDV, Qiskit o HiGHS no instalan | Media | Alto | F0 lo detecta el día 1 con un script que **ejecuta**, no que importa. Respaldo de SDV implementado y probado, no documentado |
| R4 | **QAOA no escala** al Nivel B | **Certeza (100%)** | Medio | No es un riesgo, es un hecho físico (§2.4). Ya está incorporado al diseño y al informe |
| R5 | El MILP de N=45 no cierra en 600 s | Media | Bajo | Se reporta el gap de optimalidad, que sigue acotando la calidad del annealing |
| R6 | Todos los brazos empatan estadísticamente | Media | Bajo | Ya previsto: el empate **es** el resultado. La estabilidad (F6), la equidad (F7) y la interpretabilidad (§5) sostienen el trabajo por sí solas |
| R7 | El presupuesto de cómputo se dispara (10 semillas × CV anidada) | Media | Medio | D13 separa los protocolos: anidada en 3 semillas, holdout en 10. Resultados intermedios cacheados en disco por semilla |
| R8 | Sobreinterpretar resultados de datos sintéticos | Media | Alto para la nota | El informe abre con la advertencia. Los resultados son sobre **el método**, no sobre el riesgo crediticio ecuatoriano |
| R9 | La auditoría de equidad no encuentra nada porque los protegidos son independientes | Media | Medio | Neutralizado por diseño en D15: la dependencia se inyecta a propósito |
| R10 | Deriva de versiones de Qiskit (la API cambió mucho en 1.0) | Media | Medio | Versiones fijadas exactamente; verificar la documentación vigente al implementar, no confiar en la memoria del modelo |

---

## 7. Honestidad metodológica — límites que el informe debe declarar

1. Los datos son **sintéticos**, calibrados con rangos públicos, no muestreados de cartera real. Ninguna cifra de este trabajo describe a ninguna cooperativa.
2. La relación entre variables y default **la construimos nosotros**. Que un método recupere las variables "correctas" mide su capacidad de recuperar *nuestra* estructura, no la realidad.
3. **"Cuántico-inspirado" significa recocido simulado clásico.** El brazo de QAOA es un **circuito cuántico simulado clásicamente**, no ejecutado en hardware. **No se usó ningún procesador cuántico.** Cualquier ventaja observada es de la *formulación*, no de física cuántica.
4. QAOA es, en este proyecto, **más caro que enumerar todas las soluciones a la fuerza**. Se incluye para medir esa brecha, no para proponerlo como método práctico.
5. La estructura de correlación la impone nuestra semilla programática; **SDV la reproduce, no la descubre**.
6. Los intervalos de confianza son de bootstrap sobre el test; la variabilidad entre semillas se reporta por separado. Son dos fuentes de incertidumbre distintas y no se deben sumar ingenuamente.
7. Las referencias públicas (SEPS, burós de crédito, INEC) se usaron como **rango orientativo** para fijar mínimos, máximos y medias plausibles — no como fuente de datos individuales. Todo supuesto no verificable está marcado `[SUPUESTO]` en `data/external/referencias_publicas.md`.
8. La auditoría de equidad **mide sobre datos sintéticos con sesgo que nosotros inyectamos**. Detecta si un método propaga sesgo indirecto; **no** dice nada sobre el sesgo real del crédito en Ecuador.

---

## 8. Definición de "listo"

- [ ] `dataset.parquet` con 18 variables, tasa de default ≈ 8%, AUC base en [0,72 · 0,82] y las 13 correlaciones objetivo dentro de tolerancia
- [ ] Reporte de fidelidad SDMetrics + tabla comparativa de tres sintetizadores + justificación escrita de D14
- [ ] `results.json` con todos los brazos, en Nivel A y Nivel B, a través de 10 semillas
- [ ] **C1 (ExactSolver) y C2 (MILP) coinciden exactamente** en Nivel A — prueba cruzada de la formulación
- [ ] Curva de brecha-vs-`p` de QAOA con el costo computacional asociado
- [ ] Tabla de estabilidad de selección (frecuencia por variable × método × semilla) + prueba de Friedman
- [ ] Tabla de equidad por grupo + resultados de detección de proxies
- [ ] Curvas ROC superpuestas, gráfico KS, matrices de confusión, diagrama de diferencia crítica en `reports/figures/`
- [ ] `streamlit run app/Inicio.py` levanta las 5 pestañas sin error
- [ ] `pytest` en verde
- [ ] `INFORME.md` con conclusión explícita según la tabla de decisión pre-registrada, sin haberla modificado después de ver los resultados
- [ ] Revisión final contra la lista de anti-criterios de §2.7
- [ ] `README.md` con reproducción en tres comandos, verificada en un directorio limpio
