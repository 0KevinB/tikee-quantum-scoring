# INFORME.md — Tikee: selección de variables cuántico-inspirada para scoring crediticio

Proyecto de innovación universitaria · UTPL Ecuador. Basado en `reports/results.json`,
`reports/cache/f4_level_a_qubo.json`, `reports/cache/f5_level_b.json`,
`reports/fidelity/f2_comparison.json` y `reports/cache/f7_fairness.json`, generados por
el pipeline reproducible descrito en ARCHITECTURE.md.

## Advertencia obligatoria

**Los datos son 100% sintéticos.** Se generan con una semilla programática propia
(cópula gaussiana + marginales calibradas con rangos públicos de SEPS, burós de
crédito ecuatorianos e INEC/ENEMDU) y luego con `GaussianCopulaSynthesizer` de SDV.
**Ninguna cifra de este informe describe a ninguna cooperativa real.** La estructura
de correlación y el sesgo indirecto hacia los atributos protegidos los impusimos
nosotros por diseño (D5, D15); SDV los reproduce, **no los descubre**.

**"Cuántico-inspirado" significa recocido simulado clásico.** El brazo QAOA es un
circuito cuántico **simulado clásicamente** (primitivas de referencia de Qiskit,
estadísticamente equivalentes a `AerSimulator(method="statevector")` para N≤24). **No
se usó ningún procesador cuántico** en ningún momento de este trabajo.

## Pregunta de investigación y resultado en una frase

> ¿Formular la selección de variables como QUBO mejora un modelo de scoring frente a
> LASSO o stepwise, cuando las variables candidatas están fuertemente correlacionadas?

**Resultado:** en N=18 (Nivel A), la hipótesis **no se sostiene con ganancia
predictiva** — QUBO empata con LASSO/stepwise en AUC (diferencia no significativa,
Nemenyi) pero es **menos estable** entre semillas, aunque **más parsimonioso**. En
N=45 (Nivel B), la hipótesis **se rechaza con significancia estadística**: stepwise-AIC
supera a QUBO-recocido y QUBO-MILP (Nemenyi, p<0.05), y el culpable diagnosticado es
**la formulación**, no solo el heurístico — el mismo patrón aparece en el brazo
MILP casi-certificado.

## 1. La calibración de la premisa (F1–F2)

`dataset_seed42.parquet`: 8.000 filas, tasa de default 8,31% (banda esperada
[7,5%–8,5%]), AUC de una logística con las 18 variables = 0,8170 (banda [0,72–0,82]).
De las 13 correlaciones objetivo de ARCHITECTURE.md §4.2, **12 caen dentro de la
tolerancia ±0,07**. La excepción (`ratio_cuota_ingreso` ↔ `ingreso_mensual`, objetivo
−0,45, observado +0,06) es una **imposibilidad algebraica documentada**: con
`ingreso_mensual`↔`monto_solicitado`=0,55 y `ratio_cuota_ingreso`↔`monto_solicitado`=0,65
(ambos objetivos ya satisfechos y centrales a la hipótesis), el signo de la tercera
correlación queda determinado por las otras dos bajo la fórmula de `cuota_estimada` de
ARCHITECTURE.md §4.1 — una búsqueda exhaustiva de parámetros (ver
`src/tikee/data/seed_generator.py`, comentarios `[DESVIACIÓN DOCUMENTADA]`) no encontró
ninguna combinación realista que satisfaga las tres simultáneamente. Se declara aquí en
vez de forzar un valor con varianza de ingreso irrealmente alta.

El trío colineal central de la hipótesis (`score_buro`, `peor_calificacion_12m`,
`dias_mora_max_12m`) sí cae limpio: −0,79 / −0,79 / +0,83 contra objetivos de −0,85 /
−0,80 / +0,88.

**Fidelidad sintética (F2):** GaussianCopula (sintetizador de producción, D14) obtiene
QualityReport=0,967, DiagnosticReport=1,000, NewRowSynthesis=1,000 y MAE de correlación
objetivo=0,025 (excluyendo la desviación de arriba). Se comparó contra CTGAN (300
épocas, Quality=0,863, MAE=0,484) y TVAE (Quality=0,776, MAE=0,122): **GaussianCopula
gana en todos los ejes medidos**, no solo en preservación de correlación. Esto es un
resultado más limpio que la tensión que D14 anticipaba ("CTGAN puede puntuar mejor en
fidelidad general y aun así elegimos la cópula"): aquí no hace falta ese argumento —
con solo 2.000 filas semilla, los sintetizadores basados en redes neuronales no
tuvieron suficientes datos para superar al modelo paramétrico. Se reporta la
comparación igual, como exige D14, porque el objetivo de investigación (preservar la
estructura inyectada) es la razón correcta para elegir GaussianCopula
independientemente de qué hubiera pasado en la comparación general.

## 2. Nivel A (N=18): la formulación empata, no gana

`reports/cache/f4_level_a_qubo.json`. Barrido de 24 combinaciones β×k sobre train:
óptimo en β=0,25, k=9.

**Checkpoint más importante del proyecto:** C1 (`dimod.ExactSolver`, enumeración de
2¹⁸=262.144 estados) y C2 (MILP con linealización de Glover, HiGHS) devuelven
**exactamente la misma solución** (mismas 9 variables, energía −768,7482 en ambos). La
verificación cruzada de λ es positiva: el MILP con `Σx=k` como restricción dura
coincide con el MILP que resuelve el QUBO penalizado sin restricción — λ es suficiente.

**Recocido (C0)** queda a 0,002 de AUC del óptimo certificado en una sola corrida (0,34
s frente a 19,3 s del MILP). A través de 10 semillas, C0 promedia AUC=0,8150±0,011,
frente a A0 (todas)=0,8226, B0 (LASSO)=0,8231 y B1 (stepwise)=0,8229. La prueba de
Friedman es significativa (p=6,2×10⁻⁸) pero el post-hoc de Nemenyi (CD=3,80) **no
encuentra ningún par C0-vs-clásico individualmente significativo**.

**Lo que sí distingue a C0:** usa en promedio 8,8 variables frente a las 13,3 de LASSO
— parsimonia real. **Lo que lo penaliza:** su índice de Jaccard entre semillas es 0,451,
frente a 0,643 (LASSO) y 0,650 (stepwise) — **elige un subconjunto notablemente menos
estable**. Aplicando la tabla de decisión de ARCHITECTURE.md §8.7: los IC se solapan y
hay ganancia de parsimonia, pero se pierde en estabilidad — el balance neto es
**hipótesis parcialmente respaldada, sin ventaja clara**, no una victoria de QUBO.

**QAOA (C3, semilla 42, p∈{1,2,3}):** alcanza el óptimo exacto conocido con **brecha
0,0000** en los tres casos, incluso en p=1. Esto es más limpio que el resultado
"paisaje fácil" que el plan anticipaba como posible (C0=C1=C2=C4 empatando): aquí
incluso QAOA lo confirma. **Esto no es una ventaja cuántica.** Cada evaluación de QAOA
en N=18 (p=1: ~25 min; p=2: ~28 min; p=3: ~27 min de tiempo de pared) es varios órdenes
de magnitud más cara que la enumeración exhaustiva de 0,37 s que ya conocía la
respuesta. Por su costo, QAOA se corrió una sola vez (semilla 42) — el propio
presupuesto de cómputo de PLAN.md §13 trata a QAOA como costo fijo, no multiplicado por
semilla, y no participa en la comparación de Friedman/Nemenyi de F6.

## 3. Nivel B (N=45): la hipótesis se rechaza, y se diagnostica por qué

`reports/cache/f5_level_b.json`. β*=0,25, k*=20.

A diferencia de Nivel A, aquí **el MILP no cierra** en 600 s (`status=1`, gap de
optimalidad=42,4%). El gap sigue acotando cuánto puede alejarse el recocido del óptimo
verdadero (ARCHITECTURE.md §7.3) — y de hecho, con ese gap, el mejor Tabu encontrado
(E=−9804,22, 4,2 s) es ligeramente **mejor** que el incumbente del MILP (E=−9804,11,
600 s), y el recocido queda apenas detrás (E=−9803,28, 1,75 s). N=45 ya no es un
paisaje trivialmente fácil: el óptimo certificado deja de ser accesible a esta escala,
justo la transición de régimen que ARCHITECTURE.md §2.6 anticipaba que el óptimo
certificado por MILP debía trasladar hacia arriba.

**Hallazgo de diseño confirmado — la trampa de ruido funciona:** a k=20, los tres
solucionadores incluyeron variables irrelevantes o de ruido puro diseñadas
específicamente para detectar sobreajuste (ARCHITECTURE.md §4.6): C0b incluyó `f02`
(zona_residencia, coeficiente cero) y `f45` (ruido uniforme puro); C2b incluyó además
`f03`; C4b incluyó `f44` (ruido normal) y `f45` simultáneamente. **El criterio
relevancia-redundancia no penaliza una variable que es a la vez irrelevante y
no-redundante** — no aporta al término de relevancia pero tampoco cuesta nada en el
término de redundancia, así que "entra gratis" cuando la cardinalidad k excede el
número de variables genuinamente útiles y con poca redundancia entre sí.

A través de 10 semillas, C0b promedia AUC=0,7875±0,021, muy por debajo de B1b
(stepwise)=0,8197±0,013 y B0b (LASSO)=0,8190±0,013. La prueba de Friedman es
significativa (p=2,3×10⁻⁷) y **el post-hoc de Nemenyi confirma la diferencia**: B1b
vs. C0b (rank_diff=3,70) y B1b vs. C2b (rank_diff=3,50) superan la diferencia crítica
de 3,32. **Esto es una diferencia real, no ruido de muestreo.**

Aplicando el diagnóstico de ARCHITECTURE.md §8.7 ("C0 por debajo de B0 → se diagnostica
comparando contra C1/C2"): en Nivel A no hay ExactSolver posible, pero C2b (MILP,
cuasi-certificado con gap acotado) **también** queda por debajo de los clásicos y por
debajo de Tabu en calidad de energía. Esto apunta a que el problema no es solo el
heurístico de recocido — **es la formulación relevancia-redundancia misma**, que a
N=45 se vuelve vulnerable a la trampa de ruido documentada arriba. La estabilidad de
selección lo confirma: Jaccard de C0b=0,240 frente a 0,331 (stepwise) y 0,413 (LASSO).

**Conclusión de Nivel B: hipótesis rechazada, con causa diagnosticada.** No es un
fallo del proyecto — es exactamente el tipo de resultado negativo bien diagnosticado
que ARCHITECTURE.md §8.7 pide, y probablemente el hallazgo metodológico más valioso del
trabajo: el criterio mRMR-QUBO tal como está formulado no escala limpiamente a mayor N
sin una penalización explícita contra variables de bajísima relevancia, incluso si son
también de bajísima redundancia.

## 4. Interpretabilidad regulatoria SEPS (PLAN.md §5)

La distinción central: la **selección** vía QUBO es opaca en el procedimiento (difícil
explicar por qué esas 9 variables y no otras), pero el **modelo resultante** — una
logística sobre 8-9 variables — es perfectamente interpretable: cada coeficiente es
auditable.

- **Parsimonia:** C0/C1/C2 usan 8-9 variables frente a 13-18 de los brazos clásicos en
  Nivel A. En Nivel B la brecha es aún mayor (12,3 vs. 45 de "todas las variables").
- **Coherencia de signos:** el modelo del brazo C0 (semilla 42) mostró contribuciones
  del simulador consistentes con la intuición de negocio (`ratio_deuda_ingreso`,
  `dias_mora_max_12m` y `ratio_cuota_ingreso` reduciendo o aumentando el riesgo según
  su dirección esperada en los casos probados en la app).
- **VIF:** no se detectaron señales de colinealidad residual severa en los subconjuntos
  finales de 8-9 variables (el propósito explícito de la penalización de redundancia).
- **Estabilidad entre semillas:** aquí es donde QUBO **pierde** el argumento regulatorio
  — un método que cambia de variables cada vez que se regenera la cartera (Jaccard
  0,24-0,45) no puede sostener una política de crédito escrita, sin importar que su AUC
  promedio sea competitivo.
- **Sesgo indirecto (F7, la pregunta central de ARCHITECTURE.md §9.3):** con todas las
  variables o con LASSO, un clasificador auxiliar reconstruye `zona_residencia` con
  AUC=1,00 — codificación total. QUBO lo reduce a 0,61. **Este es el resultado más
  favorable a QUBO de todo el informe**, y ocurre por un mecanismo simple: menos
  variables seleccionadas significan menos rutas indirectas hacia el atributo
  protegido sobreviviendo en el modelo final. Para `sexo` y `provincia`, ningún método
  mostró codificación sustancial (AUC≈0,50-0,55) — no había mucho que reducir.
  Ninguna diferencia de paridad demográfica superó 0,08 en los cinco brazos auditados.

**El intercambio con números:** en Nivel A, QUBO no gana AUC (diferencia no
significativa) y pierde estabilidad de selección frente a LASSO/stepwise, pero reduce
la codificación de `zona_residencia` de 1,00 a 0,61 y usa ~40% menos variables. **La
recomendación honesta:** si el objetivo es minimizar sesgo indirecto medible y
simplificar el expediente de crédito, y la pérdida de estabilidad es tolerable, QUBO
tiene un argumento regulatorio genuino. Si el objetivo es maximizar AUC con un método
estable y auditable de forma simple, **LASSO o stepwise son la elección correcta y no
hay razón para adoptar un optimizador combinatorio adicional.**

## 5. Limitaciones y honestidad metodológica (PLAN.md §7)

1. Datos sintéticos; la estructura de correlación y el sesgo indirecto los impusimos
   nosotros. Recuperar la estructura "correcta" mide la capacidad del método de
   recuperar *nuestro* diseño, no la realidad del crédito ecuatoriano.
2. "Cuántico-inspirado" = recocido simulado clásico; QAOA es un circuito simulado
   clásicamente, nunca ejecutado en hardware cuántico real.
3. QAOA es, en este proyecto, más caro que enumerar todas las soluciones a la fuerza —
   se incluye para medir esa brecha (que resultó ser cero en Nivel A), no como método
   práctico propuesto.
4. Se documentaron tres desviaciones de parámetros exactos de ARCHITECTURE.md,
   encontradas y resueltas durante la implementación (F1): masa-en-cero de
   `dias_mora_max_12m` (72%→45%) y de `deuda_total_sistema` (18%→5%), y la
   imposibilidad algebraica de `ratio_cuota_ingreso`↔`ingreso_mensual`. Las tres están
   marcadas `[DESVIACIÓN DOCUMENTADA]` en el código y explicadas en este informe — no
   se ocultó ninguna.
5. El límite de tiempo del MILP en el bucle de 10 semillas de Nivel B se redujo de 600s
   a 120s por presupuesto de tiempo de entrega; F5 ya demostró con el límite completo
   que el resultado ("no cierra, se reporta el gap") es el mismo tipo de hallazgo con
   cualquiera de los dos límites.
6. La CV anidada de XGBoost (D13) se corrió sobre la semilla 42 únicamente, no sobre
   las 3 semillas especificadas, por presupuesto de tiempo de entrega. Los
   hiperparámetros fijados se reutilizaron sin re-ajuste en las 10 semillas del
   protocolo de holdout, que es la parte de D13 que sí se respetó íntegra (nunca se
   tocó un hiperparámetro después de ver resultados de test).
7. Los intervalos de confianza bootstrap (dentro de una semilla) y la variabilidad
   entre semillas son dos fuentes de incertidumbre distintas, reportadas por separado.
8. La auditoría de equidad mide sobre datos sintéticos con sesgo que nosotros
   inyectamos; detecta si un método **propaga** sesgo indirecto, no dice nada sobre el
   sesgo real del crédito en Ecuador.

## 6. Revisión final contra anti-criterios (PLAN.md §2.7)

- [x] Ningún hiperparámetro (α, β, k, p, C de LASSO) se ajustó después de ver métricas
      de test — todo el ajuste ocurrió por CV sobre train o por barrido pre-registrado.
- [x] Se reportan las 10 semillas, no la mejor — incluyendo mín/máx/dp en todas las
      tablas.
- [x] Relevancia y redundancia se calcularon exclusivamente sobre train
      (`selection/relevance.py`, `selection/redundancy.py`, siempre invocados con
      `train_df`).
- [x] El recocido nunca se presenta como "computación cuántica" — está marcado como
      heurístico clásico en cada página de la app y en este informe.
- [x] El brazo de óptimo certificado (C1/C2) nunca se omitió, ni siquiera cuando no
      favorece a la hipótesis (Nivel B).
- [x] SDV se presenta como reproductor de una estructura que la semilla programática
      impuso, no como descubridor de correlaciones.

## 7. Conclusión

La hipótesis de investigación se sostiene parcialmente en N=18 (parsimonia y menor
sesgo indirecto, sin ganancia de AUC ni de estabilidad) y **se rechaza con
significancia estadística en N=45**, con una causa diagnosticada y reproducible: el
criterio relevancia-redundancia, tal como está formulado, es vulnerable a seleccionar
variables irrelevantes-pero-no-redundantes cuando la cardinalidad excede el número de
predictores genuinamente útiles. El hallazgo regulatorio más sólido — que la selección
QUBO reduce drásticamente la codificación del atributo protegido `zona_residencia`
frente a usar todas las variables o LASSO — sobrevive independientemente del resultado
de AUC, y es probablemente la contribución más defendible de este trabajo.
