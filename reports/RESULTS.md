# RESULTS.md — Tikee Quantum-Inspired Scoring

Generado a partir de `reports/results.json`, `reports/cache/f4_level_a_qubo.json`,
`reports/cache/f5_level_b.json`, `reports/fidelity/f2_comparison.json` y
`reports/cache/f7_fairness.json`. Semilla principal de referencia para cifras de un
solo corte: 42. Los promedios son sobre las 10 semillas {42, 101, 202, 303, 404, 505,
606, 707, 808, 909}.

## Nivel A (N=18) — AUC media ± dp a través de 10 semillas

| Brazo | Selección | Clasif. | AUC media | dp | min | max | #Vars (semilla 42) | Jaccard |
|---|---|---|---|---|---|---|---|---|
| A0 | Ninguna | Logística | 0.8226 | 0.0135 | 0.8007 | 0.8422 | 18 | 1.000 |
| A1 | Ninguna | XGBoost | 0.8136 | 0.0125 | 0.7921 | 0.8329 | 18 | 1.000 |
| B0 | LASSO | Logística | 0.8231 | 0.0128 | 0.8011 | 0.8399 | 13.3 (media) | 0.643 |
| B0x | LASSO | XGBoost | 0.8142 | 0.0130 | 0.7909 | 0.8329 | 13.3 (media) | 0.643 |
| B1 | Stepwise-AIC | Logística | 0.8229 | 0.0133 | 0.8032 | 0.8421 | 8.6 (media) | 0.650 |
| **C0** | **QUBO + recocido** | **Logística** | **0.8150** | 0.0113 | 0.8010 | 0.8355 | 8.8 (media) | **0.451** |
| C1 | QUBO + ExactSolver | Logística | 0.8205 | 0.0122 | 0.8034 | 0.8399 | 8.8 (media) | 0.507 |
| C2 | QUBO + MILP-Glover | Logística | 0.8205 | 0.0122 | 0.8034 | 0.8399 | 8.8 (media) | 0.507 |
| C3 (solo semilla 42, p=1/2/3) | QUBO + QAOA simulado | Logística | 0.8190 | — | — | — | 9 | — |
| R | k al azar × 100 | Logística | 0.7717 | 0.0173 | 0.7406 | 0.7913 | 9 | — |

**Prueba de Friedman:** estadístico=49.07, p=6.2×10⁻⁸ (significativo). **Nemenyi CD=3.80**:
ningún par C0-vs-clásico (A0/B0/B1) es significativo individualmente.

**Checkpoint F4 (innegociable):** C1 == C2 exactamente (misma solución, misma energía
−768.7482). Recocido (C0) queda a 0,002 de AUC del óptimo certificado.

**QAOA (semilla 42, p ∈ {1,2,3}):** brecha respecto al óptimo = 0,0000 en los tres
casos — alcanza el óptimo exacto, más caro que la enumeración exhaustiva que ya lo
conocía. Ver INFORME.md para el encuadre (D17/D18).

## Nivel B (N=45) — AUC media ± dp a través de 10 semillas

| Brazo | Selección | Clasif. | AUC media | dp | min | max | #Vars (media) | Jaccard |
|---|---|---|---|---|---|---|---|---|
| A0b | Ninguna | Logística | 0.8179 | 0.0147 | 0.7887 | 0.8382 | 45 | 1.000 |
| A1b | Ninguna | XGBoost | 0.8117 | 0.0131 | 0.7867 | 0.8297 | 45 | 1.000 |
| B0b | LASSO | Logística | 0.8190 | 0.0128 | 0.7936 | 0.8353 | 20.0 | 0.413 |
| **B1b** | **Stepwise-AIC** | **Logística** | **0.8197** | 0.0131 | 0.7948 | 0.8371 | 13.1 | 0.331 |
| **C0b** | **QUBO + recocido** | **Logística** | **0.7875** | 0.0213 | 0.7552 | 0.8290 | 12.3 | **0.240** |
| C2b | QUBO + MILP (límite 120s en el bucle de 10 semillas) | Logística | 0.7967 | 0.0291 | 0.7329 | 0.8287 | 12.3 | 0.260 |
| C4b | QUBO + Tabu | Logística | 0.7974 | 0.0231 | 0.7552 | 0.8293 | 12.3 | 0.282 |
| Rb | k al azar × 100 | Logística | 0.7563 | 0.0236 | 0.7130 | 0.7864 | 20 | — |

**Prueba de Friedman:** estadístico=43.87, p=2.3×10⁻⁷ (significativo). **Nemenyi
CD=3.32**: **B1b vs. C0b (rank_diff=3.70) y B1b vs. C2b (rank_diff=3.50) son
significativos** — stepwise-AIC supera a ambos brazos QUBO.

**MILP en N=45 (semilla 42, límite de 600s):** no cierra. `status=1`, gap de
optimalidad = 42,4%. El gap sigue acotando cuán lejos puede estar el recocido
(ARCHITECTURE.md §7.3): con este gap, el incumbente del MILP (E=−9804,11) es en
realidad ligeramente peor que el mejor encontrado por Tabu (E=−9804,22) en 4 s.

**Trampas de ruido (k*=20, semilla 42):** C0b incluyó `f02` (irrelevante) y `f45`
(ruido puro); C2b incluyó `f02`, `f03`, `f45`; C4b incluyó `f03`, `f44` y `f45`. Los
tres solucionadores cayeron en la trampa diseñada en ARCHITECTURE.md §4.6.

## Fidelidad sintética (E11, E12)

| Sintetizador | Converge | Tiempo | Quality (global) | MAE correlación objetivo |
|---|---|---|---|---|
| **GaussianCopula (producción)** | Sí | 158,8 s | **0,9674** | **0,0248** (excl. 1 desviación documentada) |
| CTGAN (300 épocas) | Sí | 26,9 s | 0,8632 | 0,4842 |
| TVAE (300 épocas) | Sí | 12,6 s | 0,7764 | 0,1217 |

DiagnosticReport = 1,0000; NewRowSynthesis = 1,0000 (ambos sobre el umbral). GaussianCopula
gana en **todos** los ejes medidos — no solo en preservación de correlación.

## Auditoría de equidad y detección de proxies (semilla 42, Nivel A)

| Brazo | #Vars | AUC_proxy(sexo) | AUC_proxy(zona_residencia) | AUC_proxy(provincia) |
|---|---|---|---|---|
| A0 | 18 | 0,554 | **1,000** | 0,513 |
| B0 (LASSO) | 17 | 0,555 | **1,000** | 0,513 |
| **C0 (QUBO)** | 8 | 0,504 | **0,611** | 0,513 |
| C1 / C2 | 8 | 0,516 | 0,610 | 0,495 |

**Pregunta central de F7 respondida:** para `zona_residencia`, usar todas las
variables o LASSO permite reconstruir el atributo protegido con AUC=1,00 (codificación
total); QUBO lo reduce a 0,61 simplemente por seleccionar menos variables. Para `sexo`
y `provincia`, ningún método muestra codificación sustancial (AUC≈0,50–0,55) en
ninguno de los brazos.

Diferencias de paridad demográfica: todas < 0,08 en los cinco brazos auditados de
Nivel A; ninguna dispara la regla del 80% de impacto dispar.
