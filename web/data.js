// Datos pre-computados del experimento (10 semillas, pipeline reproducible).
// Fuente: reports/results.json, reports/RESULTS.md, reports/INFORME.md,
// reports/cache/f7_fairness.json, reports/fidelity/f2_comparison.json,
// reports/cache/app_artifacts.joblib (coeficientes del brazo C0, extraídos y
// validados: AUC reconstruido con estos coeficientes = AUC original, corr=0.999999).
const TIKEE_DATA = {
  nivelA: [
    { brazo: "A0", seleccion: "Ninguna", clasif: "Logística", aucMean: 0.8226, aucSd: 0.0135, aucMin: 0.8007, aucMax: 0.8422, vars: "18", jaccard: 1.000 },
    { brazo: "A1", seleccion: "Ninguna", clasif: "XGBoost", aucMean: 0.8136, aucSd: 0.0125, aucMin: 0.7921, aucMax: 0.8329, vars: "18", jaccard: 1.000 },
    { brazo: "B0", seleccion: "LASSO", clasif: "Logística", aucMean: 0.8231, aucSd: 0.0128, aucMin: 0.8011, aucMax: 0.8399, vars: "13.3 (media)", jaccard: 0.643 },
    { brazo: "B0x", seleccion: "LASSO", clasif: "XGBoost", aucMean: 0.8142, aucSd: 0.0130, aucMin: 0.7909, aucMax: 0.8329, vars: "13.3 (media)", jaccard: 0.643 },
    { brazo: "B1", seleccion: "Stepwise-AIC", clasif: "Logística", aucMean: 0.8229, aucSd: 0.0133, aucMin: 0.8032, aucMax: 0.8421, vars: "8.6 (media)", jaccard: 0.650 },
    { brazo: "C0", seleccion: "QUBO + recocido", clasif: "Logística", aucMean: 0.8150, aucSd: 0.0113, aucMin: 0.8010, aucMax: 0.8355, vars: "8.8 (media)", jaccard: 0.451, highlight: true },
    { brazo: "C1", seleccion: "QUBO + ExactSolver", clasif: "Logística", aucMean: 0.8205, aucSd: 0.0122, aucMin: 0.8034, aucMax: 0.8399, vars: "8.8 (media)", jaccard: 0.507 },
    { brazo: "C2", seleccion: "QUBO + MILP-Glover", clasif: "Logística", aucMean: 0.8205, aucSd: 0.0122, aucMin: 0.8034, aucMax: 0.8399, vars: "8.8 (media)", jaccard: 0.507 },
    { brazo: "C3", seleccion: "QUBO + QAOA (semilla 42)", clasif: "Logística", aucMean: 0.8190, aucSd: null, aucMin: null, aucMax: null, vars: "9", jaccard: null },
    { brazo: "R", seleccion: "k al azar × 100", clasif: "Logística", aucMean: 0.7717, aucSd: 0.0173, aucMin: 0.7406, aucMax: 0.7913, vars: "9", jaccard: null },
  ],
  nivelB: [
    { brazo: "A0b", seleccion: "Ninguna", clasif: "Logística", aucMean: 0.8179, aucSd: 0.0147, aucMin: 0.7887, aucMax: 0.8382, vars: "45", jaccard: 1.000 },
    { brazo: "A1b", seleccion: "Ninguna", clasif: "XGBoost", aucMean: 0.8117, aucSd: 0.0131, aucMin: 0.7867, aucMax: 0.8297, vars: "45", jaccard: 1.000 },
    { brazo: "B0b", seleccion: "LASSO", clasif: "Logística", aucMean: 0.8190, aucSd: 0.0128, aucMin: 0.7936, aucMax: 0.8353, vars: "20.0", jaccard: 0.413 },
    { brazo: "B1b", seleccion: "Stepwise-AIC", clasif: "Logística", aucMean: 0.8197, aucSd: 0.0131, aucMin: 0.7948, aucMax: 0.8371, vars: "13.1", jaccard: 0.331, winner: true },
    { brazo: "C0b", seleccion: "QUBO + recocido", clasif: "Logística", aucMean: 0.7875, aucSd: 0.0213, aucMin: 0.7552, aucMax: 0.8290, vars: "12.3", jaccard: 0.240, highlight: true },
    { brazo: "C2b", seleccion: "QUBO + MILP (120s)", clasif: "Logística", aucMean: 0.7967, aucSd: 0.0291, aucMin: 0.7329, aucMax: 0.8287, vars: "12.3", jaccard: 0.260 },
    { brazo: "C4b", seleccion: "QUBO + Tabu", clasif: "Logística", aucMean: 0.7974, aucSd: 0.0231, aucMin: 0.7552, aucMax: 0.8293, vars: "12.3", jaccard: 0.282 },
    { brazo: "Rb", seleccion: "k al azar × 100", clasif: "Logística", aucMean: 0.7563, aucSd: 0.0236, aucMin: 0.7130, aucMax: 0.7864, vars: "20", jaccard: null },
  ],
  friedman: {
    nivelA: { stat: 49.07, p: "6.2×10⁻⁸", cd: 3.80, conclusion: "Significativo, pero ningún par C0-vs-clásico (A0/B0/B1) es significativo individualmente en el post-hoc de Nemenyi." },
    nivelB: { stat: 43.87, p: "2.3×10⁻⁷", cd: 3.32, conclusion: "Significativo. B1b vs. C0b (rank_diff=3.70) y B1b vs. C2b (rank_diff=3.50) SÍ superan la diferencia crítica: stepwise-AIC gana de forma demostrable." },
  },
  fidelity: [
    { synth: "GaussianCopula (producción)", converge: "Sí", tiempo: "158.8 s", quality: 0.9674, mae: 0.0248, best: true },
    { synth: "CTGAN (300 épocas)", converge: "Sí", tiempo: "26.9 s", quality: 0.8632, mae: 0.4842 },
    { synth: "TVAE (300 épocas)", converge: "Sí", tiempo: "12.6 s", quality: 0.7764, mae: 0.1217 },
  ],
  fairness: [
    { brazo: "A0 (todas)", vars: 18, sexo: 0.554, zona: 1.000, provincia: 0.513 },
    { brazo: "B0 (LASSO)", vars: 17, sexo: 0.555, zona: 1.000, provincia: 0.513 },
    { brazo: "C0 (QUBO)", vars: 8, sexo: 0.504, zona: 0.611, provincia: 0.513, highlight: true },
    { brazo: "C1 / C2", vars: 8, sexo: 0.516, zona: 0.610, provincia: 0.495 },
  ],
  checkpoints: [
    { nombre: "F1 — AUC base en banda [0.72, 0.82]", valor: "AUC = 0.8170 (dataset final, semilla 42)", ok: true },
    { nombre: "F1 — Tasa de default en [7.5%, 8.5%]", valor: "8.31%", ok: true },
    { nombre: "F1 — 13 correlaciones objetivo (tolerancia ±0.07)", valor: "12/13 dentro de tolerancia; 1 excepción algebraica documentada", ok: true },
    { nombre: "F4 — C1 (ExactSolver) == C2 (MILP-Glover)", valor: "Misma solución exacta, misma energía −768.7482", ok: true },
    { nombre: "F4 — Recocido (C0) vs. óptimo certificado", valor: "A 0.002 de AUC del óptimo, en 0.34s vs. 19.3s del MILP", ok: true },
    { nombre: "Tests automatizados", valor: "26/26 pytest en verde", ok: true },
  ],
  // Modelo C0 (QUBO + recocido simulado, Nivel A, semilla 42) — logística sobre 8 variables.
  // Extraído de reports/cache/app_artifacts.joblib. Validado: AUC reconstruido con estos
  // coeficientes = AUC original del pipeline (corr=0.999999996).
  simulator: {
    threshold: 0.0887888001237543,
    intercept: -2.9546390011073362,
    features: [
      { key: "antiguedad_socio_meses", label: "Antigüedad como socio (meses)", mean: 70.486250, std: 62.815527, coef: -0.2618686748563462, min: 1, max: 300, step: 1, default: 60 },
      { key: "score_buro", label: "Score de buró (1-999)", mean: 667.679643, std: 177.424796, coef: -0.5329673464041045, min: 1, max: 999, step: 1, default: 650 },
      { key: "peor_calificacion_12m", label: "Peor calificación últimos 12m (1=A1 … 9=E)", mean: 4.610000, std: 2.554784, coef: 0.0784953484169015, min: 1, max: 9, step: 1, default: 4 },
      { key: "dias_mora_max_12m", label: "Días de mora máx. últimos 12m", mean: 24.832143, std: 34.828676, coef: 0.3775513784704433, min: 0, max: 180, step: 1, default: 10 },
      { key: "plazo_meses", label: "Plazo solicitado (meses)", mean: 25.750714, std: 14.587867, coef: -0.03155311048061631, min: 6, max: 60, step: 1, default: 24 },
      { key: "nivel_educacion", label: "Nivel de educación (1=primaria … 5=posgrado)", mean: 2.662500, std: 1.229375, coef: -0.03232639306469469, min: 1, max: 5, step: 1, default: 3 },
      { key: "ratio_cuota_ingreso", label: "Ratio cuota/ingreso", mean: 0.373965, std: 0.262959, coef: 0.5912323584346971, min: 0.02, max: 0.90, step: 0.01, default: 0.30 },
      { key: "ratio_deuda_ingreso", label: "Ratio deuda total/ingreso", mean: 10.042196, std: 4.880385, coef: 0.20796583979009198, min: 0, max: 20, step: 0.1, default: 8 },
    ],
    note: "nivel_educacion es una variable diseñada como ruido (coeficiente ≈ 0 en el proceso generador). El QUBO la seleccionó igual, pero la logística le asignó un coeficiente casi nulo (−0.032) — el modelo se autocorrigió. Se incluye en el simulador precisamente para mostrar eso.",
  },
  figures: [
    { file: "roc_overlay_nivel_a.png", title: "Curvas ROC superpuestas — Nivel A" },
    { file: "roc_overlay_nivel_b.png", title: "Curvas ROC superpuestas — Nivel B" },
    { file: "ks_curve_nivel_a.png", title: "Curva KS — Nivel A" },
    { file: "ks_curve_nivel_b.png", title: "Curva KS — Nivel B" },
    { file: "auc_boxplot_nivel_a.png", title: "Distribución de AUC (10 semillas) — Nivel A" },
    { file: "auc_boxplot_nivel_b.png", title: "Distribución de AUC (10 semillas) — Nivel B" },
    { file: "critical_difference_nivel_a.png", title: "Diagrama de diferencia crítica (Nemenyi) — Nivel A" },
    { file: "critical_difference_nivel_b.png", title: "Diagrama de diferencia crítica (Nemenyi) — Nivel B" },
    { file: "confusion_matrices_nivel_a.png", title: "Matrices de confusión — Nivel A" },
    { file: "confusion_matrices_nivel_b.png", title: "Matrices de confusión — Nivel B" },
    { file: "qaoa_gap_curve.png", title: "Brecha QAOA vs. óptimo conocido (p=1,2,3)" },
  ],
};
