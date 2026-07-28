# Tikee — Selección de variables cuántico-inspirada para scoring crediticio

Proyecto de innovación universitaria · UTPL Ecuador
Inspirado en el pitch de la fintech Tikee. **Datos 100% sintéticos.**

## Pregunta de investigación

¿Formular la selección de variables como un problema **QUBO** mejora un modelo de scoring crediticio frente a LASSO o stepwise, cuando las variables candidatas están fuertemente correlacionadas entre sí?

El mismo QUBO se resuelve con **cinco solucionadores** para separar tres preguntas que la literatura suele confundir:

1. ¿Es buena **la formulación** QUBO relevancia-redundancia? → recocido vs. LASSO
2. ¿Aporta algo **el heurístico**? → recocido vs. óptimo certificado (enumeración exacta / MILP)
3. ¿Aporta algo **el paradigma de circuitos**? → QAOA simulado vs. recocido

## Advertencias

- Los datos son **sintéticos**, generados con SDV y calibrados con rangos públicos (SEPS, burós de crédito, INEC) como referencia orientativa. Ninguna cifra describe a ninguna cooperativa real.
- **"Cuántico-inspirado" = recocido simulado clásico.** El brazo de QAOA es un circuito cuántico **simulado clásicamente**. **No se usa hardware cuántico.**
- QAOA aquí **no puede ganar**: en N=18 el óptimo global ya se conoce por enumeración, y simular el circuito es órdenes de magnitud más caro. Se incluye para medir esa brecha, que es un resultado legítimo. En N=45 ni siquiera puede ejecutarse (requeriría 563 TB de RAM).

## Estado

**Fase 1 — planificación: completa** (v2.0, alcance ampliado). Ver [PLAN.md](PLAN.md) y [ARCHITECTURE.md](ARCHITECTURE.md).
**Fase 2 — implementación: completa.** F0–F8 ejecutadas end-to-end, `pytest` en verde (26/26), app Streamlit de 5 pestañas funcional. Ver [reports/INFORME.md](reports/INFORME.md) y [reports/RESULTS.md](reports/RESULTS.md) para el análisis y las cifras finales.

Tres desviaciones documentadas respecto a ARCHITECTURE.md, encontradas durante la
calibración de F1 y explicadas en `reports/INFORME.md` §5 y en los comentarios
`[DESVIACIÓN DOCUMENTADA]` de `src/tikee/data/seed_generator.py`. Ninguna afecta los
dos checkpoints innegociables (F1: AUC en banda; F4: C1==C2 exacto), ambos en verde.

## Reproducción

```bash
make setup && make verify && make experiment && make app
```

`make experiment` corre el pipeline completo de 10 semillas (~1 h). Los resultados de
esta entrega ya están cacheados en `reports/` y `reports/cache/`; `make app` funciona
directamente sobre ellos sin necesidad de recorrer `make experiment` de nuevo.

## Documentos

| Archivo | Contiene |
|---------|----------|
| [PLAN.md](PLAN.md) | Calibración de plazo, alcance, decisiones D1–D20, cronograma por fases, riesgos, honestidad metodológica |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Stack, estructura, flujo de datos, esquema de 18 variables + expansión a 45, sesgo inyectado, formulación QUBO, cinco solucionadores, diseño experimental, auditoría de equidad |
| [data/external/referencias_publicas.md](data/external/referencias_publicas.md) | Trazabilidad de cada rango de calibración |
