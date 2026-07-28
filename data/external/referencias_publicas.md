# Referencias públicas de calibración

Regla: **rango orientativo, no dato**. Cada fila declara fuente, año, valor tomado y
transformación aplicada. Lo no verificable dentro de esta sesión de trabajo se marca
`[SUPUESTO]`: es un valor plausible elegido por el equipo, no una cifra citada de una
fuente consultada directamente.

| Variable / parámetro | Valor usado | Fuente | Transformación | Estado |
|----------------------|-------------|--------|-----------------|--------|
| tasa base de default | 8% (`default = 1` si mora > 90 días en 12 meses) | SEPS — boletines del sector cooperativo, morosidad de cartera segmento 1 | Fijada como objetivo de calibración de `beta0` (D4) | `[SUPUESTO]` orientativo, valor no leído literal de un boletín específico |
| escala `score_buro` | 1–999, sesgada a la derecha (masa en 700–900) | Estructura típica de burós de crédito de Ecuador (Equifax/DataCredito, material público) | Beta(4.5, 2.0) reescalada a [1,999] | `[SUPUESTO]` forma paramétrica |
| distribución de ingresos | lognormal, mediana ≈ 620 USD, rango 460–3.500 | INEC — ENEMDU, ingreso laboral promedio | `lognorm(s=0.55, scale=560)` truncada al rango del esquema | `[SUPUESTO]` parámetros de forma |
| brecha de ingresos por sexo | `sexo=F` desplaza `log(ingreso)` en −0.18 | INEC/ENEMDU, brecha salarial de género reportada en Ecuador (orden de magnitud 15-20%) | multiplicación por `exp(-0.18)` | `[SUPUESTO]` magnitud exacta |
| informalidad por sexo | P(`tipo_empleo`=independiente_informal) +8pp si `sexo=F` | INEC/ENEMDU, mayor informalidad laboral femenina | ajuste directo de probabilidad categórica | `[SUPUESTO]` magnitud exacta |
| informalidad y ocupación rural | P(`tipo_empleo`=agricultor) mucho mayor en `zona_residencia=rural` | INEC, estructura del empleo rural ecuatoriano (agricultura predomina) | ajuste directo de probabilidad categórica (+30pp) | `[SUPUESTO]` magnitud exacta |
| bancarización rural | `zona_residencia=rural` desplaza `score_buro` en −60 puntos | Menor densidad de agencias/historial crediticio más corto en zonas rurales (hecho documentado, magnitud no citada) | resta directa, recortada a [1,999] | `[SUPUESTO]` magnitud exacta |
| antigüedad de vínculo rural | `zona_residencia=rural` multiplica `antiguedad_socio_meses` por 1.25 | Cooperativas de ahorro y crédito con arraigo histórico en zonas rurales (segmento 1 SEPS) | multiplicación directa | `[SUPUESTO]` magnitud exacta |
| reparto urbano/rural | 62% urbana / 38% rural | INEC, censo de población (orden de magnitud nacional) | muestreo categórico `p=[0.62, 0.38]` | `[SUPUESTO]` proporción aproximada, no desagregada por segmento cooperativo |
| tasa de la cuota | 16.5% anual | Junta de Política y Regulación Monetaria y Financiera, tasa referencial de consumo | usada en `cuota_estimada = monto·(i/12)/(1-(1+i/12)^-plazo)` | `[SUPUESTO]` valor de referencia, no la tasa vigente de un período específico |
| plazos ofertados | {6,12,18,24,36,48,60} meses | Práctica estándar de crédito de consumo/microcrédito en cooperativas segmento 1 | categórica con `p` decreciente hacia plazos largos | `[SUPUESTO]` distribución de probabilidad |
| estructura de hogares | `carga_familiar` 0–6, moda baja | INEC, tamaño medio de hogar ecuatoriano | discretización uniforme recortada | `[SUPUESTO]` forma de la distribución |

**Nota metodológica:** ninguna de estas cifras describe a una cooperativa real. Se
usan como puntos de anclaje plausibles para que el dataset sintético tenga una
estructura de riesgo creíble, no como datos verificados de una institución
específica. Ver PLAN.md §7 (honestidad metodológica) y ARCHITECTURE.md §4.7.
