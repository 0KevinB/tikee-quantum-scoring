# Guía de usuario — app web de Tikee

**App en vivo:** [web-production-e797c.up.railway.app](https://web-production-e797c.up.railway.app)

Esta guía es para cualquiera que visite la app por primera vez y no haya leído
el código ni los documentos técnicos: qué estás viendo, pestaña por pestaña, y
qué significa cada número. Si buscas el análisis completo con todas las cifras,
revisa [reports/INFORME.md](../reports/INFORME.md); si buscas el diseño técnico,
revisa [ARCHITECTURE.md](../ARCHITECTURE.md).

## Antes de entrar: dos advertencias que hay que leer

1. **Los datos son 100% sintéticos.** No hay información de ninguna persona ni
   de ninguna cooperativa real. Todo se generó por software, calibrado con
   rangos públicos (SEPS, INEC) solo como referencia de qué números son
   realistas — no como fuente de datos individuales.
2. **"Cuántico-inspirado" no es computación cuántica.** El método usa un
   algoritmo clásico llamado *recocido simulado* (inspirado vagamente en cómo
   se enfría un metal). Hay un experimento adicional con un *circuito cuántico
   simulado por software* (nunca en hardware cuántico real) — se incluye para
   medir honestamente qué tan lejos está esa promesa de la práctica, no porque
   la app use una computadora cuántica.

## Qué problema resuelve este proyecto (en una frase)

Un modelo de crédito necesita decidir **qué variables del solicitante usar**
(edad, ingresos, historial de pagos...). Este proyecto compara un método
"inteligente" de optimización (QUBO) contra los métodos clásicos que ya usa la
industria (LASSO, selección paso a paso), para responder con honestidad: ¿el
método nuevo realmente ayuda, o solo suena más sofisticado?

**Resultado corto:** el método nuevo no gana en precisión. Donde sí gana es en
algo más importante para un regulador financiero: **usa menos variables y
esconde menos información sensible** (como la zona donde vive alguien) que
sobrevive de forma indirecta en el modelo.

## Las 6 pestañas

### 1 · Datos

Qué hay en el conjunto de datos sintético: cuántas solicitudes (8.000), qué
porcentaje termina en mora (≈8%), y cómo se relacionan las variables entre sí
(el mapa de calor de correlación). Al final de la página hay una comparación de
tres formas distintas de generar los datos sintéticos — la que se usó es la que
mejor preserva la estructura de los datos reales que estamos simulando.

**Qué mirar:** el mapa de calor — los colores intensos (rojo/azul) muestran qué
variables van de la mano (por ejemplo, el historial de pagos tiende a moverse
junto con el puntaje de buró, que es exactamente el tipo de redundancia que el
proyecto está probando si el método sabe manejar).

### 2 · Selección

Qué variables eligió cada método de selección, para dos escenarios: uno simple
(18 variables candidatas) y uno más realista y difícil (45 variables,
incluyendo interacciones y dos columnas de "ruido" a propósito, para ver si
algún método se deja engañar).

**Qué mirar:**
- Las listas de variables elegidas por cada método — fíjate que los métodos
  "certificados" (que garantizan la mejor solución posible) coinciden exacto
  entre sí. Eso es una prueba de que el método está implementado correctamente.
- La advertencia en rojo sobre las "variables de ruido" — es honesta: en el
  escenario difícil, algunos métodos sí cayeron en la trampa. Se reporta sin
  esconderlo.
- La curva del circuito cuántico simulado — muestra que llega al mejor
  resultado posible, pero tardando muchísimo más que simplemente probar todas
  las combinaciones una por una. Ese es el punto: medir la brecha, no vender
  una ventaja que no existe.

### 3 · Comparación

Las curvas ROC (qué tan bien distingue el modelo entre quien paga y quien no)
y las matrices de confusión de cada método, lado a lado. Puedes cambiar entre
"Nivel A" (18 variables) y "Nivel B" (45 variables) con el selector de arriba.

**Qué mirar:** todas las curvas están muy pegadas entre sí — eso confirma
visualmente que ningún método gana por mucho margen. La diferencia real está
en cuántas variables usa cada uno, no en qué tan buena es la curva.

### 4 · Simulador

La parte más interactiva: completa el formulario de un solicitante hipotético
(edad, ingresos, historial...) y el modelo calcula al instante una probabilidad
de mora y una decisión de aprobar/rechazar. Debajo aparecen **las 3 razones
principales** que más pesaron en ese cálculo — así es como un asesor de crédito
le explicaría a un socio real por qué se aprobó o se negó su solicitud.

**Cómo usarlo:**
1. Elige qué modelo probar en "Modelo (brazo)" (por defecto, el modelo elegido
   por el método QUBO — solo usa 8 variables).
2. Mueve los controles para describir a un solicitante.
3. Presiona "Calcular probabilidad de default".
4. Lee la probabilidad, la decisión, y las 3 razones — todo el cálculo ocurre
   en el servidor con un modelo real ya entrenado, sin inventar nada.

### 5 · Estabilidad

Aquí está la evidencia estadística seria: cómo se comporta cada método a través
de **10 corridas independientes** (para no confiar en un solo resultado con
suerte), el resultado de la prueba estadística que confirma si las diferencias
son reales o casualidad, y el panel de equidad.

**Qué mirar, en orden de importancia:**
1. **Panel de equidad y detección de proxies** (al final de la página): la
   cifra más importante del proyecto. Muestra qué tan bien se puede adivinar
   la zona de residencia de alguien SOLO mirando las variables que el modelo
   usó — sin que esa zona haya sido nunca una variable de entrada. Con todas
   las variables o con el método clásico, se adivina casi perfecto (1.00). Con
   el método QUBO, baja a 0.61, simplemente por usar menos variables.
2. **Índice de Jaccard**: qué tan parecidas son las variables elegidas entre
   una corrida y otra. Más alto es mejor — un método que cambia de variables
   cada vez que se le da información nueva no sirve para escribir una política
   de crédito estable.
3. El resultado de Friedman/Nemenyi (el diagrama de puntos) confirma con
   estadística si las diferencias de precisión entre métodos son reales.

### 6 · Documentación

Todo lo que no cabe en un gráfico: esta misma guía, el informe completo con
todas las cifras, la tabla de resultados en bruto, las referencias públicas
usadas para calibrar los datos sintéticos, y la documentación técnica
(arquitectura del sistema y plan del proyecto) para quien quiera revisar o
replicar el trabajo — todo dentro de la app, sin tener que ir a GitHub.

## Preguntas frecuentes

**¿Esto usa una computadora cuántica de verdad?**
No. Todo corre en un servidor normal. Hay una simulación de un circuito
cuántico (para el experimento QAOA), pero se ejecuta con software clásico, no
con hardware cuántico.

**¿Los datos son de alguna cooperativa ecuatoriana real?**
No. Son 100% generados por software. Se usaron rangos públicos de fuentes como
la SEPS y el INEC solo para que los números parezcan realistas — nunca se usó
información de una persona o institución real.

**¿"AUC" qué es?**
Una medida de 0.5 a 1.0 de qué tan bien distingue el modelo entre quien va a
pagar y quien no. 0.5 es "adivinar al azar"; 1.0 es "perfecto". Los modelos de
este proyecto rondan 0.80-0.82, un nivel realista para un scoring de crédito.

**¿Por qué el modelo "QUBO" no es simplemente el mejor si es más sofisticado?**
Porque más sofisticado no significa mejor — y este proyecto existe justamente
para comprobarlo con números en vez de asumirlo. Los resultados dicen que no
gana en precisión, y esa es información honesta y útil, no un fracaso del
proyecto.

**¿Dónde veo todos los números completos, no solo la app?**
[reports/INFORME.md](../reports/INFORME.md) tiene el análisis completo con
todas las cifras y las limitaciones declaradas explícitamente.
