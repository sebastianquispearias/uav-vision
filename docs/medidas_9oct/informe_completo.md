---
title: "Qué sostiene una placa embebida en vuelo"
subtitle: "Medición del 9 de octubre de 2026 · sistema de geolocalización por visión monocular · LAC, PUC-Rio"
date: "9 de octubre de 2026"
lang: es
---

# 0. Para qué sirve este documento

Hay tres lectores posibles y el texto sirve a los tres sin cambiar de registro.

El primero sos vos dentro de seis meses, cuando no te acuerdes por qué la Pi 4 quedó
configurada a una imagen por segundo. El segundo es Bruno y el grupo, que hicieron
preguntas concretas el 2 de octubre y merecen respuestas con números. El tercero es
quien te entreviste para un puesto de *Edge AI* o *Perception Engineer*, que va a
querer ver no el resultado sino **cómo llegaste a él**.

Por eso cada número viene con su cuenta desarrollada. Un número suelto se olvida y no
se puede defender; una cuenta se reconstruye.

---

# 1. Qué pidió el grupo, textual

De la reunión del 2 de octubre, *Evolução do Sistema BeeSwarm*. Cito literal y
traduzco a qué significa en términos de trabajo.

## 1.1 Medir cuánto cuesta

> *"Vamos ver o quanto isso pesa em processamento. Aí nós sentamos com o resultado e
> vemos o próximo passo."*

> *"vamos rodar o que a gente tem hoje"*

**Qué significa.** Es un pedido de medición, no de funcionalidad. "Correr lo que
tenemos hoy" es explícito: no agregues capacidades, cuantificá las que hay. Este
documento es la respuesta.

## 1.2 Explicar la anomalía de los FPS

> *"Eu tô vendo que lá em cima tá dando 4 FPSs. Esse é uma média entre os dois
> Raspberries ou é... cada um tá conseguindo 4 FPSs?"*

> *"Uai, o Raspberry 4 e o 5 tão dando a mesma coisa? **Não faz sentido**."*

> *"**A gente tem que saber explicar por que que funcionou.**"*

**Qué significa.** Bruno detectó una inconsistencia real y nadie supo explicarla ese
día. Tenía razón en sospechar: dos computadoras con una generación de diferencia no
pueden rendir igual. La respuesta está en la sección 5 y es que **la pantalla mostraba
la tasa pedida, no la entregada**. Eran dos números distintos con el mismo nombre.

## 1.3 La duda sobre el vector de apariencia

Planteada dos veces, la segunda citando que Bruno ya la había hecho antes:

> *"A gente passou um bounding box, a gente deu uma cropada e a gente fez o vetor. Mas
> na hora de comparação do vetor... se você ver uma imagem de um lado e ver uma imagem
> do outro... elas tão invertidas, **uma pessoa vista da direita, uma pessoa vista da
> esquerda. É a mesma pessoa. Mas a imagem é invertida**."*

> *"é a mesma dúvida que o Bruno tinha tido do vetor de embedding pra comparar
> diferentes detecções... **isso pode variar muito dependendo da direção**... não era
> melhor tipo usar meio que a geoloca...?"*

La respuesta que diste en la reunión fue correcta pero cualitativa:

> *"É uma heurística juntada com isso, então é... **é os dois**, né?"*

**Qué significa.** Bruno está señalando el problema de **invariancia de punto de
vista** en re-identificación. Y tiene razón empírica: el número está medido y es
incómodo. Lo desarrollo en la sección 3.4.

## 1.4 Dos drones, y esperar antes de integrar

> *"Eu só tenho uma sugestão aqui. **Vamos trabalhar só com dois drones por enquanto**,
> Sebastian? Porque a gente nem tem o terceiro com câmera direito."*

> *"vamos esperar um pouquinho antes de botar as coisas na nossa ground station"*

> *"**Espera mais uma semana ou duas** pra gente ter essa conversa. Vamos ver isso
> rodando no mundo real."*

**Qué significa.** Dos aeronaves, no tres. La integración con la estación del grupo
queda postergada **por decisión de ellos**, no por falta tuya. Y hay una fecha: una o
dos semanas desde el 2 de octubre vencen entre el 9 y el 16.

---

# 2. Qué es Edge AI y qué se mide de verdad

Tu pregunta de fondo: *¿es solo latencia y capacidad del modelo?* No. Son **siete
ejes**, y el error más común es medir uno y declarar el sistema.

## 2.1 Los siete ejes

**(1) Tasa sostenida, no de ráfaga.** Cuántas inferencias por segundo puede mantener
el sistema **durante el tiempo que dura la misión**, no durante treinta segundos. Es
el eje que más se falsea, porque medir corto es barato y da números buenos. La
sección 5.3 muestra el caso exacto: a los dos minutos la Pi 4 parecía aprobada y a los
ocho estaba frenándose.

**(2) Latencia por etapa, con percentiles.** La tasa dice *cada cuánto* produce el
sistema. La latencia dice *qué tan vieja* es cada respuesta. Son independientes: un
sistema puede entregar a 10 Hz con 3 segundos de retraso. Y hay que medirla **por
etapa**, porque el total no dice dónde optimizar.

**(3) Energía por inferencia.** En joules por cuadro. Es lo que traduce una decisión de
cómputo en minutos de autonomía, que en un dron es la restricción que manda sobre todas
las demás.

**(4) Memoria y comportamiento bajo presión.** No solo cuánta usa, sino qué pasa cuando
falta. Si el sistema entra en memoria de intercambio, **la tasa media casi no se mueve**
mientras la latencia individual se dispara. Es un fallo que solo se ve mirando la cola.

**(5) Precisión en cada configuración.** Toda optimización de velocidad paga en algo.
La pregunta no es "¿cuánto más rápido?" sino "¿cuánto más rápido **y a qué costo de
detección**?". Esto exige verdad de campo, que es caro de conseguir, y por eso casi
nadie lo publica.

**(6) La cola, no el promedio.** El percentil 95 y el 99. Un sistema rápido en
diecinueve cuadros de veinte y trabado en el veinte tiene un promedio excelente y
pierde objetivos. El promedio es exactamente el estadístico que esconde este fallo.

**(7) Las decisiones de modelo y de runtime.** Cuantización (INT8 contra FP32),
resolución de entrada, número de hilos, motor de inferencia (NCNN, TFLite, ONNX
Runtime). Estas son las palancas, y solo sirven si sabés sobre qué eje querés mover.

## 2.2 Dónde estás parado

De los siete, tenés **cinco medidos**:

| Eje | Estado | Dónde |
|---|---|---|
| 1. Tasa sostenida | **Medido** | §5.3, curva térmica de 8 minutos |
| 2. Latencia por etapa | **Medido** | §5.1, mediana de ciclo y de detector |
| 3. Energía por inferencia | **Falta** | §7.3 |
| 4. Memoria | **Medido** | §5.4, 3,1 GB libres, sin intercambio |
| 5. Precisión por configuración | **Medido** | §5.5, sobre vuelo etiquetado |
| 6. La cola (p95) | **Instrumentado, aún sin dato válido** | §7.2 |
| 7. Modelo y runtime | Parcial | NCNN elegido; sin barrido de cuantización |

Cinco de siete, con verdad de campo real en el quinto, es **más de lo que la mayoría de
los proyectos puede mostrar**. Los dos que faltan son justamente los que un
entrevistador técnico va a preguntar, así que conviene tener la respuesta preparada:
*"no está medido, y sé exactamente cómo medirlo"* es una respuesta sólida; improvisar
no lo es.

## 2.3 Lo que no es Edge AI

Vale decirlo porque aclara el resto. **No** es entrenar modelos; eso pasa en la nube,
antes. **No** es elegir la arquitectura con mejor mAP en un *benchmark*; esa elección
se hace bajo restricciones que el benchmark no tiene. Edge AI es el trabajo de hacer
que un modelo ya entrenado **cumpla un contrato bajo restricciones físicas**: tantos
vatios, tantos grados, tantos milisegundos, tanta memoria, tanta batería. La
ingeniería está en el contrato, no en el modelo.

---

# 3. Cómo es tu sistema

## 3.1 La cadena, de punta a punta

```
cámara  →  detector  →  seguidor  →  apariencia  →  rayo  →  suelo  →  identidad  →  reporte
ArduCam    YOLO        BoT-SORT     OSNet        pinhole  z=0     fusión        radio
1920×1080  ncnn        + CMC        ReID                                        GrADyS
```

Qué hace cada etapa, con lo que entra y lo que sale:

**Cámara.** ArduCam Module 3 (IMX708) en el puerto CAM1 de la Raspberry, leída con
`picamera2`. Calibrada con tablero: focal 1407,0 px, centro (945,7 / 547,1), error de
reproyección **4,25 px sobre 111 fotos válidas**. Esto importa: la focal entra
directamente en el cálculo del rayo, y un error ahí se propaga a metros en el suelo.

**Detector.** YOLO compilado a NCNN, que es el motor de inferencia pensado para ARM.
Sale una lista de cajas con clase y confianza.

**Seguidor.** BoT-SORT con compensación de movimiento de cámara (CMC). Le asigna a cada
caja un identificador que persiste entre cuadros. El CMC importa en un dron porque la
cámara se mueve: sin él, el movimiento propio se confunde con movimiento del objetivo.

**Apariencia.** OSNet produce un vector por recorte. Es lo que permite decir "esta
persona de ahora es la misma de hace veinte segundos" cuando el seguidor perdió la
pista.

**Rayo.** El paso geométrico. Un píxel y la pose de la aeronave dan una semirrecta en
el espacio.

**Suelo.** Esa semirrecta se corta con el plano horizontal a la altura declarada. El
corte es la posición estimada del objetivo.

**Identidad.** Acumula impactos por pista, decide si una pista está madura, y fusiona
pistas distintas que probablemente sean la misma persona.

**Reporte.** Un mensaje JSON por el plano de datos de GrADyS, difundido a quien escuche.

## 3.2 La geometría, con la cuenta

Del píxel al rayo. Con `(u, v)` el píxel, `(cx, cy)` el centro óptico y `f` la focal
en píxeles, la dirección en el sistema de la cámara es

```
d_cam = normalizar( u - cx ,  v - cy ,  f )
```

Esa dirección se rota al sistema del mundo por el rumbo de la aeronave y el cabeceo del
montaje (−55° en el vuelo de referencia), más alabeo y cabeceo del fuselaje cuando hay
fuente de actitud.

Del rayo al suelo. Con origen `o` (la posición de la aeronave) y dirección
`d`, el corte con el plano `z = z_suelo` está en

```
k = (z_suelo - o_z) / d_z          p = o + k * d
```

Si `d_z >= 0` el rayo apunta al horizonte o por encima y no hay corte: esa detección
se descarta.

**Por qué esto importa para entender el sistema.** La precisión en el suelo depende del
**rango inclinado**, no de la altura. A 35 m de altura y −55° de cabeceo, el rango
inclinado es 35 / sen(55°) = 42,7 m. Un error de rumbo de 1° desplaza el
impacto en 42,7 * tan(1°) = 0,75 m. Esa es la razón de que el radio de
fusión no sea una constante.

## 3.3 El radio de fusión, que es una fórmula y no un número

```
fusion_radius_m = gps_sigma + slant_range * yaw_sigma
```

Dos términos, dos fuentes de error distintas. El primero es la incertidumbre de la
posición de la aeronave, que no depende de a dónde mire. El segundo es la
incertidumbre de rumbo **multiplicada por la distancia**: cuanto más lejos mira, más
cuesta un grado de error.

Esto es lo que separa un parámetro derivado de una constante mágica. Todos los
términos están en la telemetría, así que el radio se recalcula por detección en vez de
ajustarse a mano.

**No todos los parámetros son así, y conviene no mentir sobre eso.** `MIN_MEASUREMENTS`,
`DUTY_MIN`, `COOCURRENCIA_MIN` y el umbral del detector **no** se derivan de los datos:
codifican el costo de equivocarse, que cambia según la misión. Un rescate tolera más
falsos positivos que una inspección. Presentarlos como físicamente derivados sería
falso.

## 3.4 La duda de Bruno, con el número

Bruno preguntó si el vector de apariencia aguanta el cambio de punto de vista. La
medición sobre vectores promediados en ventanas de cuatro segundos dice:

> **Con el umbral puesto, el 35 % de los pares de personas DISTINTAS pasa el filtro de
> apariencia.**

Leído de frente: **la apariencia sola no distingue personas.** Lo que sostiene el
sistema es que **la distancia filtra primero**, y la apariencia refina dentro de lo que
la distancia ya dejó pasar. Tu respuesta en la reunión, *"é os dois"*, era correcta.
Ahora tiene un número detrás.

Esto no invalida nada, pero sí invalida **una frase**: no se puede decir "la apariencia
distingue personas". Se puede decir "la distancia separa candidatos y la apariencia
decide entre los que quedan cerca".

---

# 4. Cómo se midió, y por qué así

El método es la parte defendible. Un resultado sin método es una anécdota.

## 4.1 Instrumentar la cadena, no cronometrarla desde afuera

Un cronómetro externo mide el intervalo entre reportes, que incluye las esperas. Lo
que interesa es el trabajo. Por eso cada reporte que sale de la aeronave lleva ahora:

```
lat_ciclo_p50_ms      la mediana del ciclo completo de una mirada
lat_ciclo_p95_ms      el percentil 95 del mismo ciclo
lat_detector_p50_ms   la mediana del detector solo
lat_muestras          cuántas miradas entraron en el intervalo
mem_mb                memoria disponible
```

## 4.2 Percentil 95 y no promedio

Ya está argumentado en §2.1(6) y es la decisión metodológica más importante. El
promedio esconde exactamente el fallo que importa.

## 4.3 Vaciar la ventana en cada reporte

Los acumuladores se vacían al publicar. Un percentil acumulado sobre toda la misión
queda dominado por el arranque, cuando se están cargando el detector y el modelo de
apariencia, y seguiría informando ese arranque minutos después de que la placa se
asentó. El operador necesita saber qué está haciendo la aeronave **ahora**.

## 4.4 Reloj de pared, no el del simulador

El reloj del proveedor lo mueve la simulación cuando se reproduce un vuelo grabado, e
informaría lo que un cuadro **debía** tardar. La pregunta es cuánto tardó, así que se
usa `time.perf_counter()`.

## 4.5 Las dos placas, la misma tasa, al mismo tiempo

Lo que vale no es el techo de cada placa aislada sino **dónde divergen**. La Pi 5 plana
en 47 °C mientras la Pi 4 trepa hasta frenarse es todo el argumento, y ese contraste
solo existe si a las dos se les pidió lo mismo.

## 4.6 Diez minutos por punto

No es exceso de celo. Es lo que tarda la temperatura en mostrar su pendiente. La
sección 5.3 muestra el caso en que dos minutos mienten.

## 4.7 La precisión, sin hardware

El vuelo etiquetado se vuelve a correr submuestreado: una de cada N imágenes. La tasa
declarada **se deriva de la lista de cuadros**, no se configura aparte, de modo que no
hay dos perillas que puedan desincronizarse. El vuelo se grabó a 1,64 imágenes por
segundo, así que la única dirección honesta es hacia abajo: tasas mayores no se pueden
simular porque esos cuadros no existen.

---

# 5. Resultados, con los cálculos desarrollados

## 5.1 Latencia: el número que explica todo lo demás

Veinte muestras, las dos placas a 0,5 imágenes por segundo, cámara y detector reales.

| | Pi 5 | Pi 4 |
|---|---|---|
| mediana del ciclo | **103,8 ms** | **721,1 ms** |
| rango observado | 94 – 148 ms | 680 – 764 ms |
| muestras | 10 | 7 |
| mediana del detector | 103,8 ms | 721,1 ms |

```
721,1 ms / 103,8 ms = 6,95
```

**La Pi 4 tarda casi siete veces más por imagen que la Pi 5.**

Y el segundo dato de esa tabla es tan importante como el primero: **la mediana del
detector es igual a la del ciclo completo**, en las dos placas. La proyección del rayo,
el corte con el suelo, el seguidor y la capa de identidad **juntos no suman un
milisegundo medible**. Si hay que optimizar algo, es el detector, y nada más. Eso
descarta de un golpe cualquier trabajo de optimización sobre la geometría o la
asociación.

## 5.2 El techo de cada placa, con la cuenta

Si cada imagen cuesta `T` segundos, el máximo sostenible es `1/T`:

```
Pi 5:   1 / 0,1038 s = 9,63 img/s
Pi 4:   1 / 0,7211 s = 1,39 img/s
```

**Aquí está la respuesta a Bruno.** La misión pedía 3 imágenes por segundo, lo que
exige terminar cada ciclo en menos de 1/3 = 333 ms. La Pi 4 tarda 721. **No es que
vaya lenta: es aritméticamente incapaz.** Los 0,78 que se midieron el 7 de octubre son
ese techo de 1,39 castigado por el frenado térmico.

Y la pantalla decía 4 FPS para las dos porque mostraba **la tasa pedida**, no la
entregada. Dos cantidades distintas con el mismo nombre en la misma casilla.

## 5.3 Por qué una se calienta y la otra no

El ciclo de trabajo es la fracción del tiempo que el procesador está ocupado. Pidiendo
0,5 imágenes por segundo, el período es 2000 ms:

```
Pi 5:   103,8 ms / 2000 ms =  5,2 %
Pi 4:   721,1 ms / 2000 ms = 36,1 %
```

Siete veces más trabajo por unidad de tiempo, y por eso siete veces más calor
disipado. No hace falta ninguna otra explicación: **una placa estabiliza en 47 °C y la
otra no estabiliza.**

### La curva térmica, y la trampa

A **una** imagen por segundo, ocho minutos de banco:

| minuto | Pi 4 | Pi 5 |
|---|---|---|
| 0,0 | 70,1 °C | 51,2 °C |
| 2,0 | 74,5 °C | 51,8 °C |
| 4,0 | 75,5 °C | 53,5 °C |
| 6,0 | 77,9 °C | 52,0 °C |
| 8,0 | **79,9 °C** | 52,0 °C |

La Pi 5 es plana. La Pi 4 sube aproximadamente (79,9 - 70,1) / 8 = 1,2 °C por
minuto **y no se aplana**. El límite blando de una Raspberry Pi 4 son 80 °C, donde el
firmware empieza a bajar la frecuencia. Lo cruza a los ocho minutos. **Un vuelo dura
quince o veinte.**

> **La trampa, que es la lección metodológica del día.** A los dos minutos la Pi 4
> marcaba 74,5 °C y entregaba el 100 % de lo que declaraba. Parecía aprobada. Una
> medición de treinta segundos habría dicho que funciona. **El rendimiento sostenido y
> el de ráfaga no son el mismo número**, y en un vuelo solo cuenta el primero.

### La prueba dura, del firmware

Al apagar las placas:

```
Pi 5:  temp=47,2 °C   throttled=0x0
Pi 4:  temp=75,0 °C   throttled=0xe0000
```

`0xe0000` son tres bits: `0x20000` frecuencia del ARM limitada, `0x40000` frenado
ocurrido, `0x80000` límite térmico blando alcanzado. Los tres son bits de "ocurrió
alguna vez", y están puestos. **Esto no es una inferencia desde la curva: la placa se
frenó y lo dice ella.** La Pi 5 cerró limpia.

## 5.4 Memoria

```
Pi 5:  3190 MB disponibles de 4050    Pi 4:  3132 MB disponibles de 3797
```

Se usa `MemAvailable`, no `MemFree`, que son preguntas distintas. `MemFree` cuenta solo
lo que nadie tiene tomado, y en un Linux con rato encendido siempre es poco porque el
núcleo se queda con la caché de páginas; una placa sana con dos gigas de caché
parecería a punto de morir. `MemAvailable` es la estimación del propio núcleo de lo que
una asignación nueva realmente conseguiría.

**Conclusión: la memoria no es un cuello de botella.** No hay intercambio. El eje 4
queda cerrado.

## 5.5 Precisión contra tasa: el resultado contraintuitivo

Vuelo del 2 de agosto, submuestreado. Siete personas con etiqueta humana como verdad de
campo.

| imágenes/s | cuadros usados | personas encontradas | falsos positivos |
|---|---|---|---|
| 1,64 | 1381 | 5 de 7 | 6 |
| 0,82 | 691 | 5 de 7 | 4 |
| 0,55 | 461 | 4 de 7 | 5 |
| 0,41 | 346 | **5 de 7** | **3** |

**Usando una cuarta parte de los cuadros, el sistema encuentra a las mismas cinco
personas y produce la mitad de los falsos positivos.**

El mecanismo es plausible y vale explicarlo: menos cuadros son menos oportunidades para
que una pista espuria acumule evidencia suficiente para madurar. Y como los umbrales de
madurez se escalan por la tasa declarada, veinte segundos de evidencia siguen siendo
veinte segundos reales de reloj, no menos.

### Una limitación del metodo que acota lo que esto afirma

El submuestreo se aplica **despues** de cargar las pistas, y esas pistas se calcularon
con el vuelo completo a 1,64 imagenes por segundo. En el codigo del replay son las
lineas 291 y 411: primero se cargan, despues se tiran cuadros.

**Entonces el seguidor nunca sufrio la tasa baja.** Y el seguidor es justamente lo que
mas sufre: a 0,41 imagenes por segundo pasan 2,4 segundos entre cuadros, la persona se
movio mucho mas, y el solapamiento entre cajas se desploma, igual que en el ejemplo de
la seccion 9.4 donde el IoU caia a 0,231 y la pista se partia en dos.

Lo que la tabla mide, dicho con precision:

> Si la capa de identidad recibe menos observaciones **de pistas calculadas a tasa
> completa**, sigue encontrando 5 de 7.

Lo que la tabla **no** mide:

> Si la camara hubiera capturado a 0,41 imagenes por segundo, el sistema encontraria
> 5 de 7.

**El resultado es optimista.** A tasa realmente baja el seguidor tambien se degradaria.
Para que la afirmacion valga hay que volver a correr el seguidor sobre los cuadros
submuestreados en vez de heredar las pistas; es una corrida mas y no necesita hardware.

### Que se puede decir mientras tanto

Que **la capa de acumulacion tolera menos observaciones sin perder personas**, y que los
falsos positivos bajan. Eso es verdadero y es un resultado util: dice que el cuello de
botella no esta en la maduracion. Si convierte el calor de la Pi 4 de bloqueante en
parametro depende de la corrida que falta.

## 5.6 Latencia contra error de posición: una cuenta que falta hacer

Un POI reportado con `T` segundos de retraso describe dónde estaba la persona hace T
segundos. Con una persona caminando a `v` metros por segundo, el error introducido es
$v \times T$.

A paso normal, v = 1,4 m/s:

```
Pi 5:   1,4 m/s * 0,104 s = 0,15 m
Pi 4:   1,4 m/s * 0,721 s = 1,01 m
```

**El error mediano de todo el sistema es 2,39 m.** La latencia de la Pi 4 aporta sola
un metro de eso, un 42 % del presupuesto de error, **sobre objetivos en movimiento**.
La de la Pi 5 aporta quince centímetros.

Esta cuenta no estaba hecha antes de hoy porque la latencia no se medía. Es el tipo de
cosa que no se descubre razonando.

---

# 6. Qué significa para volar

La decisión ya no es una opinión:

**La Pi 5 vuela a 3 imágenes por segundo.** Entrega 2,93–3,26, temperatura plana en
47–53 °C, sin slots perdidos. Su techo es 9,6, así que está trabajando al 31 % de su
capacidad. Hay margen de sobra.

**La Pi 4 no vuela a 3.** Es aritméticamente imposible. Y a 1 cruza el límite térmico a
los ocho minutos.

Tres salidas, en orden de lo que yo haría:

**(a) Un disipador o ventilador en la Pi 4.** El problema es térmico y la solución de un
problema térmico cuesta cinco dólares. Hay que probarlo antes de descartar la placa:
resolvería el eje térmico y dejaría la placa en su techo de cómputo de 1,39.

**(b) Volar la Pi 4 a 0,5 imágenes por segundo.** Ciclo de trabajo del 36 % contra el
72 % que tendría a 1, y la sección 5.5 dice que no cuesta detecciones. Hay que medir si
a esa tasa se estabiliza bajo 80.

**(c) Volar un solo dron.** El grupo pidió dos, pero un dron que cumple su contrato vale
más que dos donde uno miente sobre lo que entrega.

---

# 7. Qué no sabemos

Esta sección existe porque un informe sin ella no es creíble.

## 7.0 El seguidor no sufrió la tasa baja

Desarrollado en §5.5. El submuestreo se aplica después de cargar pistas calculadas a
tasa completa, así que la tabla de precisión mide la tolerancia de la capa de
acumulación, no el comportamiento del sistema a esa tasa de captura. Es optimista y
hay que rehacerlo antes de afirmar lo segundo.

## 7.1 La tabla de precisión es un vuelo y siete personas

Cuatro puntos, n = 7. El 4 de 7 a 0,55 img/s **es ruido**: con siete personas, la
diferencia entre cuatro y cinco es una sola persona. Sirve para decidir hoy; no para
afirmar que la relación vale en general. Para eso harían falta varios vuelos y más
personas.

## 7.2 El percentil 95 todavía no significa nada

A media imagen por segundo entra **una sola mirada** en cada intervalo de reporte, de
modo que el p95 y la mediana son literalmente la misma muestra. Los percentiles
recién tienen sentido a 3 o 4 imágenes por segundo, donde entran seis u ocho miradas
por intervalo. La instrumentación está lista; falta la corrida.

## 7.3 La energía no está medida

Falta el eje 3 completo. Se puede aproximar con la caída de tensión de la batería a lo
largo de una corrida larga, que ya se está registrando en el CSV, pero una
aproximación por tensión es mala: la curva de descarga de una LiPo es plana en el medio
de su rango, que es justo donde se vuela. Lo correcto es un medidor de corriente en
serie.

## 7.4 La Pi 4 a 0,5 img/s no se midió térmicamente

Se midió a 1. El barrido completo de cinco tasas se cortó por tiempo con veinte
muestras. Es la corrida que falta para cerrar la tabla.

## 7.5 La SIYI no está calibrada

La cámara SIYI A8 mini quedó funcionando por RTSP el 8 de octubre, pero sus parámetros
intrínsecos son **estimados de las especificaciones** (focal 1130 px calculada como
4,47 * 1920 / 7,6), no medidos con tablero. La ArduCam sí está calibrada, con
4,25 px de error de reproyección. Esa diferencia se propaga directamente al error en
metros, así que para geolocalizar hay que usar la ArduCam hasta que la SIYI se calibre.

---

# 8. Guión para Bruno

En portugués, que es como vas a hablarle. Pregunta por pregunta.

## *"Por que o Raspberry 4 e o 5 davam a mesma coisa? Não faz sentido."*

> **Você estava certo, não fazia sentido.** A tela mostrava a taxa **pedida**, não a
> entregue. Eram dois números diferentes com o mesmo nome.
>
> Medimos o tempo real por imagem: o Pi 5 leva 104 ms, o Pi 4 leva 721 ms. Quase sete
> vezes mais. Três imagens por segundo exigem fechar cada ciclo em menos de 333 ms, e o
> Pi 4 leva 721. **Não é que ele seja lento: é aritmeticamente incapaz.** O teto dele
> é 1,39 imagens por segundo.
>
> E agora a tela mostra as duas taxas lado a lado, então isso não pode voltar a
> acontecer sem que apareça.

## *"Vamos ver o quanto isso pesa em processamento."*

> O detector é praticamente o ciclo inteiro nas duas placas. A projeção do raio, o
> rastreador e a camada de identidade **juntos não somam um milissegundo mensurável**.
>
> Em termos de carga: pedindo meia imagem por segundo, o Pi 4 trabalha 36 % do tempo e
> o Pi 5 trabalha 5 %. É por isso que um esquenta e o outro não.
>
> E o firmware do próprio Pi 4 confirma: ao desligar reportou `throttled=0xe0000`, que
> são três bits — limite térmico atingido, freada ocorrida, frequência do ARM limitada.
> Não é dedução da curva, é a placa dizendo.

## *"O vetor de embedding muda muito dependendo da direção?"*

> **Muda, e você tinha razão em desconfiar.** Medido sobre vetores médios em janelas de
> quatro segundos: com o limiar que usamos, **35 % dos pares de pessoas DIFERENTES
> passam no filtro de aparência**.
>
> Ou seja, a aparência sozinha não distingue pessoas. O que sustenta o sistema é que **a
> distância filtra primeiro**, e a aparência decide entre os que já estão perto. Aquilo
> que eu disse na reunião, *"é os dois"*, estava certo — agora tem número.
>
> O que isso proíbe é uma frase: não dá para dizer "a aparência distingue pessoas".

## *"Vamos trabalhar só com dois drones."*

> Combinado. E aí está o problema: **o Pi 4 é o que limita**. Não voa a 3 imagens por
> segundo, e a 1 ele cruza os 80 °C em oito minutos.
>
> Mas tem uma coisa boa, e é contraintuitiva: medimos quanto custa baixar a taxa, e
> **não custa detecções**. Com um quarto das imagens o sistema acha as mesmas cinco
> pessoas de sete e produz **metade dos falsos positivos**.
>
> Então o calor deixa de ser um impedimento e vira um parâmetro. Minha proposta é um
> dissipador no Pi 4, que custa cinco dólares, e voar os dois.

## Si pregunta qué falta

> E tem uma coisa que eu nao vou apresentar ainda: medi quanto custa baixar a taxa,
> mas o rastreador no sofreu a taxa baixa na minha medicao, entao o numero esta
> otimista. Preciso refazer antes de afirmar.
>
> Falta a energia por imagem, que é o que traduz taxa em minutos de autonomia, e o
> percentil 95 da latência — hoje a instrumentação está pronta mas a meia imagem por
> segundo entra uma amostra só por relatório, então o p95 é a mediana. Preciso rodar a
> 3 e 4 para isso ter sentido.

---

---

# 9. Cómo explicar el sistema, con las cuentas hechas

Esta sección existe porque la pregunta *"¿cómo funciona tu sistema?"* se contesta mal
casi siempre, y la razón es una sola: **se empieza por los componentes.** Decir "uso
YOLO, BoT-SORT y OSNet" no enseña nada del sistema, enseña tres nombres. Y si el otro
pregunta algo, hay que improvisar.

El orden correcto es: **el problema físico, el truco que lo resuelve, la cadena, el
número.** Las bibliotecas van al final y casi nunca hacen falta.

Todo lo que sigue usa los parámetros reales del sistema.

## 9.1 Por qué una cámara sola no mide distancia

Es el punto de partida y el que hace entender todo lo demás.

Un sensor de imagen registra **de qué dirección vino la luz**, no desde qué distancia.
Dos personas, una de 1,80 m a 40 metros y otra de 0,90 m a 20 metros, ocupan
exactamente los mismos píxeles. La imagen es idéntica. **La información de profundidad
se perdió en la proyección y no hay procesamiento que la recupere de un solo cuadro.**

Por eso el término correcto para este tipo de sistema es **geolocalización
*bearing-only***: solo tenés el rumbo hacia el objetivo, no la distancia.

De ahí salen las tres maneras de recuperar la distancia:

| método | cómo | costo |
|---|---|---|
| estéreo | dos cámaras separadas, por paralaje | dos sensores, calibración entre ambos, peso |
| lidar | mide el tiempo de vuelo de un pulso | caro, pesado, consume |
| **geometría** | **cortar el rayo con un plano conocido** | **gratis, pero exige conocer la pose** |

El sistema usa la tercera. Esa es la decisión de diseño, y el precio que paga es que
**todo el error de pose se convierte en error de posición**, que es de lo que trata
la sección 9.6.

## 9.2 De un píxel a un punto en el suelo: la cuenta completa

Partimos de datos reales del sistema:

```
camara ArduCam Module 3, calibrada con tablero
  focal           f  = 1407,0 px
  centro optico   cx = 945,7 px ,  cy = 547,1 px
  imagen          1920 x 1080

aeronave
  posicion        (0, 0, 35)      metros, z hacia arriba
  rumbo           0 grados        apuntando al norte
  cabeceo montaje -55 grados      la camara mira 55 grados hacia abajo

deteccion
  pixel           (1100, 700)
```

### Paso 1. Del píxel a una dirección en el sistema de la cámara

Se resta el centro óptico y se pone la focal como tercera componente:

```
u - cx = 1100 - 945,7 = 154,3
v - cy =  700 - 547,1 = 152,9
f      = 1407,0
```

La norma de ese vector:

```
|d| = raiz( 154,3^2 + 152,9^2 + 1407,0^2 )
    = raiz( 23 808,5 + 23 378,4 + 1 979 649,0 )
    = raiz( 2 026 835,9 )
    = 1423,67
```

Y normalizado:

```
d_cam = ( 0,10838 ,  0,10740 ,  0,98828 )
```

**Qué significa cada número.** La tercera componente, 0,98828, es casi 1: el objetivo
está casi sobre el eje óptico. Las otras dos son las desviaciones. En ángulos:

```
horizontal:  arctan(154,3 / 1407,0) = 6,26 grados a la derecha del eje
vertical:    arctan(152,9 / 1407,0) = 6,20 grados por debajo del eje
```

Como la cámara ya mira 55° hacia abajo, el rayo baja en total unos 61°.

### Paso 2. De la cámara al mundo

El sistema de la cámara tiene `x` a la derecha, `y` hacia abajo en la imagen, y `z`
hacia adelante sobre el eje óptico. Con rumbo 0 y cabeceo de montaje −55°, esos tres
ejes expresados en coordenadas del mundo (x=este, y=norte, z=arriba) son:

```
z_c  (eje optico, 55 grados abajo) = ( 0 ,  0,5736 , -0,8192 )
x_c  (derecha)                     = ( 1 ,  0      ,  0      )
y_c  (abajo en la imagen)          = ( 0 , -0,8192 , -0,5736 )
```

La dirección en el mundo es la combinación lineal:

```
d_mundo = 0,10838 * x_c  +  0,10740 * y_c  +  0,98828 * z_c

  este  :  0,10838
  norte :  0,10740*(-0,8192) + 0,98828*(0,5736) = -0,08798 + 0,56688 =  0,47890
  arriba:  0,10740*(-0,5736) + 0,98828*(-0,8192) = -0,06160 - 0,80960 = -0,87120

d_mundo = ( 0,10838 , 0,47890 , -0,87120 )
```

Comprobación de que sigue siendo unitario:
`0,011746 + 0,229345 + 0,758990 = 1,00008`. Correcto.

### Paso 3. Del rayo al suelo

La aeronave está en `(0, 0, 35)` y el suelo en `z = 0`:

```
k = (0 - 35) / (-0,87120) = 40,175

p = (0, 0, 35) + 40,175 * (0,10838 , 0,47890 , -0,87120)
  = ( 4,354 , 19,240 , 0,0 )
```

**La persona está 4,35 m al este y 19,24 m al norte del punto del suelo bajo la
aeronave.**

Y como `d_mundo` es unitario, **`k` es directamente el rango inclinado: 40,175 m.** La
distancia horizontal es `raiz(4,354² + 19,240²) = 19,73 m`.

Esa separación entre **19,73 m horizontales** y **40,18 m de rango inclinado** no es un
detalle: es la que gobierna el error, y se usa en 9.6.

## 9.3 Qué hace el seguidor, con números

Esto es lo que no te quedó claro, así que va despacio.

**El problema.** El detector no tiene memoria. En el cuadro 100 encuentra tres cajas.
En el 101 encuentra tres cajas. **No sabe cuál de las tres del 101 es cuál de las tres
del 100.** Para el detector son seis cajas independientes.

Sin resolver eso no se puede acumular evidencia: cada detección sería una persona
nueva, y nunca habría dos miradas de la misma.

**La solución: asociar por solapamiento.** El seguidor compara cada caja nueva con cada
caja anterior y mide cuánto se superponen, con la **intersección sobre unión**:

```
IoU = area de la interseccion / area de la union
```

Con números reales. Una persona a 40 metros ocupa una caja de unos 40 x 90 px:

```
cuadro 100   caja A: centro (1100, 700), 40 x 90
             o sea  x de 1080,0 a 1120,0   y de 655,0 a 745,0

cuadro 101   caja B: centro (1108, 704), 41 x 92
             o sea  x de 1087,5 a 1128,5   y de 658,0 a 750,0
```

La intersección:

```
en x:  de 1087,5 a 1120,0  ->  32,5 px
en y:  de  658,0 a  745,0  ->  87,0 px
area de la interseccion = 32,5 * 87,0 = 2827,5 px2
```

La unión:

```
area A = 40 * 90 = 3600 px2
area B = 41 * 92 = 3772 px2
union  = 3600 + 3772 - 2827,5 = 4544,5 px2
```

Y el resultado:

```
IoU = 2827,5 / 4544,5 = 0,622
```

Con un umbral típico de 0,3, **0,622 pasa**: el seguidor decide que son la misma
persona y le pone a la caja B el mismo identificador que tenía A, por ejemplo `7`.

**Eso es "poner un identificador".** Es una decisión de asociación, no una propiedad de
la imagen. A partir de ahí, la capa de identidad puede decir "la pista 7 lleva 40
observaciones" en vez de "hay 40 detecciones sueltas".

## 9.4 Por qué hace falta compensación de movimiento, con números

Acá está la razón de que un seguidor común no sirva en un dron.

Una persona **quieta** en el suelo, vista desde una aeronave que **se mueve**, cambia de
lugar en la imagen. No porque ella se mueva, sino porque la cámara sí.

A 40 m de rango, un desplazamiento lateral de la aeronave de 0,7 m entre dos cuadros
desplaza la proyección aproximadamente:

```
desplazamiento en pixeles = f * (0,7 / 40) = 1407 * 0,0175 = 24,6 px
```

Recalculemos el IoU con la misma caja de 40 x 90 corrida 25 px, sin que la persona se
haya movido:

```
interseccion en x:  40 - 25 = 15 px
interseccion en y:  90 px (no cambia)
area de la interseccion = 15 * 90 = 1350 px2

union = 3600 + 3600 - 1350 = 5850 px2

IoU = 1350 / 5850 = 0,231
```

**0,231 está por debajo del umbral de 0,3.** El seguidor concluye que la pista 7
desapareció y que hay una persona nueva, que recibe el identificador 8.

Resultado: la misma persona quieta se parte en dos pistas, ninguna acumula evidencia
suficiente, y el sistema no reporta a nadie. O peor: reporta dos.

**Eso es lo que corrige la compensación de movimiento de cámara.** Estima el
desplazamiento global de la imagen entre cuadros, lo resta, y recién entonces compara.
Con la corrección aplicada, el IoU vuelve a ser el 0,622 del ejemplo anterior.

> Esto no es teórico. Medido en este proyecto el 5 de octubre: al corregir el manejo de
> la compensación, el seguidor `ocsort` pasó de **6 detecciones con identificador a 487
> de 764**. La tabla preliminar de `NOTES.md` no estaba midiendo trackers, estaba
> midiendo un error de configuración.

## 9.5 El vector de apariencia, y qué significa su distancia

El seguidor funciona **entre cuadros consecutivos**. Si la persona se oculta cinco
segundos detrás de un árbol, el solapamiento es cero y la pista se corta. Para volver a
reconocerla hace falta otra cosa.

**El modelo de apariencia** toma el recorte de la caja y produce un vector de 512
números, normalizado a longitud 1. La idea es que dos recortes de la misma persona
den vectores que apunten casi en la misma dirección.

La comparación es la **distancia coseno**:

```
distancia = 1 - coseno(angulo entre los dos vectores)
```

Qué valores toma:

```
distancia = 0,00   los dos vectores apuntan exactamente igual
distancia = 0,20   muy parecidos
distancia = 1,00   perpendiculares, sin relacion
distancia = 2,00   opuestos
```

Los umbrales reales del sistema:

```
EMB_DIST_MAX_MEDIDO = 0,95    para asociar dentro de la fusion
EMB_DIST_REUNE      = 0,63    mas estricto, para reunir una pista cortada
```

### El dato incómodo que contesta a Bruno

Bruno preguntó si ese vector aguanta el cambio de punto de vista. La medición sobre
vectores promediados en ventanas de cuatro segundos, en este sistema, dice:

> **El 35 % de los pares de personas DISTINTAS da una distancia por debajo del umbral.**

O sea: **si solo se mirara la apariencia, una de cada tres parejas de personas
diferentes se fusionaría en una sola.** La apariencia por sí sola no distingue
personas.

Lo que sostiene el sistema es el orden: **la distancia filtra primero.** Dos
detecciones separadas 20 metros no pueden ser la misma persona, sin importar cuánto se
parezcan sus vectores. La apariencia decide solo entre las candidatas que la geometría
ya dejó cerca.

Y hay una razón mecánica para que el vector sufra: a 40 metros de rango una persona
ocupa unos **40 x 90 píxeles**. El modelo de apariencia fue entrenado con recortes de
vigilancia bastante más grandes. Con noventa píxeles de alto, lo que queda es color de
ropa y proporción, no textura ni rasgos.

## 9.6 Cómo se propaga el error, y de dónde sale el radio de fusión

Esta es la parte que convierte el sistema en ingeniería y no en una demostración.

Como la distancia se obtiene de la geometría, **todo error en la pose de la aeronave se
convierte en error de posición en el suelo**. Hay tres fuentes y no pesan igual.

### Error de posición de la aeronave

Se traslada uno a uno. Si el GPS está 1,5 m corrido, el impacto está 1,5 m corrido. No
depende de a dónde mire la cámara.

```
GPS_SIGMA_M = 1,5 m
```

### Error de rumbo

Gira el rayo alrededor del eje vertical. El impacto se mueve sobre un arco de radio
igual a la **distancia horizontal**:

```
desplazamiento = distancia_horizontal * angulo_en_radianes
               = 19,73 m * 0,017453 rad        (1 grado)
               = 0,344 m
```

### Error de cabeceo

Este es el que más pesa, y no es obvio. La distancia horizontal depende del ángulo de
depresión según `R = h / tan(theta)`. Derivando:

```
dR/dtheta = -h / sin^2(theta)
          = -35 / (0,8763)^2
          = -45,58 m por radian

para 1 grado:  45,58 * 0,017453 = 0,795 m
```

**Un grado de cabeceo cuesta más del doble que un grado de rumbo** en esta geometría.
La razón es que un error de cabeceo empuja el impacto **radialmente**, alejándolo o
acercándolo, mientras que el de rumbo lo mueve lateralmente sobre un arco.

### El radio de fusión

La fórmula del código es:

```
fusion_radius_m = gps_sigma + slant_range * yaw_sigma
```

Con las constantes reales del sistema:

```
bias_sigma_m = 2,4 / 1,1774 = 2,0384 m     (el error medido, convertido a sigma)
GPS_SIGMA_M  = 1,5 m
RANGO_REFERENCIA_M = 20,3 m

yaw_sigma = raiz( 2,0384^2 - 1,5^2 ) / 20,3
          = raiz( 4,1551 - 2,2500 ) / 20,3
          = raiz( 1,9051 ) / 20,3
          = 1,3803 / 20,3
          = 0,06799 rad   =  3,90 grados
```

Y para la detección del ejemplo, con rango inclinado 40,175 m:

```
fusion_radius = 1,5 + 40,175 * 0,06799
              = 1,5 + 2,731
              = 4,23 m
```

**Dos personas separadas menos de 4,23 metros en esa geometría no se pueden distinguir
con confianza, y el sistema las trata como candidatas a ser la misma.**

Y fijate lo que hace el segundo término: a 20 m de rango el radio sería
`1,5 + 1,36 = 2,86 m`, y a 80 m sería `1,5 + 5,44 = 6,94 m`. **El sistema es más
tolerante cuando mira lejos, porque sabe que ahí es menos preciso.** Eso es lo que una
constante fija no puede hacer.

> **Una observación honesta sobre la fórmula.** Usa el **rango inclinado** (40,18 m)
> donde la geometría del error de rumbo pediría la **distancia horizontal** (19,73 m),
> lo que la hace conservadora por un factor de 2,04 en este caso. Pero como el error de
> cabeceo contribuye 0,795 m contra los 0,344 m del rumbo, usar el rango inclinado
> funciona como una aproximación agrupada de las dos fuentes. Vale saberlo: no es un
> error, es una simplificación que conviene poder explicar si alguien la señala.

## 9.7 Cuándo una pista está madura

Una detección no es una persona. La capa de identidad acumula y decide. Los umbrales
reales:

```
MIN_MEASUREMENTS   = 8      minimo de impactos antes de estimar una posicion
DUTY_MIN           = 0,10   fraccion minima de cuadros en que hay que verla
report_dur_s       = 36     segundos de evidencia antes de reportar (modo span)
report_min_looks   = 20     miradas independientes (modo looks)
```

El **ciclo de trabajo** es la parte sutil. Con `report_dur_s = 36` a 3 imágenes por
segundo, el lapso son `36 * 3 = 108` cuadros. Pero no alcanza con que la pista exista
108 cuadros: hay que **haberla visto** en al menos el 10 %:

```
n_reporte = max(5, redondeo(0,10 * 108)) = 11 observaciones
```

**Por qué existe ese piso.** Una pista detectada en 3 de los 108 cuadros que abarca no
está siendo seguida: está siendo **redescubierta**. El lapso solo la haría parecer
madura. El ciclo de trabajo separa seguimiento de coincidencia.

Y por eso los umbrales se escalan por la tasa declarada: a 1 imagen por segundo el
mismo `report_dur_s = 36` son 36 cuadros, no 108. **Treinta y seis segundos de
evidencia siguen siendo treinta y seis segundos reales.** Esa es la razón de que bajar
la tasa no destruya la precisión, como mide la sección 5.5.

## 9.8 El guión, por capas

La respuesta se da en capas y se para en cuanto el otro tiene suficiente.

### Una frase

> "Un dron con una sola cámara encuentra personas en el suelo y dice dónde están, en
> metros."

### Treinta segundos: el mecanismo

> "Una cámara sola no mide distancia: un píxel te da una **dirección**, no un punto.
>
> Lo que hace el sistema es **cortar esa dirección con el suelo**. Si sé dónde está el
> dron, a qué altura y hacia dónde mira, el corte entre el rayo y el plano del suelo es
> una posición concreta. Saco profundidad de la geometría en vez de un sensor.
>
> Y como cada corte tiene error, **acumulo en el tiempo**: muchas miradas de la misma
> persona desde posiciones distintas, y recién con evidencia suficiente digo que ahí
> hay alguien."

### Dos minutos: la cadena y el número

> "La cámara saca una imagen, un detector marca las cajas, un seguidor asocia las cajas
> entre cuadros para que la misma persona sea una sola pista, y un modelo de apariencia
> saca un vector del recorte para reconocerla si el seguidor la pierde.
>
> Con la caja y la pose calculo el rayo, lo corto con el suelo, y ese impacto se acumula
> por pista. Una capa de identidad decide cuándo hay evidencia suficiente, si el
> objetivo está quieto o se mueve, y si dos pistas son la misma persona. Lo que madura
> se difunde.
>
> **El error mediano sobre un vuelo real es 2,39 metros.** Todo corre a bordo, en la
> Raspberry: a la estación le llegan las coordenadas, no el video."

### Si preguntan más

| pregunta | respuesta corta | el número |
|---|---|---|
| ¿Cómo sabés que es la misma persona? | distancia primero, apariencia después | 35 % de los pares distintos pasan el filtro de apariencia |
| ¿Cuánto error tiene? | depende del rango, no de la altura | 2,39 m de mediana; 1° de cabeceo cuesta 0,80 m a 40 m |
| ¿Por qué no estéreo o lidar? | peso y consumo en un dron | una cámara, profundidad por geometría |
| ¿Cuántas personas encuentra? | con verdad de campo | 5 de 7, con 6 falsos positivos |
| ¿Corre a bordo o en tierra? | a bordo, entero | 104 ms por imagen en la Pi 5 |

### Una palabra que conviene no usar mal

Lo que hace el sistema **no es triangulación**. Triangular es cruzar dos rayos entre sí
para hallar su punto de encuentro. Acá se corta **un** rayo contra **el suelo**, que es
un plano conocido. Son cosas distintas, y notar la diferencia deja mejor parado que
usar la palabra grande.

En la propia reunión del 2 de octubre alguien se corrigió solo:

> *"É, eu tô chamando de triangularização, mas é, desculpa, **não é uma
> triangularização, de jeito nenhum**."*

Tenía razón. Y vale la pena agregar que el cruce de rayos **sí existe** en el
repositorio, como estimador alternativo conmutable: `fusion.py`, cableado el 8 de
octubre. Sirve cuando el objetivo no está sobre el plano declarado, y falla cuando los
rayos son casi paralelos. Medido: con un blanco 5 m por encima del plano, el plano se
equivoca 3,30 m y el cruce de rayos 0,00; con la aeronave casi quieta, el plano da
2,86 m y el cruce 16,60.

---

# Anexo A. El núcleo del código

Cuatro piezas. No cuarenta.

## A.1 Del píxel al rayo

```python
def pixel_to_ray(pos, yaw_deg, pixel, pitch_deg, f_px, w, h, principal,
                 body_pitch_deg=0.0, body_roll_deg=0.0):
    """Un píxel y una pose dan una semirrecta en el mundo."""
    cx, cy = principal
    u, v = pixel
    # Dirección en el sistema de la cámara. La focal va en z porque el plano
    # imagen está a f píxeles del centro óptico.
    d_cam = np.array([u - cx, v - cy, f_px], dtype=float)
    d_cam /= np.linalg.norm(d_cam)
    R = _world_to_camera_rotation(yaw_deg, pitch_deg, body_pitch_deg, body_roll_deg)
    return np.asarray(pos, dtype=float), R.T @ d_cam
```

**Por qué así.** La focal entra como tercera componente, no como escala: el plano imagen
está a `f` píxeles del centro óptico, y esa es la definición del modelo pinhole. La
rotación se traspone porque `R` lleva del mundo a la cámara y acá se va en la dirección
contraria.

## A.2 Del rayo al suelo

```python
def _ground_impact(origin, direction, ground_z):
    """Corta la semirrecta con el plano horizontal. None si no lo alcanza."""
    if direction[2] >= 0:          # apunta al horizonte o arriba
        return None
    k = (ground_z - origin[2]) / direction[2]
    if k <= 0:                     # el plano queda detrás de la cámara
        return None
    return origin + k * direction
```

**Por qué las dos guardas.** La primera descarta rayos que nunca cortan el plano. La
segunda descarta el corte que existe matemáticamente pero está **detrás** de la cámara:
una semirrecta no es una recta, y olvidarlo pone objetivos a espaldas del dron.

## A.3 El radio de fusión

```python
# fusion_radius_m = gps_sigma + slant_range * yaw_sigma
#
# Dos fuentes de error que no se comportan igual. La del GPS no depende de a
# dónde mire la aeronave. La de rumbo se multiplica por la distancia: un grado
# cuesta 0,75 m a 42 m de rango inclinado, y 1,7 m a 100 m.
self.yaw_sigma_rad = math.sqrt(max(0.0, bias_sigma_m**2 - gps_sigma_m**2)) / RANGO_REFERENCIA_M
```

**Por qué una fórmula y no un número.** Todos los términos están en la telemetría, así
que el radio se recalcula por detección en vez de ajustarse a mano para un vuelo y
quedar mal en el siguiente.

## A.4 La instrumentación de latencia

```python
_t0_ciclo = time.perf_counter()
_t0_det = time.perf_counter()
detecciones = self.camera.detect(self._position, yaw)
self._lat_detector.append((time.perf_counter() - _t0_det) * 1000.0)
...
self._lat_ciclo.append((time.perf_counter() - _t0_ciclo) * 1000.0)

def _latencias(self):
    """Lo que costó el lazo en el ÚLTIMO intervalo, y después lo olvida."""
    d = {"ciclo_p50": self._percentil(self._lat_ciclo, 0.50),
         "ciclo_p95": self._percentil(self._lat_ciclo, 0.95),
         "detector_p50": self._percentil(self._lat_detector, 0.50),
         "n": len(self._lat_ciclo)}
    self._lat_ciclo = []
    self._lat_detector = []
    return d
```

**Tres decisiones en diez líneas.** `perf_counter` y no el reloj del proveedor, porque
en una reproducción ese reloj informaría lo que un cuadro *debía* tardar. Dos
cronómetros anidados, porque el total no dice dónde está el costo. Y el vaciado al
publicar, porque un percentil acumulado queda dominado por el arranque.

---

# Anexo B. Cómo reproducir todo

```bash
# El barrido de tasas, sobre hardware
python scripts/medir/barrido_fps.py --estacion <ip:puerto> \
    --placas pi@<pi5> pi@<pi4> --tasas 0.5 1 2 3 4 --minutos 10

# La precisión por tasa, sin hardware
python scripts/replay_vuelo3.py --pistas=demo/data/pistas_bot_cmc_sof.npz \
    --preliminares --submuestreo=N --candidatos=salida.json
python scripts/personas_encontradas.py --candidatos salida.json
```

**Dos avisos que cuestan una tarde si no se saben.** Las pistas externas y
`--preliminares` no son opcionales: con el seguidor interno del replay la línea base da
1 de 7 en vez de 5 de 7, y entonces nada de lo que sigue es comparable. Y diez minutos
por punto no es prudencia: a los dos minutos la Pi 4 miente.

Datos crudos, registros de las dos placas y de la estación: `docs/medidas_9oct/`.
