# Notas de decisiones y mediciones

Bitácora del proyecto: de dónde sale cada número del código y por qué se
tomó cada decisión. El código no repite esta historia — los docstrings
dicen QUÉ hace cada cosa; este archivo dice POR QUÉ y CUÁNDO.

## Calibración de cámara

- `focal_px = 1407.0` y `principal_point = (945.7, 547.1)` salen de los
  `run_info.json` grabados por `onboard.py` en los tres vuelos reales
  (26-jul, 01-ago y 02-ago 2026). Calibración formal con tablero de
  ajedrez sigue pendiente.
- **02-ago-2026: cambió el montaje.** La ArduCam se giró 180° sobre el
  eje óptico y el pitch pasó de −45° a −55°. El ISP endereza la imagen
  en captura (hflip+vflip), lo que refleja el punto principal:
  `(945.7, 547.1) → (973.3, 531.9)` con la fórmula `tamaño − 1 − c`.
  Ese es el motivo de `CameraConfig.rotated_180()` y de que
  `SimulatedCamera.pitch_deg` no tenga valor por defecto: un default
  escondido con el pitch viejo produjo 6.287 m de error en su momento.

| Vuelo | focal_px | principal_point | pitch |
|---|---|---|---|
| 26-jul | 1407.0 | 945.7, 547.1 | −45° |
| 01-ago | 1407.0 | 945.7, 547.1 | −45° |
| 02-ago | 1407.0 | 973.3, 531.9 (reflejado) | −55° |

## Modelos de ruido de píxel

- `heuristic`: sigma = 5.0 / conf (el del paper).
- `visdrone_1d`: sigma = 1.13 / conf, ajuste empírico sobre
  VisDrone2019-DET-val + YOLOv8s. La ablación del paper mostró que el
  ranking de métodos no depende de cuál se use.

## Huellas de apariencia (OSNet)

- Modelo: `osnet_x0_25_msmt17` vía boxmot, CPU, un batch por imagen,
  vectores normalizados.
- Separación medida sobre el vuelo 02-ago (coseno): misma identity
  0.63–0.87; identidades distintas 0.33–0.46. No se solapan.
  - Umbral de fusión `emb_dist_max = 0.95` = punto medio del hueco
    (coseno 0.545) convertido a distancia L2.
  - Umbral del gemelo `EMB_DIST_GEMELO = 0.70` = zona profunda de
    misma-identity (misma ≤ 0.86, distintas ≥ 1.04 en distancia).
- 24-ago-2026: verificado que `OnboardCamera._fingerprints` reproduce
  exactamente las huellas del análisis offline (coseno 1.000000).
- Decisión (23-ago-2026): la huella se calcula EN la cámara; la imagen
  no sale del módulo. Por radio viajan 512 float32 (2048 B) o 128 B con
  PCA int8 (validado offline en la tesis de handoff). Alternativa
  futura: crop bajo demanda (2–5 KB una vez) cuando la Ground
  Station quiera verificar un POI con un detector pesado.

## Umbrales de la identity

- Reglas y valores base validados offline sobre el vuelo 02-ago
  (`drone-geolocation/entrenamiento/correr_botsort.py`): radio de
  fusión 3.5 m, mínimos de 6/20/25 observaciones a la cadencia 0.7 FPS
  de ese análisis. En el módulo se expresan como duraciones
  (8.6 / 29 / 36 s) × tasa declarada, que reproducen esos conteos.
- **Lección C-5 (24-ago-2026, vuelos 1-2):** la tasa que importa es la
  de OBSERVACIONES reales, no la de cámara. El vuelo 1 capturaba a
  8.7 FPS pero la persona se detectaba en ~30% de los frames
  (2.45 obs/s); con fps de cámara los umbrales exigían evidencia
  imposible y salían 0 candidates.
- Posición actual de un MÓVIL: la mediana de la ventana reciente
  retrasa al caminante media ventana (medido: 7.40 m de retraso a
  1 m/s); el ajuste lineal evaluado en la última muestra lo reduce a
  0.48 m. De ahí `_current_position`.
- Umbral MÓVIL 4.0 m ≈ 1.15 × radio de fusión: una pista quieta solo
  "tiembla" por ruido de proyección; más que eso, caminó.
- `COOCURRENCIA_MIN = 3`: dos pistas vistas juntas en esa cantidad de
  cuadros son dos cosas distintas, diga lo que diga la posición y la
  apariencia. Nada aparece dos veces en la misma foto.
- `DUTY_MIN = 0.10`: fracción mínima de los cuadros que abarca una pista
  en los que tiene que haber sido detectada de verdad. Por debajo, la
  pista es un puñado de avistajes estirados y su extensión en cuadros
  exagera la evidencia que hay detrás. **Ojo:** este número se probó como
  regla de madurez y se descartó, ver `docs/DESCARTADO.md`.
- `POS_FRAC_GEMELO = 0.4`, junto con `EMB_DIST_GEMELO`: la excepción al
  veto de co-ocurrencia. Los detectores a veces emiten cajas duplicadas
  para una persona y el seguidor las vuelve dos pistas co-ocurrentes. El
  veto se levanta solo cuando la evidencia dice "misma persona": posición
  casi idéntica Y distancia de apariencia bien dentro de la zona medida de
  misma identity.
- `RADIO_95_2D = 2.4477`: radio del círculo que contiene el 95% de una
  gaussiana isótropa en dos dimensiones, en sigmas, = `sqrt(-2 ln 0.05)`.
  Derivado, no ajustado. Convierte una incertidumbre por eje en algo que
  un operador puede dibujar en un mapa y caminar.
- `RANGO_REFERENCIA_M = 20.3`: el alcance oblicuo al que se midió el error
  mediano de 2.4 m de la cadena, o sea la distancia mediana cámara-objetivo
  de las observaciones del operador en el vuelo 3 (p10 10.1 m, p90 53.0 m,
  a una altura mediana de 17.1 m). Un margen de posición citado sin el
  alcance al que vale está incompleto: un error de rumbo mueve el impacto
  por alcance × ángulo.
- `GPS_SIGMA_M = 1.5`: desviación estándar horizontal supuesta para un
  receptor de consumo, por eje. **No medida acá:** con un punto de
  calibración y dos fuentes de error, una hay que suponerla, y esta es la
  mejor conocida de las dos. El error de rumbo sale después del piso
  medido.

### Madurez: `span` mide la cosa equivocada

`span` es el tiempo entre el primer y el último avistaje, y es la regla que
usó la cadena hasta que se midió contra un vuelo. Se conserva para que
esos números sigan siendo reproducibles, pero mide mal. En el vuelo 3:

- un objeto estático que el detector confunde con una persona se avista
  **86 veces en 758 s y madura**;
- una persona vista **continuamente 30 s** en un barrido **nunca** madura;
- el operador, ubicado a menos de 3 m de la verdad **a los 2 s**, se
  reporta recién **a los 76 s**.

`looks` cuenta avistajes independientes: los intervalos distintos de
`look_s` en los que la cosa se detectó. Cuadros consecutivos de un segundo
son UN avistaje, porque comparten el mismo error de pose y el mismo fondo;
un segundo avistaje un minuto después es otro.

**Ninguno de los dos modos separa un objetivo real de un falso positivo
persistente.** Medido: las señales propias de la cadena no lo hacen (tasa
de detección en vista, confianza, tamaño aparente, y un clasificador de
apariencia que no transfiere entre vuelos). Esa decisión es de quien mira
el recorte.

- `track_min_looks = 3`, `report_min_looks = 20`: **decisiones, no
  mediciones.** Codifican cuánto cuesta un reporte falso contra uno tarde.
  El 20 es el valor que, en el vuelo 3 con el seguidor del vuelo, confirmó
  los mismos objetivos que la regla `span` en **22 s en vez de 52**, y
  cumplió los requisitos provisionales de misión
  (`docs/requisitos_mision.json`). Con 5 confirma en 6 s pero **también
  confirmó a un no-persona**.
- `fps` es hoy un FALLBACK y la historia vale conservarla, porque este
  parámetro estuvo mal de **tres maneras.** Primero significaba la tasa de
  detección, que nadie puede saber de antemano porque depende de lo
  intermitente que sea la escena. El 25-ago-2026 se redefinió como la tasa
  de CUADROS, con el argumento de que quien pone el temporizador la conoce
  exacta. Medido esa misma noche, también era falso: el lazo se reagendaba
  como `now + period` y entregaba **2.31 cuadros por segundo contra 3.00
  configurados**, y todo umbral derivado de la tasa declarada se estiraba un
  tercio. El lazo y los llamadores están arreglados, pero un número que
  estuvo mal tres veces es un número del que hay que dejar de depender. Un
  reloj no se puede desconfigurar.

### El margen que más avistajes no pueden promediar (`bias_sigma_m`)

Desviación estándar por eje del error que más avistajes **no** promedian:
GPS y rumbo, compartido por cada avistaje de un vuelo. El default sale del
error mediano medido de la cadena en vuelos reales, **2.4 m**: para una
gaussiana en dos dimensiones el error radial mediano es 1.1774 sigma. Vale
al alcance al que se midió, `RANGO_REFERENCIA_M`. Cuando las observaciones
traen `range_m` el error compartido se modela como
`sqrt(gps_sigma_m² + (range × yaw_sigma)²)`, con el error de rumbo resuelto
para que el modelo devuelva `bias_sigma_m` a ese alcance.

| alcance | radio del 95% |
|---|---|
| 12 m | ~4.2 m |
| 20 m | 5.0 m |
| 90 m | ~15 m |

### Movimiento: la ventana, la velocidad y la extrapolación

- `motion_window_s = 5.0`: hasta dónde atrás mira "dónde está ahora y a qué
  velocidad va". Un objetivo que se mueve tiene que describirse por su
  pasado reciente. Sobre toda su vida, una lancha que patrulla de ida y
  vuelta tiene su mediana en el medio de la patrulla, y una recta por su
  último cuarto de avistajes (**20 a 29 s, 81 a 235 m** en el vuelo 3 con un
  objetivo sintético a 4 y 8 m/s) cruza los giros y apunta ahí también: se
  reportaba a **10-14 m** de donde estaba.
- `mobile_speed_mps = 0.5`: **decisión, no medición** (una persona caminando
  va a ~1.4 m/s). La velocidad sola no alcanza en una pista corta: en el
  vuelo 3 el operador **parado** dejó pistas de 6-21 avistajes en 1-5 s con
  velocidades ajustadas de **0.75 a 1.95 m/s**, y cada una se volvió un
  candidato móvil aparte: **ocho puntos para una persona.** Por eso el
  movimiento ajustado tiene que llevar al objetivo más lejos que el
  desplazamiento móvil a lo largo de los avistajes sobre los que se ajustó.
- `extrapolation_max_s = 3.0`: hasta dónde adelante se arrastra un
  candidato móvil desde su último avistaje. Es aparte de la ventana sobre la
  que se estima su velocidad: un objetivo que gira sigue andando en el papel
  mientras esto lo permita. Con 5 s, una patrulla sintética que giraba cada
  6.25 s a 8 m/s se reportaba pasado su giro. Medido sobre esa patrulla
  (error mediano / p90):

| | 1.5 m/s | 4 m/s | 8 m/s |
|---|---|---|---|
| 5 s | 0.35 / 11.39 | 2.80 / 21.38 | 7.91 / 17.09 |
| **3 s** | 2.30 / 14.40 | 2.82 / 13.44 | 6.69 / 17.09 |
| 2 s | 2.56 / 15.90 | 1.89 / 11.29 | 6.69 / 17.91 |

  Ningún valor gana a todas las velocidades. 3 s tiene el peor caso más
  chico, y esa es la elección: **una decisión sobre qué fallo tolerar, no un
  óptimo.**

### Qué recorte lleva un candidato (`crop_choice`)

`confidence` (default) guarda el avistaje más confiado. `appearance` guarda,
por pista, el avistaje cuyo vector está más cerca de la apariencia media de
la pista, así que la foto muestra lo que la evidencia mayormente es.

La diferencia es el caso de una caja que agarró dos personas, o una persona
al lado de desorden: el detector está más seguro ahí, y el filtro de
apariencia en tierra le da la razón al detector. En el vuelo 3, con el
seguidor de reemplazo, el único candidato confirmado que mayormente **no** es
una persona llevaba una caja así bajo `confidence` y pasó el filtro; bajo
`appearance` lleva una de sus cajas no-persona y queda filtrado.

Es opt-in porque la misma medición muestra el costo: los recortes elegidos
son menos confiados (**mediana 0.41 contra 0.77**), y pasa el filtro un
no-persona preliminar más.

### La densidad que separa fantasmas de personas

De los cuadros en que esta capa recibió una detección mientras el candidato
estaba vivo, la fracción en que se vio al candidato mismo. Un falso positivo
estático es un parpadeo estirado en mucho tiempo; una persona seguida es
densa mientras está en vista. Medido en el vuelo 3 contra las letras que un
humano puso en cada caja: los **6 fantasmas techan en 0.397** y los **7
candidatos de persona real pisan en 0.714**, sin nada en el medio.

**Avistajes por segundo hace lo mismo en este vuelo y NO se debe usar:** no
es libre de escala. Una cifra de 5.7 avistajes por segundo existe solo
porque esta grabación es a ráfagas a 1.64 FPS; en una placa estable a 3 FPS
la misma cantidad no puede pasar de 3. **Los cuadros entregados son la
unidad de oportunidad; los segundos no.**

Un cuadro en que el detector no encontró nada nunca llega a `observe`, así
que el denominador son los cuadros en que la detección estaba produciendo
algo, no los que la cámara capturó. Es la más estricta de las dos lecturas y
la única disponible sin un segundo canal desde la cámara.

**Se probó como regla de madurez y se cayó**, ver `docs/DESCARTADO.md`.

### Cómo bajó el error del operador, en tres pasos

El punto del operador en el vuelo 02-ago, a medida que se arreglaron tres
reglas distintas. Cada paso es un contraste que `tests/test_sin_mezcla.py`
fija hoy:

| | error | qué cambió |
|---|---|---|
| **2,18 m** | hasta describir a los móviles por su pasado RECIENTE | algunos caminantes del vuelo se clasifican móviles y dejan de fundirse en el candidato del operador (**670 → 519 impactos**) |
| **2,25 m** | hasta exigir que el movimiento ajustado llevara al objetivo más lejos que el ruido de proyección | las pistas cortas del operador **parado** dejan de ser móviles y vuelven a fundirse en un candidato |
| **2,27 m** | hasta que una pista móvil tuviera que preguntar si la persona ya tenía candidato antes de abrir uno | la rama móvil era el único camino que nunca consultaba el emparejador, así que un operador parado cuyas cajas duplicadas finjen velocidad se volvía varios puntos |

Medido por identidad contra las etiquetas a mano de ese vuelo, el operador
pasó de **cuatro candidatos a tres**, con las mismas 5 personas encontradas y
los mismos 4 fantasmas, y el punto quedó más cerca.

### Sin compensación de movimiento de cámara, el mapa se ve MEJOR

Es el contraste que justifica que exista `scripts/personas_encontradas.py`.
El mismo detector y las mismas cajas, con la única diferencia de
`use_cmc=False`: de **2637 cajas solo 852 reciben id**, contra **1608 con
CMC**, y la evidencia que nunca llega a la capa de identity no forma
candidato.

Lo que hay que mirar es que el mapa **sin** CMC se ve **más limpio** y tiene
**dos personas menos**. Juzgando por fantasmas, la versión vieja gana. Por eso
el marcador cuenta personas y no cajas.

### `EMB_DIST_REUNE = 0.63`: el número que NO es seguro

Reasociar un candidato en movimiento es el único caso donde la posición
argumenta por separar y la apariencia por juntar, así que decide la
apariencia, y se le exige más que el 0.95 que decide una fusión estática.

**Por eso la reunión está apagada por defecto.** Medido sobre el vuelo
02-ago, pista por pista, contra las letras que un humano puso en cada caja:

| | distancias |
|---|---|
| trozos de la MISMA persona | 0.26 (el operador) y 0.41 0.41 0.60 0.61 0.64 0.81 (el caminante) |
| personas DISTINTAS | 0.64 (B vs C) · 0.67 (H vs C) · 0.68 (H vs B) · 0.73 … |

Los dos rangos **se solapan 0.18**: siete pares de personas distintas están
más cerca que el par más lejano de la misma persona. Ningún umbral los
separa. Lo que junta al caminante sin fusionar a nadie es una ventana entre
0.61 y 0.67, y 0.63 cae ahí, elegido mirando el vuelo sobre el que se
juzga, que es el error que este repositorio ya pagó una vez. A 0.70 el
chico del balcón es absorbido por otro y **desaparece del mapa**, que en una
búsqueda es el peor fallo que hay: al operador no se le dice que hay una
persona ahí.

Queda apagado hasta que exista un segundo vuelo sobre el que elegir el
número.

### Separación mínima por clase (`SEPARACION_MINIMA_M`)

Distancia centro a centro más chica a la que dos miembros de una clase
siguen siendo dos cosas. Es **geometría de estacionamiento, no ruido:** una
plaza estándar mide 2.4-2.6 m de ancho, así que dos coches lado a lado
están a 2.5 m y un radio de fusión de 3.5 m (el valor ajustado para
personas) los reporta como un solo coche. Nada aguas abajo puede
deshacerlo, porque a esa altura ya hay un candidato.

**Las personas están ausentes a propósito,** y la razón vale decirla: dos
personas también pueden estar a 0.6 m, pero nada las pone ahí como las
plazas ponen a los coches. Aplicarle un piso de 0.6 m a las personas
encogería el radio con el que se midió cada vuelo hasta hoy, para comprar
una separación que la escena no impone. Los vehículos son el caso donde la
geometría es lo bastante regular para codificarla.

El canje que hace va para un lado a propósito. Por debajo del radio de
ruido, un coche bajo ruido de proyección puede fragmentarse en dos
candidatos a un par de metros; por encima, dos coches se fusionan en uno.
La fragmentación reporta lo mismo dos veces casi en el mismo lugar, que un
operador resuelve de un vistazo. La conflación hace **desaparecer** un
vehículo, y nada en la pantalla lo dice.

## La segunda opinión de CLIP en la estación (`--clip-descarta`)

Por defecto, lo que CLIP duda va al **final** de la cola y nada más se mueve:
el orden es estable y un POI sin puntaje nunca se degrada. Con
`--clip-descarta` se **quita** en vez de degradarse.

Medido en el vuelo 02-ago, quitar en vez de degradar saca la bolsa y el cono y
deja a todas las personas en pie: **fantasmas 2 → 1, con 5 de 5 personas
conservadas.**

Está apagado por defecto por dos razones, y la segunda es la que importa:
esconder un punto que el operador nunca vio es decisión del operador y no del
sistema, y **CLIP sigue perdiendo el desorden más difícil**. El objeto rojo de
ese vuelo puntuó **1.81, por encima del umbral**, y además se parece a una
persona por tamaño.

## Fundir los reportes de dos drones (`flota.fundir`)

- **El margen de un pin fundido es el MENOR de los dos.** Medido sobre los
  candidatos del 02-ago, el radio del 95% de un objetivo estático es **82 a
  99.9% sesgo** y casi nada de dispersión, y ese sesgo es `gps_sigma` más el
  alcance oblicuo por `yaw_sigma`. Un objetivo visto desde 14 m y desde 37 m
  no tiene una incertidumbre: tiene la del dron más cercano. Quedarse con el
  que llegó primero era reportar la peor de dos respuestas sin motivo.
- **3-oct-2026, la apariencia del pin fundido.** Un pin se quedaba con los
  campos del primer reporte que lo creó, y si ESE dron no calculaba
  apariencia el pin quedaba sin vector aunque el otro sí lo trajera. Medido
  con una placa calculando apariencia y la otra no: el veredicto del
  operador **se registraba en disco y no llegaba a ningún dron**, porque
  `gs_mapa.plantilla_para` filtra por `emb`. Un pin fundido sin vector
  convierte el botón del operador en un botón que no hace nada.

## Umbrales de la cámara

### `EMB_DIST_OBJETIVO = 0.85`: el otro número que NO es seguro

Distancia de apariencia bajo la cual una caja que el detector dudó se
conserva porque **se parece** al objetivo que el operador señaló. A
diferencia de la ventana, no pregunta dónde está la caja, y ese es el
punto: la ventana es la proyección de una posición en el suelo y falla
justo cuando la actitud de la aeronave es menos cierta.

Medido sobre 773 cuadros de la misión 02-ago (2746-3700), 1673 cajas de
personas y 484 del objetivo, conservando las cajas puntuadas entre 0.10 y
0.25 que hoy se tiran:

| | recall objetivo | recall total | precisión |
|---|---|---|---|
| solo conf ≥ 0.25, como vuela | 91.1% | 48.3% | 57.9% |
| **+ las dudosas que SE PARECEN (0.85)** | **94.4%** | **50.1%** | **55.6%** |
| lo mismo a 1.00 | 97.7% | 52.5% | 52.3% |
| lo mismo a 1.20 | 97.7% | 61.1% | 40.3% |

No se reentrena ningún peso y no se adapta nada: la plantilla es el
embedding del recorte que el operador clicó, así que apagar esto devuelve
el sistema exactamente a lo que era.

**EL NÚMERO NO ES SEGURO TODAVÍA, por la misma razón que `EMB_DIST_REUNE`
no lo es:** se eligió mirando el tramo sobre el que se juzga. Hay que
reelegirlo sobre los vuelos del 01-ago antes de poder defenderlo, y hasta
entonces la compuerta queda cerrada salvo que el llamador pida una
distancia explícita.

### `BANDA_BAJA = 0.2`: el piso de la banda BYTE

El puntaje más bajo que se le pide al detector cuando hay un seguidor
corriendo. Es 0.2 porque **es el propio `track_low_thresh` de BoT-SORT**, y
una caja bajo el piso del seguidor se descarta antes de asociar, así que
pedirla es pagar OSNet (~33 ms por caja) por algo que nadie va a mirar.

Medido en el vuelo 02-ago: de **2443 cajas entre 0.10 y 0.20, exactamente
cero** recibieron identificador de pista; entre 0.20 y 0.25, **30 de 628**
sí.

Pedir por debajo del umbral de reporte no cuesta inferencia extra, porque
el detector ya puntuó esas cajas y las estaba tirando. La banda 0.2-0.3
cierra **77 de los 226 huecos** donde una persona está en un cuadro y falta
en el siguiente.

### Pasada por mosaicos (`slice_every`)

El cuadro se reduce a la entrada del modelo antes de inferir, así que una
persona de 55 px de alto llega como 28 y las que ya estaban en el límite
desaparecen. La pasada por mosaicos a resolución nativa se saltea esa
reducción. Cuesta **seis veces la inferencia**, que es por qué no corre en
cada cuadro, y por qué no necesita hacerlo: la capa de identity pide once
avistajes en treinta y seis segundos, un décimo de los cuadros, así que una
pasada de cada cinco mantiene el promedio cerca del presupuesto mientras la
pasada barata sigue corriendo siempre.

Medido en el vuelo 02-ago con el umbral de mosaico elegido sobre los vuelos
del 01-ago: recall **55.0 → 57.3%** total y **39.5 → 42.4%** donde el dron
está alto, a la misma precisión.

### Compensación de movimiento de cámara (CMC)

El dron se mueve, así que cada caja se desplaza en la imagen entre cuadros
y la predicción del seguidor pierde la caja siguiente si no se quita primero
el movimiento del fondo.

Medido con los parámetros del seguidor del vuelo y etiquetas a mano:

| | ids sobre el operador parado |
|---|---|
| sin CMC, ventanas etiquetadas del 02-ago | 32 |
| con flujo óptico disperso (`sof`) | **5** |

En los vuelos 2a / 2b de otro día los ids sobre personas **se
redujeron a la mitad** (10 → 6, 22 → 11) sin absorber ninguna caja de
no-persona.

Costo en la Raspberry Pi 5 a la escala de imagen 0.15 por defecto de boxmot:
**+8.1 ms medianos por cuadro** (1.0 → 9.1 ms), un ~4% de los 206 ms de la
cadena, sin throttling.

El método por defecto de boxmot es `ecc` y es peor en los dos ejes: sobre el
mismo vuelo y los mismos umbrales calibrados recuperó muchos menos ids
(**IDF1 0.376 contra 0.656** con `sof`) y costó más en la Pi (**p90 20.5 ms
contra 10.4 ms**).

### `pausa_s`: la pausa entre los arranques pesados

Default 0, sin cambio para nada que esté enchufado a la pared. Con batería
es la única palanca que tiene el software contra el fallo medido el
**25-ago-2026**: la placa murió **3 s después** de abrir la cámara y cargar
los dos modelos, con el pack **LLENO**, consumiendo 3.47 W. Eso no está ni
cerca de saturar un UBEC de 5 A, así que lo que la mata es **el escalón, no
el nivel.** Abrir la cámara y cargar los modelos seguido apila esos
escalones; esto los separa.

### 4-oct-2026: el reintento de cámara podía mentir, y lo encontró un linter

Al configurar ruff (`[tool.ruff]` en `pyproject.toml`) saltó un `B023`,
*function definition does not bind loop variable*, en
`OnboardCamera._first_frame`. Era un bug de verdad, en el código que vuela.

El bucle era así:

```python
for intento in range(1, self.INTENTOS_ENCENDIDO + 1):
    listo = threading.Event()
    def capture():
        try: picam.capture_array()
        finally: listo.set()        # cierra sobre la CELDA, no sobre el valor
    threading.Thread(target=capture, daemon=True).start()
    if listo.wait(self.ESPERA_PRIMER_FRAME_S):
        return
```

`listo` es local de `_first_frame`, así que **las dos definiciones de
`capture` comparten una sola celda**. Secuencia que falla:

1. el intento 1 se cuelga dentro de `capture_array` (el modo de fallo que este
   watchdog existe para cubrir: `Camera frontend has timed out`);
2. vence el plazo, el hilo queda vivo porque es daemon, a propósito;
3. el intento 2 crea un `Event` **nuevo** y rebinda `listo`;
4. el hilo del intento 1 se desatasca y llama `set()` **sobre el `Event` del
   intento 2**;
5. `listo.wait()` del intento 2 devuelve `True` y `_first_frame` **vuelve como
   si la cámara hubiera entregado un cuadro**, con el intento 2 todavía colgado.

Reproducido con una cámara de mentira que se atasca en la primera captura y se
desatasca durante la segunda:

```
como estaba -> VUELVE diciendo que el intento 2 entrego un cuadro
arreglado   -> levanta RuntimeError: la camara no entrego un frame
```

El arreglo es ligar el `Event` **por intento** como argumento por omisión
(`def capture(ev=listo)`), así que un hilo tardío solo puede poner el suyo.

**La lección vale más que el bug:** ninguno de los 50 gates lo detectaba, porque
ninguno simula una captura que se cuelga y después se desatasca. Lo encontró una
herramienta en treinta segundos. Por eso ruff está configurado y corre en CI.

### La cámara que se enumera pero no transmite

El sensor se detecta por I2C y se enumera mucho antes de que vaya a
transmitir de verdad, y a veces no transmite en absoluto: libcamera reporta
`Camera frontend has timed out` y la llamada de captura **no vuelve nunca**.

Observado el **25-ago-2026**: cinco fallos seguidos dentro del minuto
posterior al arranque y justo después de matar un proceso a mitad de
captura, y después cinco éxitos de cinco una vez que la placa llevaba unos
minutos encendida. Nada cambió en la configuración.

Sin el reintento, ese modo de fallo es una misión perdida sin diagnóstico:
la Pi viva, el protocolo corriendo su temporizador, y cero detecciones para
siempre porque la primera captura nunca volvió.

### Los canales llegan RGB, no BGR

picamera2 etiqueta esta configuración como `BGR888`, pero ese nombre es de
libcamera y lista las componentes en el orden **opuesto** al que el arreglo
llega. Lo que vuelve es R,G,B. Todo aguas abajo tiene forma de OpenCV y
espera B,G,R: ultralytics lo asume para un arreglo crudo, el modelo de ReID
lo asume, y `cv2.imencode` lo asume al escribir el recorte. Sin tocarlo, el
rojo y el azul quedan cambiados para los tres a la vez.

Medido en la placa real (**25-ago-2026**): una persona que el detector
encontró a 0.887 con los canales cambiados puntúa **0.909** una vez
corregido. Es poco, y **no es** la razón por la que los pesos de VisDrone no
encuentran nada en interior, que es una brecha de dominio. La razón para
arreglarlo es el **recorte**: es la fotografía que un operador mira para
decidir si manda a alguien a ese punto, y estaba llegando con piel azul.

Convertir el cuadro entero una vez es lo que mantiene a los tres
consumidores de acuerdo. Cuesta **1.70 ms contra 184 ms** de inferencia en
la Pi 5, un 0.9% del cuadro.

## Umbrales del protocolo de visión

Todos estos vivían como comentarios dentro de `vision_protocol.py` hasta
el 4-oct-2026. Son decisiones, no derivaciones: su valor podría ser otro
en otra misión sin que el sistema funcione distinto, así que el docstring
dice QUÉ son y esta sección dice de dónde salen.

- `STATION_TIMEOUT_S = 0.3` y `STATION_RETRY_S = 10.0`: **sin medir.** El
  timeout es una fracción del período del lazo de visión (0.25 s a 4 Hz),
  porque una consulta a la estación no puede costarle al lazo más que una
  astilla de su período. El retry evita preguntarle a una estación caída
  en cada reporte.
- `MIN_MEASUREMENTS = 8`: **no derivable de los datos.** Mínimo de
  impactos antes de que el consenso RANSAC hable. Codifica cuánta
  evidencia hace falta para no inventar un punto.
- `RODEO_TOLERANCIA_M = 5.0`: es el orden del sesgo de GPS que el sistema
  ya contempla (`GPS_SIGMA_M` 1.5 m, y el radio del 95% de un candidato
  estático en el vuelo 3 es 2.4 m). Es un radio y no una coincidencia de
  coordenada porque una aeronave en posición deriva.
- `RODEO_PLAZO_S = 60.0`: **no derivado, es una guarda.** Un tramo de una
  órbita de 30 m son decenas de segundos a las velocidades de esta
  aeronave; protege contra una aeronave a la que se le mandó un punto que
  no puede alcanzar.
- `report_preliminary` apagado por defecto: un dron en órbita puede
  esperar a estar seguro. Se enciende para un BARRIDO, y la razón está
  medida sobre el vuelo 3: una pasada de 30 s sobre una persona **no
  madura nunca** un candidato, así que una búsqueda que cruza cada punto
  una vez y sigue no reporta nada.
- `attitude_source`: la cámara está fija al cuerpo. A 30 m, 10° de
  cabeceo sin contabilizar ponen el impacto a **7-10 m** de distancia
  (`tests/test_actitud.py`). `None` deja solo el ángulo de montaje, que
  es correcto para un dron en órbita e incorrecto para uno que escolta.

### La ventana del objetivo: la ganancia no sobrevive a la cadena

`FOCO_RADIO_PX = 320` es la mitad del cuadrado de 640 px sobre el que se
midió la ganancia, en un cuadro de 1920x1080. `FOCO_UMBRAL = 0.10` es el
piso al que el detector ya estaba puntuando.

Contra cajas de verdad la ventana sube el recall sobre el objetivo de
**43.9% a 60.7%** y baja la precisión de **71.2% a 53.3%**. Pero corrida
por la cadena entera sobre el vuelo del 02-ago y medida en personas y
fantasmas **no cambia nada**: las mismas 5 personas y los mismos 6
fantasmas, en las cuatro variantes probadas (objetivo sobre el operador y
sobre el candidato más frágil, con el seguidor tal como vuela y con su
piso bajado para aceptar la banda).

El mecanismo está en los números: 1673 cajas cayeron dentro de la ventana
y solo **29** terminaron con identificador de pista. BoT-SORT descarta
todo lo que está bajo su umbral bajo de 0.20 **antes** de asociar, y una
detección sin pista no llega a la capa de identity. De las 841 cajas
entre 0.10 y 0.15 y las 494 entre 0.15 y 0.20, **cero** fueron seguidas.
Y por encima de 0.20 tampoco agrega, porque `camera.py` ya abre esa banda
sobre el cuadro entero (`low_band=0.2`).

Lo que queda es la fontanería para actuar sobre el clic del operador.
`docs/RESULTADOS.md` publica el 43.9 → 60.7 como "sin implementar": esa
fila está desactualizada, hoy es `fix_target()`, y omite que en personas y
fantasmas no mueve nada.

### Extensión en el suelo por clase (`GROUND_EXTENT_M`)

Dimensiones nominales de la cosa, **no correcciones ajustadas.** La
extensión a lo largo de la línea de visión está entre el ancho y el largo
del objeto según cómo quedó estacionado; cada valor es el punto medio de
ese par, así que el residuo está acotado por la mitad de su diferencia:
**±1.3 m para un coche, ±4.7 m para un autobús.** La alternativa es un
sesgo de la media longitud entera, siempre hacia el dron.

**Las personas están ausentes a propósito:** una persona de pie cubre
unos 0.4 m de suelo, así que la corrección sería de 0.2 m contra un error
de sistema de 2.4 m, una veinteava parte del ruido. Y todo resultado de
vuelo hasta hoy se midió con el borde inferior.

## Salud eléctrica de la placa

- `THROTTLED` se lee como archivo en vez de llamar a `vcgencmd`: **3 ms
  contra lanzar un proceso** en cada reporte.
- La máscara tiene dos mitades que significan cosas distintas. Los bits
  bajos son AHORA: actuar sobre ellos es que la aeronave está en problemas
  este segundo. Los altos son DESDE EL ARRANQUE: quedan puestos después de
  que la caída pasó, que es lo que los hace útiles en tierra, porque las
  caídas duran un instante y nadie está mirando en ese momento.
- **3-oct-2026, por qué existe esto:** una placa se quedó sin tensión
  durante cuatro horas y nada lo mostró. El kernel venía registrando
  `Undervoltage detected!` desde las 21:56; la estación, el operador y los
  logs que se miraban decían que la aeronave estaba sana, y a la 01:58 se
  cortó a mitad de una línea y no volvió. La causa se encontró al día
  siguiente moviendo su SD a otra placa y leyendo su journal. En el aire
  eso es un dron que desaparece sin explicación.

## 5-oct-2026: el YAML de boxmot movia dos compuertas de apariencia en silencio

Al meter el adaptador de trackers (`uav_vision/trackers.py`) en
`camera._build_tracker`, la puerta de equivalencia -- construir el tracker de
las dos formas y comparar atributo por atributo -- atrapo esto **antes** de que
el cambio entrara:

| | como estaba | por create_tracker |
|---|---|---|
| `appearance_thresh` | **0.25** | 0.6188818853936099 |
| `proximity_thresh` | **0.5** | 0.6084297894561342 |

`create_tracker` de boxmot mezcla los ajustes del llamador **sobre un YAML por
tracker**, y esos archivos traen el resultado de una busqueda de
hiperparametros sobre MOT (de ahi los dieciseis decimales). Son dos compuertas
de **apariencia** que este vuelo nunca calibro, para parametros que nadie aqui
fija. `appearance_thresh` es el coseno con el que el ReID decide que dos cajas
son la misma persona: de 0.25 a 0.62 cambia el seguimiento de verdad.

**Y no lo habria visto el gate de 2.39 m**, porque el replay usa sus propias
pistas de un `.npz` y no construye el tracker de la camara. Se habria
descubierto volando.

Por eso `construir()` instancia la clase **directo** y el YAML no se carga
nunca: los unicos valores que no son defecto de la clase son los que se pasan.

**Y es lo mismo que hace que comparar trackers signifique algo.** Si el YAML
entrara, cada tracker del zoo correria con el tuning de otro para peatones a
nivel del suelo, y la tabla mediria eso y no los trackers.

## 5-oct-2026: los diez trackers de boxmot sobre el vuelo 02-ago

Generada por `scripts/comparar_trackers.py --cmc=on`, que corre los tres pasos de
cada fila. **No está escrita a mano**, y eso es deliberado: la primera versión de
esta tabla sí lo estaba, y era falsa por tres motivos que resultaron ser todos
nuestros. La historia está al final de la sección, porque es lo que más vale.

| tracker | CMC | personas | fantasmas | sin juzgar | con id / 2637 | pistas | mediana/max cajas |
|---|---|---|---|---|---|---|---|
| botsort | on | **5 de 7** | 6 | 2 | 1608 | 36 | 13 / 336 |
| sam2mot | ninguno | **5 de 7** | 8 | 1 | 1891 | 298 | 2 / 136 |
| occluboost | on | 4 de 7 | **1** | 0 | 1392 | 6 | 192 / 586 |
| deepocsort | on | 4 de 7 | 2 | 2 | 643 | 31 | 7 / 120 |
| hybridsort | on | 4 de 7 | 3 | 1 | 1185 | 17 | 12 / 287 |
| boosttrack | on | 4 de 7 | 3 | 1 | 951 | 51 | 7 / 274 |
| strongsort | on | 4 de 7 | 5 | 3 | 1576 | 106 | 5 / 190 |
| bytetrack | ninguno | 3 de 7 | 0 | 0 | 728 | 109 | 4 / 84 |
| ocsort | ninguno | 2 de 7 | 0 | 0 | 437 | 77 | 3 / 71 |
| sfsort | ninguno | 1 de 7 | 0 | 0 | 1549 | 704 | 1 / 109 |

**La columna CMC dice lo que CORRIÓ, no lo que se pidió**, y la diferencia no es
cosmética: boxmot expone la compensación de movimiento por cuatro mecanismos
distintos y la trae ENCENDIDA por defecto, así que un tracker que ignora el
pedido no queda apagado, queda como boxmot quiera. Leído de la instancia
construida con `uav_vision.trackers.cmc_real`, el reparto es a tres bandas:

- **5 conmutables**: `botsort`, `boosttrack`, `occluboost` por `use_cmc`;
  `deepocsort` por `cmc_off`, que es el mismo booleano **invertido**;
  `hybridsort` solo por `cmc_method`, porque `create_cmc` devuelve `None` para un
  método `None` y entonces el método ES el interruptor.
- **4 sin mecanismo**: `bytetrack`, `ocsort`, `sfsort`, `sam2mot`. Y `ninguno` no
  es `off`: juntarlos cuenta a un tracker que no tiene compensación como prueba
  de que apagarla no hizo daño.
- **1 forzado**: `strongsort` hace `create_cmc("ecc")` en `__init__` sin
  parámetro y lo aplica sin guarda (`strongsort.py:68` y `:83`). **Compensa
  siempre y no hay forma de apagarlo.**

### Cuánto vale la compensación, medido con las dos tablas

`scripts/comparar_trackers.py --cmc=off` da la otra mitad del contraste. Solo
tiene sentido para los cinco conmutables:

| tracker | CMC off | CMC on | personas | fantasmas |
|---|---|---|---|---|
| botsort | 3 de 7 / 1 fantasma | **5 de 7 / 6** | +2 | +5 |
| boosttrack | 2 de 7 / 0 | 4 de 7 / 3 | +2 | +3 |
| deepocsort | 3 de 7 / 0 | 4 de 7 / 2 | +1 | +2 |
| occluboost | 3 de 7 / 2 | 4 de 7 / 1 | +1 | −1 |
| hybridsort | 4 de 7 / 3 | 4 de 7 / 3 | 0 | 0 |

La compensación vale **dos personas** en el tracker que vuela, y **cuesta cinco
fantasmas**. Las dos mitades se publican juntas a propósito: el marcador del
producto son los dos números, y quedarse con el que conviene es cómo una mejora
deja de ser una mejora. `hybridsort` es el caso que impide leer esto como una
regla: su corrida cambia (1033 → 1185 detecciones con id) y su marcador no.

### Lo que esta tabla NO autoriza a concluir

**1. Es UN vuelo, un día, un lugar, siete personas.** Diez trackers sobre un solo
escenario multiplican las formas de sobreajustarse a él. El entregable es la
máquina de intercambiar piezas; esta comparación es de un vuelo.

**2. "Más detecciones con id" no es mejor, y las dos puntas lo muestran.**
`sfsort` asigna 1549 y encuentra UNA persona, porque las parte en 704 pistas de
mediana 1 caja. `occluboost` asigna 1392 en **6 pistas** de mediana 192: está
fusionando identidades, que es el fallo peligroso, porque una persona fusionada
DESAPARECE del mapa. Su único fantasma no es precisión, es que casi todo cayó en
el mismo contacto.

**3. Siete de los diez no reciben la calibración completa.** Una fila que no la
recibió no se compara en los mismos términos que una que sí, así que las listas
van al lado de los números y no debajo:

| tracker | nuestros 5 ajustes, sin traducir | con traducir | sin equivalente |
|---|---|---|---|
| botsort | 5 de 5 | 5 de 5 | — |
| occluboost | 2 de 5 | 5 de 5 | — |
| sfsort | 0 de 5 | 5 de 5 | — |
| bytetrack | 2 de 5 | 4 de 5 | `new_track_thresh` |
| ocsort | 0 de 5 | 4 de 5 | `new_track_thresh` |
| strongsort | 0 de 5 | 4 de 5 | `new_track_thresh` |
| sam2mot | 1 de 5 | 4 de 5 | `track_low_thresh` |
| boosttrack | 0 de 5 | 3 de 5 | `new_track_thresh`, `track_low_thresh` |
| deepocsort | 0 de 5 | 3 de 5 | `new_track_thresh`, `track_low_thresh` |
| hybridsort | 0 de 5 | 3 de 5 | `new_track_thresh`, `track_low_thresh` |

Renombres que **cambian el valor**: `match_thresh=0.85 → iou_threshold=0.15`.
Renombres solo por rol: `track_high_thresh → det_thresh`.

### Por qué la primera versión de esta tabla era falsa, y los tres motivos eran nuestros

Importa más que los números, porque los números son de un vuelo y esto no.

**1. `match_thresh → iou_threshold` iba INVERTIDO, no "aproximado".** BoT-SORT
limita un COSTE de `1−IoU` (`matching.py:79` lo construye, `matching.py:35` lo
pasa a `lap.lapjv` como `cost_limit`), así que `match_thresh=0.85` significa
"asociá desde IoU ≥ 0,15" y es el ajuste **más permisivo** que lleva esta
calibración. En todo el resto de boxmot el mismo número es un PISO sobre el IoU
(`stages.py:152`, `boost.py:196`, `hybrid.py:343`, `hybrid.py:363`,
`occluboost.py:956`), así que cinco trackers recibieron "las cajas tienen que
solaparse 0,85 para ser la misma persona". A 25 m eso no pasa nunca. La
conversión es `1 − v` y es **derivada**: es el valor donde las dos expresiones
admiten los mismos pares. Vive en `uav_vision.trackers.CONVERSIONES` y la
sección 9 de `tests/test_elegir_tracker.py` falla si se pierde.

**2. "Ninguna fila corrió con CMC" era falso para tres de las nueve.**
`strongsort`, `hybridsort` y `deepocsort` compensaban, y `strongsort` encabezaba
la tabla. La premisa que supuestamente invalidaba las comparaciones era ella
misma incorrecta.

**3. El docstring de `botsort_pistas.py` afirmaba lo contrario del vuelo.** Decía
"configured exactly as `_build_tracker` builds it for the flight mission
(camera-motion compensation off)", y `_build_tracker` la tiene ON con método
`sof` (`camera.py:479-480`, `compensate_motion=True`). Esa línea es el origen de
que la tabla diera `botsort` 3 de 7 mientras la portada del repo dice 5 de 7.

**El efecto combinado: el orden se dio vuelta entero.** El tracker que vuela
pasó del cuarto puesto al primero, `ocsort` dejó de ser "no funciona para este
caso" (6 detecciones con id de 2637) y pasó a trackear el 16,6 % del vuelo, y
`sam2mot`, que la tabla omitía sin decir por qué, empata en personas con el que
vuela. **La tabla preliminar sugería cambiar el tracker del vuelo por
`strongsort`, y eso era un artefacto de dos bugs nuestros.** Las tres fallas
tenían la misma forma: una señal que no medía lo que uno creía. Un ajuste
"honrado" que llegaba invertido, un ajuste "ignorado" cuyo efecto seguía
encendido, y un comentario que contradecía al código que describía. Ninguna
rompía nada; las tres producían una tabla publicable. Lo que las encontró fue
mirar la INSTANCIA construida y el fuente de la librería, no las firmas.

La fila de una tabla se regenera con dos órdenes:

    V=../drone-geolocation/entrenamiento/venv/Scripts/python.exe
    "$V" scripts/comparar_trackers.py --trackers=botsort --cmc=on
    "$V" scripts/medir/umbral_invertido.py      # el diagnóstico de ocsort


## Rastreador (BoT-SORT)

- Parámetros validados offline: high 0.35 / low 0.2 / new 0.4 /
  buffer 40 frames / match 0.85, con huellas externas y CMC.
- El buffer se declara en segundos porque 40 frames son 24 s a la
  cadencia del vuelo 02-ago pero solo 8 s a los 5 Hz del dron.
- CMC (compensación de movimiento de cámara) **ENCENDIDA por defecto en el
  dron**, método `sof` y no el `ecc` de boxmot (`camera.py:267` y `:479`).
  Nada la apaga: `compensate_motion` no se pone en `False` en ninguna misión
  ni en ningún script. Su coste YA está dentro de los 199 ms / 5,04 FPS que
  `scripts/medir/medir_vuelo.py` mide en la Pi 5, porque esa sonda construye
  la cámara sin tocar el interruptor; lo que no está medido es su parte
  aislada. Vale +2 personas y +5 fantasmas sobre el vuelo 02-ago (tabla de
  los diez trackers, más arriba).
- BotSort con `with_reid=True` exige huellas; sin ReID configurado el
  módulo lo crea en modo solo-movimiento.

## Presupuesto de la Raspberry Pi (banco 22-ago-2026)

- torch 3.74 FPS / 258 ms · NCNN 12.47 FPS / 77 ms (3.3×).
- Térmico: 3 fps → 47 °C / 20% CPU · 5 fps → 52 °C / 33% ·
  10 fps → 58 °C / 76%.
- Con UBEC 5 A el voltaje se hunde sobre ~5 FPS y el sistema colapsa;
  por eso la tasa de visión por defecto es 4 Hz (bajada de 5 a 4 el
  24-ago-2026 para operar con margen bajo el punto de colapso). **La
  prueba con el de 7 A está hecha**, y es el experimento de una variable
  de `docs/deploying-yolo-reid-tracking-on-a-raspberry-pi.md:45-52`: el de
  7 A sobrevivió los 87 s con 0 filas en bajo voltaje y el de 5 A murió a
  los 55 s con 153, **consumiendo menos** (p95 1,19 A contra 1,50 A). No
  era la corriente, era el escalón.
- Ensayo cadena completa en vivo (24-ago-2026, alimentación de pared):
  detector solo 7.23 FPS / CPU 54% · +BoT-SORT 8.84 FPS / 65% ·
  +OSNet 6.59 FPS / 75.5%. OSNet ≈ +40 ms/frame con 1 persona. La
  capacidad (6.59) supera la demanda (4 Hz) incluso con huellas.
- 24-ago-2026: `OnboardCamera` corrió en la Pi real a 5.58 FPS
  (contrato probado en hardware). Config: `camera_auto_detect=0` +
  `dtoverlay=imx708` en config.txt (el auto-detect no reconoce el
  IMX708 de Arducam); `numpy<2` requerido por picamera2/simplejpeg.

## Validación acumulada (vuelo 02-ago salvo indicado)

- Geometría: rayos del protocolo vs rayos del vuelo real = 0.000° de
  diferencia (268 muestras).
- Replay con identity: operador como POI separado a 2.32 m
  (análisis offline: 2.49 m); la caja de equipos ya no roba el consenso.
- Stack con RF-DETR como detector: 2.39 m, 1106 obs, conf 0.80, caja
  ausente — empate en precisión (el piso ~2.3–2.5 m lo pone el sesgo
  GPS/yaw), mejora clara en robustez. Rol asignado: verificador en la
  Ground Station.
- Vuelos sanos 1-2 (C-5): candidato dominante limpio en las 3 corridas
  — 0.41 / 3.13 / 1.43 m (fusión simple: 1.03 / 2.08 / 0.59).
- Gates sintéticos de la identity (`tests/test_identity.py`):
  estático 0.09 m; móvil sin retraso; veto de co-ocurrencia mantiene 2;
  gemelas fusionan a 1.
