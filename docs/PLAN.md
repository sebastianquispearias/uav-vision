# Qué sigue, en orden, con su criterio de aceptación

Cada punto dice **cuándo está listo**, para que el trabajo tenga final y no sea una cinta sin fin.
Lo que ya está medido y descartado vive en `DESCARTADO.md`; los números actuales en `RESULTADOS.md`.

## Verificación activa: el bloque nuevo, y va antes que el resto

Las tres piezas de abajo son un solo mecanismo, no tres funciones sueltas. El operador dice qué
busca, el sistema mide cuánta evidencia tiene y cuánta le falta, y el dron va a conseguir la que
falta. En la literatura eso se llama **active sequential hypothesis testing**, y la regla de parada
es el test secuencial de Wald (1945): se acumula evidencia y se para al cruzar uno de dos umbrales,
uno para confirmar y otro para descartar.

Van antes que los puntos 1 a 6 porque ninguna necesita volar, y porque la medición del 30sep (abajo,
en lo que no hay que hacer) sacó de la mesa la justificación que se les suponía.

### A. La barra de certeza, visible

Hoy un candidato está maduro o no está, y el operador no ve nada mientras se decide. Medido en el
vuelo 3: un barrido de 30 s sobre una persona **nunca** madura un candidato, y uno de 60 s lo hace
el 47 % de las veces. Durante esos 30 s el sistema está a mitad de camino y el operador no lo sabe.

La cuenta ya existe: `identity.py` calcula `c["n"]` contra `n_reporte` y la cobertura temporal en
cada cuadro. Falta mostrarla, y faltan dos cosas de fondo. Primero, **ponderar cada avistamiento**:
hoy el décimo desde el mismo ángulo vale igual que el primero desde un ángulo nuevo, y en un test
secuencial cada observación aporta según cuán informativa es. Segundo, **el umbral de abajo**: hoy
la barra sólo sube, y un candidato que acumula evidencia en contra debería vaciarse y descartarse
solo en vez de quedar colgado como preliminar.

**Listo cuando:** la estación muestra, por candidato, la fracción de evidencia reunida; un candidato
con evidencia en contra se descarta solo; y el marcador de producto no empeora.

### B. El click que define el objetivo

El veredicto del operador existe y hoy hace poco: "es lo que busco" baja el umbral en esa ventana, y
"no es" borra un punto del mapa. Lo primero está medido y no sirve (ver punto 5). Lo segundo tira la
señal más valiosa que hay.

La vía medida es la plantilla de apariencia: el recorte del click es la huella, y una detección
dudosa que se le parece se acepta. El objetivo sube de 91,1 a 94,4 % sin tocar un peso y con vuelta
atrás inmediata. **No se entrenan pesos en vuelo**, por lo que ya está en la lista de abajo.

Y el "no es" se guarda como negativo difícil. En aprendizaje activo el ejemplo más valioso no es el
positivo sino el negativo que el modelo clasificó mal con confianza, que es exactamente lo que ese
botón produce.

**Listo cuando:** el click del operador cambia qué candidatos se reportan en el replay del 02ago, los
veredictos negativos quedan guardados y utilizables, y el marcador de producto lo juzga.

### C. La maniobra: mandar un dron a mirar desde otro lado

Es la fase 2 de la misión original ("los drones van, circulan y mantienen posición") y nunca se
construyó. Hoy la estación propone una segunda mirada y el piloto decide.

**No se justifica por recall.** La medición del 30sep lo cierra: el detector pierde personas más
grandes que las que encuentra en la misma imagen, así que ir a buscar más píxeles no arregla nada.
Lo que la justifica es la imagen para que decida el operador, seguir un blanco que se desplaza, y el
caso fuera de distribución (SeaDronesSee: 0,866 con la cámara a 36 grados contra 0,067 apuntando
recto hacia abajo, con barcos de 74 a 160 px).

Lo que falta no es el criterio sino su objetivo. `view_selection.py` puntúa vistas por **diversidad
geométrica**, que es lo correcto para triangular y lo equivocado para reconocer. Para verificar qué
es algo, el criterio tiene que ser la reducción esperada de incertidumbre **de clasificación**.
Primer obstáculo material: `correr.py` es el único que importa `view_selection.py` y `fusion.py`, y
no arranca (importa `CamaraSimulada`, que no existe: `camera.py` define `SimulatedCamera`).

**Listo cuando:** el operador señala un candidato, un dron no líder llega a una posición calculada y
devuelve una imagen desde otro ángulo, y queda medido cuánto movió la barra del punto A.

## Producto

### 1. Volar otra vez, a 20-25 m, otro día y otro sitio

**Por qué es el primero y no se puede saltear:** no solo da la primera evaluación limpia. Hoy **no
tenemos un solo frame de validación en el régimen que importa**, porque todo el material alto que
existe o es el test o son sus vecinos inmediatos. Por eso la validación eligió mal tres veces
seguidas y **no podemos ni elegir entre dos modelos** antes de tocar el test.

**Listo cuando:** hay un vuelo de 10 minutos a 20-25 m, con al menos dos personas además del
operador, y al menos una posición topografiada que no sea la del operador.

**Ayuda que existe:** la página `/plan` de la herramienta de etiquetado dibuja la altura contra los
frames y dice, del tramo que elijas, cuánto es nuevo y de qué tamaño se vería la persona ahí.

### 2. Medir en la Pi el cuadro bajo demanda

Nada de lo nuevo corrió nunca en el aire. La Pi tarda 206 ms por cuadro.

**Listo cuando:** `<= 330 ms/frame` medido en la Pi con el modelo congelado y el cuadro bajo demanda
respondiendo.

**Las fichas salieron de este punto el 17sep.** Medirlas en la Pi era caro y ya no hace falta
decidirlo ahí: por el marcador de producto no compran ni una persona, no quitan ningún fantasma y
alejan el punto 57 cm, a cambio de seis veces el cómputo. El número está en `RESULTADOS.md`. Si
alguna vez se encienden, se mide entonces.

### 3. Juzgar las mejoras nuevas con el marcador de producto — HECHO el 17sep

Se temía que los 12 puntos de precisión que paga la adaptación fueran fantasmas nuevos. **Lo eran:**
de 2 a 8. Y a cambio el punto se acerca un 42 %, de 1,92 a 1,12 m, así que es un compromiso que hay
que elegir, no una mejora que se aplica. Las fichas no compran ninguna persona.

La tabla está en `RESULTADOS.md`. El marcador dejó de vivir en una carpeta temporal: es
`scripts/personas_encontradas.py`, con `tests/test_personas_encontradas.py` fijándolo, y los
candidatos se regeneran con `scripts/replay_vuelo3.py --candidatos=`, así que puntúa el código de hoy
y no una corrida congelada.

**Lo que queda abierto:** decidir si la adaptación se enciende, y con qué salvaguarda. Como dice el
final de este documento, el modo de fallo es silencioso y hay que poder volver al modelo original.

### 4. RF-DETR en tierra, de punta a punta — HECHO el 17sep (`39f8f86`)

El botón existe y está fijado por `tests/test_gs_segunda.js`, que recorre el flujo entero:

```
  clic en el del dron 2  : {"ruta":"/mirar","cuerpo":{"dron":"2"}}
  mientras espera        : se ven las dos fases y ninguna afirma un resultado
  contesta               : 4 personas en tierra, en 1.44 s
  y solo al dron que miro: la tarjeta del dron 2 sigue sin respuesta
```

1,44 s contra el criterio de 5 s. El botón va junto al dron y no junto al punto del mapa, porque la
segunda opinión se le pide a **un dron**: es su cuadro el que se mira.

**Lo que queda, y es del portafolio y no del producto:** el video comparativo (punto 9) está citado
en `docs/README.md` y no incrustado en ningún lado.

### 5. Modo "objetivo fijado" — HECHO el 17sep, NEGATIVO

Bajar el umbral solo dentro de la ventana del objetivo lleva el recall sobre el objetivo de 43,9 a
60,7 %, sin un milisegundo extra de cómputo y reversible. **Pasado por la cadena entera no cambia
nada**: las mismas cinco personas y los mismos seis fantasmas, en las cuatro variantes probadas, con
el objetivo puesto en el operador y en el candidato más frágil (commit `e186c50`).

Diecisiete puntos de recall que no compran una persona. Es el techo medido de la versión barata de
"que el click enseñe", y el motivo por el que la vía es la plantilla de apariencia del punto B.

### 6. Re-unir a una persona tras un hueco — CONSTRUIDO Y APAGADO, por el umbral

El criterio **se cumple**, y `tests/test_reunir_movil.py` lo mide:

```
  como viene (apagado)               A(1) B(1) C(1) G(3) H(1)   6 fantasmas
  encendido (0.63, 30 s)             A(1) B(1) C(1) G(1) H(1)   6 fantasmas
  un poco mas suelto (0.70)          A(1) B(1) G(1) H(1)        6 fantasmas
```

G pasa de tres puntos a uno sin perder a nadie y sin un fantasma más. H ya era uno.

**Por qué sigue apagado**, y es el motivo que no hay que saltearse: entre unir a la que camina y
borrar al chico del balcón hay **0,07** de distancia OSNet, en una escala donde dos pedazos de la
misma persona llegan a estar a 0,81. El `EMB_DIST_REUNE = 0,63` se eligió mirando el vuelo contra el
que se lo juzga, y su propio comentario lo declara: *"THIS NUMBER IS NOT SAFE AND THAT IS WHY
REJOINING IS OFF BY DEFAULT"*.

**Listo cuando:** el 0,63 se re-elija sobre un vuelo que no sea el del test. Es el punto 1 otra vez.

## Portafolio

El objetivo declarado es conseguir trabajo como ingeniero de visión o percepción. Para eso el sistema
**no necesita estar terminado**, necesita ser defendible y entendible rápido.

### 7. Una página que se entienda en 40 segundos — EN PIE, falta el medio

El `README.md` ya abre con la afirmación y el número, y ya tiene la sección **What it does not do**,
que es la parte que un lector que contrata valora: las 5 personas de 7 con sus 6 fantasmas, el 44 %
que el detector pierde, la medición de que acercarse no lo arregla, los 46,2 % contra 90,5 % del
modelo que no puede volar, que el sistema nunca toca el vuelo, y que sobre agua nunca se probó.

**Lo que falta:** el medio. Entre el titular y las limitaciones hay secciones de uso y de contrato de
cámara, que son para quien va a trabajar en el repo, no para quien lo está juzgando en 40 segundos.
Y el video comparativo (punto 9) está citado pero no incrustado.

### 8. Los entrenamientos que perdieron, contados como resultado — HECHO

Son cuatro, no tres, y `DESCARTADO.md` ya los cuenta como argumento y no como registro: la tabla con
el número de cada uno, y debajo la lección que no era obvia, que el entrenamiento 3 ganaba mirando
recall y precisión y perdía mirando personas, **y que por eso existe el marcador de personas y
fantasmas**. Eso es el argumento: el fracaso explica por qué el sistema se mide como se mide.

Y los cuatro estaban lastrados por una línea de Ultralytics que reinicializa la cabeza de
clasificación, así que ninguno estaba afinando nada. Está dicho ahí con el mensaje literal.

El `README.md` manda a ese documento diciendo para qué sirve: *"the document to read first if you
are judging the method rather than the result"*.

### 9. El video comparativo como pieza central — FALTA UN PASO QUE NO ES DE CÓDIGO

`docs/yolo26_vs_rfdetr.mp4` muestra 46,2 % contra 90,5 % con el acumulado corriendo en pantalla, y
ahora está citado en la sección **What it does not do** del `README.md`, que es donde lo va a ver
quien juzgue el repo.

**Lo que falta es subirlo.** Los binarios no entran al repositorio, por la misma convención que
`demo.gif`: se sube a un issue de GitHub y se pega la URL. El hueco con la instrucción ya está en el
README, en el sitio exacto. Es un paso manual y nadie más que el autor lo puede dar.

**Listo cuando:** el video se reproduce dentro del README sin que el archivo esté versionado.

## Lo que NO hay que hacer

Está medido y agotado, el detalle en `DESCARTADO.md`:

- **Más entrenamientos con otra receta sobre los mismos datos.** Tres perdieron.
- **Más etiquetado de los vuelos que ya tenemos.** El material está exprimido: lo que queda sin
  etiquetar por encima de 12 m son vecinos inmediatos de lo ya etiquetado.
- **RF-DETR como maestro mientras no haya metraje nuevo.** Pseudo-etiquetar los mismos vuelos
  triplicaría el sobreajuste, no la variedad.
- **Acercarse volando para que el detector vea mejor.** Medido el 30sep sobre las 764 pérdidas
  etiquetadas del 02ago: no son chicas (mediana 49,3 px, sólo el 8,2 % bajo los 28 px para los que el
  detector fue entrenado). Y comparando **dentro del mismo cuadro**, donde la altura del dron y la
  escena son idénticas, en el 59,1 % de los 176 cuadros el detector perdió a una persona **más
  grande** que una que encontró en la misma imagen (f2586: perdió 186 px y encontró 141). El fallo no
  es de resolución, así que más píxeles no lo arreglan. El zoom digital ya se había medido peor
  (43,9 → 11,2 → 2,3 %). La señal que queda es la **forma**: relación alto/ancho 1,27 en las perdidas
  contra 1,79 en las encontradas, y más anchas que altas el 20,0 % contra el 0,9 %. Para recall la
  vía es el detector (punto 4, segunda opinión en tierra), no la maniobra.
- **Adaptación de pesos en vuelo sin poder revertir.** La adaptación funciona, pero el modo de fallo
  es silencioso: hay que tener los dos modelos y poder volver al original.

## Trabajo futuro: servocontrol visual (cerrar el lazo)

Hoy el sistema es **percepción en lazo abierto**: mira, calcula la posición y la reporta, pero
nunca toca el vuelo. Los únicos comandos que un protocolo GrADyS puede emitir son `GotoCoords`,
`GotoGeoCoords` y `SetSpeed`: coordenadas y velocidad, nada de imagen.

Cerrar el lazo se llama **servocontrol visual** (*visual servoing*). Dos variantes:
**IBVS** (*image-based*), donde el error se mide en pixeles, y **PBVS** (*position-based*), que
primero convierte la imagen en una posicion. El sistema ya hace la mitad dificil de PBVS:
`pinhole_local.py` + `identity.py` convierten imagen en metros. Falta usar ese resultado para
mandar un comando.

**Las primitivas ya existen en `uav_api`**, solo que no estan expuestas como comando de GrADyS:

    /drive_body, /drive_body_wait    velocidad en el sistema del propio dron
    /travel_at_ned                   velocidad
    /set_heading, /set_yaw_rate      rumbo y velocidad de giro

**El caso que lo justifica:** a 40 m una persona son 42 px y el detector falla pasados los 24 m.
Un lazo que mantenga al objetivo a un tamano fijo en pixeles (por ejemplo 80 px de alto) elige la
altura solo, en vez de fijarla a mano antes de despegar.

**Lo que lo hace dificil, y por que no esta hecho:**
- Hay que exponer un comando nuevo en GrADyS, o llamar a uav_api directo desde el protocolo
  salteandose la abstraccion, que es justo lo que la interfaz existe para evitar.
- Un lazo cerrado a 3 FPS con un detector que pierde el 45 % de los cuadros se queda sin senal
  cada vez que el detector falla. Hace falta decidir que hace el controlador mientras tanto.
- Cambia lo que el sistema ES: de "observo y reporto" a "observo y vuelo", que es mucho mas
  dificil de defender como seguro ante un operador.
