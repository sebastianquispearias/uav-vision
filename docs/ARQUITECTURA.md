# Cómo está armado el sistema

Los docstrings del fuente dicen **qué es** cada cosa y cómo se corre.
[NOTES.md](../NOTES.md) dice **de dónde sale cada número**. Este documento dice **cómo encajan
las piezas**: lo que no cabe en una cabecera de archivo sin que la cabecera deje de ser legible.

Tres componentes tienen bastante diseño para merecer una sección: la estación de tierra, el
replay del vuelo, y la herramienta de etiquetado.

La cadena entera, en una línea por etapa:

```
camera.py        captura -> YOLO -> BoT-SORT -> OSNet -> recorte
pinhole_local.py pixel -> rayo de rumbo
                 interseccion con el suelo
identity.py      acumula por pista, clasifica estatico/movil, funde pistas en candidatos
vision_protocol.py  difunde los POI (es el protocolo de GrADyS)
gs_mapa.py       la estacion de tierra, con mapa
```

Para leer la cadena entera corriendo en un solo archivo, sin abrir cinco:
`scripts/sistema_esencial.py`.

---

## La estación de tierra (`scripts/banco_embedded/gs_mapa.py`)

### Qué se importa del paquete, y por qué no se copia

`fundir` y `pedidos_de_verificacion` hacen la fusión entre drones, que necesita **los mismos
números que usa el dron** para fundir sus propias pistas, y `EMB_DIST_OBJETIVO` es la distancia
con la que viaja el clic del operador. Los dos se **importan** en vez de copiarse, porque un
umbral escrito a mano en dos sitios es como la estación y los drones terminan sin estar de
acuerdo sobre qué cuenta como el mismo objeto, o sobre qué significó el clic: un bug que a nadie
se le ocurriría buscar.

Sin el paquete, porque este archivo está pensado para poder dejarse caer en cualquier parte, la
estación **sigue corriendo**: muestra los objetivos de los dos drones sin fundir, y el clic sigue
significando solo una posición, que es lo que significaba antes.

`R_TIERRA` está ahí porque los POI llegan en **metros locales**, x al este e y al norte desde el
origen de la misión, y con las coordenadas del origen esos mismos puntos se vuelven lat/lng. Esa
conversión supuestamente estaba bloqueada esperando acordar un formato con el grupo. No lo está:
los dos extremos de este enlace son nuestros.

### El estado de la estación (el dict `ESTADO`)

Las claves cuyo nombre no se explica solo:

| clave | qué es |
|---|---|
| `pois` | la última lista recibida, anotada. Lo que ve el operador |
| `historia` | cada reporte, para el rastro |
| `drones` | id de dron → lo último que se le oyó |
| `veredictos_dir` | dónde se guardan los veredictos del operador y sus recortes |
| `origen`, `origen_cli`, `desacuerdo` | el origen **en uso**, que es el del dron si declara uno; lo que **tecleó** el operador, guardado para poder contrastarlo; y los metros entre los dos cuando no coinciden |
| `nodos` | las direcciones del plano de datos de los drones, de `--nodos`, para empujarles cada orden. Vacío significa que los drones la aprenden consultando `/buscar` |
| `objetivo` | el objetivo que el operador fijó con "es lo que busco", en metros, o `None` |

**`buscar`, `buscar_v`, `buscar_epoca`.** Lo que el operador le pide **al dron** que busque. Esto
**no** es el filtro de la pantalla: esconder una clase solo deja de dibujarla, mientras que esto
cambia lo que el detector reporta en absoluto. `None` significa "lo que el dron arrancó con", y
el contador le permite a un dron notar un cambio sin comparar listas. La **época** es qué corrida
de esta estación emitió la orden: una estación reiniciada cuenta otra vez desde cero, y sin ella
un dron que tomó la versión 7 ignoraría toda orden nueva por debajo.

**`pois_por_dron`.** Una lista por dron. Una sola lista compartida se reemplazaba en cada reporte,
así que un segundo dron borraba los objetivos del primero y el mapa parpadeaba entre las dos
vistas.

**`rastros`, `frame_actual`, `frames_dir`.** En qué cuadro va el replay del banco, y dónde viven
los cuadros. Solo el banco manda esto: un dron de verdad manda coordenadas, no fotos, y el
enlace no podría cargarlas. Existe para que una demo pueda poner lo que vio la cámara al lado de
lo que el mapa hizo con eso.

**`segunda`.** La segunda opinión de tierra, una entrada por dron: qué pidió el operador y qué
volvió. **Nunca la imagen**, que se sirve desde disco por `/segunda.jpg`. Un cuadro de 1920x1080
son 300 KB, y llevarlo dentro de un sondeo de estado que corre una vez por segundo serían cuatro
megabits de la misma imagen durante todo el tiempo que el operador la mire.

`SEGUNDA` corre la segunda opinión en un **proceso aparte** a propósito: RF-DETR vive en el venv
de entrenamiento y esta estación tiene que poder dejarse caer en una laptop sin nada instalado.
`CLIP` es la segunda opinión opcional de CLIP sobre cada recorte (`filtro_clip.Anotador`), que
enciende `--clip`; `None` significa que la estación nunca arrancó una, y entonces ningún POI
lleva campo de puntaje.

### Las constantes que son decisiones del operador

Todas se pueden cambiar desde la línea de comandos.

**`DRON_CALLADO_S` (`--callado-s`).** Un dron que lleva este tiempo sin reportar no está viendo
nada ahora, y sus objetivos dejan de contar para el mapa: un pin rotulado "visto por 1+2" no
puede sobrevivir a que uno de los dos se calle. Es la misma ventana que los drones aplican a lo
que se oyen entre ellos.

**`CLIP_DESCARTA` (`--clip-descarta`).** Quitar lo que CLIP llama no-persona en vez de degradarlo.

**`OBJETIVO_RADIO_M`.** A qué distancia del clic puede estar un candidato y aun tomarse como el
que el operador quiso decir. Mismo razonamiento que el veredicto, que solo se aplica al POI más
cercano: un clic es un gesto con la precisión de un dedo sobre un mapa, y alcanzar más lejos le
entregaría al dron la apariencia de alguien **parado al lado** de la persona que se señaló.

**`RODEO_RADIO_M`, `RODEO_ALTURA_M`.** Dónde quiere el operador que se pare una aeronave cuando se
la manda a mirar un objetivo desde otro lado. Decisiones de misión, y nada las deriva: lo bastante
cerca para que una persona sea más que una forma, lo bastante lejos para no estar sobre la cabeza
de nadie, dentro de lo que permita el espacio aéreo. Viven en el instrumento del operador porque
**la capa que vuela se niega a inventarlas**.

**`RODEO_PUNTOS`.** Cuántas paradas. **UNA** por omisión, y el razonamiento importa más que el
número: la pregunta que está haciendo el operador es "¿eso es una persona?", y una fotografía
desde un ángulo que nadie tiene la contesta. Doce paradas son doce fotografías del mismo punto,
once de ellas contestando una pregunta que nadie hizo, y a este radio cada tramo lleva decenas de
segundos, así que la vuelta entera son minutos durante los cuales esa aeronave no patrulla nada.
La órbita sigue ahí y el protocolo acepta cualquier número, porque "ir y quedarse encima" está en
la misión y se va a querer; lo que no es es **el valor por omisión**, porque el valor por omisión
tiene que ser la respuesta barata a la pregunta frecuente.

**`EN_BANCO` (`--banco`).** Esta estación está manejando placas sobre un escritorio, no aeronaves.
Solo afecta lo que la página dice sobre la posición, y dice exactamente eso en vez de imprimir un
número que nadie debería creer.

### El arranque, que no tiene `main()` que documentar

`--radio-rodeo`, `--altura-rodeo` y `--puntos-rodeo` son argumentos del instrumento del operador
en vez de números en el fuente, porque los tres son decisiones de misión y la capa que vuela se
niega a inventarlos.

`--origen` se guarda **aparte** del origen que de verdad se usa: el valor tecleado es lo que el
operador **cree**, y el punto entero es poder distinguir los dos en cuanto un dron declare el
suyo. Hasta que alguno hable, el valor tecleado es todo lo que hay, así que siembra el que está
en uso.

CLIP se importa solo cuando se pide, para que la estación siga pudiendo dejarse caer en cualquier
parte sin torch. La segunda opinión se arranca **antes del primer reporte** y nunca en el primer
clic, porque cargar RF-DETR lleva unos 17 s y el criterio de esa función es que el operador espere
menos de cinco.

El puerto se sondea antes de atarlo. **Windows deja que una segunda estación ate un puerto que ya
tiene una**, y entonces los reportes van al socket que acepte primero; el síntoma es un mapa que
se queda vacío mientras el vuelo claramente corre, y ha costado dos sesiones. Negarse en vez de
adivinar.

---

## El replay del vuelo (`scripts/replay_vuelo3.py`)

### De dónde salen las entradas

Las tres viven en el archivo del vuelo por omisión. `demo/demo.py` apunta `UAV_VISION_DATOS` a
una copia autocontenida, así que el replay corre desde un clon sin archivo y sin dron.

`CONF_MIN` es el mismo corte que usa el análisis de identidad, y las filas de `embs_osnet.npy`
corresponden, en orden, a las detecciones por encima de él. `LAT0`, `LNG0` son el origen ENU, que
es el poste topografiado que usa cada análisis del vuelo 3. `PIES` es la posición topografiada del
operador y `OBJ` la caja de equipos, el ladrón del vuelo 3 que se robaba el consenso único.

Las detecciones de personas en caché preceden a que la clase llegara al reporte, así que no llevan
clase propia. Nombrarlas no cuesta nada cuando están solas, porque una clase nunca se contradice
consigo misma, y es lo que le permite a la estación distinguirlas de los vehículos una vez que los
dos están en el mismo mapa.

`fusion_radius_m` es el ruido de proyección esperado de **esta** escena (sigma del GPS más alcance
oblicuo por error de rumbo a su altura) y es el valor validado offline.

### Los interruptores, y por qué la corrida por omisión es intocable

La corrida de **solo personas es EL GATE DE EQUIVALENCIA de este repositorio**: tiene que seguir
imprimiendo 2,39 m, así que nada de ella cambia salvo que se pida. Todos los interruptores son
por eso opt-in.

| interruptor | qué hace |
|---|---|
| `--vehiculos` | agrega el camino de vehículos. Implícito en `--vivo`, que solo tiene sentido si hay algo a lo que cambiar |
| `--dron`, `--pasada` | qué dron dice ser este replay y qué mitad del vuelo vuela. El vuelo hizo dos pasadas sobre el mismo suelo con varios minutos de diferencia, y el paper mide que el sesgo de GPS entre ellas es **independiente**, así que la pasada 1 y la 2 hacen de dos aeronaves. Es el protocolo de pseudo-enjambre, usado acá para ejercitar dos drones con una cámara |
| `--refuerzo` | deja que pistas demasiado cortas para abrir un candidato refuercen uno que ya abrió una pista que duró |
| `--span`, `--miradas-min=N` | la madurez por miradas independientes es la que rige. `--span` reproduce la regla con la que se midió cada número anterior, que es la regla sobre la que está fijado el gate de 2,39 m |
| `--preliminares` | muestra los candidatos que se formaron y no maduraron. Para un dron en órbita son ruido; un vehículo que el dron cruza una vez en un barrido es exactamente el caso para el que existen |

**`UAV_VISION_GS`, `--vivo`, `--velocidad`.** Con `UAV_VISION_GS` puesto, los reportes que produjo
el protocolo se empujan a un `gs_mapa` corriendo, con el ritmo que tendrían durante el vuelo en
vez de aparecer todos de golpe. El vuelo duró unos 11 min, así que el 20x por omisión lo deja bajo
los 35 s, que alcanza para hacer clic durante.

**`--pistas=file.npz`.** Reemplaza el seguidor de reemplazo por ids calculados en otra parte;
`scripts/botsort_pistas.py` escribe los que da el BoT-SORT del vuelo. Una detección que ese
seguidor dejó sin id llega al protocolo sin id, exactamente como en el dron, y la capa de identity
no la ve nunca.

**`--evidencia-min=X`.** Separa **ASOCIAR** de **EVIDENCIA**. El seguidor ya vio cada caja y dio
ids con todas; una caja por debajo de X conserva el id que ayudó a construir pero nunca llega al
protocolo, así que no agrega ni una observación a la capa de identity ni un impacto a la
geolocalización. Los cuadros volados y la cadencia se dejan en paz, así que el piso de evidencia es
lo único que cambia.

**`--foco=x,y`.** Fija un objetivo como lo hace el "es lo que busco" del operador, para que el modo
de objetivo fijo se pueda juzgar por personas y fantasmas y no solo por cajas. Sin él, nunca se
carga nada por debajo del umbral de reporte y la corrida es byte a byte la que fija el gate. El
piso es **el de la cámara**, importado y no reescrito: este replay cargaba desde 0,10, por debajo
tanto de la banda de la cámara como del propio `track_low_thresh` del seguidor, así que servía
cajas que el dron nunca le habría dado a un seguidor en primer lugar.

**`--plantilla=f.npy`.** La otra mitad del clic del operador: a qué **se parece** el objetivo. Con
ella una caja dudada se conserva donde caiga y no solo dentro de la ventana proyectada. El vector
se lee de un archivo para que este script siga sin saber nada de a quién pertenecen las letras del
vuelo; construir la plantilla desde las etiquetas a mano es trabajo del script que mide, no del
replay.

**`--sintetico=V`.** Agrega un objetivo con **verdad conocida** que patrulla este-oeste a V m/s
por la escena, durante todo el vuelo. Se proyecta en cada cuadro en el aire con las poses del
propio vuelo, y un cuadro que lo tiene en vista lo detecta con la probabilidad medida para un
objetivo real en vista en este vuelo, con algo de ruido de píxel, bajo un solo id de pista como
sustituto de un seguidor que lo sostiene. Se llama 'boat' para que sus reportes se distingan de
los del vuelo. Nada aguas abajo sabe que es sintético, así que lo que sale es lo que la cadena
hace con un objetivo **en movimiento**: si lo reporta, cuándo, dónde y en cuántos pedazos.

**`--actitud`, `--actitud-roll=+1/-1`.** Mete el cabeceo del fuselaje grabado en `frames.csv` en
cada rayo, y suma el alabeo con ese signo. El signo es un interruptor porque el análisis de agosto
no pudo resolverlo con este vuelo.

**`--candidatos=file.json`.** Escribe cada candidato junto con los ids de las pistas fundidas en
él. Ese enlace es lo que permite rastrear un candidato hasta las detecciones que lo alimentaron, y
de ahí hasta la etiqueta que un humano le dio a cada caja, que es cómo
`scripts/personas_encontradas.py` puntúa la cadena **por persona** en vez de por caja. Se escribe
desde la corrida viva en vez de guardarse como archivo en disco, así que el marcador siempre
describe la capa de identity tal como está, no como estuvo.

### En qué se diferencian los dos modos

**SIN `--vivo`** nada del lazo cambia: corre tan rápido como puede y los reportes se publican al
final, que es lo que mide el gate de equivalencia. A la estación se le habla con la envoltura del
transporte y no con el reporte crudo: `{"message": <json string>, "source": <node id>}`.

**CON `--vivo`** el vuelo va al ritmo del reloj de pared para que haya tiempo de hacer clic a
mitad, el dron le pregunta a la estación qué debería estar buscando, y cada reporte sale **a
medida que se produce**. Esa última parte es lo que hace visible un cambio de clases: un lote
enviado al final mostraría la respuesta final y esconderría el momento en que cambió. La orden se
aplica por el **propio handler del protocolo**, el que corre un dron, así que el replay ejercita
el camino de código que vuela en vez de una copia de él, y una orden rancia o repetida se ignora
igual. Lo que se imprime es el segundo **del vuelo** y no el reloj de pared, porque si una orden
llegó a tiempo es una pregunta sobre el vuelo, no sobre el operador.

Con `--foco`, el operador señala lo que mostró **el mapa**, no la verdad topografiada: fijar la
posición verdadera mediría un modo que nadie puede usar. Con `--pasada`, el vuelo se corta en el
punto medio de su **tiempo** y no en una cuenta de cuadros, porque la cadencia varía y la mitad
de los cuadros no es la mitad del vuelo.

Lo que el reporte final **deja afuera** informa tanto como lo que lleva: un candidato que se formó
y nunca maduró es un objetivo que el dron cruzó una vez y sobre el que no se demoró, que es una
propiedad del plan de vuelo y no del detector.

---

## La herramienta de etiquetado (`scripts/etiquetar_grupos.py`)

### Los vuelos que abre

`VUELOS_LISTOS` son los vuelos cuyas cajas y embeddings ya están en disco, así que `--vuelo`
rellena las cuatro rutas. `VUELOS` son los vuelos con candidatos del detector de
`proponer_cajas.py`, donde `--vuelo` rellena además `--lista-frames` y `--nombre`, y las etiquetas
van a `etiquetas_detector_<vuelo>.json`, aparte de cualquier otro etiquetado de ese vuelo.

- **02ago** es el vuelo de prueba, convertido por `scripts/convertir_02ago.py` para que también se
  pueda revisar.
- **02ago_alto** son sus cuadros 9315-9865: el único material sin etiquetar a altura alta que hay,
  y del mismo día que la prueba, así que entrenar con él **halaga** el puntaje de la prueba.
- **02ago_huecos** son los cuadros **entre** las ventanas de la prueba, pegados a ella, así que
  sirven para **medir** y nunca para entrenar: sin ellos, un candidato que viva ahí no se puede
  juzgar en ningún sentido.
- **14jun** es el único material con una segunda persona dentro.

`_PERSONA_POR_ALTURA` es cuántos píxeles de alto salió una persona, por altura, **medido** sobre
las cajas etiquetadas de estos vuelos en vez de derivado de la óptica: la cámara mira adelante y
abajo, así que a poca altura la persona está lejos por el suelo y no crece como una vista nadir
predeciría. La tabla está en [NOTES.md](../NOTES.md).

### Lo que ve quien etiqueta

`MUESTRA` es cuántos recortes muestra un grupo: suficientes para ver qué es, pocos para cargar
rápido. `LADO` es el lado del recorte en píxeles, y a 96 el texto de una caja de contexto es un
borrón. Los tres colores son BGR: amarillo la caja que se está etiquetando, magenta las
detecciones propias del vuelo (las de `--contexto`), y cian las otras cajas del CSV que se está
etiquetando en el mismo cuadro.

`REVISION` tiene las etiquetas de la revisión por cuadro. **"duplicado"** es una segunda caja sobre
una persona que ya tiene una, y se tira. **"ignorar"** es algo que no se puede llamar de una forma
ni de otra, como un pie solo o una persona cortada a una astilla por el borde del cuadro: la
exportación lo borra, así que al detector no se lo premia ni se lo castiga por encontrarlo.

`REPASO` es la fracción de cuadros revisados que la re-comprobación a ciegas vuelve a preguntar, y
`SEMILLA_REPASO` está fija para que reabrir la herramienta pregunte por los mismos.
`LADO_MIN_NUEVA` es el lado en píxeles por debajo del cual una caja dibujada es un resbalón del
ratón.

### Lo que guarda la página

Una caja dibujada es `[x1, y1, x2, y2]` para una persona, o `[x1, y1, x2, y2, "ignorar"]`. Una
caja del CSV a la que se le arrastraron las esquinas **también** se guarda: la etiqueta pertenece
a la caja, así que la caja tiene que ser arreglable, o una detección que cubre solo las piernas
queda mal para siempre. Los pares confirmados son los que quien etiqueta dijo que son **dos
personas paradas juntas** y no una encajada dos veces; sin registrarlos quedan marcadas para
siempre y el cuadro no deja nunca de contar como un problema.

Los recortes se ordenan **primero los sin etiquetar y después los más grandes**, así que el
siguiente clic siempre es el que etiqueta más. Cada recorte lleva **su propia** etiqueta final,
así que una caja corregida aparte de su grupo se ve. Un **hueco** es un cuadro sin nadie entre dos
que sí tienen a alguien, y se encola con todo otro tipo de cuadro pendiente.

Los recortes se cortan con un margen, porque una caja dibujada justa a altura corta el contexto
que distingue una persona de un poste, y los bordes se dibujan **después** de redimensionar, así
que su grosor son píxeles del thumbnail cualquiera sea el tamaño de la caja.
