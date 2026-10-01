# Lo que se probó y no funcionó, con sus números

Este documento existe porque una idea descartada sin número se vuelve a intentar. Cada entrada dice
qué se probó, qué dio y **por qué falló**, que es lo único que evita repetirla.

## Entrenamientos

### Cuatro entrenamientos seguidos perdieron

| Entrenamiento | Datos | Resultado | Por qué |
|---|---|---|---|
| 1 | solo vuelos bajos | 30,9 / 20,9 | cajas mal escaladas: a IoU 0,5 se derrumbaba a 11,0. Entrenó a 165 px para evaluar a 66 px |
| 2 | + VisDrone mezclado | 55,4 / 46,9 | arregló la escala pero no superó la línea base (54,3 / 60,9) |
| 3 | + 184 frames a 25 m | 57,5 / 76,6 | **ganaba en totales y perdía dos personas**: se sobreajustó a la única persona de esos 184 frames |
| 4 | + 14jun y todo mezclado | 51,9 / 94,0 | precisión espectacular, **4 personas de 7**. Pierde a G |

**La lección que no era obvia:** el entrenamiento 3 ganaba mirando recall y precisión, y perdía
mirando personas. Por eso existe el marcador de personas y fantasmas.

### Todos estaban lastrados por un detalle de Ultralytics

```
Overriding model.yaml nc=10 with nc=1
```

El modelo base tiene las 10 clases de VisDrone. Al afinarlo con un dataset de **una** clase,
Ultralytics **tira la cabeza de clasificación y la reinicializa**. O sea que ninguno de los cuatro
estaba afinando: estaban reentrenando una cabeza nueva desde cero con nuestras pocas imágenes.

**Sin verificar:** si los cuatro mejoran conservando las 10 clases. Es lo que puede cambiar los cuatro
veredictos y no se probó.

## Inferencia

| Idea | Resultado | Por qué falló |
|---|---|---|
| Correr a 1280 en vez de 960 | 52,9 / 56,2 (era 55,0 / 62,7) | el modelo está especializado en personas de unos 28 px y empeora con personas más grandes |
| Correr a 1920, sin encoger nada | 44,4 / 47,4 | lo mismo, peor |
| Ampliar la imagen sobre el objetivo | recall 43,9 a **11,2** | mismo motivo, en su forma más brutal |
| SAHI solo con fichas, reemplazando el cuadro | 55,0 / 47,0 | una ficha decide sobre un pedazo de escena y llama persona a una sombra más fácilmente |
| Subir el umbral sobre SAHI | 37,2 / 46,6 en el balcón | peor que el control **en los dos ejes** |
| TTA (multiescala y espejo) | imposible | esta versión de Ultralytics no lo implementa para yolo26 |

**Lo que sí funcionó** fue no reemplazar sino **sumar**: cuadro entero más fichas, que está en
`RESULTADOS.md`.

## Filtros contra los fantasmas

| Idea | Resultado |
|---|---|
| CLIP como segundo juez | **funciona**: fantasmas 2 a 1, sin perder personas |
| Filtro por tamaño físico en metros | bueno a nivel de caja (30 % de la basura por 2 % de la gente) pero **no supera a CLIP** a nivel de candidato |
| Bajar el NMS del detector (iou 0,45) | **cero efecto**: el NMS es por clase y las clases 0 y 1 de VisDrone son ambas personas |
| NMS entre clases | quita 44 cajas de 2655 y **no cambia ni un candidato** |

**El fantasma que nadie mata:** un objeto rojo que mide 1,56 m (como una persona) y puntúa 1,81 en
CLIP (por encima del umbral). Ninguna técnica probada lo toca.

**Dato útil de por qué el tamaño físico es ruidoso:** un cono midió 2,11 m porque la caja incluye su
sombra. El detector no encierra al objeto, encierra al objeto más su sombra.

## Capa de identidad

| Idea | Resultado |
|---|---|
| Apagar la regla de rapidez | arregla la fragmentación pero **pierde todos los móviles**: G deja de marcarse como móvil |
| Exigir cobertura temporal a la rapidez | conserva el móvil **falso** (el operador quieto) y pierde el **verdadero** (G) |
| Que los móviles pregunten antes de abrir candidato | **funciona**: el operador de 2,37 a 1,92 m. Commiteado |
| Ajustes de ReID | empeoran: fusionan objetos distintos |
| Exigir densidad para confirmar (piso de ciclo de trabajo en modo miradas) | **revertido**: ver abajo |

### El piso de densidad: separa en un vuelo y se cae al cambiar de seguidor

En modo `looks`, que es el que vuela, confirmar un candidato es solo contar veinte miradas de un
segundo. Eso premia la permanencia, y lo más permanente de una escena es un fantasma estático. Se
cruzó, candidato por candidato, contra las letras que un humano puso en cada caja del 02ago:

```
de los 4 candidatos confirmados, 3 son fantasmas y 1 es el operador
de las personas reales B, C, G y H, ninguna confirma
```

La densidad (en qué fracción de los cuadros con detección de su propia vida se vio al candidato)
los separaba con un hueco limpio: fantasmas hasta **0,397**, personas desde **0,714**, nada en el
medio. Y el piso en 0,55 quitaba los tres fantasmas confirmados sin tocar al operador.

**Por qué se revirtió.** Un test de la suite corre con el seguidor de reserva en vez de BoT-SORT, y
ahí el operador saca densidad 0,126 y queda en el medio del pelotón. La causa es la fragmentación:
un objetivo partido en pedazos conserva un span largo y reparte sus avistamientos, así que su
densidad se hunde. Pistas por candidato: con compensación de movimiento el operador tiene 5 y los
fantasmas 1 a 3; con el seguidor de reserva el operador tiene 14.

**Y el vuelo que voló tenía la compensación apagada**, o sea que el régimen donde la regla falla es
el real. La variante por pista (máximo de las pistas del candidato) no separa en ninguno de los dos
casos: un fantasma con tres pistas tiene una perfectamente densa en su propia ventana y marca 1,000.

**Qué haría falta:** un segundo vuelo con la compensación encendida, o una cantidad que no dependa
de cómo el seguidor parta las pistas. La medición de la densidad sí quedó, como medición y sin
umbral, y la barra de la estación la muestra.

## Etiquetado

| Idea | Resultado |
|---|---|
| Resolver pares encimados con "la más grande gana" (tecla X) | falla **25 %** a IoU 0,3 y 7,6 % a IoU 0,5, y solo tocaría unos 100 pares en un vuelo entero |
| Etiquetar más de los vuelos que ya tenemos | agotado: lo que queda por encima de 12 m son vecinos inmediatos de lo ya etiquetado |
| Etiquetar los tramos de 8 a 12 m | la persona mide 168 px ahí, prácticamente lo mismo que a 3-8 m (183 px): más de lo que ya sobra |

## Errores de método que costaron tiempo

- **Leer el archivo de letras como si fuera la verdad.** Las letras cubren solo las cajas que los
  detectores propusieron; dos tercios de la verdad del balcón no tiene letra porque se dibujó a mano.
  Eso produjo dos falsas alarmas: "el balcón está sin etiquetar" y "el recall está inflado". Las dos
  eran falsas.
- **Puntuar candidatos fuera de las ventanas revisadas.** Marcaba como fantasma a una persona real.
- **Elegir un umbral mirando el test.** Se corrigió re-eligiéndolo en validación, que dio el mismo valor.
- **Medir con 200 frames muestreados.** La ganancia de la adaptación sobre el objetivo parecía +7,6
  puntos y con la misión entera es +2,9.
- **Cortar un entrenamiento con `| head`.** La tubería cerrada mató el proceso a la primera época.
- **Medir la banda baja contra el marcador de producto, sobre el replay.** No da cero: no puede dar
  nada. El archivo de pistas tiene 2.637 filas y las detecciones son 5.708, exactamente las de
  confianza ≥ 0,25, porque el seguidor se corrió sobre ese corte. Las 3.071 cajas de la banda llegan
  al protocolo **sin `track_id`**, y la capa de identidad solo ve detecciones con pista. Así que ni
  la ventana del objetivo ni la plantilla de apariencia pueden mover un candidato en ese replay,
  hagan lo que hagan: los tres corridos dan `n_obs` total **1299**, a la unidad. Cualquier
  conclusión sobre la banda sacada por esa vía mide el cableado, no la idea.
- **Y re-correr el seguidor con el piso bajo tampoco alcanza, por un motivo peor.** Se corrió
  (`scripts/botsort_pistas.py --piso 0.10`, dos configuraciones) y el resultado deja la idea sin
  nada que medir en este vuelo:

  | tramo de confianza | cajas | con `track_id` | |
  |---|---|---|---|
  | 0,10 – 0,20 | 2.443 | **0** (0,0 %) | por debajo del `track_low_thresh` de BoT-SORT, que las descarta antes de asociar |
  | 0,20 – 0,25 | 628 | **30** (4,8 %) | la única parte de la banda que el seguidor puede reclamar |
  | ≥ 0,25 | 2.637 | 1.284 (48,7 %) | sobre el umbral de reporte |

  De las 3.071 cajas de la banda, **30 pueden llegar a la capa de identidad**: el 1 %. Contra 1.299
  observaciones del corrido base, eso no mueve un candidato ni podría. Y el seguimiento de todo lo
  demás empeora al alimentarle la banda: con los umbrales del vuelo asigna **860** pistas contra las
  **1.608** del piso en 0,25, así que tampoco hay un A/B limpio, porque la línea base cambia con el
  cambio.

  **El detalle de diseño que sale de ahí:** el piso de la banda y el piso del seguidor tienen que
  coincidir y no coinciden. El replay carga la banda desde 0,10 y `OnboardCamera` la pide desde 0,20,
  que es justo el `track_low_thresh`. Pedirle al detector cajas por debajo del piso del seguidor es
  pagar OSNet (~33 ms por caja) por cajas que se descartan sin mirarlas.
