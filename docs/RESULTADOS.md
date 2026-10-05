# Qué hace el sistema, medido

Todo lo de acá está medido sobre datos propios y se puede reproducir con los comandos indicados.
Lo que no está medido dice "sin medir" en vez de una estimación.

**La advertencia que va con todo el documento:** salvo donde se indique otra cosa, los números salen
de **un vuelo, un día, un lugar y siete personas** (el vuelo del 2 de agosto). No hay una segunda
evaluación que los confirme.

## La métrica que importa: personas y fantasmas

El recall y la precisión por caja no dicen si el sistema sirve. Lo que sufre un operador es cuántas
personas reales aparecen en su mapa y cuántos puntos falsos tiene que ir a descartar. Esa métrica se
construyó el 16sep con las letras que el usuario asignó caja por caja.

| | detector que vuela |
|---|---|
| personas reportadas | **5 de 7** |
| fantasmas | **6** |
| error del mejor POI | **2,29 m** |
| candidatos formados | 15 |

Esos son los números del vuelo tal como quedó grabado, que es lo que trae el repositorio y lo que
corre quien lo clone. Hasta el 17sep este documento publicaba 2 fantasmas y 1,92 m: esa fila no salía
del vuelo sino de una **re-detección offline** de los mismos fotogramas, con 2655 cajas en vez de las
2637 que se grabaron. Se reproduce, y está más abajo junto a la adaptación y las fichas, pero no es
lo que voló ni lo que sale al correr este repositorio.

Las dos personas que faltan (D y E) tienen 2 y 5 cajas en todo el vuelo: no se encuentran **ni con un
detector perfecto**, porque el umbral de evidencia las descarta a propósito.

Se reproduce con `scripts/personas_encontradas.py`, y `tests/test_personas_encontradas.py` lo fija.

## De quién es cada problema

Experimento del detector perfecto: se alimentó la cadena solo con las cajas que el usuario marcó como
personas reales, y se miró qué seguía fallando.

| | detector real | detector perfecto |
|---|---|---|
| fantasmas | 4 | **0** |
| el operador aparece en | 3 puntos | **1 punto** |
| G y H aparecen en | 2 puntos cada una | **2 puntos cada una** |

- **Los fantasmas son 100 % del detector.** Con cajas perfectas desaparecen.
- **La fragmentación del operador también.** La causan cajas duplicadas.
- **Que G y H se partan en dos sobrevive al detector perfecto**: eso sí es del algoritmo, y la vía
  sería re-unir por apariencia tras un hueco. Nunca se intentó.
- **El error en metros no lo mueve ninguno de los dos.** Lo pone la geometría (GPS y yaw).

## El detector

Línea base del modelo que vuela (yolo26n VisDrone, imgsz 960, conf 0.25, IoU 0.3), contra la verdad
corregida de 1851 cajas:

```
2551-2641   recall 98.9 %   precision 85.4 %
2746-2952   recall 76.1 %   precision 61.1 %
3000-3700   recall 41.8 %   precision 55.5 %   <- el balcon, a 24 m: donde falla
TOTAL       recall 54.3 %   precision 60.9 %
```

**El tamaño de la persona explica casi todo**, y no sigue la óptica porque la cámara mira adelante y
abajo:

```
a  3- 8 m: 183 px    a  8-12 m: 168 px    a 12-18 m: 109 px
a 18-30 m:  62 px    a 30-99 m:  42 px
```

## Un detector que no puede volar encuentra el doble

RF-DETR con fichas, corriendo en tierra, sobre 201 frames del balcón:

```
YOLO26 a bordo    46.2 % de recall, 59.0 % de precision,   35 ms/frame
RF-DETR en tierra 90.5 % de recall, 63.4 % de precision, 1410 ms/frame
```

Es el número más grande que tenemos y no requiere entrenar nada. RF-DETR **nunca va a volar** (40x más
lento), pero puede correr en tierra sobre un cuadro que el dron mande bajo demanda, que ya está cableado.

Video: `docs/yolo26_vs_rfdetr.mp4`.

## Mejoras medidas, sin reentrenar

| Mejora | Qué da | Estado |
|---|---|---|
| CLIP como segundo juez | fantasmas 2 → 1, sin perder personas | en la estación, opcional con `--clip-descarta` |
| Cuadro entero + fichas cada 5 frames | recall 55,0 → 57,3 total y 39,5 → 42,4 en el balcón, misma precisión, pero **ni una persona más** en el marcador de producto | en `camera.py`, apagado por defecto |
| Umbral bajo solo en la ventana del objetivo | recall 43,9 → 60,7 % sobre el objetivo **contra cajas, pero cero cambio en personas y fantasmas**: las mismas 5 de 7 y los mismos 6, en 4 variantes | en `vision_protocol.py` (`fix_target()`, `FOCO_RADIO_PX=320`), lo dispara el clic del operador |

El umbral de las fichas (0,55) se eligió en los vuelos del 01ago, no en el test, y validación eligió
el mismo valor de forma independiente.

## Adaptación en vuelo

Adaptar el detector a la escena durante el vuelo, 91 frames, 3 vueltas, tronco congelado, 89 s:

```
                                 recall OBJETIVO   recall TODOS   precision
el que vuela                            91.1 %         49.6 %       58.3 %
adaptado a la escena                    94.0 %         56.1 %       46.3 %
adaptado con clicks simulados           94.0 %         56.1 %       46.3 %
```

**Adaptar desde clicks del operador da exactamente lo mismo que desde cajas dibujadas a mano**, porque
de las 1851 cajas que salen del click, 1217 son cajas del propio detector: el operador señala a quién,
el detector ya sabía dónde estaba el borde.

Requisito sin el cual no funciona: **conservar las 10 clases de VisDrone** en el dataset. Con una sola
clase Ultralytics reinicializa la cabeza de clasificación y todo colapsa.

Video: `docs/base_vs_adaptado.mp4`.

## Las dos mejoras, juzgadas por personas y fantasmas

Medido el 17sep. Los tres caches de detecciones se regeneraron sobre **los mismos 1659 fotogramas**,
porque cambiar el conjunto cambiaría la cadencia que ve el seguidor y la comparación dejaría de ser
sobre el detector. Los tres pasan por el seguidor que vuela (BoT-SORT con compensación de movimiento)
y por la misma cadena de identidad.

```
                        base (re-deteccion)   + fichas cada 5   adaptado a la ESCENA
  cajas conf>=0.25            2655                 2678              3629  (+37 %)
  pistas / con id           44 / 1889            43 / 1899         75 / 2222
  candidatos                    13                   12                22
  personas reportadas       5 de 7               5 de 7            5 de 7
  cuales                    A B C G H            A B C G H         A B C G H
  FANTASMAS                      2                    2                 8
  sin juzgar                     2                    2                 5
  error del POI            1,92 m               2,49 m            1,12 m
  fragmentacion            A(3) G(2)            A(2) G(2)         A(3) G(1)
```

**Las fichas no pagan.** Suman 23 cajas sobre 2655, no encuentran ni una persona más, no agregan
fantasmas, y el punto sale 57 cm peor. Lo único que mejoran es que el operador aparece en dos puntos
en vez de tres. Eso cuesta seis veces el cómputo en el fotograma con fichas.

**La adaptación es un compromiso, no una mejora.** No encuentra ni una persona más, las mismas cinco
y las mismas letras; **cuadruplica los fantasmas**, de 2 a 8; y **acerca el punto un 42 %**, de 1,92 a
1,12 m. Los 12 puntos de precisión por caja que cuesta adaptar sí se convirtieron en puntos falsos que
alguien tiene que ir a descartar. Quién gana ese compromiso depende de la misión, no de la métrica.

Se reproduce con `scripts/personas_encontradas.py` sobre los tres caches, que quedaron en
`../drone-geolocation/entrenamiento/cadena_02ago/`.

## La cadena completa

```
error del mejor POI respecto al operador topografiado:  1,92 m
POIs reportados al final del tramo:                     6 (antes 15)
cajas con id de pista:                                  80 % (antes 63 %)
```

El gate del repo (`python demo/demo.py --sin-mapa`) imprime **2,39 m** y es lo que fija que la cadena
no cambió sin querer.

## Carga del operador

```
falsas confirmadas: 0 a 4 por hora    (requisito: <= 6)   CUMPLE
cola por verificar: 27,7 -> 4,0 por hora con CLIP
```

## Lo que NO está medido

- El costo en la Pi de todo lo nuevo (fichas, cuadro bajo demanda, adaptación).
- Nada de lo nuevo corriendo en vuelo real.
- Cualquier cosa en un segundo vuelo, otro día, otro lugar.
