# uav_vision

Detección y geometría de cámara para geolocalización de objetivos desde UAV.

La misma interfaz corre en **simulación** (GrADyS-SIM) y en el **dron real**
(Raspberry Pi + ArduCam). Este paquete no importa nada de GrADyS, del
simulador ni de MAVLink: se puede instalar solo.

---

## Manifiesto — entrada y salida

Todo el módulo se reduce a un método. Las dos clases lo exponen igual:

```python
camara.ver_alvo(pos, yaw) -> list[dict]
```

**Entrada**

| Nombre | Tipo | Qué es |
|---|---|---|
| `pos` | `(x, y, z)` float, metros | posición de la cámara, marco local ENU (x=Este, y=Norte, z=Arriba) |
| `yaw` | float, grados | rumbo. 0 = Norte, 90 = Este, sentido horario |

**Salida**

Lista de detecciones. Lista vacía = no se detectó nada en ese instante.

```python
[{'px': 960.0, 'py': 805.0, 'conf': 0.846}, ...]
```

| Campo | Obligatorio | Qué es |
|---|---|---|
| `px` | sí | centro horizontal de la detección, en píxeles |
| `py` | sí | **borde inferior**, no el centro — el punto donde el objeto toca el suelo |
| `conf` | sí | confianza de la detección, en (0, 1] |
| `emb` | **no** | huella de apariencia: 512 `float32` normalizados (OSNet) |

`emb` solo aparece si la cámara puede calcularla. Pedila siempre con
`det.get('emb')`, nunca con `det['emb']`.

`py` es el borde inferior porque la persona está parada y lo que apoya en el
piso son los pies. Si se usara el centro, el rayo apuntaría a la cintura y
cruzaría el suelo más lejos. Es lo que ya hacía `onboard.py` en el dron
(`bearing_py = y2`).

`conf` no es decorativa: alimenta `select_best_views`, que con `alpha=0.2` le
da el 80 % del peso. Sin ella el selector del paper no funciona.

---

## Uso

```python
from uav_vision.camera import CamaraSimulada, CamaraArduCam

# simulacion
camara = CamaraSimulada(alvo=(0.0, 0.0, 0.0), pitch_deg=-55.0)

# dron real
camara = CamaraArduCam(modelo="/home/pi/yolov8s.pt", umbral=0.3)

# de aca para abajo el codigo es identico
for det in camara.ver_alvo(pos, yaw):
    ...
```

Los constructores son distintos a propósito — la simulada necesita saber
dónde está el objetivo, la real necesita un modelo. Lo único que tiene que
coincidir es `ver_alvo`.

`CamaraArduCam` importa `picamera2` y `ultralytics` recién en la primera
foto, así que este paquete se puede importar en una laptop que no los tenga.

Dos detalles con historia detrás:

- **`clases`** (no `clase`): el filtro acepta un conjunto de nombres y por
  defecto incluye `person` (COCO), `pedestrian` y `people` (VisDrone). Con
  un solo nombre, cambiar del modelo COCO al fine-tune VisDrone hacía que
  el filtro descartara *todas* las detecciones en silencio.
- **`ruido_pixel`** en `CamaraSimulada` (por defecto encendido): aplica
  `sigma = C / conf` al píxel, el modelo del paper. Sin ruido la simulación
  es geométricamente perfecta y RANSAC no tiene nada que rechazar.

---

## La huella (`emb`)

512 números que resumen cómo se ve un recorte. No es la imagen: no se puede
reconstruir la foto desde ahí. Sirve para decir *"estas dos detecciones son la
misma cosa"* sin mirar la posición.

Medido sobre el vuelo del 02ago (coseno, 1.0 = idénticas):

| | parecido |
|---|---|
| misma identidad, distintas fotos | 0.63 – 0.87 |
| entre identidades distintas | 0.33 – 0.46 |

No se solapan. Por eso separa al operador de los falsos positivos estáticos
—la caja blanca del equipo, con 930 observaciones— que en el vuelo 3 le
robaron el consenso a RANSAC y produjeron 4-5 m de error.

**Se calcula en la cámara** (decisión del 23ago2026). Motivo: así la imagen
nunca sale del módulo. Lo que puede viajar por radio son 512 `float32`
(2 048 B), o 128 B comprimiendo con PCA int8 — el resultado de la tesis de
handoff, que iguala al descriptor completo con 16× de compresión.

Mismo modelo que `entrenamiento/rehuella_osnet.py`: `boxmot` + OSNet
`osnet_x0_25_msmt17`, CPU, un batch de cajas por imagen, vector normalizado.

### A futuro (pendiente de medir)

Dos alternativas quedan abiertas, ninguna descartada:

1. **Mandar el recorte** en vez de la huella, y calcularla en tierra. Descarga
   a la Raspberry pero mueve imágenes por radio.
2. **Mandar la huella comprimida con PCA int8** (128 B). Es lo que la tesis de
   handoff ya validó offline; falta probarlo en el enlace real.

El costo de OSNet en la Raspberry **no está medido**. Hasta que corra
`revision2/bench_rpi.py`, `reid_modelo` es opcional y por defecto está apagado.

---

## Instalación

```bash
pip install -e .              # laptop / simulacion
pip install -e ".[dron]"      # + ultralytics (picamera2 ya viene en Raspberry Pi OS)
```

Prueba del contrato:

```bash
python tests/test_contrato.py
```

---

## Contenido

| Archivo | Qué es |
|---|---|
| `camera.py` | las dos cámaras. El contrato |
| `pinhole_local.py` | píxel ↔ rayo. Matemática pura, sin dependencias |
| `camera_config.py` | intrínsecos de cada cámara |
| `confidence.py` | modelo de confianza para simulación |
| `vision_protocol.py` | el protocolo GrADyS: cámara adentro, POIs afuera. **Único archivo que importa `gradys_embedded`** |

---

## El protocolo (`VisionProtocol`)

Es un protocolo GrADyS **solo-observador**: nunca manda comandos de
movilidad, así que convive con cualquier protocolo de misión. Cada 200 ms
(5 Hz, la tasa segura en la Raspberry) llama `camara.ver_alvo(pos, yaw)`,
convierte cada detección en rayo, cruza el rayo con el suelo y guarda el
impacto. Cada 2 s corre RANSAC sobre los impactos acumulados y transmite
el POI dominante como JSON (`BroadcastMessageCommand`).

- **posición**: llega sola por `handle_telemetry` (marco local de GrADyS).
- **yaw**: la `Telemetry` de GrADyS no lo trae. En el dron real lo da
  `UavApiYaw`, que consulta el uav_api por HTTP en localhost
  (`GET /telemetry/general`, campo `heading`) — el uav_api es el único
  dueño del puerto serial MAVLink. En simulación se inyecta una función.
- **configuración**: `instantiate()` de GrADyS llama `cls()` sin
  argumentos, así que la configuración va por
  `VisionProtocol.with_config(camera=..., pitch_deg=..., yaw_source=...)`,
  que devuelve una clase lista para dársela al runner.

Prueba de punta a punta (no necesita el simulador instalado; usa un
proveedor falso y encuentra `../gradys-embedded` solo):

```bash
python tests/test_vision_protocol.py
```

Limitación deliberada por ahora: todos los impactos van a UN solo RANSAC,
o sea el protocolo reporta el POI dominante (la persona más observada).
La capa de identidad incremental (multi-POI, móviles con trayectoria,
huellas) es el siguiente paso.

---

## Por qué existe este repo

La matemática de `pixel_to_ray` estaba escrita en cuatro lugares distintos
(`onboard/onboard.py`, `ground_station/pipeline_offline.py`,
`real_flight_14jun/rpi/onboard.py`, y el showcase del simulador) con firmas
distintas y constantes distintas. Una copia con el pitch equivocado produce
**6.287 m** de error en la posición estimada — más grande que el error total
que reporta el paper (2.34 m).

Con un solo módulo importado por todos, esa divergencia no puede ocurrir.

---

## Pendientes conocidos

**1. Punto principal — RESUELTO (24ago2026).** `CameraConfig` lleva
`calibrated_principal_point` (el de la ArduCam es `[945.7, 547.1]`, de los
`run_info.json`), y `rotated_180()` lo refleja a `[973.3, 531.9]` para el
montaje girado — la misma fórmula `tamaño - 1 - c` que aplica `onboard.py`.
`CamaraArduCam(rot180=True)` hace la reflexión sola.

**2. Calibración con tablero de ajedrez.** `focal_px = 1407.0` está tomado de
los `run_info.json` de los vuelos, no de una calibración formal.

**3. `CamaraArduCam` nunca se probó en vuelo.** Los tres vuelos corrieron con
`detect_onboard = False`: YOLO se ejecutó en tierra, no a bordo. Falta el
test de banco en la Raspberry (`revision2/bench_rpi.py`), bloqueado por falta
de una fuente de 5 V / 5 A.

---

## Valores de referencia (de los vuelos reales)

| Vuelo | `focal_px` | `principal_point` | `camera_pitch_deg` |
|---|---|---|---|
| 26jul | 1407.0 | 945.7, 547.1 | −45.0 |
| 01ago | 1407.0 | 945.7, 547.1 | −45.0 |
| 02ago | 1407.0 | 973.3, 531.9 | −55.0 |

El montaje cambió el 02ago2026: la ArduCam se giró 180° sobre el eje óptico
y el pitch pasó de −45 a −55. Por eso `pitch_deg` **no tiene valor por
defecto** en `CamaraSimulada` — quien crea la cámara declara el que usó.
