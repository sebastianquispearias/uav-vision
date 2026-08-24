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

| Campo | Qué es |
|---|---|
| `px` | centro horizontal de la detección, en píxeles |
| `py` | **borde inferior**, no el centro — el punto donde el objeto toca el suelo |
| `conf` | confianza de la detección, en (0, 1] |

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

**1. Punto principal.** Los `run_info.json` de los tres vuelos registran
`principal_point = [945.7, 547.1]` (y `[973.3, 531.9]` para el montaje
girado 180°). Pero `CameraConfig.image_center` lo calcula como
`(ancho/2, alto/2) = (960, 540)`, y `pinhole_local.project_to_pixel` hace lo
mismo internamente. O sea: **el dron usa un punto principal calibrado y la
simulación usa el centro geométrico.** Son ~14 px de diferencia. Falta
decidir si `CameraConfig` lleva `cx`/`cy` explícitos.

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
