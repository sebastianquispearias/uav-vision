# Correr la misma misión en el dron real

El comportamiento es **el mismo archivo** que en simulación:
`mi_mision_cuadrado.py`. No se toca.

Lo único que cambia son los **dos argumentos** de `construir()`:
la cámara y de dónde sale el rumbo.

---

## 1. El archivo que va a la Raspberry

```python
# mision_cuadrado_real.py
from uav_vision.camera import OnboardCamera
from uav_vision.vision_protocol import UavApiYaw
from mi_mision_cuadrado import construir

MiProtocolo = construir(
    camara=OnboardCamera(
        model="/home/pi/modelos_visdrone/y960_ncnn_model",
        threshold=0.25,
        tracker=True,
        reid_model="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=3.0,
        crops=True,
    ),
    yaw_source=UavApiYaw("http://localhost:8000"),   # rumbo REAL del Pixhawk
    velocidad=5.0,
)
```

Comparalo con `correr_sim.py`:

| | simulación | dron |
|---|---|---|
| cámara | `SimulatedCamera(target=..., pitch_deg=-55)` | `OnboardCamera(model=...)` |
| rumbo | `lambda: 0.0` | `UavApiYaw("http://localhost:8000")` |
| comportamiento | `mi_mision_cuadrado.py` | **el mismo archivo** |

---

## 2. Copiar los dos archivos a la Pi

```bash
scp examples/mi_mision_cuadrado.py    pi@192.168.1.120:~/gradys_protocols/
scp examples/mision_cuadrado_real.py  pi@192.168.1.120:~/gradys_protocols/
```

> El runner carga los protocolos desde `~/gradys_protocols/`, **no** desde
> `~/uav_vision/scripts/`. Copiarlo al sitio equivocado hace que el runner
> no lo vea.

---

## 3. Que estén corriendo los dos servicios

`mavlink-routerd` arranca solo con la Pi (lo gestiona Rpanion). Los otros
dos van a mano:

```bash
ssh pi@192.168.1.120
date                                    # el reloj llega atrasado SIEMPRE

cd ~/uav_api && setsid nohup python3 -m uav_api.run_api \
    --connection_type udpin --uav_connection 127.0.0.1:14552 \
    --sysid 3 --port 8000 > ~/uav_api_udp.log 2>&1 < /dev/null &

setsid nohup env PYTHONPATH=~/gradys-embedded python3 -m gradys_embedded.runner.cli \
    --config ~/runner_pi.toml > ~/runner.log 2>&1 < /dev/null &
```

El puerto es **14552**, no 14540: el 14540 lo reserva Rpanion y al
reiniciar gana la carrera, matando al `uav_api` con `Errno 98`.

---

## 4. Las cuatro peticiones

```bash
curl -X POST localhost:8100/mission/load -H "Content-Type: application/json" \
  -d '{"protocol":"mision_cuadrado_real:MiProtocolo",
       "initial_position":[0,0,30],
       "origin_gps_coordinates":[-22.9793,-43.2325,0],
       "x_axis_degrees":0,
       "node_ip_dict":{"1":"192.168.1.120:8200","2":"192.168.1.119:8300"},
       "communication_protocol":"http",
       "label":"cuadrado"}'

curl -X POST localhost:8100/mission/setup     # ⚠️ ARMA Y DESPEGA
curl -X POST localhost:8100/mission/start     # empieza a llamar al protocolo
curl -X POST localhost:8100/mission/stop      # lo apaga y manda RTL (aterriza)
```

`node_ip_dict` **no es opcional**: sin él el runner contesta `400 Bad Request`
con *"node_ip_dict is required"* y la misión ni se carga. El `1` es la propia
Pi (su puerto de datos) y el `2` es la estación de tierra en la laptop.

**No hay ningún `main`.** El runner ya estaba encendido; estas cuatro
peticiones son todo.

---

## 5. Si `mission/setup` no arma

Desde la laptop, para ver el motivo:

```bash
ssh pi@192.168.1.120 "timeout 80 python3 ~/prearm.py"
```

Dejarlo correr los 80 s: ArduPilot emite los `PreArm:` cada ~31 s, y una
escucha corta puede caer entre dos ráfagas y dar cero sin que nada esté mal.

Lo que dio en banco:

```
PreArm: Hardware safety switch      <- el botón redondo del módulo GPS, sin pulsar
PreArm: Check mag field: 158, ...   <- probablemente hierro bajo techo
PreArm: GPS 1: Bad fix              <- esperable bajo techo
```

También se ve en QGroundControl desde el celular: conectarse al WiFi
`rpanion` (clave `rpanion123`) y en QGC añadir un enlace UDP con
*Server Address* `10.0.2.100:14550`.

---

## Y la estación de tierra, en la laptop

```bash
cd lac/uav_vision/scripts/banco_embedded
python gs_mapa.py --puerto 8300 "--origen=-22.9793,-43.2325"
```

El `=` de `--origen` no es opcional: sin él, argparse se come el signo menos.
Abrir `http://localhost:8300`.

**Una sola `gs_mapa.py` a la vez.** En Windows un segundo proceso se queda con
el mismo puerto 8300 sin quejarse (`allow_reuse_address` de `HTTPServer`), y las
peticiones caen en cualquiera de los dos: se ve el estado de la corrida de ayer
y parece que el dron no reporta. Comprobar antes con `netstat -ano | findstr :8300`
y matar por PID lo que sobre.

**Sin POI, la estación no imprime nada y `/estado` dice `reportes: 0`.** No es un
fallo: un mensaje con `latido` actualiza la ficha del dron y vuelve sin tocar el
histórico. Lo que hay que mirar es `drones` -- si `frames_seen` sube y `fps_real`
está en 3.0, la cadena entera funciona aunque no haya nadie que ver.
