# Running the same mission on the real drone

The behaviour is **the same file** as in simulation: `mi_mision_cuadrado.py`.
It is not touched.

The only thing that changes is the **two arguments** of `construir()`:
the camera and where the heading comes from.

---

## 1. The file that goes to the Raspberry

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
    yaw_source=UavApiYaw("http://localhost:8000"),   # REAL heading from the Pixhawk
    velocidad=5.0,
)
```

Compare it with `correr_sim.py`:

| | simulation | drone |
|---|---|---|
| camera | `SimulatedCamera(target=..., pitch_deg=-55)` | `OnboardCamera(model=...)` |
| heading | `lambda: 0.0` | `UavApiYaw("http://localhost:8000")` |
| behaviour | `mi_mision_cuadrado.py` | **the same file** |

---

## 2. Copy the two files to the Pi

```bash
scp examples/mi_mision_cuadrado.py    pi@192.168.1.120:~/gradys_protocols/
scp examples/mision_cuadrado_real.py  pi@192.168.1.120:~/gradys_protocols/
```

> The runner loads protocols from `~/gradys_protocols/`, **not** from
> `~/uav_vision/scripts/`. Copying it to the wrong place makes the runner
> not see it.

---

## 3. Have both services running

`mavlink-routerd` starts with the Pi on its own (Rpanion manages it). The other
two go by hand:

```bash
ssh pi@192.168.1.120
date                                    # the clock comes up wrong ALWAYS

cd ~/uav_api && setsid nohup python3 -m uav_api.run_api \
    --connection_type udpin --uav_connection 127.0.0.1:14552 \
    --sysid 3 --port 8000 > ~/uav_api_udp.log 2>&1 < /dev/null &

setsid nohup env PYTHONPATH=~/gradys-embedded python3 -m gradys_embedded.runner.cli \
    --config ~/runner_pi.toml > ~/runner.log 2>&1 < /dev/null &
```

The port is **14552**, not 14540: 14540 is reserved by Rpanion and on a restart
it wins the race, killing `uav_api` with `Errno 98`.

---

## 4. The four requests

```bash
curl -X POST localhost:8100/mission/load -H "Content-Type: application/json" \
  -d '{"protocol":"mision_cuadrado_real:MiProtocolo",
       "initial_position":[0,0,30],
       "origin_gps_coordinates":[-22.9793,-43.2325,0],
       "x_axis_degrees":0,
       "node_ip_dict":{"1":"192.168.1.120:8200","2":"192.168.1.119:8300"},
       "communication_protocol":"http",
       "label":"cuadrado"}'

curl -X POST localhost:8100/mission/setup     # WARNING: ARMS AND TAKES OFF
curl -X POST localhost:8100/mission/start     # starts calling the protocol
curl -X POST localhost:8100/mission/stop      # shuts it down and sends RTL (lands)
```

`node_ip_dict` **is not optional**: without it the runner answers `400 Bad Request`
with *"node_ip_dict is required"* and the mission does not even load. The `1` is the
Pi itself (its data port) and the `2` is the ground station on the laptop.

**There is no `main`.** The runner was already up; these four requests are all of it.

---

## 5. If `mission/setup` does not arm

From the laptop, to see the reason:

```bash
ssh pi@192.168.1.120 "timeout 80 python3 ~/prearm.py"
```

Let it run the full 80 s: ArduPilot emits the `PreArm:` messages about every 31 s,
and a short listen can fall between two bursts and report zero with nothing
actually wrong.

What it gave on the bench:

```
PreArm: Hardware safety switch      <- the round button on the GPS module, unpressed
PreArm: Check mag field: 158, ...   <- probably iron indoors
PreArm: GPS 1: Bad fix              <- expected indoors
```

It can also be seen in QGroundControl from the phone: join the WiFi network
`rpanion` (key `rpanion123`) and in QGC add a UDP link with
*Server Address* `10.0.2.100:14550`.

---

## And the ground station, on the laptop

```bash
cd lac/uav_vision/scripts/banco_embedded
python gs_mapa.py --puerto 8300 "--origen=-22.9793,-43.2325"
```

The `=` in `--origen` is not optional: without it, argparse eats the minus sign.
Open `http://localhost:8300`.

**One `gs_mapa.py` at a time.** On Windows a second process takes the same port
8300 without complaining (`allow_reuse_address` on `HTTPServer`), and requests land
on either of the two: you see the state of yesterday's run and it looks as if the
drone is not reporting. Check first with `netstat -ano | findstr :8300` and kill by
PID whatever is left over.

**With no POI the station prints nothing and `/estado` says `reportes: 0`.** That is
not a failure: a message carrying `latido` updates the drone's card and returns
without touching the history. What to look at is `drones` -- if `frames_seen` is
rising and `fps_real` sits at 3.0, the whole chain works even with nobody to see.
