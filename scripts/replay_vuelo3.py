"""Replay flight 3 (02ago2026) through VisionProtocol, no drone needed.

The recorded flight provides everything the protocol would get live:
frames.csv has the pose per frame (lat/lng/alt/yaw) and the cached
detections (examen_v3_datos.npz) are what the detector saw. A replay
camera hands those detections to the protocol exactly as CamaraArduCam
would, and a fake provider plays the clock. The protocol then does its
real job: pixel -> ray -> ground impact -> RANSAC -> POI report.

Two-stage verification, in order of trust:

  1. RAY CHECK: the npz rows carry the rays the real pipeline computed
     during analysis. For every detection we compare the protocol's ray
     against the recorded one. If the angles don't match, the frames,
     the calibration or the yaw convention are wired wrong and the POI
     means nothing -- so this check gates the rest.

  2. POI CHECK: the final reported POI against the two known ground
     truths: PIES = (-1.3, 8.8) is the operator, OBJ = (2.5, 4.4) is
     the equipment box that stole RANSAC's consensus in the original
     flight-3 failure. The protocol currently fuses everything into ONE
     POI, so an honest replay is expected to reproduce that failure --
     that is the motivation for the incremental identity layer, not a
     surprise.

Run from lac/uav_vision:   python scripts/replay_vuelo3.py
Needs ../gradys-embedded and ../drone-geolocation next to this repo.
"""
import csv
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
_LAC = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_LAC, "gradys-embedded"))

import numpy as np

from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.vision_protocol import VisionProtocol

FLIGHT = os.path.join(_LAC, "drone-geolocation", "data", "flight_02ago",
                      "20260802_133309")
DETS_NPZ = os.path.join(_LAC, "drone-geolocation", "entrenamiento",
                        "examen_v3_datos.npz")

# ENU origin: the GT post used by every flight-3 analysis.
LAT0, LNG0 = -22.978029946, -43.23214256266666
R_EARTH = 6378137.0

PIES = np.array([-1.3, 8.8])      # operator (ground truth)
OBJ = np.array([2.5, 4.4])        # equipment box (the flight-3 thief)

PITCH = -55.0
CONF_MIN = 0.25                   # same cut the identity analysis uses


def enu(lat, lng):
    x = math.radians(lng - LNG0) * R_EARTH * math.cos(math.radians(LAT0))
    y = math.radians(lat - LAT0) * R_EARTH
    return x, y


class CamaraReplay:
    """Feeds recorded detections to the protocol, one frame at a time.

    Same ver_alvo contract as the other cameras. The current frame is
    set externally; each frame is served once so extra timer ticks
    between frames don't duplicate measurements.
    """

    def __init__(self, dets_por_frame, camara):
        self.dets_por_frame = dets_por_frame
        self.camara = camara
        self.frame = None
        self._servido = True

    def set_frame(self, frame):
        self.frame = frame
        self._servido = False

    def ver_alvo(self, pos, yaw):
        del pos, yaw
        if self._servido:
            return []
        self._servido = True
        salida = []
        for d in self.dets_por_frame.get(self.frame, []):
            x1, y1, x2, y2 = d[2:6]
            salida.append({
                "px": float((x1 + x2) / 2),
                "py": float(y2),          # bottom edge: the feet
                "conf": float(d[1]),
            })
        return salida


class FakeProvider:
    def __init__(self):
        self.time = 0.0
        self.timers = []
        self.sent = []

    def schedule_timer(self, timer, timestamp):
        self.timers.append((timestamp, timer))

    def cancel_timer(self, timer):
        self.timers = [t for t in self.timers if t[1] != timer]

    def send_communication_command(self, command):
        self.sent.append(command)

    def current_time(self):
        return self.time

    def get_id(self):
        return 3

    tracked_variables = {}

    def fire_due(self, protocol):
        due = sorted(t for t in self.timers if t[0] <= self.time)
        self.timers = [t for t in self.timers if t[0] > self.time]
        for _, name in due:
            protocol.handle_timer(name)


# ---------------------------------------------------------------- data --
poses = {}
with open(os.path.join(FLIGHT, "frames.csv")) as f:
    for r in csv.DictReader(f):
        poses[int(r["frame"])] = r

D = np.load(DETS_NPZ)
dets = D["dets"]
dets = dets[dets[:, 1] >= CONF_MIN]
por_frame = {}
for d in dets:
    por_frame.setdefault(int(d[0]), []).append(d)

frames_aire = sorted(
    f for f, p in poses.items()
    if float(p["alt_agl"]) > 3.0 and f in por_frame)
print(f"{len(dets)} detecciones (conf>={CONF_MIN}) en "
      f"{len(frames_aire)} frames de vuelo")

# --------------------------------------------- stage 1: the ray check --
camara_cfg = ARDUCAM_MODULE_3.rotated_180()
Protocolo = VisionProtocol.with_config(
    camera=CamaraReplay(por_frame, camara_cfg),
    pitch_deg=PITCH,
    yaw_source=lambda: state["yaw"],
    see_period_s=0.1,
    report_period_s=2.0,
)
state = {"yaw": 0.0}

from uav_vision.pinhole_local import pixel_to_ray

errores_ang = []
for f in frames_aire[::10]:
    p = poses[f]
    x, y = enu(float(p["lat"]), float(p["lng"]))
    pos = (x, y, float(p["alt_agl"]))
    yaw = float(p["yaw"])
    for d in por_frame[f]:
        px, py = (d[2] + d[4]) / 2, d[5]
        _, dir_mio = pixel_to_ray(
            pos, yaw, (px, py), PITCH,
            camara_cfg.focal_length_px, camara_cfg.image_width,
            camara_cfg.image_height, camara_cfg.principal_point)
        dir_real = d[9:12] / np.linalg.norm(d[9:12])
        cosang = float(np.clip(np.dot(dir_mio, dir_real), -1, 1))
        errores_ang.append(math.degrees(math.acos(cosang)))

errores_ang = np.array(errores_ang)
print(f"\nCHEQUEO DE RAYOS ({len(errores_ang)} muestras, 1 de cada 10 frames)")
print(f"  angulo protocolo vs vuelo real: mediana {np.median(errores_ang):.3f} deg"
      f", p90 {np.percentile(errores_ang, 90):.3f} deg, max {errores_ang.max():.3f} deg")
if np.median(errores_ang) > 0.5:
    print("  >> DESALINEADO: no seguir hasta resolver la convencion")
    sys.exit(1)
print("  >> rayos del protocolo coinciden con los del vuelo real")

# --------------------------------------------- stage 2: the replay -----
provider = FakeProvider()
protocol = Protocolo.instantiate(provider)
protocol.initialize()
camara = protocol.camera

t0 = float(poses[frames_aire[0]]["t_mono"])
for f in frames_aire:
    p = poses[f]
    provider.time = float(p["t_mono"]) - t0
    x, y = enu(float(p["lat"]), float(p["lng"]))
    state["yaw"] = float(p["yaw"])
    protocol.handle_telemetry(Telemetry(current_position=(x, y, float(p["alt_agl"]))))
    camara.set_frame(f)
    provider.fire_due(protocol)
protocol.finish()

reportes = [json.loads(c.message) for c in provider.sent]
assert reportes, "el protocolo no reporto nada"
ultimo = reportes[-1]["pois"][0]
poi = np.array([ultimo["x"], ultimo["y"]])

d_pies = float(np.linalg.norm(poi - PIES))
d_obj = float(np.linalg.norm(poi - OBJ))
print(f"\nREPLAY ({len(reportes)} reportes, {ultimo['n_obs']} obs, "
      f"{ultimo['n_inliers']} inliers)")
print(f"  POI final: ({poi[0]:.2f}, {poi[1]:.2f})")
print(f"  distancia al operador (PIES {tuple(PIES)}): {d_pies:.2f} m")
print(f"  distancia a la caja   (OBJ  {tuple(OBJ)}): {d_obj:.2f} m")
quien = "la CAJA (fallo original reproducido)" if d_obj < d_pies else "el OPERADOR"
print(f"  >> el POI dominante es {quien}")
print("\nConclusion honesta: con UN solo RANSAC el protocolo reporta el")
print("cluster mas observado, igual que el vuelo original. La capa de")
print("identidad incremental es la que separa operador/objetos/moviles.")
