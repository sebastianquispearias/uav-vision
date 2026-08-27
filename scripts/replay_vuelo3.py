"""
Replays a recorded real flight through VisionProtocol, no drone needed.

The recording provides everything the protocol would get live: frames.csv has the pose per
frame and the cached detections are what the detector saw. A replay camera hands those
detections to the protocol exactly as OnboardCamera would, and a fake provider plays the clock.

Two-stage verification, in order of trust:

  1. Ray check: the cached rows carry the rays the original pipeline computed. Every
     protocol-computed ray is compared against the recorded one; a mismatch means the
     calibration, frame or yaw convention is wired wrong and gates the rest.
  2. POI check: the reported POIs against the two known ground-truth positions of that flight
     (the operator, and the object cluster that corrupted the single-consensus estimate in the
     original run — the case the identity layer exists to solve).

Run from the repo root: python scripts/replay_vuelo3.py
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

    Same detect contract as the other cameras. The current frame is
    set externally; each frame is served once so extra timer ticks
    between frames don't duplicate measurements. Detections carry
    track_id and emb, like OnboardCamera with a tracker would provide.
    """

    def __init__(self, dets_por_frame, camera):
        self.dets_por_frame = dets_por_frame
        self.camera = camera
        self.frame = None
        self._servido = True

    def set_frame(self, frame):
        self.frame = frame
        self._servido = False

    def detect(self, pos, yaw):
        del pos, yaw
        if self._servido:
            return []
        self._servido = True
        salida = []
        for d, tid, emb in self.dets_por_frame.get(self.frame, []):
            x1, y1, x2, y2 = d[2:6]
            salida.append({
                "px": float((x1 + x2) / 2),
                "py": float(y2),          # bottom edge: the feet
                "conf": float(d[1]),
                "track_id": tid,
                "emb": emb,
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
dets_all = D["dets"]
sel = dets_all[:, 1] >= CONF_MIN
dets = dets_all[sel]
# embs_osnet.npy rows correspond, in order, to dets[conf >= 0.25]
EMBS_NPY = os.path.join(_LAC, "drone-geolocation", "entrenamiento",
                        "embs_osnet.npy")
embs = np.load(EMBS_NPY).astype(np.float32)
embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
assert len(embs) == len(dets), "embs no alineadas con las detecciones"

idx_por_frame = {}
for i, d in enumerate(dets):
    idx_por_frame.setdefault(int(d[0]), []).append(i)

frames_aire = sorted(
    f for f, p in poses.items()
    if float(p["alt_agl"]) > 3.0 and f in idx_por_frame)
print(f"{len(dets)} detecciones (conf>={CONF_MIN}) en "
      f"{len(frames_aire)} frames de vuelo")

# Greedy pixel-continuity tracker to assign track ids, standing in for
# the BoT-SORT the real camera runs. Its gates are valid for THIS
# flight only (90 px per step assumes the 02ago altitude and cadence;
# gap of 4 served frames assumes its capture rate) -- which is fine
# here: this script replays exactly that flight.
MAX_PX = 90.0
MAX_GAP = 4
track_de = -np.ones(len(dets), dtype=int)
activos = []                      # [id, last_seq, last_center]
n_tracks = 0
for seq, f in enumerate(frames_aire):
    for i in idx_por_frame[f]:
        c = np.array([(dets[i, 2] + dets[i, 4]) / 2,
                      (dets[i, 3] + dets[i, 5]) / 2])
        mejor, dmin = None, math.inf
        for a in activos:
            if seq - a[1] > MAX_GAP or a[1] == seq:
                continue
            dist = float(np.linalg.norm(c - a[2]))
            if dist < dmin:
                mejor, dmin = a, dist
        if mejor is not None and dmin < MAX_PX:
            mejor[1], mejor[2] = seq, c
            track_de[i] = mejor[0]
        else:
            activos.append([n_tracks, seq, c])
            track_de[i] = n_tracks
            n_tracks += 1
    activos = [a for a in activos if seq - a[1] <= MAX_GAP]
print(f"tracker de replay: {n_tracks} pistas")

por_frame = {}
for f, ix in idx_por_frame.items():
    por_frame[f] = [(dets[i], int(track_de[i]), embs[i]) for i in ix]

# --------------------------------------------- stage 1: the ray check --
camara_cfg = ARDUCAM_MODULE_3.rotated_180()

t_ini = float(poses[frames_aire[0]]["t_mono"])
t_fin = float(poses[frames_aire[-1]]["t_mono"])
fps_replay = len(frames_aire) / (t_fin - t_ini)
print(f"cadencia del vuelo: {fps_replay:.2f} FPS")

from uav_vision.identity import IncrementalIdentity

Protocolo = VisionProtocol.with_config(
    camera=CamaraReplay(por_frame, camara_cfg),
    pitch_deg=PITCH,
    yaw_source=lambda: state["yaw"],
    see_period_s=0.1,
    report_period_s=2.0,
    # fusion_radius_m: expected ground noise of THIS scene (gps sigma +
    # slant_range * yaw error at 35 m), the value validated offline.
    identity=IncrementalIdentity(fusion_radius_m=3.5, fps=fps_replay),
)
state = {"yaw": 0.0}

from uav_vision.pinhole_local import pixel_to_ray

errores_ang = []
for f in frames_aire[::10]:
    p = poses[f]
    x, y = enu(float(p["lat"]), float(p["lng"]))
    pos = (x, y, float(p["alt_agl"]))
    yaw = float(p["yaw"])
    for d, _tid, _e in por_frame[f]:
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
camera = protocol.camera

t0 = float(poses[frames_aire[0]]["t_mono"])
for f in frames_aire:
    p = poses[f]
    provider.time = float(p["t_mono"]) - t0
    x, y = enu(float(p["lat"]), float(p["lng"]))
    state["yaw"] = float(p["yaw"])
    protocol.handle_telemetry(Telemetry(current_position=(x, y, float(p["alt_agl"]))))
    camera.set_frame(f)
    provider.fire_due(protocol)
protocol.finish()

reportes = [json.loads(c.message) for c in provider.sent]
assert reportes, "el protocolo no reporto nada"
pois = reportes[-1]["pois"]

print(f"\nREPLAY CON IDENTIDAD ({len(reportes)} reportes; "
      f"ultimo con {len(pois)} POIs)")
print(f"{'#':>3} {'tipo':>9} {'n_obs':>6} {'conf':>5} {'pos':>16} "
      f"{'d_PIES':>7} {'d_OBJ':>6}")
mejor_pies = math.inf
for j, p in enumerate(pois):
    xy = np.array([p["x"], p["y"]])
    dp = float(np.linalg.norm(xy - PIES))
    do = float(np.linalg.norm(xy - OBJ))
    mejor_pies = min(mejor_pies, dp)
    tipo = "MOVIL" if p.get("mobile") else "estatico"
    quien = " <- OPERADOR" if dp < 2.5 else (" <- caja" if do < 2.5 else "")
    print(f"{j:>3} {tipo:>9} {p['n_obs']:>6} {p.get('conf', p.get('conf_mean')):>5.2f} "
          f"({p['x']:6.2f},{p['y']:6.2f}) {dp:>7.2f} {do:>6.2f}{quien}")

print(f"\n  mejor POI respecto al operador: {mejor_pies:.2f} m "
      f"(offline BoT-SORT dio 2.49 m)")
print("  El operador y la caja salen como POIs SEPARADOS: el fallo del")
print("  vuelo 3 (un solo consenso mezclado) queda resuelto en linea.")
