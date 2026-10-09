"""Replays a recorded real flight through VisionProtocol, no drone needed.

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

Where the inputs come from, every switch this script accepts, and how the two modes
differ: docs/ARQUITECTURA.md.
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

from uav_vision.camera import (
    BANDA_BAJA,
    EMB_DIST_OBJETIVO,
    OnboardCamera,
    dentro_del_foco,
    se_parece_al_objetivo,
)
from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.vision_protocol import VisionProtocol

_DATOS = os.environ.get("UAV_VISION_DATOS")
if _DATOS:
    FRAMES_CSV = os.path.join(_DATOS, "frames.csv")
    DETS_NPZ = os.path.join(_DATOS, "examen_v3_datos.npz")
    EMBS_NPY = os.path.join(_DATOS, "embs_osnet.npy")
else:
    FLIGHT = os.path.join(_LAC, "drone-geolocation", "data", "flight_02ago",
                          "20260802_133309")
    FRAMES_CSV = os.path.join(FLIGHT, "frames.csv")
    DETS_NPZ = os.path.join(_LAC, "drone-geolocation", "entrenamiento",
                            "examen_v3_datos.npz")
    EMBS_NPY = os.path.join(_LAC, "drone-geolocation", "entrenamiento",
                            "embs_osnet.npy")

_GS = os.environ.get("UAV_VISION_GS")

DRON = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--dron=")), 1)
PASADA = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--pasada=")), 0)

VIVO = "--vivo" in sys.argv
ROLL_SIGNO = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--actitud-roll=")), 0.0)
CON_VEHICULOS = "--vehiculos" in sys.argv or VIVO
VELOCIDAD = next((float(a.split("=")[1]) for a in sys.argv
                  if a.startswith("--velocidad=")), 20.0)
VEHICULOS_NPZ = os.path.join(_DATOS or os.path.join(_HERE, "demo", "data"),
                             "vehiculos.npz")

LAT0, LNG0 = -22.978029946, -43.23214256266666
R_EARTH = 6378137.0

PIES = np.array([-1.3, 8.8])
OBJ = np.array([2.5, 4.4])

PITCH = -55.0
CONF_MIN = 0.25


def enu(lat, lng):
    x = math.radians(lng - LNG0) * R_EARTH * math.cos(math.radians(LAT0))
    y = math.radians(lat - LAT0) * R_EARTH
    return x, y


class CamaraReplay:
    """Feeds recorded detections to the protocol, one frame at a time.

    Same detect contract as the other cameras. The current frame is
    set externally; each frame is served once so extra timer ticks
    between frames don't duplicate measurements. Detections carry
    track_id, emb and cls, like OnboardCamera with a tracker would provide.

    cls is None in the people-only replay, which is what the cached
    detections of the original run recorded: they predate the class
    reaching the report, and the protocol treats None as "no opinion".

    banda_baja holds the boxes the detector scored between the camera's floor and the reporting
    threshold and threw away. They enter only through the window the operator's verdict opens,
    which is the whole point of the fixed-target mode: no extra inference, just evidence that
    was already computed. A box is kept when it is inside the window AND above its floor, OR
    when it looks like the target, and the second is not a second window, because it does not
    ask where the box is.

    set_focus is the real camera's, so what is measured here is what would fly. set_classes is
    the real camera's behaviour too: the real one receives every class the detector knows and
    drops the ones nobody asked for, and this stand-in serves cached detections, so it has to
    do the same or the switch would be a lie -- the operator would see the drawing change while
    the sensor kept reporting everything. _ALIAS exists because this replay labels people
    "pedestrian", to keep them apart from the vehicle path, while the operator's console speaks
    the names the detector emits. A cached source whose cls is None has no OPINION on class and
    is served whatever is asked, exactly as the protocol treats a missing cls.
    """

    def __init__(self, dets_por_frame, camera, banda_baja=None):
        self.dets_por_frame = dets_por_frame
        self.banda_baja = banda_baja or {}
        self.foco = None
        self.camera = camera
        self.frame = None
        self._servido = True
        self.clases = None
        self.servidas = {}

    set_focus = OnboardCamera.set_focus

    def set_frame(self, frame):
        self.frame = frame
        self._servido = False

    _ALIAS = {"person": "pedestrian"}

    def set_classes(self, classes=None):
        if not classes:
            self.clases = None
            return
        self.clases = {self._ALIAS.get(c, c) for c in classes}

    @property
    def classes(self):
        """What is being served, in the cached detections' names; None means everything."""
        return self.clases

    def _pasa(self, cls):
        return self.clases is None or cls is None or cls in self.clases

    def detect(self, pos, yaw):
        del pos, yaw
        if self._servido:
            return []
        self._servido = True
        salida = []
        for d, tid, emb, cls in self.dets_por_frame.get(self.frame, []):
            if not self._pasa(cls):
                continue
            self.servidas[cls] = self.servidas.get(cls, 0) + 1
            x1, y1, x2, y2 = d[2:6]
            det = {
                "px": float((x1 + x2) / 2),
                "py": float(y2),
                "conf": float(d[1]),
                "track_id": tid,
                "emb": emb,
            }
            if cls is not None:
                det["cls"] = cls
            salida.append(det)
        if self.foco is not None:
            for d, emb, cls in self.banda_baja.get(self.frame, []):
                if not self._pasa(cls):
                    continue
                x1, y1, x2, y2 = d[2:6]
                det = {"px": float((x1 + x2) / 2), "py": float(y2), "conf": float(d[1]),
                       "emb": emb}
                en_ventana = (dentro_del_foco(det, self.foco)
                              and det["conf"] >= self.foco["umbral"])
                if not en_ventana and not se_parece_al_objetivo(det, self.foco):
                    continue
                if cls is not None:
                    det["cls"] = cls
                self.servidas[cls] = self.servidas.get(cls, 0) + 1
                salida.append(det)
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


poses = {}
with open(FRAMES_CSV) as f:
    for r in csv.DictReader(f):
        poses[int(r["frame"])] = r

D = np.load(DETS_NPZ)
dets_all = D["dets"]
sel = dets_all[:, 1] >= CONF_MIN
dets = dets_all[sel]
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

MAX_PX = 90.0
MAX_GAP = 4


def pistas(dets, idx_por_frame, frames, base_id=0):
    """Assigns a track id per detection by nearest-centre continuity.

    Run once per class group rather than over everything at once: a car and a
    person that pass within 90 px of each other are one track for a tracker that
    only knows pixels, and a track that changes class is exactly what the
    identity layer's class vote exists to prevent. base_id keeps the ids of
    separate runs from colliding.

    Its gates are valid for THIS FLIGHT ONLY: MAX_PX assumes the 02ago altitude and cadence,
    and MAX_GAP assumes its capture rate. That is fine here, because this script replays
    exactly that flight.

    Each entry of 'activos' is [id, last_seq, last_center].
    """
    track_de = -np.ones(len(dets), dtype=int)
    activos = []
    n = 0
    for seq, f in enumerate(frames):
        for i in idx_por_frame.get(f, []):
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
                activos.append([base_id + n, seq, c])
                track_de[i] = base_id + n
                n += 1
        activos = [a for a in activos if seq - a[1] <= MAX_GAP]
    return track_de, n


track_de, n_tracks = pistas(dets, idx_por_frame, frames_aire)
print(f"tracker de replay: {n_tracks} pistas")

PISTAS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pistas=")), None)
if PISTAS:
    track_de = np.load(PISTAS)["track"]
    assert len(track_de) == len(dets), "las pistas externas no corresponden a estas detecciones"
    n_tracks = int(track_de.max()) + 1 if (track_de >= 0).any() else 0
    print(f"pistas externas ({os.path.basename(PISTAS)}): {len(set(track_de[track_de >= 0]))} "
          f"pistas, {int((track_de >= 0).sum())}/{len(track_de)} detecciones con id")

_CLASE_PERSONA = "pedestrian" if CON_VEHICULOS else None

EVIDENCIA_MIN = next((float(a.split("=", 1)[1]) for a in sys.argv
                      if a.startswith("--evidencia-min=")), None)

FOCO = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--foco=")), None)
PLANTILLA = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--plantilla=")), None)
EMB_DIST = next((float(a.split("=", 1)[1]) for a in sys.argv
                 if a.startswith("--emb-dist=")), None)
_plantilla = np.load(PLANTILLA).astype(np.float32) if PLANTILLA else None
banda_baja = {}
if FOCO is not None:
    baja = dets_all[(dets_all[:, 1] < CONF_MIN) & (dets_all[:, 1] >= BANDA_BAJA)]
    embs_all = D["embs"].astype(np.float32)
    embs_all /= (np.linalg.norm(embs_all, axis=1, keepdims=True) + 1e-9)
    emb_baja = embs_all[(dets_all[:, 1] < CONF_MIN) & (dets_all[:, 1] >= BANDA_BAJA)]
    for i, d in enumerate(baja):
        banda_baja.setdefault(int(d[0]), []).append((d, emb_baja[i], _CLASE_PERSONA))
    print("objetivo fijado en (%s): %d cajas de la banda baja disponibles en %d frames"
          % (FOCO, len(baja), len(banda_baja)))

por_frame = {}
for f, ix in idx_por_frame.items():
    por_frame[f] = [(dets[i], int(track_de[i]) if track_de[i] >= 0 else None, embs[i],
                     _CLASE_PERSONA) for i in ix
                    if EVIDENCIA_MIN is None or dets[i, 1] >= EVIDENCIA_MIN]
if EVIDENCIA_MIN is not None:
    n_ev = sum(len(v) for v in por_frame.values())
    print(f"evidencia >= {EVIDENCIA_MIN}: {n_ev}/{len(dets)} detecciones llegan al protocolo")

if CON_VEHICULOS:
    V = np.load(VEHICULOS_NPZ)
    v_dets, v_embs, v_clases = V["dets"], V["embs"].astype(np.float32), V["clases"]
    v_embs /= (np.linalg.norm(v_embs, axis=1, keepdims=True) + 1e-9)
    v_idx = {}
    for i, d in enumerate(v_dets):
        v_idx.setdefault(int(d[0]), []).append(i)
    v_frames = sorted(
        f for f, p in poses.items()
        if float(p["alt_agl"]) > 3.0 and f in v_idx)
    v_track, v_n = pistas(v_dets, v_idx, v_frames, base_id=n_tracks)
    for f, ix in v_idx.items():
        por_frame.setdefault(f, []).extend(
            (v_dets[i], int(v_track[i]), v_embs[i], str(v_clases[i])) for i in ix)
    unicos, cuentas = np.unique(v_clases, return_counts=True)
    print(f"vehiculos: {len(v_dets)} detecciones en {len(v_idx)} frames, "
          f"{v_n} pistas, "
          + ", ".join(f"{u} {c}" for u, c in sorted(zip(unicos, cuentas),
                                                    key=lambda p: -p[1])))
    frames_aire = sorted(
        f for f, p in poses.items()
        if float(p["alt_agl"]) > 3.0 and f in por_frame)

SINTETICO_V = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--sintetico=")), None)

# --submuestreo=N keeps one frame of every N, which is how this recording answers a question
# about rates it was not recorded at. The flight was captured at 1.64 frames per second, so the
# only honest direction is DOWN: N=2 asks what the chain finds at 0.8 FPS, N=3 at 0.55. Faster
# than 1.64 cannot be simulated at all, because those frames do not exist.
#
# It is one knob and not two on purpose. The declared rate the identity layer scales every
# maturity threshold by is computed from this same list a few lines below, so thinning the list
# moves both together and they cannot drift -- which is the failure mode that had to be fixed in
# mision_banco_lab.py the same day this was written.
SUBMUESTREO = next((int(a.split("=", 1)[1]) for a in sys.argv
                    if a.startswith("--submuestreo=")), 1)
SINTETICO_ID = 900000
SINTETICO_X0, SINTETICO_X1, SINTETICO_Y = -25.0, 25.0, 25.0
posicion_sintetica = None
if SINTETICO_V is not None:
    from uav_vision.pinhole_local import project_to_pixel
    _cfg = ARDUCAM_MODULE_3.rotated_180()
    _rng = np.random.default_rng(11)
    _emb = _rng.normal(size=512).astype(np.float32)
    _emb /= np.linalg.norm(_emb)
    _aire = sorted(f for f, q in poses.items() if float(q["alt_agl"]) > 3.0)
    _t_origen = float(poses[_aire[0]]["t_mono"])

    def posicion_sintetica(t_mono):
        """Ground truth: a triangle wave between X0 and X1 along y = Y, at SINTETICO_V m/s."""
        largo = SINTETICO_X1 - SINTETICO_X0
        recorrido = (SINTETICO_V * (t_mono - _t_origen)) % (2 * largo)
        x = SINTETICO_X0 + (recorrido if recorrido <= largo else 2 * largo - recorrido)
        return x, SINTETICO_Y

    _n_vista = _n_det = 0
    for f in _aire:
        q = poses[f]
        x, y = posicion_sintetica(float(q["t_mono"]))
        dx, dy = enu(float(q["lat"]), float(q["lng"]))
        px = project_to_pixel((dx, dy, float(q["alt_agl"])), (x, y, 0.0), float(q["yaw"]), PITCH,
                              _cfg.focal_length_px, _cfg.image_width, _cfg.image_height,
                              _cfg.principal_point)
        if px is None:
            continue
        _n_vista += 1
        if _rng.random() > 0.28:
            continue
        u, v = px[0] + _rng.normal(0, 2.0), px[1] + _rng.normal(0, 2.0)
        e = _emb + 0.03 * _rng.normal(size=512).astype(np.float32)
        por_frame.setdefault(f, []).append(
            (np.array([f, 0.6, u - 10.0, v - 40.0, u + 10.0, v]), SINTETICO_ID, e / np.linalg.norm(e), "boat"))
        _n_det += 1
    frames_aire = sorted(f for f, q in poses.items() if float(q["alt_agl"]) > 3.0 and f in por_frame)
    print(f"blanco sintetico a {SINTETICO_V} m/s: en cuadro en {_n_vista} frames, detectado en {_n_det}")

camara_cfg = ARDUCAM_MODULE_3.rotated_180()

# Applied HERE and not next to any of the three places frames_aire is built, because which of
# them ran depends on the flags: this is the one point where the list is final whichever branch
# produced it. Guarded, so the default path is untouched and not merely equivalent -- [::1] is a
# copy, and the replay that prints 2.39 m is the gate this file lives under.
if SUBMUESTREO > 1:
    _antes = len(frames_aire)
    frames_aire = frames_aire[::SUBMUESTREO]
    print(f"submuestreo 1 de cada {SUBMUESTREO}: {_antes} -> {len(frames_aire)} cuadros")

t_ini = float(poses[frames_aire[0]]["t_mono"])
t_fin = float(poses[frames_aire[-1]]["t_mono"])
fps_replay = len(frames_aire) / (t_fin - t_ini)
print(f"cadencia del vuelo: {fps_replay:.2f} FPS")

from uav_vision.identity import IncrementalIdentity

Protocolo = VisionProtocol.with_config(
    camera=CamaraReplay(por_frame, camara_cfg, banda_baja),
    pitch_deg=PITCH,
    yaw_source=lambda: state["yaw"],
    attitude_source=((lambda: (ROLL_SIGNO * state["roll"], state["pitch"]))
                     if "--actitud" in sys.argv else None),
    see_period_s=0.1,
    report_period_s=2.0,
    identity=IncrementalIdentity(fusion_radius_m=3.5, fps=fps_replay,
                                 reinforce_with_fragments="--refuerzo" in sys.argv,
                                 maturity="span" if "--span" in sys.argv else "looks",
                                 report_min_looks=next((int(a.split("=", 1)[1]) for a in sys.argv
                                                        if a.startswith("--miradas-min=")), 20)),
    report_preliminary="--preliminares" in sys.argv,
)
state = {"yaw": 0.0, "roll": 0.0, "pitch": 0.0}

from uav_vision.pinhole_local import pixel_to_ray

errores_ang = []
for f in frames_aire[::10]:
    p = poses[f]
    if f not in por_frame:
        continue
    x, y = enu(float(p["lat"]), float(p["lng"]))
    pos = (x, y, float(p["alt_agl"]))
    yaw = float(p["yaw"])
    for d, _tid, _e, _c in por_frame[f]:
        if len(d) < 13:
            continue
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

provider = FakeProvider()
protocol = Protocolo.instantiate(provider)
protocol.initialize()
if FOCO is not None:
    _fx, _fy = (float(v) for v in FOCO.split(","))
    protocol.fix_target(_fx, _fy, plantilla=_plantilla,
                        emb_dist=(EMB_DIST if EMB_DIST is not None
                                  else (EMB_DIST_OBJETIVO if _plantilla is not None else None)))
camera = protocol.camera

t0 = float(poses[frames_aire[0]]["t_mono"])

if PASADA:
    ts = [float(poses[f]["t_mono"]) - t0 for f in frames_aire]
    corte = (ts[0] + ts[-1]) / 2.0
    frames_aire = [f for f, t in zip(frames_aire, ts)
                   if (t < corte) == (PASADA == 1)]
    print(f"pasada {PASADA}: {len(frames_aire)} frames "
          f"({'antes' if PASADA == 1 else 'despues'} del segundo {corte:.0f})")

def _consultar_orden():
    """Ask the station what to look for. A dead link leaves things as they are."""
    import urllib.request
    try:
        with urllib.request.urlopen(_GS.rstrip("/") + "/buscar", timeout=0.4) as r:
            d = json.loads(r.read())
    except Exception:
        return
    if not protocol.apply_search_order(d.get("clases"), d.get("v"), d.get("epoca")):
        return
    print("  [operador] segundo %.0f del vuelo: ahora se busca %s"
          % (provider.time, d.get("clases") or "(todo)"), flush=True)


_ULTIMO_AVISO = {"t": 0.0}


def _decir_frame(f):
    """Tell the bench station which frame this is, so it can show it.

    Rate limited against the wall clock, not the frame count. One POST per frame slowed the
    replay by a factor of three -- and it was wasted anyway: the panel polls four times a
    second, so telling it thirty times buys nothing. Tying it to the clock keeps the panel as
    smooth as it can be and leaves the flight running at the speed it claims.

    Bench only. A drone in the air sends coordinates, and the link could not carry pictures
    anyway; this exists so a demo can put the camera's view beside the map's reading of it.
    """
    import time as _tt
    import urllib.request
    ahora = _tt.time()
    if ahora - _ULTIMO_AVISO["t"] < 0.15:
        return
    _ULTIMO_AVISO["t"] = ahora
    try:
        urllib.request.urlopen(urllib.request.Request(
            _GS.rstrip("/") + "/frame_actual",
            data=json.dumps({"n": int(f)}).encode("utf-8"),
            headers={"Content-Type": "application/json"}), timeout=0.4).read()
    except Exception:
        pass


def _mandar_nuevos(desde):
    """Post the reports produced since `desde`; returns how many are out."""
    import urllib.request
    for c in provider.sent[desde:]:
        cuerpo = json.dumps({"message": c.message, "source": DRON}).encode("utf-8")
        pedido = urllib.request.Request(
            _GS, data=cuerpo, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(pedido, timeout=2).read()
        except Exception as e:
            print("  no se pudo enviar: %s" % e, flush=True)
            break
    return len(provider.sent)


if VIVO:
    import time as _t
    if not _GS:
        sys.exit("--vivo necesita la estacion de tierra: exporta GS_URL o usa demo.py")
    print("")
    print("EN VIVO a %gx. Abri %s y pulsa los botones de BUSCANDO."
          % (VELOCIDAD, _GS))
    _consultar_orden()
    _reloj = _t.time()
    _fuera = 0
    for i, f in enumerate(frames_aire):
        p = poses[f]
        provider.time = float(p["t_mono"]) - t0
        _retraso = provider.time / VELOCIDAD - (_t.time() - _reloj)
        if _retraso > 0:
            _t.sleep(min(_retraso, 0.25))
        if i % 10 == 0:
            _consultar_orden()
        _decir_frame(f)
        x, y = enu(float(p["lat"]), float(p["lng"]))
        state["yaw"] = float(p["yaw"])
        state["roll"], state["pitch"] = float(p["roll"]), float(p["pitch"])
        protocol.handle_telemetry(
            Telemetry(current_position=(x, y, float(p["alt_agl"]))))
        camera.set_frame(f)
        provider.fire_due(protocol)
        if len(provider.sent) > _fuera:
            _fuera = _mandar_nuevos(_fuera)
    protocol.finish()
    _mandar_nuevos(_fuera)
else:
    for f in frames_aire:
        p = poses[f]
        provider.time = float(p["t_mono"]) - t0
        x, y = enu(float(p["lat"]), float(p["lng"]))
        state["yaw"] = float(p["yaw"])
        state["roll"], state["pitch"] = float(p["roll"]), float(p["pitch"])
        protocol.handle_telemetry(
            Telemetry(current_position=(x, y, float(p["alt_agl"]))))
        camera.set_frame(f)
        provider.fire_due(protocol)
    protocol.finish()

reportes = [json.loads(c.message) for c in provider.sent]
assert reportes, "el protocolo no reporto nada"
pois = reportes[-1]["pois"]

todos = protocol.identity.candidates(preliminary=True)
maduros = sum(1 for c in todos if c.get("mature"))

_CANDS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--candidatos=")), None)
if _CANDS:
    _guardables = ("x", "y", "n_obs", "conf", "mature", "tracks", "mobile", "cls",
                   "looks", "looks_min", "evidence", "duty")
    with open(_CANDS, "w") as _f:
        json.dump([{k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
                    for k, v in c.items() if k in _guardables}
                   for c in protocol.identity.candidates(preliminary=True, with_tracks=True)],
                  _f, indent=1, default=str)
    print("candidatos con sus pistas -> %s" % _CANDS)
print("")
if VIVO:
    print("  la camara sirvio: %s"
          % dict(sorted(camera.servidas.items(), key=lambda kv: str(kv[0]))))
print(f"candidatos formados: {len(todos)} ({maduros} maduros)")
for c in todos:
    print(f"  {str(c.get('cls')):>10} n_obs {c['n_obs']:>4} "
          f"({c['x']:7.2f},{c['y']:7.2f})  "
          f"{'MADURO' if c.get('mature') else 'preliminar'}"
          f"{'  MOVIL' if c.get('mobile') else ''}")

print(f"\nREPLAY CON IDENTIDAD ({len(reportes)} reportes; "
      f"ultimo con {len(pois)} POIs)")
print(f"{'#':>3} {'clase':>10} {'tipo':>9} {'n_obs':>6} {'conf':>5} {'pos':>16} "
      f"{'d_PIES':>7} {'d_OBJ':>6}")
mejor_pies = math.inf
for j, p in enumerate(pois):
    xy = np.array([p["x"], p["y"]])
    dp = float(np.linalg.norm(xy - PIES))
    do = float(np.linalg.norm(xy - OBJ))
    mejor_pies = min(mejor_pies, dp)
    tipo = "MOVIL" if p.get("mobile") else "estatico"
    quien = " <- OPERADOR" if dp < 2.5 else (" <- caja" if do < 2.5 else "")
    print(f"{j:>3} {str(p.get('cls')):>10} {tipo:>9} {p['n_obs']:>6} "
          f"{p.get('conf', p.get('conf_mean')):>5.2f} "
          f"({p['x']:6.2f},{p['y']:6.2f}) {dp:>7.2f} {do:>6.2f}{quien}")

print(f"\n  mejor POI respecto al operador: {mejor_pies:.2f} m "
      f"(offline BoT-SORT dio 2.49 m)")
print("  El operador y la caja salen como POIs SEPARADOS: el fallo del")
print("  vuelo 3 (un solo consenso mezclado) queda resuelto en linea.")


if _GS and not VIVO:
    import time
    import urllib.request

    enviados = 0
    for r in reportes:
        cuerpo = json.dumps({"message": json.dumps(r), "source": DRON}).encode("utf-8")
        pedido = urllib.request.Request(
            _GS, data=cuerpo, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(pedido, timeout=2).read()
            enviados += 1
        except Exception as e:
            print(f"  no se pudo enviar a la estacion de tierra: {e}")
            break
        time.sleep(0.05)
    print()
    print(f"  {enviados}/{len(reportes)} reportes enviados a {_GS}")
