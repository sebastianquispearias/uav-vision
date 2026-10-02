"""Two drones on a desk: the real runner, the real network, and the flight's own detections.

Everything between the camera and the radio is the code that flies: the gradys-embedded runner,
its HTTP data plane between the two boards, VisionProtocol with its identity layer, its listening
to neighbours and its search orders. Two things are replayed from the 02-ago flight instead of
sensed: what the camera saw (the cached detections, with the replay's stand-in track ids) and where
the drone was when it saw it (the flight's GPS position and yaw). The fake uav_api on each board
only lets the runner start; the hovering pose it reports is not used.

One board replays the first takeoff and the other the second, started together, so both see the
same scene within the same minutes. It is the pseudo-swarm protocol of the paper, across two
machines on a network instead of two processes on a laptop.

Configured by environment variables, so one file serves both boards:
    BANCO_DATOS      folder with examen_v3_datos.npz, embs_osnet.npy, frames.csv and
                     pistas_sustituto_02ago.npz (default ~/banco/datos)
    BANCO_DESDE_S    flight second this board starts replaying from: 0 on one, 700 on the other
    BANCO_ESTACION   ground station URL to poll for search orders (optional)

Loaded like any mission, from ~/gradys_protocols:
    POST /mission/load {"protocol": "mision_banco_dos_drones:ProtocoloVisionBanco", ...}
"""
import csv
import math
import os

import numpy as np

from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol

LAT0, LNG0, R_TIERRA = -22.978029946, -43.23214256266666, 6378137.0
PITCH = -55.0
CONF_MIN = 0.25
# The rate at which the chain processed this flight. The identity layer converts its time
# thresholds into observation counts with it, exactly as the replay does.
FPS_CADENA = 1.638
DATOS = os.path.expanduser(os.environ.get("BANCO_DATOS", "~/banco/datos"))
DESDE_S = float(os.environ.get("BANCO_DESDE_S", "0"))


def enu(lat, lng):
    """Local metres east and north of the flight's reference post, as the replay computes them."""
    return (math.radians(lng - LNG0) * R_TIERRA * math.cos(math.radians(LAT0)),
            math.radians(lat - LAT0) * R_TIERRA)


class VueloGrabado:
    """The recorded flight as a clock: which processed frames are due, and what each one saw."""

    def __init__(self, datos, desde_s):
        dets = np.load(os.path.join(datos, "examen_v3_datos.npz"))["dets"]
        dets = dets[dets[:, 1] >= CONF_MIN]
        embs = np.load(os.path.join(datos, "embs_osnet.npy")).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        pistas = np.load(os.path.join(datos, "pistas_sustituto_02ago.npz"))["track"]
        self.poses = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(datos, "frames.csv")))}
        self.por_frame = {}
        for i, d in enumerate(dets):
            tid = int(pistas[i]) if pistas[i] >= 0 else None
            self.por_frame.setdefault(int(d[0]), []).append((d, tid, embs[i]))
        aire = sorted(f for f in self.por_frame
                      if f in self.poses and float(self.poses[f]["alt_agl"]) > 3.0)
        t0 = float(self.poses[aire[0]]["t_mono"])
        self.linea = [(float(self.poses[f]["t_mono"]) - t0, f) for f in aire]
        self.linea = [(t, f) for t, f in self.linea if t >= desde_s]
        self.desde_s = desde_s
        self.k = 0
        self.inicio = None

    def due(self, ahora):
        """
        Every recorded frame whose flight time has been reached since the last call, in order.

        The flight clock starts at the first call, so a board replays from the moment its mission
        starts. All due frames are returned, not only the newest: skipping one would drop its
        detections, which the recording made on purpose.
        """
        if self.inicio is None:
            self.inicio = ahora
        t = self.desde_s + (ahora - self.inicio)
        frames = []
        while self.k < len(self.linea) and self.linea[self.k][0] <= t:
            frames.append(self.linea[self.k][1])
            self.k += 1
        return frames

    def pose(self, frame):
        p = self.poses[frame]
        x, y = enu(float(p["lat"]), float(p["lng"]))
        return (x, y, float(p["alt_agl"])), float(p["yaw"])


class CamaraBanco:
    """Serves the recorded detections of the frame being replayed, under the camera contract."""

    def __init__(self, vuelo, camara):
        self.vuelo = vuelo
        self.camera = camara
        self.frame = None
        self.classes = None

    def set_classes(self, classes=None):
        # The cached detections carry no class, and a detection without one passes any search, as
        # it does in the protocol. The order is still taken and acknowledged in the report.
        self.classes = frozenset(classes) if classes else None

    def ultimo_marco_jpeg(self, calidad: int = 85):
        """The recorded frame this camera is on, if its picture was put on the board.

        Without this the bench can fly a drone to another side of a target and has nothing to show
        when it gets there, which is the whole point of having flown. The recording's pictures are
        2.9 GB, so only the window each board replays is copied; outside it this returns None and
        the station says there is no frame instead of showing a stale one.

        The quality argument is ignored: the file is already a JPEG and re-encoding it would cost
        time on the board to make the picture worse.
        """
        del calidad
        ruta = os.path.join(DATOS, "frames", "frame_%04d.jpg" % int(self.frame))
        try:
            with open(ruta, "rb") as f:
                return f.read()
        except OSError:
            return None

    def detect(self, pos, yaw):
        del pos, yaw
        salida = []
        for d, tid, emb in self.vuelo.por_frame.get(self.frame, []):
            x1, _y1, x2, y2 = d[2:6]
            salida.append({"px": float((x1 + x2) / 2), "py": float(y2), "conf": float(d[1]),
                           "track_id": tid, "emb": emb, "cls": None})
        return salida


class ProtocoloBanco(VisionProtocol):
    """VisionProtocol whose pose and camera are driven by the recorded flight."""

    _pose_placa = None

    def handle_telemetry(self, telemetry):
        # The fake autopilot's pose is deliberately kept out of the ray casting: rays have to be
        # cast from where the drone was when each frame was taken, which is the recording's pose.
        # But it IS where this board thinks it is flying, so the orbit needs it: without this the
        # drone is told to go somewhere, the fake autopilot walks it there, and nothing ever
        # notices it arrived.
        self._pose_placa = telemetry.current_position
        if self._rodeo is not None:
            self._llego_al_rodeo()

    def _posicion_para_vuelo(self):
        return self._pose_placa

    def _see(self):
        for frame in VUELO.due(self.provider.current_time()):
            self._position, ESTADO["yaw"] = VUELO.pose(frame)
            CAMARA.frame = frame
            super()._see()


VUELO = VueloGrabado(DATOS, DESDE_S)
ESTADO = {"yaw": 0.0}
CAMARA = CamaraBanco(VUELO, ARDUCAM_MODULE_3.rotated_180())

ProtocoloVisionBanco = ProtocoloBanco.with_config(
    camera=CAMARA,
    pitch_deg=PITCH,
    yaw_source=lambda: ESTADO["yaw"],
    see_period_s=0.1,
    report_period_s=2.0,
    identity=IncrementalIdentity(fusion_radius_m=3.5, fps=FPS_CADENA),
    report_preliminary=True,
    station_url=os.environ.get("BANCO_ESTACION"),
)
