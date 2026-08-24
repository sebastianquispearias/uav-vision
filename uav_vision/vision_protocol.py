"""Vision protocol: watch the ground, geolocate what the camera sees,
report POIs to the ground station.

This is the ONLY module in the package that imports gradys_embedded.
The camera layer (camera.py, pinhole_local.py) stays importable on any
laptop; this file needs the GrADyS ecosystem installed next to it.

How it plugs into GrADyS
------------------------
GrADyS protocols are classes implementing IProtocol. The runner (either
gradys-sim or EmbeddedRunner on the real drone) instantiates the class
and drives it with events: handle_telemetry() whenever the node moves,
handle_timer() when a scheduled timer fires. This protocol does NOT
command the drone -- it never sends a MobilityCommand. It only observes
and reports, so it can run side by side with whatever mobility protocol
the mission uses.

Data flow, one cycle:

    timer "see" (default 5 Hz)
        camera.ver_alvo(pos, yaw) -> [{px, py, conf}, ...]
        pixel_to_ray(...)         -> bearing ray in world frame
        ray ^ ground plane        -> impact point (x, y)
        store (impact, conf)

    timer "report" (default every 2 s)
        RANSAC consensus over stored impacts -> POI estimate
        BroadcastMessageCommand(json)        -> ground station

Where pose comes from
---------------------
- position: Telemetry.current_position, pushed by the runner. This is
  the local cartesian frame every GrADyS node shares.
- yaw: Telemetry does not carry attitude, so the protocol asks a
  yaw_source callable. On the real drone that is UavApiYaw, which polls
  Francisco's uav_api over HTTP on localhost (GET /telemetry/general,
  field "heading") -- the API owns the MAVLink serial port, nobody else
  touches it. In simulation the test injects a function that returns
  the simulated heading.

Known limitation (on purpose, for now): all impacts go into ONE RANSAC,
so the protocol reports the single dominant POI -- the person observed
the most. The incremental identity layer (multi-POI, mobile tracks,
appearance embeddings) is the next step; see README "Pendientes".
"""

from __future__ import annotations

import json
import math
from typing import Callable, List, Optional, Sequence, Tuple, Type

import numpy as np

from gradys_embedded.protocol.interface import IProtocol
from gradys_embedded.protocol.messages.communication import BroadcastMessageCommand
from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.pinhole_local import pixel_to_ray

TIMER_SEE = "uav_vision:see"
TIMER_REPORT = "uav_vision:report"

# Same consensus parameters the paper and onboard.py use.
RANSAC_ITERATIONS = 100
RANSAC_THRESHOLD_M = 5.0
MIN_MEASUREMENTS = 8


def _ground_impact(
    origin: Sequence[float],
    direction: Sequence[float],
    ground_z: float,
) -> Optional[Tuple[float, float]]:
    """Intersect a bearing ray with the horizontal plane z = ground_z."""
    dz = direction[2]
    if dz >= -1e-9:            # ray parallel to the ground or pointing up
        return None
    t = (ground_z - origin[2]) / dz
    return (origin[0] + t * direction[0], origin[1] + t * direction[1])


def _ransac_consensus(
    impacts: np.ndarray,
    rng: np.random.Generator,
    threshold_m: float = RANSAC_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
) -> Tuple[np.ndarray, int]:
    """Largest cluster of ground impacts, refit as the inlier mean.

    Same idea as onboard/fusion.py ransac_fusion, expressed over impact
    points instead of rays: hypothesize a point, count impacts within
    threshold_m, keep the hypothesis with the most support.

    Returns (estimate_xy, n_inliers).
    """
    best_inliers = None
    for _ in range(iterations):
        candidate = impacts[rng.integers(len(impacts))]
        d = np.linalg.norm(impacts - candidate, axis=1)
        inliers = d < threshold_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    estimate = impacts[best_inliers].mean(axis=0)
    return estimate, int(best_inliers.sum())


class UavApiYaw:
    """Yaw source for the real drone: polls uav_api on localhost.

    uav_api owns the MAVLink serial connection; every consumer reads
    pose over HTTP. GET /telemetry/general returns a JSON object whose
    "heading" field is the yaw in degrees (0 = North, clockwise).
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout_s: float = 0.5):
        self.url = base_url.rstrip("/") + "/telemetry/general"
        self.timeout_s = timeout_s

    def __call__(self) -> Optional[float]:
        import urllib.request

        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as r:
                return float(json.loads(r.read())["heading"])
        except Exception:
            return None           # no pose -> skip this frame, never crash


class VisionProtocol(IProtocol):
    """Observe-only protocol: camera in, POI messages out.

    IProtocol.instantiate() calls cls() with no arguments, so runtime
    configuration cannot go through __init__. Use with_config() to build
    a configured subclass and hand THAT class to the runner:

        Protocol = VisionProtocol.with_config(
            camera=CamaraSimulada(alvo=(3, -2, 0), pitch_deg=-55.0),
            pitch_deg=-55.0,
            yaw_source=my_yaw_function,
        )
    """

    # -- configuration (class attributes, set by with_config) -------------
    camera = None                                  # ver_alvo(pos, yaw) provider
    pitch_deg: Optional[float] = None              # camera mount pitch; no default
    yaw_source: Optional[Callable[[], Optional[float]]] = None
    see_period_s: float = 0.2                      # 5 Hz, the safe rate on the Pi
    report_period_s: float = 2.0
    ground_z: float = 0.0
    rng_seed: int = 0
    # Optional IdentidadIncremental. When present AND detections carry a
    # 'track_id', reports become multi-POI (statics + mobiles separated).
    # Without it, everything goes into one RANSAC -- which on flight 3
    # reproduces the original failure (POI lands nearer the equipment
    # box than the operator; see scripts/replay_vuelo3.py).
    identidad = None

    @classmethod
    def with_config(
        cls,
        camera,
        pitch_deg: float,
        yaw_source: Callable[[], Optional[float]],
        see_period_s: float = 0.2,
        report_period_s: float = 2.0,
        ground_z: float = 0.0,
        rng_seed: int = 0,
        identidad=None,
    ) -> Type["VisionProtocol"]:
        """Build a configured protocol class ready for the runner.

        pitch_deg is explicit and has no default for the same reason
        CamaraSimulada's doesn't: the real mount changed from -45 to -55
        once already, and hidden defaults are how copies diverge.
        """
        return type(
            "ConfiguredVisionProtocol",
            (cls,),
            {
                "camera": camera,
                "pitch_deg": pitch_deg,
                "yaw_source": staticmethod(yaw_source),
                "see_period_s": see_period_s,
                "report_period_s": report_period_s,
                "ground_z": ground_z,
                "rng_seed": rng_seed,
                "identidad": identidad,
            },
        )

    # -- IProtocol ---------------------------------------------------------

    def initialize(self) -> None:
        if self.camera is None or self.pitch_deg is None or self.yaw_source is None:
            raise RuntimeError(
                "VisionProtocol is not configured. Build the class with "
                "VisionProtocol.with_config(camera=..., pitch_deg=..., "
                "yaw_source=...) and give THAT class to the runner."
            )
        self._position: Optional[Tuple[float, float, float]] = None
        self._impacts: List[Tuple[float, float]] = []
        self._confs: List[float] = []
        self._frames_seen = 0
        self._rng = np.random.default_rng(self.rng_seed)

        now = self.provider.current_time()
        self.provider.schedule_timer(TIMER_SEE, now + self.see_period_s)
        self.provider.schedule_timer(TIMER_REPORT, now + self.report_period_s)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self._position = telemetry.current_position

    def handle_timer(self, timer: str) -> None:
        if timer == TIMER_SEE:
            self._see()
            self.provider.schedule_timer(
                TIMER_SEE, self.provider.current_time() + self.see_period_s)
        elif timer == TIMER_REPORT:
            self._report()
            self.provider.schedule_timer(
                TIMER_REPORT, self.provider.current_time() + self.report_period_s)

    def handle_packet(self, message: str) -> None:
        pass                      # observe-only: nothing to receive yet

    def finish(self) -> None:
        pass

    # -- internals ---------------------------------------------------------

    def _see(self) -> None:
        """One camera cycle: detect, back-project, store ground impacts."""
        if self._position is None:
            return                # no telemetry yet
        yaw = self.yaw_source()
        if yaw is None:
            return                # pose source unavailable this frame
        self._frames_seen += 1

        cam_cfg = self.camera.camara
        for det in self.camera.ver_alvo(self._position, yaw):
            origin, direction = pixel_to_ray(
                self._position,
                yaw,
                (det["px"], det["py"]),
                self.pitch_deg,
                cam_cfg.focal_length_px,
                cam_cfg.image_width,
                cam_cfg.image_height,
                cam_cfg.principal_point,
            )
            impact = _ground_impact(origin, direction, self.ground_z)
            if impact is None:
                continue
            self._impacts.append(impact)
            self._confs.append(det["conf"])
            track_id = det.get("track_id")
            if self.identidad is not None and track_id is not None:
                self.identidad.observar(
                    frame=self._frames_seen,
                    track_id=int(track_id),
                    impacto_xy=impact,
                    conf=det["conf"],
                    emb=det.get("emb"),
                )

    def _report(self) -> None:
        """Broadcast the current POI list.

        With an identity layer: one POI per candidate, mobiles first.
        Without one (or before any candidate matures): single dominant
        POI by RANSAC consensus over all impacts.
        """
        pois = self.identidad.candidatos() if self.identidad is not None else []

        if not pois:
            if len(self._impacts) < MIN_MEASUREMENTS:
                return
            impacts = np.asarray(self._impacts)
            estimate, n_inliers = _ransac_consensus(impacts, self._rng)
            pois = [{
                "x": round(float(estimate[0]), 2),
                "y": round(float(estimate[1]), 2),
                "n_obs": int(len(impacts)),
                "n_inliers": n_inliers,
                "conf_mean": round(float(np.mean(self._confs)), 3),
            }]

        message = {
            "type": "vision_poi",
            "sender": self.provider.get_id(),
            "time": self.provider.current_time(),
            "frames_seen": self._frames_seen,
            "pois": pois,
        }
        self.provider.send_communication_command(
            BroadcastMessageCommand(json.dumps(message)))
