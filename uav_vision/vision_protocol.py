"""
Vision protocol: runs the camera, geolocates detections and reports POIs.

This is the only module in the package that imports gradys_embedded. The camera layer stays
importable on machines without the GrADyS ecosystem installed.

The protocol is observe-only: it never sends mobility commands, so it can run alongside
whatever mobility protocol the mission uses. Its cycle:

    timer "see" (default 4 Hz):
        camera.detect(pos, yaw) -> detections -> pixel_to_ray -> ground impact -> store
    timer "report" (default every 2 s):
        candidate list (identity layer) or single RANSAC consensus -> broadcast JSON

Pose sources:
    - Position arrives through handle_telemetry (the local cartesian frame all GrADyS nodes
      share).
    - Telemetry does not carry attitude, so yaw comes from a yaw_source callable. On the real
      drone that is UavApiYaw, which polls the uav_api HTTP service on localhost; the API owns
      the MAVLink serial connection and every other process reads pose over HTTP. In simulation
      the caller injects a function returning the simulated heading.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional, Sequence, Tuple, Type

import numpy as np

from gradys_embedded.protocol.interface import IProtocol
from gradys_embedded.protocol.messages.communication import BroadcastMessageCommand
from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.pinhole_local import pixel_to_ray

TIMER_SEE = "uav_vision:see"
TIMER_REPORT = "uav_vision:report"

RANSAC_ITERATIONS = 100
RANSAC_THRESHOLD_M = 5.0
MIN_MEASUREMENTS = 8


def _ground_impact(
    origin: Sequence[float],
    direction: Sequence[float],
    ground_z: float,
) -> Optional[Tuple[float, float]]:
    """Intersects a bearing ray with the horizontal plane z = ground_z."""
    dz = direction[2]
    if dz >= -1e-9:  # ray parallel to the ground or pointing up
        return None
    t = (ground_z - origin[2]) / dz
    return (origin[0] + t * direction[0], origin[1] + t * direction[1])


def _ransac_consensus(
    impacts: np.ndarray,
    rng: np.random.Generator,
    threshold_m: float = RANSAC_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
) -> Tuple[np.ndarray, int]:
    """
    Finds the largest cluster of ground impacts and refits it as the inlier mean. Same consensus
    idea as fusion.ransac_fusion, expressed over impact points instead of rays.

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
    """
    Yaw source for the real drone: polls the uav_api service on localhost.

    GET /telemetry/general returns {"result": "Success", "info": {"heading": <deg>, ...}} —
    the heading (0 = North, clockwise) lives inside "info". Verified against the real service
    on 2026-08-24; a mock that serves a flat {"heading": ...} is also accepted. Returns None on
    any failure so a transient HTTP error skips one frame instead of crashing the protocol.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout_s: float = 0.5):
        self.url = base_url.rstrip("/") + "/telemetry/general"
        self.timeout_s = timeout_s

    def __call__(self) -> Optional[float]:
        import urllib.request

        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as r:
                payload = json.loads(r.read())
                return float(payload.get("info", payload)["heading"])
        except Exception:
            return None


class VisionProtocol(IProtocol):
    """
    Observe-only protocol: camera in, POI messages out.

    IProtocol.instantiate() calls cls() with no arguments, so runtime configuration cannot go
    through __init__. Use with_config() to build a configured subclass and hand that class to
    the runner:

        Protocol = VisionProtocol.with_config(
            camera=OnboardCamera(...),
            pitch_deg=-55.0,
            yaw_source=UavApiYaw(),
        )
    """

    # -- configuration (class attributes, set by with_config) -------------
    camera = None                                  # detect(pos, yaw) provider
    pitch_deg: Optional[float] = None              # camera mount pitch; no default
    yaw_source: Optional[Callable[[], Optional[float]]] = None
    # 4 Hz: below the ~5 FPS voltage-collapse point measured with the 5 A UBEC,
    # so the default never operates at the edge of the power budget.
    see_period_s: float = 0.25
    report_period_s: float = 2.0
    ground_z: float = 0.0
    rng_seed: int = 0
    # Optional IncrementalIdentity. When present and detections carry a 'track_id', reports
    # become multi-POI (statics and mobiles separated). Without it, all impacts go into one
    # RANSAC and the report is the single dominant POI.
    identity = None
    # Report tracks that have formed but not yet matured, flagged mature=False. Off by default
    # because for a loitering drone it only adds noise -- it can afford to wait for certainty.
    # Turn it on for a SWEEP: measured on flight 3, a 30 s pass over a person never matures a
    # candidate, so a search that crosses each point once and moves on reports nothing at all.
    # The ground station must show these differently; they are requests for verification, not
    # finds.
    report_preliminary: bool = False

    @classmethod
    def with_config(
        cls,
        camera,
        pitch_deg: float,
        yaw_source: Callable[[], Optional[float]],
        see_period_s: float = 0.25,
        report_period_s: float = 2.0,
        ground_z: float = 0.0,
        rng_seed: int = 0,
        identity=None,
        report_preliminary: bool = False,
    ) -> Type["VisionProtocol"]:
        """
        Builds a configured protocol class ready for the runner. pitch_deg is explicit and has
        no default: the mount angle is a property of the deployment.
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
                "identity": identity,
                "report_preliminary": report_preliminary,
            },
        )

    # -- IProtocol ---------------------------------------------------------

    def initialize(self) -> None:
        if self.camera is None or self.pitch_deg is None or self.yaw_source is None:
            raise RuntimeError(
                "VisionProtocol is not configured. Build the class with "
                "VisionProtocol.with_config(camera=..., pitch_deg=..., yaw_source=...) and "
                "give that class to the runner.")
        self._position: Optional[Tuple[float, float, float]] = None
        self._impacts: List[Tuple[float, float]] = []
        self._confs: List[float] = []
        self._frames_seen = 0
        self._rng = np.random.default_rng(self.rng_seed)

        now = self.provider.current_time()
        self._t_inicio = now
        # Slots the work overran. Not a curiosity: it is the difference between a drone that
        # is keeping up and one quietly two thirds as attentive as it claims to be.
        self._slots_perdidos = 0
        # Start of the window the reported rate covers. Reset at every report.
        self._t_ventana = now
        self._frames_ventana = 0
        self._slots_ventana = 0
        self._proximo_see = now + self.see_period_s
        self._proximo_report = now + self.report_period_s
        self.provider.schedule_timer(TIMER_SEE, self._proximo_see)
        self.provider.schedule_timer(TIMER_REPORT, self._proximo_report)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self._position = telemetry.current_position

    def handle_timer(self, timer: str) -> None:
        if timer == TIMER_SEE:
            self._see()
            self._proximo_see = self._next_slot(
                self._proximo_see, self.see_period_s, contar=True)
            self.provider.schedule_timer(TIMER_SEE, self._proximo_see)
        elif timer == TIMER_REPORT:
            self._report()
            self._proximo_report = self._next_slot(
                self._proximo_report, self.report_period_s)
            self.provider.schedule_timer(TIMER_REPORT, self._proximo_report)

    def _next_slot(self, previsto: float, periodo: float,
                          contar: bool = False) -> float:
        """
        The next slot on a fixed cadence, skipping any the work ran past.

        Rescheduling as `now + period` -- which is what this did until 2026-08-25 -- makes the real
        interval `work + period`, so the loop never runs at the rate it was asked for. Measured
        on the Pi: 2.31 frames per second against 3.00 configured, on an empty scene. That
        matters beyond throughput, because the identity layer scaled every maturity threshold
        by the declared rate, so a loop slower than it claimed silently stretched what
        "36 seconds of evidence" meant -- to about 47.

        When the work does overrun a slot, the missed ones are skipped rather than queued. A
        backlog of camera frames cannot be worked off: each would be stale by the time it ran,
        and firing them back to back would starve everything else. It can only be counted and
        reported, which is what slots_perdidos is for.
        """
        ahora = self.provider.current_time()
        proximo = previsto + periodo
        if proximo > ahora:
            return proximo
        perdidos = int((ahora - proximo) // periodo) + 1
        if contar:
            self._slots_perdidos += perdidos
        return proximo + perdidos * periodo

    def handle_packet(self, message: str) -> None:
        pass  # observe-only: no incoming commands in this version

    def _ritmo(self):
        """
        Rate and misses over the LAST report interval, not over the whole mission.

        A lifetime average is the wrong statistic here, and measurably so. The camera takes
        about 15 s to wake and the models to load, and on the Pi that one-off showed up as
        2.42 fps and 46 lost slots on a loop that was in fact delivering 2.98 of a configured
        3.00 -- an operator reading it would have seen a saturated drone that was running
        perfectly. The startup is real but it is over; what matters in flight is the rate now.

        The cumulative count is kept alongside, because after the flight the total is the
        thing worth knowing.
        """
        ahora = self.provider.current_time()
        dt = ahora - self._t_ventana
        frames = self._frames_seen - self._frames_ventana
        perdidos = self._slots_perdidos - self._slots_ventana
        self._t_ventana = ahora
        self._frames_ventana = self._frames_seen
        self._slots_ventana = self._slots_perdidos
        return {
            "fps_real": round(frames / dt, 2) if dt > 0 else None,
            "slots_perdidos": perdidos,
            "slots_perdidos_total": self._slots_perdidos,
        }

    def _gps_origin(self):
        """The mission's coordinate origin as [lat, lon, alt], or None outside the runner."""
        origen = getattr(self.provider, "origin_gps_coordinates", None)
        if origen is None:
            return None
        try:
            return [float(v) for v in origen]
        except (TypeError, ValueError):
            return None

    def finish(self) -> None:
        pass

    # -- internals ---------------------------------------------------------

    def _see(self) -> None:
        """One camera cycle: detect, back-project, store ground impacts."""
        if self._position is None:
            return  # no telemetry yet
        yaw = self.yaw_source()
        if yaw is None:
            return  # pose source unavailable this frame
        self._frames_seen += 1

        cam_cfg = self.camera.camera
        for det in self.camera.detect(self._position, yaw):
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
            if self.identity is not None and track_id is not None:
                self.identity.observe(
                    frame=self._frames_seen,
                    track_id=int(track_id),
                    ground_xy=impact,
                    conf=det["conf"],
                    emb=det.get("emb"),
                    crop=det.get("crop"),
                    # The clock, so maturity is measured rather than inferred from a frame
                    # count times a rate the caller merely promised.
                    t=self.provider.current_time(),
                )

    def _report(self) -> None:
        """
        Broadcasts the current POI list. With an identity layer: one POI per candidate, mobiles
        first. Without one, or before any candidate has enough evidence: the single dominant POI
        by RANSAC consensus over all stored impacts.
        """
        import base64

        pois = (self.identity.candidates(preliminary=self.report_preliminary)
                if self.identity is not None else [])
        latido = False

        if not pois:
            if len(self._impacts) < MIN_MEASUREMENTS:
                # Nothing found -- and that is exactly when the drone must still speak. From
                # the ground, a drone that sees nobody and a drone that has died look
                # identical: both are silence. An operator who cannot tell them apart has to
                # assume the worst and abort. One empty beat per report period buys the
                # difference for a few dozen bytes.
                latido = True
            else:
                impacts = np.asarray(self._impacts)
                estimate, n_inliers = _ransac_consensus(impacts, self._rng)
                pois = [{
                    "x": round(float(estimate[0]), 2),
                    "y": round(float(estimate[1]), 2),
                    "n_obs": int(len(impacts)),
                    "n_inliers": n_inliers,
                    "conf_mean": round(float(np.mean(self._confs)), 3),
                }]

        # The identity layer hands over raw JPEG bytes; JSON needs text. Encoded here rather
        # than there so the identity layer stays free of transport concerns.
        for p in pois:
            crop = p.pop("crop", None)
            if crop:
                p["crop"] = base64.b64encode(crop).decode("ascii")

        ritmo = self._ritmo()
        message = {
            "type": "vision_poi",
            "sender": self.provider.get_id(),
            "time": self.provider.current_time(),
            "frames_seen": self._frames_seen,
            # What the loop actually delivered over the last interval, so the gap between
            # configured and real can never again be something only a stopwatch would find.
            "fps_real": ritmo["fps_real"],
            "slots_perdidos": ritmo["slots_perdidos"],
            "slots_perdidos_total": ritmo["slots_perdidos_total"],
            "latido": latido,
            # The frame these metres are measured in, so the receiver never has to be told
            # separately. It cannot be: when the mission is loaded without an origin the
            # runner resolves one from the GPS fix at that moment, which nobody can know in
            # advance to type into a ground station. Send it and the question disappears.
            #
            # Read defensively: IProvider does not declare it -- the embedded runtime's
            # provider carries it, a test harness does not. None is a valid answer and means
            # "local metres only", which is what every desk run has ever produced.
            "origen_gps": self._gps_origin(),
            "pois": pois,
        }
        self.provider.send_communication_command(
            BroadcastMessageCommand(json.dumps(message)))
