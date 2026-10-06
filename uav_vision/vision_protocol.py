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

CONSTANTS THAT ARE DECISIONS
    STATION_TIMEOUT_S, STATION_RETRY_S
        How long a poll of the ground station may take, and how long to wait before asking
        an unreachable station again. The poll must never cost the see loop more than a
        sliver of its period, and a station that is down must not be asked on every report.

    GROUND_EXTENT_M
        How much ground each class covers along the direction it is being looked at, in
        metres, used by _footprint_center to move the impact from the near edge of the object
        to the middle of its footprint. Nominal dimensions of the thing, not tuned
        corrections, so the names are VisDrone's. People are absent on purpose.

    FOCO_RADIO_PX, FOCO_UMBRAL
        The image window the operator's verdict opens around a fixed target, in pixels, and
        the detector score accepted inside it. Mission decisions and not optics: they set how
        much unvouched-for evidence is acceptable in exchange for finding the target.

    RODEO_TOLERANCIA_M, RODEO_PLAZO_S
        How close the aircraft has to stand to where it was sent before its frame counts, and
        how long one leg of the orbit may take before the orbit is abandoned. A radius and not
        a coordinate match, because an aircraft holding position drifts and demanding the
        coordinate would mean never arriving.

    THROTTLED, BITS_AHORA, BITS_ALGUNA_VEZ
        The board's own power and thermal complaints, read as a firmware bitmask. The two
        halves mean different things: the low bits are NOW, so acting on them means the
        aircraft is in trouble this second, and the high bits are EVER SINCE BOOT, so they
        stay set after a brief dip nobody was watching for.

    Every value above, what it trades off and the measurement behind it: NOTES.md.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
from gradys_embedded.protocol.interface import IProtocol
from gradys_embedded.protocol.messages.communication import BroadcastMessageCommand
from gradys_embedded.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.flota import mismo_objetivo
from uav_vision.identity import dominant_class
from uav_vision.pinhole_local import pixel_to_ray, project_to_pixel
from uav_vision.view_selection import next_best_viewpoint, orbit_waypoints

TIMER_SEE = "uav_vision:see"
TIMER_REPORT = "uav_vision:report"
STATION_TIMEOUT_S = 0.3
STATION_RETRY_S = 10.0

GROUND_EXTENT_M: Dict[str, float] = {
    "bicycle": 1.2,
    "motor": 1.4,
    "tricycle": 1.8,
    "awning-tricycle": 1.8,
    "car": 3.1,
    "van": 3.8,
    "truck": 5.3,
    "bus": 7.3,
}

RANSAC_ITERATIONS = 100
RANSAC_THRESHOLD_M = 5.0
MIN_MEASUREMENTS = 8

FOCO_RADIO_PX = 320.0
FOCO_UMBRAL = 0.10

RODEO_TOLERANCIA_M = 5.0
RODEO_PLAZO_S = 60.0


def _ground_impact(
    origin: Sequence[float],
    direction: Sequence[float],
    ground_z: float,
) -> Optional[Tuple[float, float]]:
    """Intersects a bearing ray with the horizontal plane z = ground_z.

    A ray that runs parallel to the ground or points up above the horizon never meets the
    plane, and None is the answer for it.
    """
    dz = direction[2]
    if dz >= -1e-9:
        return None
    t = (ground_z - origin[2]) / dz
    return (origin[0] + t * direction[0], origin[1] + t * direction[1])


def _footprint_center(
    origin: Sequence[float],
    impact: Tuple[float, float],
    extent_m: float,
) -> Tuple[float, float]:
    """
    Moves a ground impact from the near edge of the object to the middle of its footprint.

    'py' is the bottom edge of the detection box: the lowest pixel the object occupies, which
    on the ground is its point CLOSEST to the camera, not the centre of what it stands on. For
    a person the two are 0.2 m apart and nobody would notice. For a car they are metres apart,
    and -- this is what makes it worth correcting -- always in the same direction, towards the
    drone. A bias does not average out: fusing a hundred views of the same car from the same
    pass gives a hundred times the same wrong answer, and the report looks all the more
    confident for it.

    The correction is half the object's ground extent, along the horizontal bearing from the
    drone. Directly under the camera the bearing is undefined and there is nothing to correct
    anyway, so the impact is returned untouched.
    """
    dx = impact[0] - origin[0]
    dy = impact[1] - origin[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return impact
    k = 0.5 * extent_m / d
    return (impact[0] + k * dx, impact[1] + k * dy)


def _ransac_consensus(
    impacts: np.ndarray,
    rng: np.random.Generator,
    threshold_m: float = RANSAC_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the largest cluster of ground impacts and refits it as the inlier mean. Same consensus
    idea as fusion.ransac_fusion, expressed over impact points instead of rays.

    Returns (estimate_xy, inlier_mask). The mask and not just its count, because whatever else
    is asked of this consensus -- which class it is, which frames it came from -- has to be
    asked of the impacts that formed it, not of the ones it rejected.
    """
    best_inliers = None
    for _ in range(iterations):
        candidate = impacts[rng.integers(len(impacts))]
        d = np.linalg.norm(impacts - candidate, axis=1)
        inliers = d < threshold_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    estimate = impacts[best_inliers].mean(axis=0)
    return estimate, best_inliers


THROTTLED = "/sys/devices/platform/soc/soc:firmware/get_throttled"
BITS_AHORA = (("bajo_voltaje", 0x1), ("frecuencia_limitada", 0x2),
              ("acelerador", 0x4), ("limite_termico", 0x8))
BITS_ALGUNA_VEZ = (("bajo_voltaje", 0x10000), ("frecuencia_limitada", 0x20000),
                   ("acelerador", 0x40000), ("limite_termico", 0x80000))


def salud_electrica(ruta: str = THROTTLED) -> Optional[dict]:
    """What the firmware says about power and heat, or None where there is no such firmware.

    None is not a failure: a laptop running the replay has no Raspberry firmware to ask, and a
    station that drew a warning from a missing file would cry wolf on every desk run.

    THE VALUE IS HEXADECIMAL AND THE FILE WRITES IT WITHOUT A PREFIX, which is why the base is
    stated instead of inferred. Read with base 0, as this did, '80008' parses as the DECIMAL
    80008, which is 0x13888, and the bits land on different meanings entirely: measured on the
    Pi 4 flying the bench, a board whose real state was 'soft temperature limit' was reported to
    the station as 'under-voltage'. Worse, a value carrying a hex letter -- the same board read
    0xe0008 through vcgencmd -- raises ValueError on base 0, the except below swallows it, and
    the function returns None, which this docstring itself declares to mean 'no firmware to ask'.
    So the alarm the station shows first, because it is the only one that predicts a loss, could
    be both WRONG and ABSENT, and neither failure printed anything.

    Base 16 accepts the prefix too, so a file that writes '0x80008' decodes the same as one that
    writes '80008'. The test feeds both, and it used to feed only the prefixed form, which is
    exactly why this survived: with a prefix, base 0 is correct.
    """
    try:
        with open(ruta) as fh:
            crudo = int(fh.read().strip(), 16)
    except Exception:
        return None
    return {"crudo": crudo,
            "ahora": [n for n, b in BITS_AHORA if crudo & b],
            "alguna_vez": [n for n, b in BITS_ALGUNA_VEZ if crudo & b]}

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

    The configuration lives in class attributes, which with_config sets:

    camera
        The detect(pos, yaw) provider.

    pitch_deg
        The camera mount pitch. It has no default, because the mount angle is a property of
        the deployment and guessing it puts every impact somewhere else.

    yaw_source
        A callable returning the heading in degrees, or None when the pose is unavailable
        this frame.

    see_period_s
        How often the camera is asked for a frame. Bounded by the board's power budget rather
        than by the detector, for which see NOTES.md.

    identity
        An optional IncrementalIdentity. When present and detections carry a 'track_id',
        reports become multi-POI, with statics and mobiles separated. Without it, all impacts
        go into one RANSAC and the report is the single dominant POI.

    report_preliminary
        Report tracks that have formed but not yet matured, flagged mature=False. Off by
        default, because a loitering drone can afford to wait for certainty; on for a sweep,
        which crosses each point once and would otherwise report nothing. The ground station
        must show these differently: they are requests for verification, not finds.

    ground_extent_m
        Class -> ground extent in metres, used to correct the near-edge bias of 'py'.
        Defaults to the vehicle table; people are not in it on purpose. Pass {} to switch the
        correction off entirely.

    station_url
        Where to ask what to look for, when the ground station is not a node of the fleet.
        With it, the drone polls the station's /buscar once per report period; without it,
        orders arrive only as 'vision_buscar' packets on the data plane. Both paths end in
        apply_search_order, so they cannot disagree on what an order means.

    attitude_source
        Where the airframe's attitude comes from: a callable returning (roll_deg, pitch_deg),
        right wing down and nose up positive, or None when unavailable. The camera is fixed to
        the body, so a drone that noses down to fly forward points its camera elsewhere. None
        keeps the mount angle alone, which is right for a loitering drone and wrong for one
        that escorts.
    """

    camera = None
    pitch_deg: Optional[float] = None
    yaw_source: Optional[Callable[[], Optional[float]]] = None
    see_period_s: float = 0.25
    report_period_s: float = 2.0
    ground_z: float = 0.0
    rng_seed: int = 0
    identity = None
    report_preliminary: bool = False
    ground_extent_m: Mapping[str, float] = GROUND_EXTENT_M
    station_url: Optional[str] = None
    attitude_source: Optional[Callable[[], Optional[Tuple[float, float]]]] = None

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
        ground_extent_m: Optional[Mapping[str, float]] = None,
        station_url: Optional[str] = None,
        attitude_source: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
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
                "attitude_source": (staticmethod(attitude_source)
                                    if attitude_source is not None else None),
                "see_period_s": see_period_s,
                "report_period_s": report_period_s,
                "ground_z": ground_z,
                "rng_seed": rng_seed,
                "identity": identity,
                "report_preliminary": report_preliminary,
                "station_url": station_url,
                "ground_extent_m": (GROUND_EXTENT_M if ground_extent_m is None
                                    else dict(ground_extent_m)),
            },
        )

    def initialize(self) -> None:
        """Builds the per-flight state and starts the two timers.

        Everything here is emptied at launch rather than at class level, so a relaunched
        protocol never starts out corroborating a previous flight or carrying its candidates.

        What the state fields are, where the reason is not obvious from the name:

        _clases
            Parallel to _impacts: what the detector called each one, or None. Kept so the
            RANSAC fallback can name its POI from the impacts that actually formed it.

        _objetivo
            The target the operator pointed at, in metres, or None. Kept as a ground position
            and projected onto every frame: where it lands in the image depends on the
            aircraft's pose at that instant, which is not something the ground can send.

        _descartados
            What the operator has looked at and refused. The drone's own signals cannot tell a
            person from an object the detector keeps confusing with one: CLIP and the physical
            size of the box both miss the phantom that is person-shaped and person-sized, and
            the phantom count of docs/RESULTADOS.md is what that costs. The operator can
            tell, and saying so costs one click. Keeping the refusal here is what stops the
            drone from spending the link on the same wrong point every two seconds for the
            rest of the flight.

        _rodeo
            Where this aircraft was told to go to look at a target from another side, and the
            target it was sent to look at. None when nothing was asked. This is the one place
            in the whole package that makes the aircraft move, and it only ever moves because
            a human clicked.

        _ajenos and ventana_ajenos_s
            What the neighbours are reporting, by sender, and how long one of their reports
            stands as current. The window is loose enough to cover a missed report period,
            tight enough that it is still a claim about now.

        _orden_epoca, _orden_v, _orden_rechazo
            The last search order taken: which station session issued it, its version, and the
            camera's refusal when it could not comply. The epoch is what lets a restarted
            station, whose counter starts again from zero, still be obeyed.

        _slots_perdidos
            Slots the work overran. Not a curiosity: it is the difference between a drone that
            is keeping up and one quietly two thirds as attentive as it claims to be.

        _t_ventana
            Start of the window the reported rate covers. Reset at every report.
        """
        if self.camera is None or self.pitch_deg is None or self.yaw_source is None:
            raise RuntimeError(
                "VisionProtocol is not configured. Build the class with "
                "VisionProtocol.with_config(camera=..., pitch_deg=..., yaw_source=...) and "
                "give that class to the runner.")
        self._position: Optional[Tuple[float, float, float]] = None
        self._impacts: List[Tuple[float, float]] = []
        self._confs: List[float] = []
        self._clases: List[Optional[str]] = []
        self._frames_seen = 0
        self._objetivo = None
        self._descartados: List[dict] = []
        self._rodeo: Optional[dict] = None
        self._ajenos: Dict = {}
        self.ventana_ajenos_s = 15.0
        self._orden_epoca = None
        self._orden_v = None
        self._orden_rechazo = None
        self._proxima_consulta = float("-inf")
        self._rng = np.random.default_rng(self.rng_seed)

        now = self.provider.current_time()
        self._t_inicio = now
        self._slots_perdidos = 0
        self._t_ventana = now
        self._frames_ventana = 0
        self._slots_ventana = 0
        self._proximo_see = now + self.see_period_s
        self._proximo_report = now + self.report_period_s
        self.provider.schedule_timer(TIMER_SEE, self._proximo_see)
        self.provider.schedule_timer(TIMER_REPORT, self._proximo_report)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self._position = telemetry.current_position
        if self._rodeo is not None:
            self._llego_al_rodeo()

    def handle_timer(self, timer: str) -> None:
        """Runs whichever of the two loops fired, and books the next slot on its cadence.

        The station is asked before reporting and not after, so the report that goes out
        already says what the camera is doing rather than what it was doing a period ago.
        """
        if timer == TIMER_SEE:
            self._see()
            self._proximo_see = self._next_slot(
                self._proximo_see, self.see_period_s, contar=True)
            self.provider.schedule_timer(TIMER_SEE, self._proximo_see)
        elif timer == TIMER_REPORT:
            self._poll_station()
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
        """
        Listens to what the other drones found.

        The report goes out as a broadcast, so a neighbour's findings were already arriving
        here and being dropped. Keeping them is what lets this drone know, without a ground
        station in the middle, that a target it is unsure about has been seen by someone else.

        Nothing that moves the aircraft is acted upon. The link carries data and the stick stays
        with the pilot, so a drone that moved because a packet told it to would be a behaviour
        that cannot fly. It listens, it corroborates, and it says so in its own report. The one
        packet it obeys is a search order, which changes what the camera reports and nothing
        else.
        """
        try:
            m = json.loads(message)
        except Exception:
            return
        if m.get("type") == "vision_buscar":
            self.apply_search_order(m.get("clases"), m.get("v"), m.get("epoca"))
            return
        if m.get("type") == "vision_mirar":
            self.send_frame(m.get("para"))
            return
        if m.get("type") == "vision_rodear":
            self.rodear(m.get("x"), m.get("y"), m.get("radio_m"), m.get("altura_m"),
                        bool(m.get("movil")), int(m.get("puntos") or 1))
            return
        if m.get("type") == "vision_reiniciar":
            self.reiniciar()
            return
        if m.get("type") == "vision_descarte":
            self.descartar(m.get("x"), m.get("y"), m.get("cls"), m.get("plantilla"))
            return
        if m.get("type") == "vision_objetivo":
            self.fix_target(m.get("x"), m.get("y"), m.get("radio_px"), m.get("umbral"),
                            self._emb_de_mensaje(m.get("plantilla")), m.get("emb_dist"))
            return
        if m.get("type") != "vision_poi":
            return
        quien = m.get("sender")
        if quien is None or quien == self.provider.get_id():
            return
        self._ajenos[quien] = {
            "t": self.provider.current_time(),
            "pois": m.get("pois") or [],
        }

    def rodear(self, x=None, y=None, radio_m=None, altura_m=None,
               movil=False, puntos=1) -> bool:
        """Flies to the place from which this target has not been seen yet, and sends that frame.

        This is the second half of the mission as it was written: detect a POI, and then the
        aircraft goes, circles it and holds position. Everything before this reported and never
        touched the flight.

        The place is not "closer". Closer was measured and does not settle the detection question
        on this flight: within one frame, where altitude and light are identical, the detector
        missed people LARGER than ones it found. What is missing from a single pass is not pixels,
        it is a second direction, and that is what next_best_viewpoint returns: the position whose
        ray to the target is the most different from the ones already taken.

        What the extra view is FOR is worth being exact about, because it decides how to judge it.
        It is not a claim that the detector will do better from there; that depends on the model
        and is not measured. It is a second geometry for the same target, which tightens the
        bearing-only estimate, and a picture of the target from the side for the person who has to
        decide, who needs a posture or a face and not a shape seen from above.

        A MOVING target is refused, and that is the most important line of this method. The order
        carries a ground coordinate and not a pixel, which is why it works with the camera seeing
        nothing at the moment of the click: the position comes from the identity layer, which never
        forgets a candidate. But a walker's position goes stale while the aircraft flies, and this
        layer refuses to extrapolate a mover beyond extrapolation_max_s, three seconds, because
        'past the window its velocity was estimated over, following the line is guessing'. A flight
        of tens of seconds is an order of magnitude outside what the estimate can carry, so the
        aircraft would arrive at where somebody was and photograph empty ground. Following a mover
        needs the loop closed on the image, which is visual servoing and is not this.

        puntos > 1 asks for the whole way round instead of one stop. The orbit is a finite list on
        purpose and each leg has a deadline: an aircraft that was told to go somewhere it cannot
        reach must give up, not wait forever.

        The only viewing direction this aircraft can vouch for is its own, right now, so that is
        the only one it passes to next_best_viewpoint. A station that knows where the other
        drones are can pass theirs; this layer does not invent them.

        radio_m and altura_m are mission decisions and have no defaults in this layer: see
        next_best_viewpoint. Call with x None to cancel.
        """
        if x is None:
            self._rodeo = None
            return True
        if self.camera is None or self._position is None:
            return False
        if radio_m is None or altura_m is None:
            return False
        if movil:
            return False
        objetivo = (float(x), float(y), self.ground_z)
        d = np.array([objetivo[0] - self._position[0],
                      objetivo[1] - self._position[1],
                      objetivo[2] - self._position[2]], dtype=float)
        norma = float(np.linalg.norm(d))
        vistas = [tuple(d / norma)] if norma > 1e-9 else []
        if int(puntos) > 1:
            lista = orbit_waypoints(objetivo, vistas, float(radio_m), float(altura_m),
                                    n_points=int(puntos))
            diversidad = next_best_viewpoint(objetivo, vistas,
                                             float(radio_m), float(altura_m))[1]
        else:
            uno, diversidad = next_best_viewpoint(objetivo, vistas,
                                                  radius_m=float(radio_m),
                                                  altitude_m=float(altura_m))
            lista = [uno] if uno is not None else []
        if not lista:
            return False
        self._rodeo = {"objetivo": (objetivo[0], objetivo[1]), "puntos": lista, "i": 0,
                       "ir_a": lista[0], "diversidad": float(diversidad), "mirado": False,
                       "t_tramo": self.provider.current_time()}
        self.provider.send_mobility_command(GotoCoordsMobilityCommand(*lista[0]))
        return True

    def _posicion_para_vuelo(self):
        """Where the aircraft actually IS, which on the bench is not where it is looking from.

        Flying and seeing use different positions and only the bench makes that visible. A board on
        a desk casts its rays from the pose the recording had, because that is where the pictures
        were taken, and its own autopilot reports the desk. The orbit is about the aircraft, so it
        asks here and the bench overrides this with what the autopilot says, leaving the ray
        casting alone. In the air the two are the same position and this returns it.
        """
        return self._position

    def _llego_al_rodeo(self) -> None:
        """Sends the frame once the aircraft is standing where it was sent, and not before.

        Sending on the way would hand the ground the same view it already had, which is the whole
        point of having flown. The tolerance is a radius, not a coordinate match: an aircraft
        holding position drifts, and demanding a coordinate would mean never arriving.

        Past the deadline the leg is given up on and the orbit abandoned. Saying nothing is
        better than an aircraft parked against a waypoint it cannot reach while the operator
        waits for a picture that is never coming.
        """
        aqui = self._posicion_para_vuelo()
        if self._rodeo is None or aqui is None:
            return
        r = self._rodeo
        ahora = self.provider.current_time()
        if ahora - r["t_tramo"] > RODEO_PLAZO_S:
            self._rodeo = None
            return
        ir = r["ir_a"]
        if math.hypot(aqui[0] - ir[0], aqui[1] - ir[1]) > RODEO_TOLERANCIA_M:
            return
        self.send_frame()
        r["i"] += 1
        if r["i"] >= len(r["puntos"]):
            r["mirado"] = True
            self._rodeo = None
            return
        r["ir_a"] = r["puntos"][r["i"]]
        r["t_tramo"] = ahora
        self.provider.send_mobility_command(GotoCoordsMobilityCommand(*r["ir_a"]))

    def reiniciar(self) -> None:
        """Forgets every candidate, because the operator asked for a clean board.

        The station's clear button only ever hid pins, and a page reload brought them all back:
        the candidates live in this layer, on this aircraft, and no message reached them. So the
        button was telling the truth about the screen and a lie about the system.

        The refusals survive. Those are the operator's judgements about the world and they cost
        a person's attention to produce; the board is just how it is being drawn right now.
        """
        if self.identity is not None:
            self.identity.olvidar_todo()
        print("[vision] reinicio pedido por el operador: identidad vacia, %d descartes intactos"
              % len(self._descartados), flush=True)

    def descartar(self, x=None, y=None, cls=None, plantilla=None) -> bool:
        """Records that the operator looked at this point and said it is not what we are after.

        A refusal needs an appearance and is rejected without one. Position and class alone would
        suppress whatever stands where a refused object stood, and the thing most likely to stand
        there next is a person walking past it: mismo_objetivo falls back to class and distance
        when either side has no vector, which is right for fusing two drones' reports and wrong
        for refusing to report at all. So a refusal carries the embedding of what was refused, and
        a candidate with no embedding of its own is never suppressed.

        Call with x None to forget every refusal, which is the way back: a refusal is the
        operator's judgement and the operator has to be able to withdraw it.
        """
        if x is None:
            self._descartados = []
            return True
        vector = self._emb_de_mensaje(plantilla)
        if vector is None:
            return False
        self._descartados.append({"x": float(x), "y": float(y), "cls": cls, "emb": vector})
        return True

    def _fue_descartado(self, poi) -> bool:
        """Whether the operator already refused this target, by position AND appearance."""
        if not self._descartados or poi.get("emb") is None:
            return False
        return any(mismo_objetivo(poi, no) for no in self._descartados)

    @staticmethod
    def _emb_de_mensaje(valor):
        """Reads an appearance template off the wire, in either form it can arrive in.

        A report leaving this drone packs the embedding as base64 of float16, which is what the
        station holds and hands straight back when the operator clicks a candidate. A caller
        inside the process, a test or the replay, has the vector itself. Accepting both is what
        keeps the station from having to decode and re-encode something it never reads.
        """
        if valor is None:
            return None
        if isinstance(valor, str):
            import base64
            return np.frombuffer(base64.b64decode(valor), dtype=np.float16).astype(np.float32)
        return np.asarray(valor, dtype=np.float32)

    def fix_target(self, x=None, y=None, radio_px=None, umbral=None,
                   plantilla=None, emb_dist=None) -> bool:
        """Fixes the target the operator pointed at, or releases it when x is None.

        Fixing a target lowers the reporting threshold inside a window of the image, which costs
        no computing at all: the detector had already scored those boxes and was discarding them
        for being under the threshold. What it buys and what it costs are in
        NOTES.md, and the short version is that the box numbers improve and the
        product numbers do not move.

        Only the target's ground position travels. Turning it into a square of the image is this
        drone's job and nobody else's, because the square depends on where the aircraft is and
        where it is pointing at the instant the frame is taken, which the ground cannot know.

        What CAN travel, and is the other half of the operator's click, is what the target looks
        like: the appearance template, which is the embedding the station already received with
        the candidate. With it a doubted box is kept wherever it falls, which is what the window
        cannot do, because the window is a projection and goes wrong when the attitude is least
        certain. Both plantilla and emb_dist have to arrive for the gate to open; see
        camera.EMB_DIST_OBJETIVO for what it buys and why the distance is not settled yet.
        """
        if self.camera is None:
            return False
        if x is None:
            self._objetivo = None
            fijar = getattr(self.camera, "set_focus", None)
            if callable(fijar):
                fijar(None)
            return True
        self._objetivo = {"pos": (float(x), float(y)),
                          "radio_px": float(radio_px) if radio_px is not None else FOCO_RADIO_PX,
                          "umbral": float(umbral) if umbral is not None else FOCO_UMBRAL,
                          "plantilla": plantilla,
                          "emb_dist": emb_dist}
        return True

    def _apuntar_foco(self, yaw, alabeo, cabeceo) -> None:
        """Projects the fixed target onto this frame, so the camera knows which square to favour.

        A target that falls outside the frame clears the window instead of leaving the last one in
        place: a stale square lowers the threshold over a piece of ground nobody vouched for, which
        is precisely how a free recall gain turns into invented points.

        The appearance template is only handed over when there is one. A camera that implements
        the older set_focus, of which the test stubs are two, keeps working untouched: the
        protocol must not demand a capability it is not using, and this is the same duck typing
        the getattr already relies on.
        """
        fijar = getattr(self.camera, "set_focus", None)
        if not callable(fijar):
            return
        if self._objetivo is None:
            fijar(None)
            return
        cam_cfg = self.camera.camera
        px = project_to_pixel(
            self._position, (self._objetivo["pos"][0], self._objetivo["pos"][1], self.ground_z),
            yaw, self.pitch_deg, cam_cfg.focal_length_px, cam_cfg.image_width,
            cam_cfg.image_height, cam_cfg.principal_point,
            body_pitch_deg=cabeceo, body_roll_deg=alabeo)
        if px is None:
            fijar(None)
            return
        extra = {}
        if self._objetivo.get("plantilla") is not None:
            extra = {"plantilla": self._objetivo["plantilla"],
                     "emb_dist": self._objetivo.get("emb_dist")}
        fijar(px[0], px[1], self._objetivo["radio_px"], self._objetivo["umbral"], **extra)

    def send_frame(self, para=None) -> bool:
        """Sends one frame, once, because somebody on the ground asked to look at it.

        The detector that flies is the one that fits in the power budget, and it finds fewer people
        than one that does not have to; the measured gap between the two is in docs/RESULTADOS.md.
        The better detector will never fly, because it takes a second and a half per frame against
        thirty five milliseconds, but there is no reason the ground cannot run it on a frame the
        aircraft sends when an operator wants a second opinion about a spot.

        One frame on request, never a stream. A frame is about 300 KB, so at three per second the
        video alone is seven megabits and would sit on top of the telemetry on the same link; asked
        for by hand it is one transfer, and the operator is not asking three times a second.
        """
        if self.camera is None:
            return False
        marco = getattr(self.camera, "ultimo_marco_jpeg", None)
        datos = marco() if callable(marco) else None
        if not datos:
            return False
        import base64
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(
            {"type": "vision_marco", "sender": self.provider.get_id(), "para": para,
             "t": self.provider.current_time(),
             "jpeg": base64.b64encode(datos).decode("ascii"),
             "pos": [round(float(v), 2) for v in self._position] if self._position is not None else None,
             "yaw": round(float(self.yaw_source()), 2) if self.yaw_source is not None else None})))
        return True

    def apply_search_order(self, clases, v, epoch=None) -> bool:
        """
        Makes the camera look for what the operator asked, if the order is new.

        An order is new when it comes from another station session (epoch) or carries a higher
        version than the last one taken. Orders arrive by two paths -- a packet on the data plane,
        or a poll of the station -- and both repeat, so without that test the camera would be
        reset on every report. A class the detector cannot emit is refused rather than raised: a
        wrong button pressed on the ground must not stop the protocol in the air. The refusal is
        kept and travels in the next report, so the operator sees that the drone did not comply
        instead of assuming it did.

        Returns True when the order was taken as new, whether applied or refused.
        """
        if v is None:
            return False
        if epoch == self._orden_epoca and self._orden_v is not None and v <= self._orden_v:
            return False
        self._orden_epoca, self._orden_v = epoch, v
        try:
            self.camera.set_classes(list(clases) if clases else None)
            self._orden_rechazo = None
        except ValueError as e:
            self._orden_rechazo = str(e)
        return True

    def _poll_station(self) -> None:
        """Asks the ground station for its search order, when a station URL is configured."""
        if not self.station_url:
            return
        ahora = self.provider.current_time()
        if ahora < self._proxima_consulta:
            return
        import urllib.request
        try:
            with urllib.request.urlopen(self.station_url.rstrip("/") + "/buscar",
                                        timeout=STATION_TIMEOUT_S) as r:
                d = json.loads(r.read())
        except Exception:
            self._proxima_consulta = ahora + STATION_RETRY_S
            return
        self.apply_search_order(d.get("clases"), d.get("v"), d.get("epoca"))

    def _estado_busqueda(self) -> Dict:
        """
        What the camera is searching for now, and the last order behind it.

        The applied state, not the requested one: it is what shows an operator a drone that has
        not heard the order yet, or refused it. The names the detector can emit travel too, so
        the station offers buttons for those rather than a fixed list -- a button for a class
        the model lacks is a request that can only be refused.

        'apariencia' says whether this aircraft can honour a refusal at all. _fue_descartado
        needs the candidate's own appearance vector and returns False without one, so a board
        whose mission left reid_model unset is deaf to every verdict the operator gives. That
        used to be silent on the station, which makes a button that does nothing look broken
        rather than unavailable.
        """
        clases = getattr(self.camera, "classes", None)
        conocidas = getattr(self.camera, "known_classes", None)
        return {
            "clases": sorted(clases) if clases else None,
            "v": self._orden_v,
            "epoca": self._orden_epoca,
            "rechazo": self._orden_rechazo,
            "conocidas": sorted(conocidas) if conocidas else None,
            "apariencia": getattr(self.camera, "reid_model", None) is not None,
        }

    def _corroboracion(self, poi) -> List:
        """
        Which other drones are reporting this same target, right now.

        Only recent neighbours count. A sighting from five minutes ago says where something
        was, not that it is still there, and treating the two as one claim is how a stale
        report gets promoted into a confirmation.
        """
        ahora = self.provider.current_time()
        fuera = []
        for quien, visto in self._ajenos.items():
            if ahora - visto["t"] > self.ventana_ajenos_s:
                continue
            if any(mismo_objetivo(poi, otro) for otro in visto["pois"]):
                fuera.append(quien)
        return sorted(fuera)

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

    def _see(self) -> None:
        """One camera cycle: detect, back-project, store ground impacts.

        The cycle is skipped outright with no telemetry yet, and skipped again when the pose
        source is unavailable for this frame: a frame cast from a guessed pose is worse than
        no frame, because nothing downstream can tell the two apart.

        Each impact is handed to the identity layer with its slant range, the clock and the
        class the detector gave it. The range is camera to point on the ground, and it is what
        makes a far target worse placed than a near one: a heading error moves the impact by
        that distance times the angle. The clock travels so that maturity is measured rather
        than inferred from a frame count times a rate the caller merely promised, and the
        class travels because the identity layer votes on it across the track and refuses to
        merge two names into one candidate.
        """
        if self._position is None:
            return
        yaw = self.yaw_source()
        if yaw is None:
            return
        self._frames_seen += 1

        cam_cfg = self.camera.camera
        actitud = self.attitude_source() if self.attitude_source is not None else None
        alabeo, cabeceo = actitud if actitud is not None else (0.0, 0.0)
        self._apuntar_foco(yaw, alabeo, cabeceo)
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
                body_pitch_deg=cabeceo,
                body_roll_deg=alabeo,
            )
            impact = _ground_impact(origin, direction, self.ground_z)
            if impact is None:
                continue
            cls = det.get("cls")
            extent = self.ground_extent_m.get(cls) if cls else None
            if extent:
                impact = _footprint_center(self._position, impact, extent)
            rango = math.sqrt((impact[0] - self._position[0]) ** 2
                              + (impact[1] - self._position[1]) ** 2
                              + (self._position[2] - self.ground_z) ** 2)
            self._impacts.append(impact)
            self._confs.append(det["conf"])
            self._clases.append(cls)
            track_id = det.get("track_id")
            if self.identity is not None and track_id is not None:
                self.identity.observe(
                    frame=self._frames_seen,
                    track_id=int(track_id),
                    ground_xy=impact,
                    conf=det["conf"],
                    emb=det.get("emb"),
                    crop=det.get("crop"),
                    t=self.provider.current_time(),
                    cls=cls,
                    range_m=rango,
                )

    def _report(self) -> None:
        """
        Broadcasts the current POI list. With an identity layer: one POI per candidate, mobiles
        first. Without one, or before any candidate has enough evidence: the single dominant POI
        by RANSAC consensus over all stored impacts.

        The operator's refusals are applied here and not inside the identity layer, on purpose.
        That layer's job is to say what it has seen, and it has still seen this; what changed is
        that a human looked at it and said no. Keeping the two apart means the refusal costs
        nothing to undo and leaves the layer's measurements untouched.

        WITH an identity layer, no candidate means nothing is known yet and the drone says
        exactly that. It used to fall back to a RANSAC consensus over every impact stored,
        which on flight 3 put a point between the operator and the equipment box: the mixture
        the identity layer exists to prevent, reported at the moment the drone knows least. The
        consensus stays for a drone with no identity layer, where it is the design, and there it
        is named from the inliers only, because the rejected impacts are the ones the consensus
        decided were not this object and letting them vote would be answering with the noise.

        Nothing found is exactly when the drone must still speak. From the ground, a drone that
        sees nobody and a drone that has died look identical: both are silence, and an operator
        who cannot tell them apart has to assume the worst and abort. One empty beat per report
        period buys the difference for a few dozen bytes.

        What the message carries beyond the POIs, and why:

        fps_real, slots_perdidos
            What the loop actually delivered over the last interval, so the gap between
            configured and real can never again be something only a stopwatch would find.

        origen_gps
            The frame these metres are measured in, so the receiver never has to be told
            separately. It cannot be: when the mission is loaded without an origin the runner
            resolves one from the GPS fix at that moment, which nobody can know in advance to
            type into a ground station. Read defensively, because IProvider does not declare it
            -- the embedded runtime's provider carries it, a test harness does not. None is a
            valid answer and means local metres only, which is what every desk run produces.

        pos
            Where the drone itself is. A station that plots targets but not the aircraft asks
            the operator to hold the most basic fact in their head, and it hides the one thing
            that explains a bad fix: a drone that barely moved gives rays that barely cross.
            Metres in the same frame as the POIs, so nothing needs converting.

        salud
            What the board says about its own current and temperature. In every report and not
            one in ten, because a voltage dip lasts an instant and the next report may not
            exist.

        Each POI's crop arrives from the identity layer as raw JPEG bytes and is encoded here
        rather than there, so the identity layer stays free of transport concerns. Its
        embedding goes out at half precision: the vector is only ever compared by cosine, and
        at float16 that comparison is unchanged to six decimals while the field costs 1.4 kB
        instead of 2.7 -- and this link is a 4G dongle, not a lab cable. Corroboration is heard
        straight off the air, so a ground station that reaches only one of the drones still
        learns that two of them agree.
        """
        import base64

        pois = (self.identity.candidates(preliminary=self.report_preliminary,
                                         now=self.provider.current_time())
                if self.identity is not None else [])
        if self._descartados:
            pois = [p for p in pois if not self._fue_descartado(p)]
        latido = False

        if not pois:
            if self.identity is not None or len(self._impacts) < MIN_MEASUREMENTS:
                latido = True
            else:
                impacts = np.asarray(self._impacts)
                estimate, inliers = _ransac_consensus(impacts, self._rng)
                votos: Dict[str, int] = {}
                for cls, es_inlier in zip(self._clases, inliers):
                    if cls and es_inlier:
                        votos[cls] = votos.get(cls, 0) + 1
                pois = [{
                    "x": round(float(estimate[0]), 2),
                    "y": round(float(estimate[1]), 2),
                    "cls": dominant_class(votos),
                    "n_obs": int(len(impacts)),
                    "n_inliers": int(inliers.sum()),
                    "conf_mean": round(float(np.mean(self._confs)), 3),
                }]

        for p in pois:
            crop = p.pop("crop", None)
            if crop:
                p["crop"] = base64.b64encode(crop).decode("ascii")
            emb = p.pop("emb", None)
            if emb is not None:
                p["emb"] = base64.b64encode(
                    np.asarray(emb, dtype=np.float16).tobytes()).decode("ascii")
            otros = self._corroboracion(p)
            if otros:
                p["corroborado_por"] = otros

        ritmo = self._ritmo()
        message = {
            "type": "vision_poi",
            "sender": self.provider.get_id(),
            "time": self.provider.current_time(),
            "frames_seen": self._frames_seen,
            "fps_real": ritmo["fps_real"],
            "slots_perdidos": ritmo["slots_perdidos"],
            "slots_perdidos_total": ritmo["slots_perdidos_total"],
            "latido": latido,
            "origen_gps": self._gps_origin(),
            "pos": [round(float(self._position[0]), 2),
                    round(float(self._position[1]), 2),
                    round(float(self._position[2]), 2)] if self._position is not None else None,
            "buscando": self._estado_busqueda(),
            "salud": salud_electrica(),
            "pois": pois,
        }
        self.provider.send_communication_command(
            BroadcastMessageCommand(json.dumps(message)))
