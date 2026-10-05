"""
Camera providers for vision-based geolocation.

Two classes expose the same method, detect(pos, yaw). Consumers never learn which one they
received: simulation code gets SimulatedCamera, the real drone gets OnboardCamera, and the consumer
code is identical in both cases.

Contract of detect(pos, yaw):
    Input:
        pos: (x, y, z) in meters, local ENU frame (x=East, y=North, z=Up).
        yaw: degrees. 0 = North, 90 = East, clockwise.
    Output:
        List of detections, one dict per detection. An empty list means nothing was detected.
        {'px': float, 'py': float, 'conf': float}
        - 'px' is the horizontal center of the detection.
        - 'py' is the BOTTOM edge, not the center: the point where the object touches the ground.
        - 'conf' is the detector confidence in (0, 1]. It feeds the view selector, which weights
          it heavily; it is not informational.
    Optional fields:
        - 'cls': the name the detector gave the class, as the model spells it ('person',
          'pedestrian', 'car'). Present whenever the camera knows one. It travels through the
          identity layer to the report: with more than one class enabled, a coordinate with no
          name is not actionable, and two names must never be merged into one candidate.
        - 'emb': appearance embedding, 512 normalized float32 (OSNet). Present only when the
          camera can compute it. Read it with det.get('emb'), never det['emb'].
        - 'track_id': stable integer identity assigned by the tracker. Present only when the
          camera runs one. The identity layer (identity.py) requires it.

CONSTANTS THAT ARE DECISIONS
    EMB_DIST_OBJETIVO
        Appearance distance under which a box the detector doubted is kept because it looks
        like the target the operator pointed at. No weight is retrained and nothing is adapted:
        the template is the embedding of the crop the operator clicked, so switching this off
        returns the system to exactly what it was. THE NUMBER IS NOT SAFE YET, for the same
        reason EMB_DIST_REUNE is not: it was chosen by looking at the stretch it is judged on,
        so the gate stays shut unless a caller asks for it by passing a distance.

    BANDA_BAJA
        Floor of the BYTE band: the lowest score the detector is asked for when a tracker is
        running. It is BoT-SORT's own track_low_thresh, because a box under the tracker's floor
        is discarded before association, so asking for it is paying the appearance model for
        something nobody will look at.

Design notes and measured values behind the defaults are collected in NOTES.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from uav_vision.camera_config import ARDUCAM_MODULE_3, DEFAULT_CAMERA, CameraConfig
from uav_vision.confidence import confidence_to_pixel_sigma_model, simulate_confidence
from uav_vision.pinhole_local import project_to_pixel

Detection = Dict[str, float]


def dentro_del_foco(det, foco) -> bool:
    """Whether a detection falls inside the window the operator's target is in."""
    if not foco:
        return False
    return (abs(det["px"] - foco["cx"]) <= foco["radio"]
            and abs(det["py"] - foco["cy"]) <= foco["radio"])


EMB_DIST_OBJETIVO = 0.85

BANDA_BAJA = 0.2


def se_parece_al_objetivo(det, foco) -> bool:
    """Whether a doubted detection looks like the target the operator fixed.

    Unlike the window, this does not ask where the box is. That is the point of it: the window is
    the projection of a ground position and goes wrong exactly when the aircraft's attitude is
    least certain, while an appearance match does not care whether the geometry agrees.
    """
    if not foco:
        return False
    plantilla, tope, emb = foco.get("plantilla"), foco.get("emb_dist"), det.get("emb")
    if plantilla is None or tope is None or emb is None:
        return False
    d = np.asarray(emb, dtype=np.float32) - np.asarray(plantilla, dtype=np.float32)
    return float(np.linalg.norm(d)) <= float(tope)


def solo_confirmadas(detections, threshold, foco=None):
    """
    Drops the weak boxes the tracker did not claim, except inside the operator's window.

    With the BYTE band open the detector returns boxes below the reporting threshold. Those
    are worth having only as evidence that something already being followed is still there:
    a box the tracker attached to an existing track. One that arrives unattached is a guess,
    and it must not reach the fusion, which unlike the identity layer does not check
    'track_id' before using a detection.

    The exception is the window around a target the operator pointed at. There the guess is
    worth taking: somebody already said there is a person in that square, so a weak box there
    is far more likely to be them than to be a new invention, and a false one lands where the
    tracker will attach it to the track that is already running instead of opening a point of
    its own. Measured over the balcony window of the 02ago flight, lowering the threshold to
    0.10 inside it costs no computing at all, because the detector had already scored those
    boxes and was throwing them away; what it buys and what it costs are in NOTES.md.

    The second exception does not look at position at all: a box the detector doubted is kept if
    its appearance matches the template of the target the operator pointed at. See
    EMB_DIST_OBJETIVO for what that buys and for why the distance is not settled yet. It costs no
    extra computing either, because with the band open the embedding of a doubted box is already
    computed before it is filtered.
    """
    return [d for d in detections
            if d["conf"] >= threshold or "track_id" in d
            or (dentro_del_foco(d, foco) and d["conf"] >= foco["umbral"])
            or se_parece_al_objetivo(d, foco)]


class SimulatedCamera:
    """
    Simulation camera: there is no image, the pixel is computed geometrically by projecting a
    known target position through the camera model.

    pitch_deg has no default on purpose. The camera mount angle is a physical property of each
    deployment and hiding it as a default is how independent copies of the geometry drift apart.
    Whoever creates the camera states the pitch it models.
    """

    def __init__(
        self,
        target: Sequence[float],
        pitch_deg: float,
        camera: CameraConfig = DEFAULT_CAMERA,
        rng: Optional[np.random.Generator] = None,
        pixel_noise: bool = True,
        noise_model: str = "heuristic",
        cls: str = "person",
    ) -> None:
        self.target = tuple(target)
        self.pitch_deg = pitch_deg
        self.camera = camera
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pixel_noise = pixel_noise
        self.noise_model = noise_model
        self.cls = cls

    def detect(self, pos: Sequence[float], yaw: float) -> List[Detection]:
        pixel = project_to_pixel(
            pos,
            self.target,
            yaw,
            self.pitch_deg,
            self.camera.focal_length_px,
            self.camera.image_width,
            self.camera.image_height,
            self.camera.principal_point,
        )
        if pixel is None:
            return []

        conf = simulate_confidence(
            pixel,
            self.camera.image_center,
            self.camera.max_radius,
            self.rng,
        )
        px, py = pixel
        if self.pixel_noise:
            sigma = confidence_to_pixel_sigma_model(conf, self.noise_model)
            px = float(np.clip(px + self.rng.normal(0, sigma),
                               0, self.camera.image_width - 1))
            py = float(np.clip(py + self.rng.normal(0, sigma),
                               0, self.camera.image_height - 1))
        return [{"px": px, "py": py, "conf": conf, "cls": self.cls}]


def _solapan(a, b, umbral: float = 0.4) -> bool:
    """Whether two boxes are the same find, by intersection over union."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return union > 0 and inter / union >= umbral


class OnboardCamera:
    """
    Real camera: captures a frame with picamera2 and runs a YOLO detector on it.

    picamera2, ultralytics and boxmot are imported on the first capture, not at construction, so
    this module can be imported on machines that do not have them installed.

    Optional stages, enabled by constructor arguments:
        - reid_model: compute an OSNet appearance embedding per detection ('emb' field).
        - tracker: run BoT-SORT over the detections and attach a stable 'track_id' per
          detection. Requires fps, because the tracker's memory is measured in frames and only
          the declared rate makes a buffer duration meaningful.
        - crops: attach a small JPEG of each detection ('crop' field), for the ground
          station to verify what the onboard detector could not settle by itself.

    CLASES_PERSONA is the set of names that count as a person. Detection models trained on
    different datasets spell the class differently (COCO: 'person'; VisDrone: 'pedestrian',
    'people'), and filtering on a single name silently drops every detection when the model
    changes.

    The constructor arguments that are not self-explanatory:

    low_band
        The BYTE band. With a tracker running, the detector is asked for boxes down to this
        score, but only those the tracker attached to an existing track are reported. A person
        the detector merely doubted keeps their track alive; a box out of nowhere does not
        become a target, because new_track_thresh still guards that. None disables the band.

    rot180
        When the camera is mounted upside-down the ISP un-flips the image at capture time
        (hflip+vflip). That remapping moves the calibrated principal point, so the effective
        config must be the reflected one; see CameraConfig.rotated_180().

    compensate_motion
        Camera-motion compensation. The drone moves, so every box shifts in the image between
        frames and the tracker's prediction misses the next box unless the background motion is
        removed first. On by default, and both what it recovers and what it costs are measured
        in NOTES.md.

    startup_pause_s
        Seconds to sit idle between the heavy start-up steps. Default 0, which is no change for
        anything on mains. On battery it is the only lever software has against a board that
        dies while opening the camera and loading the models back to back: what kills it is the
        step, not the level, and this pulls the steps apart.

    crops, crop_side_px, crop_quality, crop_margin
        A crop is the cheapest thing the drone can say that the ground can check. The link
        budget is the constraint the whole architecture was built around, since video off the
        drone is not affordable, so these are sized in kilobytes and not megabytes: one small
        packet per detection rather than a stream. The margin is there because a box drawn
        tight on a person at altitude cuts off the context that makes the verifier's job
        possible, and it buys that back for almost no bytes.

    tile_every, tile_side, tile_conf
        A second pass over tiles of the frame at native resolution, every Nth frame; 0 turns it
        off. The frame is downscaled to the model's input before inference, so a person at the
        size limit disappears, and a tiled pass skips that reduction. It costs six times the
        inference, which is why it does not run on every frame and does not need to: the
        identity layer asks for a tenth of the frames. NOTES.md has what it recovers.

    The window the operator's verdict opens lives in self.foco, in pixels of the current frame,
    or None. It is set once per frame by whoever knows the geometry, which is the protocol,
    because it can project the target's ground position; never by the camera, which has no idea
    where anything is.
    """

    CLASES_PERSONA = frozenset({"person", "pedestrian", "people"})

    def __init__(
        self,
        model: str,
        threshold: float = 0.3,
        low_band: Optional[float] = BANDA_BAJA,
        classes: Optional[Sequence[str]] = None,
        camera: CameraConfig = ARDUCAM_MODULE_3,
        rot180: bool = True,
        reid_model: Optional[str] = None,
        tracker: bool = False,
        tracker_name: str = "botsort",
        fps: Optional[float] = None,
        track_buffer_s: float = 8.0,
        compensate_motion: bool = True,
        startup_pause_s: float = 0.0,
        crops: bool = False,
        crop_side_px: int = 128,
        crop_quality: int = 70,
        crop_margin: float = 0.25,
        tile_every: int = 0,
        tile_side: int = 960,
        tile_conf: float = 0.55,
    ) -> None:
        self.model = model
        self.tile_every = int(tile_every)
        self.tile_side = int(tile_side)
        self.tile_conf = float(tile_conf)
        self._n_frames = 0
        self._ultimo_frame = None
        self.threshold = threshold
        self.low_band = low_band
        self.foco = None
        self.classes = frozenset(classes) if classes is not None else self.CLASES_PERSONA
        self.camera = camera.rotated_180() if rot180 else camera
        self.rot180 = rot180
        self.reid_model = reid_model
        if tracker and fps is None:
            raise ValueError(
                "tracker=True requires fps: the tracker buffer is measured in frames and "
                "has no meaning without the capture rate.")
        self.rastreador_habilitado = tracker
        self.tracker_name = tracker_name
        self.fps = fps
        self.track_buffer_s = track_buffer_s
        self.compensate_motion = compensate_motion
        self.startup_pause_s = startup_pause_s
        self.crops = crops
        self.crop_side_px = crop_side_px
        self.crop_quality = crop_quality
        self.crop_margin = crop_margin
        self._picam: Any = None
        self._yolo: Any = None
        self._reid: Any = None
        self._tracker: Any = None

    def set_classes(self, classes: Optional[Sequence[str]] = None) -> None:
        """Changes what counts as a target, mid-flight.

        Adds no inference time: the detector is called without a class filter and
        already scores every class it knows on every frame; self.classes only decides
        which of those survive the loop in detect(), and switching reloads no model.
        What a wider search does cost is everything done per surviving box. Measured on
        the Pi 5 with the flight chain on 600 frames of the 02-ago flight
        (scripts/medir_multiclase_pi.py): about 33 ms per box, mostly the OSNet
        embedding, so adding cars to people took the median frame from 206 to 245 ms
        (4.85 to 4.08 fps) in a scene with 1.1 cars per frame, without throttling.

        Pass None to go back to people. Names must be names the model emits --
        see known_classes -- because a typo would silently report nothing.

        "person" is translated rather than looked up, because it is the name an operator uses
        and not the name every model emits: a COCO model says 'person', a VisDrone one
        'pedestrian' and 'people'. Any of the three asks for people and is answered with the
        names THIS model uses for them. Without that, the station's person button raises on the
        model that actually flies.
        """
        if classes is None:
            self.classes = self.CLASES_PERSONA
            return
        pedidas = frozenset(classes)
        conocidas = self.known_classes
        if conocidas:
            nombres_persona = self.CLASES_PERSONA & set(conocidas)
            if pedidas & self.CLASES_PERSONA and nombres_persona:
                pedidas = (pedidas - self.CLASES_PERSONA) | nombres_persona
            desconocidas = pedidas - set(conocidas)
            if desconocidas:
                raise ValueError(
                    "the detector does not emit %s; it knows %s"
                    % (sorted(desconocidas), conocidas))
        self.classes = pedidas

    @property
    def known_classes(self) -> List[str]:
        """Class names this detector can emit, or [] before the model is loaded."""
        if self._yolo is None:
            return []
        return sorted(self._yolo.names.values())

    def _power_on(self) -> None:
        """Starts the camera and loads the models. Called automatically on the first capture.

        The two-second sleep after start() is the sensor stabilising its exposure, and the
        capture that follows is a warm-up, not a frame anybody uses.
        """
        if self._picam is not None:
            return

        import time

        from libcamera import Transform
        from picamera2 import Picamera2
        from ultralytics import YOLO

        picam = Picamera2()
        tf = Transform(hflip=1, vflip=1) if self.rot180 else Transform()
        picam.configure(
            picam.create_still_configuration(
                main={"size": (self.camera.image_width, self.camera.image_height)},
                transform=tf,
            )
        )
        picam.start()
        time.sleep(2)
        self._picam = picam
        self._first_frame(picam)

        self._settle("camera arriba")
        self._yolo = YOLO(self.model)

        if self.reid_model is not None:
            self._settle("detector cargado")
            from boxmot.reid.core.reid import ReID
            self._reid = ReID(self.reid_model, device="cpu", half=False)

        if self.rastreador_habilitado:
            self._build_tracker()

    def _settle(self, tras: str) -> None:
        """Lets the supply recover before the next heavy step, when asked to."""
        if self.startup_pause_s <= 0:
            return
        import time
        print("[camera] %s: %.1f s de respiro antes del siguiente escalon"
              % (tras, self.startup_pause_s), flush=True)
        time.sleep(self.startup_pause_s)

    ESPERA_PRIMER_FRAME_S = 8.0
    INTENTOS_ENCENDIDO = 3

    def _first_frame(self, picam) -> None:
        """Warm-up capture with a watchdog: retries the camera instead of hanging on it.

        The sensor is detected over I2C and enumerated long before it will actually stream, and
        sometimes it does not stream at all: libcamera reports "Camera frontend has timed out"
        and the capture call never returns. Without a watchdog, that failure mode is a mission
        lost with no diagnosis: the board alive, the protocol running its timer, and zero
        detections forever, because the first capture never returned. Better to spend a few
        seconds retrying, and to fail loudly if it will not come. NOTES.md has when it was seen.

        The capture runs on a DAEMON thread, because a capture wedged inside the driver may
        never return and must not keep the process alive. Between attempts the device is both
        stopped and closed: stop() alone keeps it acquired, and the reopen fails too.

        The event is bound PER ATTEMPT, as a default argument, and that is not a style choice.
        Closing over it would share one cell across every attempt, so a thread left wedged by
        attempt 1 would, on unwedging, set the event of attempt 2 -- and this method would
        return as though the camera had delivered a frame when it had not. That is reachable
        exactly in the failure this watchdog exists for.
        """
        import threading
        import time

        for intento in range(1, self.INTENTOS_ENCENDIDO + 1):
            listo = threading.Event()

            def capture(ev=listo):
                try:
                    picam.capture_array()
                finally:
                    ev.set()

            threading.Thread(target=capture, daemon=True).start()
            if listo.wait(self.ESPERA_PRIMER_FRAME_S):
                return
            if intento == self.INTENTOS_ENCENDIDO:
                raise RuntimeError(
                    "la camera no entrego un frame en %d intentos de %.0f s. El sensor "
                    "responde por I2C pero no transmite: probar de nuevo en unos segundos, y "
                    "si persiste revisar el cable plano (I2C tolera un contacto marginal, las "
                    "lineas CSI no)." % (self.INTENTOS_ENCENDIDO, self.ESPERA_PRIMER_FRAME_S))
            try:
                picam.stop()
                picam.close()
            except Exception:
                pass
            time.sleep(2.0)
            picam.start()
            time.sleep(2.0)

    def _build_tracker(self) -> None:
        """Builds the tracker named by tracker_name, with the buffer converted to frames.

        The two-way import this used to carry lives in uav_vision.trackers now, together with
        the rest of the version juggling: boxmot moved its trackers between the version on one
        board and the version on the other, and it named its parameters differently in each
        tracker. Nothing about boxmot's API appears here any more, which is what lets the
        aircraft fly a different tracker without this file changing again.

        The thresholds are NOT in that module, and deliberately: they are this deployment's
        calibration, measured against the flight, and the adapter's job is to deliver them and
        to report what the target could not take. A tracker that silently did not receive them
        would be running on somebody else's defaults, so the refusal is printed.

        Two of the settings are not boxmot's defaults. reid_model stays None because the
        embeddings are supplied externally through embs=, while with_reid still has to be on
        for appearance matching, since BotSort demands embeddings when it is and must run
        motion-only when there is no ReID model at all. cmc_method is 'sof' and not boxmot's
        default 'ecc', which is worse on both axes; the measurement is in NOTES.md.
        """
        from uav_vision.trackers import construir

        self._tracker, _, ignorados, aproximados = construir(
            self.tracker_name,
            embs_propias=self.reid_model is not None,
            use_cmc=self.compensate_motion,
            cmc_method="sof",
            track_high_thresh=0.35,
            track_low_thresh=0.2,
            new_track_thresh=0.4,
            track_buffer=max(2, round(self.track_buffer_s * self.fps)),
            match_thresh=0.85,
        )
        if ignorados or aproximados:
            print("[camera] %s no recibe %s%s"
                  % (self.tracker_name, sorted(ignorados),
                     "; renombrados solo por rol: %s" % aproximados if aproximados else ""),
                  flush=True)

    def close(self) -> None:
        """Releases the camera device, both stopped and closed.

        stop() alone keeps the device acquired, and then no other Picamera2 instance can open
        it, a later OnboardCamera included.
        """
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None

    def ultimo_marco_jpeg(self, calidad: int = 85) -> Optional[bytes]:
        """The last frame the detector looked at, encoded, for whoever asks to look at it themselves.

        It is the frame the detector SAW and not a fresh capture, so what the ground judges is the same
        picture the aircraft judged: a new capture seconds later would answer a different question and
        quietly excuse a miss. Returns nothing before the first detection rather than an empty image,
        because an empty image looks like an answer.
        """
        if self._ultimo_frame is None:
            return None
        import cv2
        ok, buf = cv2.imencode(".jpg", self._ultimo_frame, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
        return buf.tobytes() if ok else None

    def _cajas_de_fichas(self, frame, ya_vistas) -> List[tuple]:
        """People the whole-frame pass missed, found by running the tiles of the frame at native size.

        Only boxes that land where the frame found nothing are returned. Replacing the frame's own
        detections with the tiles' costs precision, because a tile decides on a fragment of the scene
        and calls a shadow a person more readily; adding to them does not, which is what the flight's
        own footage showed. The tiles are also asked for more confidence than the frame is: alone they
        are the less reliable witness, and the threshold that keeps them useful was chosen on the
        01ago flights, never on the flight this is judged against.
        """
        if not self.tile_every or self._n_frames % self.tile_every:
            return []
        alto, ancho = frame.shape[:2]
        lado = min(self.tile_side, alto, ancho)
        salto = int(lado * 0.8)
        previas = [b.xyxy[0].cpu().numpy() for b in ya_vistas]
        salida: List[tuple] = []
        ys = sorted({*range(0, max(1, alto - lado + 1), salto), max(0, alto - lado)})
        xs = sorted({*range(0, max(1, ancho - lado + 1), salto), max(0, ancho - lado)})
        for y0 in ys:
            for x0 in xs:
                ficha = frame[y0:y0 + lado, x0:x0 + lado]
                for caja in self._yolo(ficha, verbose=False, conf=self.tile_conf)[0].boxes:
                    nombre = self._yolo.names[int(caja.cls[0])]
                    if nombre not in self.classes:
                        continue
                    b = caja.xyxy[0].cpu().numpy()
                    xy = np.array([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], dtype=float)
                    if any(_solapan(xy, otra) for otra in previas):
                        continue
                    previas.append(xy)
                    salida.append((xy, float(caja.conf[0]), nombre))
        return salida

    def detect(self, pos: Sequence[float], yaw: float) -> List[Detection]:
        """One capture, detected, tracked and described, as the detect() contract promises.

        pos and yaw are unused and deleted at once: the real photograph already contains what it
        contains. They are in the signature so the contract matches the simulated camera, which
        does need them.

        THE CHANNELS ARRIVE RGB. picamera2 labels this configuration "BGR888", but that name is
        libcamera's and lists the components in the opposite order to the one the array actually
        arrives in. Everything downstream is OpenCV-shaped and expects BGR: ultralytics assumes
        it for a raw array, the ReID model assumes it, and cv2.imencode assumes it when the crop
        is written. Left alone, red and blue are swapped for all three at once. Converting the
        whole frame once, here, is what keeps the three consumers agreeing, and NOTES.md has
        what it costs and what it was hiding.

        The floor handed to the detector drops below the reporting threshold whenever a tracker
        or a focus window is running. That costs no extra inference, because the detector had
        already scored those boxes and was discarding them, and the band is filtered back out
        by solo_confirmadas unless the tracker claimed the box.

        Each detection carries 'py' as the BOTTOM edge of the box, which is the point touching
        the ground, and the class name the model emitted, so a report can say WHAT it found and
        not just where: with more than one class enabled, a coordinate with no name is not
        actionable, because the ground station cannot tell a person from a car.
        """
        del pos, yaw

        import cv2

        self._power_on()
        frame = cv2.cvtColor(self._picam.capture_array(), cv2.COLOR_RGB2BGR)
        piso = self.threshold
        if self._tracker is not None and self.low_band is not None:
            piso = min(self.threshold, self.low_band)
        if self.foco is not None:
            piso = min(piso, self.foco["umbral"])
        self._ultimo_frame = frame
        resultados = self._yolo(frame, verbose=False, conf=piso)
        self._n_frames += 1
        extra = self._cajas_de_fichas(frame, resultados[0].boxes)

        detections: List[Detection] = []
        cajas: List[np.ndarray] = []
        for caja in resultados[0].boxes:
            if self._yolo.names[int(caja.cls[0])] not in self.classes:
                continue
            xyxy = caja.xyxy[0].cpu().numpy()
            x1, _y1, x2, y2 = xyxy
            detections.append({
                "px": float((x1 + x2) / 2),
                "py": float(y2),
                "conf": round(float(caja.conf[0]), 3),
                "cls": self._yolo.names[int(caja.cls[0])],
            })
            cajas.append(xyxy)
        for xyxy, conf, nombre in extra:
            x1, _y1, x2, y2 = xyxy
            detections.append({"px": float((x1 + x2) / 2), "py": float(y2),
                               "conf": round(float(conf), 3), "cls": nombre})
            cajas.append(xyxy)

        fingerprints: List[np.ndarray] = []
        if self._reid is not None and detections:
            fingerprints = self._fingerprints(frame, cajas)
            for det, emb in zip(detections, fingerprints):
                det["emb"] = emb

        if self._tracker is not None:
            self._track(frame, detections, cajas, fingerprints)

        if self.crops and detections:
            for det, caja in zip(detections, cajas):
                det["crop"] = self._crop(frame, caja)

        if piso < self.threshold:
            detections = solo_confirmadas(detections, self.threshold, self.foco)

        return detections

    def set_focus(self, cx=None, cy=None, radio_px=320.0, umbral=0.10,
                  plantilla=None, emb_dist=None) -> None:
        """Points the low threshold at one square of the image, or clears it when cx is None.

        Zooming into that square was tried first and is much worse: cropping it and enlarging it
        took recall from 43.9 % to 11.2 %, and enlarging further to 2.3 %, because the detector was
        trained on VisDrone and only recognises people of about 28 pixels. Making the target bigger
        stops it looking like a person. Lowering the threshold there costs nothing and works.

        plantilla and emb_dist are the appearance of the target and how far a doubted box may be
        from it and still be kept. Both None leaves that gate shut and the behaviour unchanged.
        """
        if cx is None:
            self.foco = None
            return
        self.foco = {"cx": float(cx), "cy": float(cy), "radio": float(radio_px),
                     "umbral": float(umbral),
                     "plantilla": (np.asarray(plantilla, dtype=np.float32)
                                   if plantilla is not None else None),
                     "emb_dist": float(emb_dist) if emb_dist is not None else None}

    def _crop(self, frame, caja) -> bytes:
        """
        Returns a small JPEG around one detection, for the ground station to verify.

        The crop is squared before scaling: a person's box is tall and thin, and letting the
        resize squash it hands the verifier a distorted body it was never trained on.
        """
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in caja]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        lado = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * self.crop_margin)
        a = max(0, int(cx - lado / 2)), max(0, int(cy - lado / 2))
        b = min(w, int(cx + lado / 2)), min(h, int(cy + lado / 2))
        parche = frame[a[1]:b[1], a[0]:b[0]]
        if parche.size == 0:
            return b""
        if max(parche.shape[:2]) > self.crop_side_px:
            e = self.crop_side_px / max(parche.shape[:2])
            parche = cv2.resize(parche, (max(1, int(parche.shape[1] * e)),
                                         max(1, int(parche.shape[0] * e))),
                                interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", parche,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.crop_quality])
        return buf.tobytes() if ok else b""

    def _track(
        self,
        frame,
        detections: List[Detection],
        cajas: List[np.ndarray],
        fingerprints: List[np.ndarray],
    ) -> None:
        """
        Runs the tracker on this frame's boxes and attaches 'track_id' to each detection.

        Called on every frame, including frames with no detections: the tracker needs the empty
        frames to age out tracks that left the scene. Detections the tracker has not confirmed
        yet get no track_id; consumers skip those until it does.
        """
        if cajas:
            dts = np.array(
                [[*c, det["conf"], 0] for c, det in zip(cajas, detections)],
                dtype="float32")
            embs = np.asarray(fingerprints, dtype="float32") if fingerprints else None
        else:
            dts = np.empty((0, 6), dtype="float32")
            embs = None

        res = np.asarray(self._tracker.update(dts, frame, embs=embs))
        for fila in res:
            det_idx = int(fila[7])
            if 0 <= det_idx < len(detections):
                detections[det_idx]["track_id"] = int(fila[4])

    def _fingerprints(self, frame, cajas: List[np.ndarray]) -> List[np.ndarray]:
        """
        OSNet embeddings for all boxes of the frame, computed in a single batch. Vectors are
        normalized to unit length so cosine similarity reduces to a dot product.
        """
        out = self._reid.process({
            "fallback": True,
            "boxes": np.asarray(cajas, dtype="float32"),
            "image": frame,
        })
        feats = np.asarray(out["_features"], dtype="float32")

        fingerprints = []
        for v in feats:
            n = float(np.linalg.norm(v))
            fingerprints.append(v / n if n > 0 else v)
        return fingerprints
