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

Design notes and measured values behind the defaults are collected in NOTES.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from uav_vision.camera_config import ARDUCAM_MODULE_3, DEFAULT_CAMERA, CameraConfig
from uav_vision.confidence import confidence_to_pixel_sigma_model, simulate_confidence
from uav_vision.pinhole_local import project_to_pixel

Detection = Dict[str, float]


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
        # A real detector does not return the exact pixel. Pixel noise follows
        # sigma = C / confidence, so low-confidence detections are noisier. Disable only for
        # pure-geometry tests: without noise the fusion stage has nothing to reject and
        # simulation results become meaningless.
        self.pixel_noise = pixel_noise
        # Noise model name, resolved by confidence.confidence_to_pixel_sigma_model.
        self.noise_model = noise_model
        # What the simulated target is. The real camera always names its detections, so a
        # simulated one that stayed silent would let class-dependent code pass in simulation
        # and fail in the air -- which is the one thing this pair of classes exists to prevent.
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
        if pixel is None:  # out of frame or behind the camera
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
    """

    # Class names that count as a person. Detection models trained on different datasets name
    # the class differently (COCO: 'person'; VisDrone: 'pedestrian', 'people'); filtering on a
    # single name silently drops every detection when the model changes.
    CLASES_PERSONA = frozenset({"person", "pedestrian", "people"})

    def __init__(
        self,
        model: str,
        threshold: float = 0.3,
        classes: Optional[Sequence[str]] = None,
        camera: CameraConfig = ARDUCAM_MODULE_3,
        rot180: bool = True,
        reid_model: Optional[str] = None,
        tracker: bool = False,
        fps: Optional[float] = None,
        track_buffer_s: float = 8.0,
        compensate_motion: bool = False,
        startup_pause_s: float = 0.0,
        crops: bool = False,
        crop_side_px: int = 128,
        crop_quality: int = 70,
        crop_margin: float = 0.25,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.classes = frozenset(classes) if classes is not None else self.CLASES_PERSONA
        # When the camera is mounted upside-down the ISP un-flips the image at capture time
        # (hflip+vflip). That remapping moves the calibrated principal point, so the effective
        # config must be the reflected one; see CameraConfig.rotated_180().
        self.camera = camera.rotated_180() if rot180 else camera
        self.rot180 = rot180
        self.reid_model = reid_model
        if tracker and fps is None:
            raise ValueError(
                "tracker=True requires fps: the tracker buffer is measured in frames and "
                "has no meaning without the capture rate.")
        self.rastreador_habilitado = tracker
        self.fps = fps
        self.track_buffer_s = track_buffer_s
        # Camera-motion compensation improves tracking from a moving camera but costs CPU;
        # disabled until measured on the target hardware.
        self.compensate_motion = compensate_motion
        # A crop is the cheapest thing the drone can say that the ground can check. The link
        # budget is the constraint the whole architecture was built around -- video off the
        # drone is not affordable -- so these are sized in kilobytes, not megabytes: a 128 px
        # JPEG at quality 70 lands around 2-5 KB, which is one small packet per detection
        # rather than a stream.
        # Seconds to sit idle between the heavy start-up steps. Default 0: no change for
        # anything on mains. On battery it is the only lever software has against the failure
        # measured on 2026-08-25 -- the board died 3 s into opening the camera and loading the two
        # models, on a FULL pack, drawing 3.47 W. That is nowhere near saturating a 5 A UBEC,
        # so what kills it is the step itself, not the level. Opening the camera and loading
        # the models back to back stacks those steps; this pulls them apart.
        self.startup_pause_s = startup_pause_s
        self.crops = crops
        self.crop_side_px = crop_side_px
        self.crop_quality = crop_quality
        # A box drawn tight on a person at altitude cuts off the context that makes the
        # verifier's job possible; a margin buys that back for almost no bytes.
        self.crop_margin = crop_margin
        self._picam: Any = None
        self._yolo: Any = None
        self._reid: Any = None
        self._tracker: Any = None

    # -- what counts as a target ------------------------------------------

    def set_classes(self, classes: Optional[Sequence[str]] = None) -> None:
        """Changes what counts as a target, mid-flight.

        Costs nothing. The detector is called without a class filter and already
        scores every class it knows on every frame; self.classes only decides which
        of those survive the loop in detect(). Switching from people to vehicles
        reloads no model and adds no inference time.

        Pass None to go back to people. Names must be names the model emits --
        see known_classes -- because a typo would silently report nothing.
        """
        if classes is None:
            self.classes = self.CLASES_PERSONA
            return
        pedidas = frozenset(classes)
        conocidas = self.known_classes
        if conocidas:
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

    # -- hardware ---------------------------------------------------------

    def _power_on(self) -> None:
        """Starts the camera and loads the models. Called automatically on the first capture."""
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
        time.sleep(2)  # the sensor needs time to stabilize exposure
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

    # The sensor is detected over I2C and enumerated long before it will actually stream, and
    # sometimes it does not stream at all: libcamera reports "Camera frontend has timed out"
    # and the capture call never returns. Observed on 2026-08-25 -- five failures in a row within a
    # minute of boot and right after a process was killed mid-capture, then five successes out
    # of five once the board had been up a few minutes. Nothing in the configuration changed.
    #
    # Without this, that failure mode is a mission lost with no diagnosis: the Pi alive, the
    # protocol running its timer, and zero detections forever, because the first capture never
    # returned. Better to spend a few seconds retrying, and to fail loudly if it will not come.
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
        """Warm-up capture with a watchdog: retries the camera instead of hanging on it."""
        import threading
        import time

        for intento in range(1, self.INTENTOS_ENCENDIDO + 1):
            listo = threading.Event()

            def capture():
                try:
                    picam.capture_array()
                finally:
                    listo.set()

            # A daemon thread, because if the capture is wedged inside the driver it may never
            # return and must not keep the process alive.
            threading.Thread(target=capture, daemon=True).start()
            if listo.wait(self.ESPERA_PRIMER_FRAME_S):
                return
            if intento == self.INTENTOS_ENCENDIDO:
                raise RuntimeError(
                    "la camera no entrego un frame en %d intentos de %.0f s. El sensor "
                    "responde por I2C pero no transmite: probar de nuevo en unos segundos, y "
                    "si persiste revisar el cable plano (I2C tolera un contacto marginal, las "
                    "lineas CSI no)." % (self.INTENTOS_ENCENDIDO, self.ESPERA_PRIMER_FRAME_S))
            # stop() alone keeps the device acquired; without close() the reopen fails too.
            try:
                picam.stop()
                picam.close()
            except Exception:
                pass
            time.sleep(2.0)
            picam.start()
            time.sleep(2.0)

    def _build_tracker(self) -> None:
        """Builds the BoT-SORT tracker, with the buffer converted from seconds to frames."""
        from boxmot.trackers.bbox.botsort import BotSort

        self._tracker = BotSort(
            reid_model=None,  # embeddings are supplied externally via embs=
            # BotSort requires embeddings when appearance matching is on; without a ReID model
            # it must run motion-only.
            with_reid=self.reid_model is not None,
            use_cmc=self.compensate_motion,
            track_high_thresh=0.35,
            track_low_thresh=0.2,
            new_track_thresh=0.4,
            track_buffer=max(2, round(self.track_buffer_s * self.fps)),
            match_thresh=0.85,
        )

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            # stop() alone keeps the device acquired; without close() no other
            # Picamera2 instance (a later OnboardCamera included) can open it.
            self._picam.close()
            self._picam = None

    # -- contract ---------------------------------------------------------

    def detect(self, pos: Sequence[float], yaw: float) -> List[Detection]:
        # pos and yaw are unused: the real photo already contains what it contains. They are in
        # the signature so the contract matches the simulated camera.
        del pos, yaw

        import cv2

        self._power_on()
        # picamera2 labels this configuration "BGR888", but that name is libcamera's and lists
        # the components in the opposite order to the one the array actually arrives in: what
        # comes back is R,G,B. Everything downstream is OpenCV-shaped and expects B,G,R --
        # ultralytics assumes it for a raw array, the ReID model assumes it, and cv2.imencode
        # assumes it when the crop is written. Left alone, red and blue are swapped for all
        # three at once.
        #
        # Measured on the real board (2026-08-25): a person the detector found at 0.887 with the
        # channels swapped scores 0.909 once corrected -- small, and not the reason the
        # VisDrone weights find nothing indoors, which is a domain gap. The reason to fix it
        # is the crop: it is the photograph an operator looks at to decide whether to send
        # someone to that point, and it was arriving with blue skin.
        #
        # Converting the whole frame once, here, is what keeps the three consumers agreeing.
        # It costs 1.70 ms against 184 ms of inference on the Pi 5 -- 0.9% of the frame.
        frame = cv2.cvtColor(self._picam.capture_array(), cv2.COLOR_RGB2BGR)
        resultados = self._yolo(frame, verbose=False, conf=self.threshold)

        detections: List[Detection] = []
        cajas: List[np.ndarray] = []
        for caja in resultados[0].boxes:
            if self._yolo.names[int(caja.cls[0])] not in self.classes:
                continue
            xyxy = caja.xyxy[0].cpu().numpy()
            x1, _y1, x2, y2 = xyxy
            detections.append({
                "px": float((x1 + x2) / 2),
                "py": float(y2),  # bottom edge: the point touching the ground
                "conf": round(float(caja.conf[0]), 3),
                # Carried from here so a report can say WHAT it found, not just where.
                # With more than one class enabled, a coordinate without a class name is
                # not actionable: the ground station cannot tell a person from a car.
                "cls": self._yolo.names[int(caja.cls[0])],
            })
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

        return detections

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
        # Clamped to the frame: a detection at the edge yields a smaller crop, not a crash.
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
