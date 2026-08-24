"""
Camera providers for vision-based geolocation.

Two classes expose the same method, ver_alvo(pos, yaw). Consumers never learn which one they
received: simulation code gets CamaraSimulada, the real drone gets CamaraArduCam, and the consumer
code is identical in both cases.

Contract of ver_alvo(pos, yaw):
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

Deteccion = Dict[str, float]


class CamaraSimulada:
    """
    Simulation camera: there is no image, the pixel is computed geometrically by projecting a
    known target position through the camera model.

    pitch_deg has no default on purpose. The camera mount angle is a physical property of each
    deployment and hiding it as a default is how independent copies of the geometry drift apart.
    Whoever creates the camera states the pitch it models.
    """

    def __init__(
        self,
        alvo: Sequence[float],
        pitch_deg: float,
        camara: CameraConfig = DEFAULT_CAMERA,
        rng: Optional[np.random.Generator] = None,
        ruido_pixel: bool = True,
        modelo_ruido: str = "heuristic",
    ) -> None:
        self.alvo = tuple(alvo)
        self.pitch_deg = pitch_deg
        self.camara = camara
        self.rng = rng if rng is not None else np.random.default_rng()
        # A real detector does not return the exact pixel. Pixel noise follows
        # sigma = C / confidence, so low-confidence detections are noisier. Disable only for
        # pure-geometry tests: without noise the fusion stage has nothing to reject and
        # simulation results become meaningless.
        self.ruido_pixel = ruido_pixel
        # Noise model name, resolved by confidence.confidence_to_pixel_sigma_model.
        self.modelo_ruido = modelo_ruido

    def ver_alvo(self, pos: Sequence[float], yaw: float) -> List[Deteccion]:
        pixel = project_to_pixel(
            pos,
            self.alvo,
            yaw,
            self.pitch_deg,
            self.camara.focal_length_px,
            self.camara.image_width,
            self.camara.image_height,
            self.camara.principal_point,
        )
        if pixel is None:  # out of frame or behind the camera
            return []

        conf = simulate_confidence(
            pixel,
            self.camara.image_center,
            self.camara.max_radius,
            self.rng,
        )
        px, py = pixel
        if self.ruido_pixel:
            sigma = confidence_to_pixel_sigma_model(conf, self.modelo_ruido)
            px = float(np.clip(px + self.rng.normal(0, sigma),
                               0, self.camara.image_width - 1))
            py = float(np.clip(py + self.rng.normal(0, sigma),
                               0, self.camara.image_height - 1))
        return [{"px": px, "py": py, "conf": conf}]


class CamaraArduCam:
    """
    Real camera: captures a frame with picamera2 and runs a YOLO detector on it.

    picamera2, ultralytics and boxmot are imported on the first capture, not at construction, so
    this module can be imported on machines that do not have them installed.

    Optional stages, enabled by constructor arguments:
        - reid_modelo: compute an OSNet appearance embedding per detection ('emb' field).
        - rastreador: run BoT-SORT over the detections and attach a stable 'track_id' per
          detection. Requires fps, because the tracker's memory is measured in frames and only
          the declared rate makes a buffer duration meaningful.
    """

    # Class names that count as a person. Detection models trained on different datasets name
    # the class differently (COCO: 'person'; VisDrone: 'pedestrian', 'people'); filtering on a
    # single name silently drops every detection when the model changes.
    CLASES_PERSONA = frozenset({"person", "pedestrian", "people"})

    def __init__(
        self,
        modelo: str,
        umbral: float = 0.3,
        clases: Optional[Sequence[str]] = None,
        camara: CameraConfig = ARDUCAM_MODULE_3,
        rot180: bool = True,
        reid_modelo: Optional[str] = None,
        rastreador: bool = False,
        fps: Optional[float] = None,
        buffer_pista_s: float = 8.0,
        compensar_camara: bool = False,
    ) -> None:
        self.modelo = modelo
        self.umbral = umbral
        self.clases = frozenset(clases) if clases is not None else self.CLASES_PERSONA
        # When the camera is mounted upside-down the ISP un-flips the image at capture time
        # (hflip+vflip). That remapping moves the calibrated principal point, so the effective
        # config must be the reflected one; see CameraConfig.rotated_180().
        self.camara = camara.rotated_180() if rot180 else camara
        self.rot180 = rot180
        self.reid_modelo = reid_modelo
        if rastreador and fps is None:
            raise ValueError(
                "rastreador=True requires fps: the tracker buffer is measured in frames and "
                "has no meaning without the capture rate.")
        self.rastreador_habilitado = rastreador
        self.fps = fps
        self.buffer_pista_s = buffer_pista_s
        # Camera-motion compensation improves tracking from a moving camera but costs CPU;
        # disabled until measured on the target hardware.
        self.compensar_camara = compensar_camara
        self._picam: Any = None
        self._yolo: Any = None
        self._reid: Any = None
        self._tracker: Any = None

    # -- hardware ---------------------------------------------------------

    def _encender(self) -> None:
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
                main={"size": (self.camara.image_width, self.camara.image_height)},
                transform=tf,
            )
        )
        picam.start()
        time.sleep(2)  # the sensor needs time to stabilize exposure

        self._picam = picam
        self._yolo = YOLO(self.modelo)

        if self.reid_modelo is not None:
            from boxmot.reid.core.reid import ReID
            self._reid = ReID(self.reid_modelo, device="cpu", half=False)

        if self.rastreador_habilitado:
            self._crear_rastreador()

    def _crear_rastreador(self) -> None:
        """Builds the BoT-SORT tracker, with the buffer converted from seconds to frames."""
        from boxmot.trackers.bbox.botsort import BotSort

        self._tracker = BotSort(
            reid_model=None,  # embeddings are supplied externally via embs=
            # BotSort requires embeddings when appearance matching is on; without a ReID model
            # it must run motion-only.
            with_reid=self.reid_modelo is not None,
            use_cmc=self.compensar_camara,
            track_high_thresh=0.35,
            track_low_thresh=0.2,
            new_track_thresh=0.4,
            track_buffer=max(2, round(self.buffer_pista_s * self.fps)),
            match_thresh=0.85,
        )

    def apagar(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam = None

    # -- contract ---------------------------------------------------------

    def ver_alvo(self, pos: Sequence[float], yaw: float) -> List[Deteccion]:
        # pos and yaw are unused: the real photo already contains what it contains. They are in
        # the signature so the contract matches the simulated camera.
        del pos, yaw

        self._encender()
        frame = self._picam.capture_array()
        resultados = self._yolo(frame, verbose=False, conf=self.umbral)

        detecciones: List[Deteccion] = []
        cajas: List[np.ndarray] = []
        for caja in resultados[0].boxes:
            if self._yolo.names[int(caja.cls[0])] not in self.clases:
                continue
            xyxy = caja.xyxy[0].cpu().numpy()
            x1, _y1, x2, y2 = xyxy
            detecciones.append({
                "px": float((x1 + x2) / 2),
                "py": float(y2),  # bottom edge: the point touching the ground
                "conf": round(float(caja.conf[0]), 3),
            })
            cajas.append(xyxy)

        huellas: List[np.ndarray] = []
        if self._reid is not None and detecciones:
            huellas = self._huellas(frame, cajas)
            for det, emb in zip(detecciones, huellas):
                det["emb"] = emb

        if self._tracker is not None:
            self._rastrear(frame, detecciones, cajas, huellas)

        return detecciones

    def _rastrear(
        self,
        frame,
        detecciones: List[Deteccion],
        cajas: List[np.ndarray],
        huellas: List[np.ndarray],
    ) -> None:
        """
        Runs the tracker on this frame's boxes and attaches 'track_id' to each detection.

        Called on every frame, including frames with no detections: the tracker needs the empty
        frames to age out tracks that left the scene. Detections the tracker has not confirmed
        yet get no track_id; consumers skip those until it does.
        """
        if cajas:
            dts = np.array(
                [[*c, det["conf"], 0] for c, det in zip(cajas, detecciones)],
                dtype="float32")
            embs = np.asarray(huellas, dtype="float32") if huellas else None
        else:
            dts = np.empty((0, 6), dtype="float32")
            embs = None

        res = np.asarray(self._tracker.update(dts, frame, embs=embs))
        for fila in res:
            det_idx = int(fila[7])
            if 0 <= det_idx < len(detecciones):
                detecciones[det_idx]["track_id"] = int(fila[4])

    def _huellas(self, frame, cajas: List[np.ndarray]) -> List[np.ndarray]:
        """
        OSNet embeddings for all boxes of the frame, computed in a single batch. Vectors are
        normalized to unit length so cosine similarity reduces to a dot product.
        """
        salida = self._reid.process({
            "fallback": True,
            "boxes": np.asarray(cajas, dtype="float32"),
            "image": frame,
        })
        feats = np.asarray(salida["_features"], dtype="float32")

        huellas = []
        for v in feats:
            n = float(np.linalg.norm(v))
            huellas.append(v / n if n > 0 else v)
        return huellas
