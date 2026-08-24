"""Proveedor de camara para geolocalizacion por vision.

Hay dos clases y las dos exponen el MISMO metodo. Quien las consume no
sabe cual le toco: en simulacion recibe CamaraSimulada, en el dron
recibe CamaraArduCam, y su codigo no cambia.

CONTRATO de  ver_alvo(pos, yaw)
-------------------------------
entra:
    pos  = (x, y, z) en metros, marco local ENU (x=Este, y=Norte, z=Arriba)
    yaw  = grados. 0 = Norte, 90 = Este, sentido horario

sale:
    lista de detecciones. Cada deteccion es un dict:
        {'px': float, 'py': float, 'conf': float}
    lista vacia = no se detecto nada en este instante

    'px' es el centro horizontal de la deteccion.
    'py' es el BORDE INFERIOR, no el centro: es el punto donde el objeto
         toca el suelo. Asi lo hace onboard.py en el dron real
         (bearing_py = y2), porque la persona esta parada y lo que apoya
         en el piso son los pies.
    'conf' alimenta select_best_views, que con alpha=0.2 le da el 80% del
         peso. No es decorativa: sin ella el selector del paper no corre.

campo OPCIONAL:
    'emb' = huella de apariencia, 512 float32 normalizados (OSNet).
         Solo aparece si la camara puede calcularla. Quien la use debe
         pedirla con det.get('emb'), nunca con det['emb'].
         Sirve para decir "estas dos detecciones son la misma cosa" sin
         mirar la posicion, y con eso separar personas de falsos
         positivos estaticos.

DECISION TOMADA (2026-08-23): la huella se calcula ACA, en la camara.
La imagen nunca sale del modulo; lo que viaja son 512 numeros (2048 B),
o 128 B si se comprime con PCA int8. Ver README, seccion "a futuro".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from uav_vision.camera_config import ARDUCAM_MODULE_3, DEFAULT_CAMERA, CameraConfig
from uav_vision.confidence import confidence_to_pixel_sigma_model, simulate_confidence
from uav_vision.pinhole_local import project_to_pixel

Deteccion = Dict[str, float]


class CamaraSimulada:
    """Camara de simulacion: no hay imagen, el pixel se calcula por geometria.

    pitch_deg NO tiene valor por defecto a proposito. El montaje real
    cambio de -45 a -55 grados el 02ago2026, y tener el numero escondido
    como default en dos archivos distintos es justamente lo que hace que
    las copias diverjan. Quien crea la camara declara el pitch que uso.
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
        # ruido_pixel: un detector real no devuelve el pixel exacto. El
        # modelo es el del paper: sigma_pixel = C / conf (menos confianza,
        # mas ruido). Sin esto la simulacion es perfecta y RANSAC no tiene
        # nada que rechazar — el resultado no dice nada. Apagable solo
        # para pruebas de geometria pura.
        self.ruido_pixel = ruido_pixel
        # modelo_ruido: "heuristic" (sigma = C / conf) o "visdrone_1d"
        # (ajuste empirico sobre VisDrone2019-DET-val + YOLOv8s).
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
        if pixel is None:          # fuera de cuadro o detras de la camara
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
    """Camara real: saca una foto con picamera2 y la pasa por YOLO.

    picamera2, ultralytics y boxmot se importan la primera vez que se
    pide una foto, no al crear el objeto. Asi este archivo se puede
    importar en una laptop que no tiene ninguno de los tres instalados.

    Si se pasa reid_modelo, cada deteccion sale ademas con su huella
    ('emb'). Cuesta CPU: en la Raspberry hay que medirlo antes de darlo
    por hecho (revision2/bench_rpi.py).
    """

    # nombres de clase que cuentan como "persona": COCO dice 'person',
    # VisDrone dice 'pedestrian' y 'people'. Con un solo nombre, cambiar
    # de modelo hacia que el filtro descartara TODO en silencio.
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
        # rot180: desde el 02ago2026 la ArduCam esta montada girada 180
        # sobre el eje optico. El ISP endereza la imagen en captura (asi
        # YOLO ve personas de pie), pero eso mueve el punto principal
        # calibrado: cada coordenada c pasa a (tamano - 1) - c. Es lo
        # mismo que hace onboard.py. Sin esta linea, los rayos del vuelo
        # rot180 salen ~1.1 grados torcidos (~0.8 m en el suelo a 35 m).
        self.camara = camara.rotated_180() if rot180 else camara
        self.rot180 = rot180
        # mismo modelo que usa entrenamiento/rehuella_osnet.py
        self.reid_modelo = reid_modelo
        # rastreador: BoT-SORT oficial (boxmot) sobre las detecciones.
        # Cada deteccion sale ademas con 'track_id', que es lo que la
        # capa de identidad (identity.py) necesita para armar POIs.
        # BoT-SORT cuenta su memoria en FRAMES, asi que el buffer se
        # declara en SEGUNDOS y se convierte con el fps declarado: el
        # mismo numero de frames significa 8 s a 5 Hz pero 24 s a la
        # cadencia del vuelo 3 -- otra constante que no generaliza sola.
        # compensar_camara (CMC) mejora el tracking cuando la camara se
        # mueve, pero su costo en la Raspberry NO esta medido: apagado
        # hasta que el banco diga que entra en el presupuesto de 5 Hz.
        if rastreador and fps is None:
            raise ValueError(
                "rastreador=True necesita fps: el buffer de BoT-SORT "
                "se mide en frames y sin la cadencia real el numero "
                "no significa nada.")
        self.rastreador_habilitado = rastreador
        self.fps = fps
        self.buffer_pista_s = buffer_pista_s
        self.compensar_camara = compensar_camara
        self._picam: Any = None
        self._yolo: Any = None
        self._reid: Any = None
        self._tracker: Any = None

    # -- hardware ---------------------------------------------------------

    def _encender(self) -> None:
        """Arranca camara y modelo. Se llama sola la primera vez."""
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
        time.sleep(2)              # el sensor necesita estabilizar exposicion

        self._picam = picam
        self._yolo = YOLO(self.modelo)

        if self.reid_modelo is not None:
            from boxmot.reid.core.reid import ReID
            self._reid = ReID(self.reid_modelo, device="cpu", half=False)

        if self.rastreador_habilitado:
            self._crear_rastreador()

    def _crear_rastreador(self) -> None:
        """Official BoT-SORT, same knobs the offline validation used
        (entrenamiento/correr_botsort.py), buffer converted from
        seconds using the declared fps."""
        from boxmot.trackers.bbox.botsort import BotSort

        self._tracker = BotSort(
            reid_model=None,                # embeddings come in via embs=
            # Appearance matching only when this camera computes
            # OSNet embeddings; otherwise BotSort refuses to run
            # ("A ReID model is required when embeddings are not
            # provided") -- motion-only mode needs with_reid=False.
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

    # -- contrato ---------------------------------------------------------

    def ver_alvo(self, pos: Sequence[float], yaw: float) -> List[Deteccion]:
        # pos y yaw NO se usan: la foto real ya contiene lo que contiene.
        # Estan en la firma para que el contrato sea identico al simulado.
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
                "py": float(y2),                        # borde inferior: los pies
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
        """Run BoT-SORT on this frame's boxes; attach 'track_id'.

        Called EVERY frame, detections or not: the tracker also needs
        the empty frames to age out tracks that left the scene. A
        detection the tracker hasn't confirmed yet gets no track_id;
        the identity layer simply skips those until it does.
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
        """Huellas OSNet de todas las cajas del frame, en una sola pasada.

        Misma receta que entrenamiento/rehuella_osnet.py: un batch por
        imagen, vector normalizado a norma 1 (asi el coseno es un simple
        producto punto).
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
