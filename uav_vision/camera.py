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
from uav_vision.confidence import simulate_confidence
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
    ) -> None:
        self.alvo = tuple(alvo)
        self.pitch_deg = pitch_deg
        self.camara = camara
        self.rng = rng if rng is not None else np.random.default_rng()

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
        return [{"px": pixel[0], "py": pixel[1], "conf": conf}]


class CamaraArduCam:
    """Camara real: saca una foto con picamera2 y la pasa por YOLO.

    picamera2, ultralytics y boxmot se importan la primera vez que se
    pide una foto, no al crear el objeto. Asi este archivo se puede
    importar en una laptop que no tiene ninguno de los tres instalados.

    Si se pasa reid_modelo, cada deteccion sale ademas con su huella
    ('emb'). Cuesta CPU: en la Raspberry hay que medirlo antes de darlo
    por hecho (revision2/bench_rpi.py).
    """

    def __init__(
        self,
        modelo: str,
        umbral: float = 0.3,
        clase: str = "person",
        camara: CameraConfig = ARDUCAM_MODULE_3,
        rot180: bool = True,
        reid_modelo: Optional[str] = None,
    ) -> None:
        self.modelo = modelo
        self.umbral = umbral
        self.clase = clase
        self.camara = camara
        # rot180: desde el 02ago2026 la ArduCam esta montada girada 180
        # sobre el eje optico. El ISP endereza la imagen en captura, de
        # forma que YOLO ve personas de pie y la geometria no cambia.
        self.rot180 = rot180
        # mismo modelo que usa entrenamiento/rehuella_osnet.py
        self.reid_modelo = reid_modelo
        self._picam: Any = None
        self._yolo: Any = None
        self._reid: Any = None

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
            if self._yolo.names[int(caja.cls[0])] != self.clase:
                continue
            xyxy = caja.xyxy[0].cpu().numpy()
            x1, _y1, x2, y2 = xyxy
            detecciones.append({
                "px": float((x1 + x2) / 2),
                "py": float(y2),                        # borde inferior: los pies
                "conf": round(float(caja.conf[0]), 3),
            })
            cajas.append(xyxy)

        if self._reid is not None and detecciones:
            for det, emb in zip(detecciones, self._huellas(frame, cajas)):
                det["emb"] = emb

        return detecciones

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
