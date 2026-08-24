"""
Runs the full geolocation pipeline in either of the two worlds.

    python correr.py sim      the target is computed geometrically, no images
    python correr.py dron     the ArduCam takes real pictures and YOLO reads them

Everything below `main` is identical in both cases. The only line that differs
is the one that builds the camera, and it lives in `elegir_camara`. That single
line is the whole point of this file: the pipeline never learns which world it
is running in, it only asks the camera for detections.

The pipeline is the one from the paper:

    camera -> detections (px, py, conf)
           -> pixel_to_ray, one bearing ray per detection
           -> select_best_views, greedy on angular diversity and confidence
           -> ransac_fusion, robust triangulation
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from uav_vision.camera_config import ARDUCAM_MODULE_3, SIYI_A8_MINI, CameraConfig
from uav_vision.fusion import ransac_fusion
from uav_vision.pinhole_local import pixel_to_ray
from uav_vision.view_selection import select_best_views

# Parameters from the paper (Table tab:params).
K_VISTAS = 30
ALPHA = 0.2
MIN_ANGLE_DEG = 10.0
RANSAC_ITER = 100
RANSAC_THRESHOLD_M = 5.0


def elegir_camara(mundo: str, alvo, pitch_deg: float, modelo: str, semilla: int):
    """
    Builds the camera for the requested world. This is the only place in the
    program that knows the difference between simulation and hardware.
    """
    if mundo == "sim":
        from uav_vision.camera import CamaraSimulada
        return CamaraSimulada(
            alvo=alvo,
            pitch_deg=pitch_deg,
            camara=SIYI_A8_MINI,
            rng=np.random.default_rng(semilla),
        )
    from uav_vision.camera import CamaraArduCam
    return CamaraArduCam(modelo=modelo, umbral=0.3, camara=ARDUCAM_MODULE_3)


def trayectoria(n: int, altura: float, radio: float) -> List[Tuple[float, float, float]]:
    """
    A circular pass around the origin, used only to feed the simulated world
    with a plausible sequence of poses. On the drone these poses come from the
    autopilot instead.
    """
    poses = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        x, y = radio * math.cos(a), radio * math.sin(a)
        yaw = (math.degrees(math.atan2(-x, -y))) % 360.0   # facing the origin
        poses.append(((x, y, altura), yaw))
    return poses


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mundo", choices=["sim", "dron"])
    ap.add_argument("--pasos", type=int, default=40)
    ap.add_argument("--altura", type=float, default=35.0)
    ap.add_argument("--radio", type=float, default=25.0)
    ap.add_argument("--pitch", type=float, default=-55.0)
    ap.add_argument("--modelo", default="yolov8n.pt", help="solo para el dron")
    ap.add_argument("--semilla", type=int, default=0, help="solo para la simulacion")
    ap.add_argument("--alvo", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = ap.parse_args(argv)

    alvo = tuple(args.alvo)
    camara = elegir_camara(args.mundo, alvo, args.pitch, args.modelo, args.semilla)

    print("=" * 64)
    print(f"  mundo   : {args.mundo}")
    print(f"  camara  : {type(camara).__name__}")
    print(f"  optica  : {camara.camara.name}  f={camara.camara.focal_length_px:.0f} px"
          f"  punto principal={camara.camara.principal_point}")
    print(f"  pitch   : {args.pitch} grados")
    print("=" * 64)

    # ---- de aca para abajo NADA depende del mundo -----------------------

    mediciones: List[tuple] = []
    confianzas: List[float] = []
    t0 = time.time()

    for i, (pos, yaw) in enumerate(trayectoria(args.pasos, args.altura, args.radio)):
        for det in camara.ver_alvo(pos, yaw):
            rayo = pixel_to_ray(
                pos, yaw, (det["px"], det["py"]),
                pitch_deg=args.pitch,
                focal_px=camara.camara.focal_length_px,
                img_w=camara.camara.image_width,
                img_h=camara.camara.image_height,
                principal_point=camara.camara.principal_point,
            )
            mediciones.append(rayo)
            confianzas.append(det["conf"])
        if (i + 1) % 10 == 0:
            print(f"  paso {i+1:3d}/{args.pasos}   rayos acumulados: {len(mediciones)}")

    duracion = time.time() - t0
    print()
    print(f"  {len(mediciones)} rayos en {duracion:.1f} s")

    if len(mediciones) < 2:
        print("  no alcanza para triangular (hacen falta 2 rayos)")
        if hasattr(camara, "apagar"):
            camara.apagar()
        return 1

    indices = select_best_views(
        mediciones, confianzas,
        k=K_VISTAS, alpha=ALPHA, min_angle_deg=MIN_ANGLE_DEG,
    )
    elegidas = [mediciones[i] for i in indices]
    print(f"  {len(elegidas)} vistas seleccionadas de {len(mediciones)}")

    elegidas_conf = [confianzas[i] for i in indices]
    estimacion = ransac_fusion(
        elegidas,
        n_iterations=RANSAC_ITER,
        threshold_m=RANSAC_THRESHOLD_M,
        rng=np.random.default_rng(args.semilla),
        ground_z=alvo[2],
        confidences=elegidas_conf,
    )

    print()
    print(f"  estimacion : ({estimacion[0]:.2f}, {estimacion[1]:.2f}, {estimacion[2]:.2f})")
    if args.mundo == "sim":
        error = float(np.linalg.norm(np.asarray(estimacion) - np.asarray(alvo)))
        print(f"  verdad     : ({alvo[0]:.2f}, {alvo[1]:.2f}, {alvo[2]:.2f})")
        print(f"  error      : {error:.2f} m")
    else:
        print("  (en el dron no hay verdad conocida: comparar con el GPS del objetivo)")

    if hasattr(camara, "apagar"):
        camara.apagar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
