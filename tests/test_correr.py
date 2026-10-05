"""The paper's chain runs, and it is the only gate that says so.

`correr.py` is the one file that imports `view_selection.py` and `fusion.py`: between them
2067 lines, 39 % of the package, and the published contribution. It had not run since the camera
classes were renamed from Spanish to English -- `python correr.py sim` died on
`ImportError: cannot import name 'CamaraSimulada'`, and nothing in the suite noticed, because
every other test exercises the chain that flies (camera -> pinhole_local -> ground intersection
-> identity -> vision_protocol) and never touches this one.

Dead code is not the problem. Dead code nobody knows is dead is the problem: a published result
that cannot be re-run is a claim, not a measurement. This file is the smoke alarm.

It is a geometric gate, not a tolerance to tune. The simulated world projects a target at a known
position through the camera model, so the error is the error of the selection and the
triangulation and nothing else. Twelve centimetres is already two orders of magnitude above what
the chain achieves when it works and far below what it reports when any stage is broken.

Run with: python tests/test_correr.py

THE IMPORT AT THE TOP IS HALF THE GATE: elegir_camara builds the camera by name, so a rename in
camera.py breaks it and no amount of running the flying chain would say so.

A chain that triangulated WITHOUT selecting would pass that gate and still have lost the
contribution, so the three stages are driven by hand too, and the checks are that the selector
really discards views and that the fusion lands on the target.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)

import numpy as np

import correr
from uav_vision.camera import SimulatedCamera
from uav_vision.fusion import ransac_fusion
from uav_vision.pinhole_local import pixel_to_ray
from uav_vision.view_selection import select_best_views

print()
print("=" * 64)
print("1. LA CADENA DEL PAPER ARRANCA Y TRIANGULA")
print("=" * 64)
codigo = correr.main(["sim", "--pasos", "40", "--semilla", "0"])
print(f"  python correr.py sim --pasos 40  ->  codigo de salida {codigo}")
assert codigo == 0, "la cadena del paper tiene que correr de punta a punta"

print()
print("=" * 64)
print("2. LAS DOS PIEZAS DEL PAPER ESTAN EN EL CAMINO, NO SOLO IMPORTADAS")
print("=" * 64)
camara = SimulatedCamera(target=(0.0, 0.0, 0.0), pitch_deg=-55.0,
                         camera=correr.SIYI_A8_MINI, rng=np.random.default_rng(0))
rayos, confianzas = [], []
for pos, yaw in correr.trayectoria(40, 35.0, 25.0):
    for det in camara.detect(pos, yaw):
        rayos.append(pixel_to_ray(
            pos, yaw, (det["px"], det["py"]), pitch_deg=-55.0,
            focal_px=camara.camera.focal_length_px,
            img_w=camara.camera.image_width,
            img_h=camara.camera.image_height,
            principal_point=camara.camera.principal_point))
        confianzas.append(det["conf"])

elegidas = select_best_views(rayos, confianzas, k=correr.K_VISTAS, alpha=correr.ALPHA,
                             min_angle_deg=correr.MIN_ANGLE_DEG)
estimacion = ransac_fusion([rayos[i] for i in elegidas],
                           n_iterations=correr.RANSAC_ITER,
                           threshold_m=correr.RANSAC_THRESHOLD_M,
                           rng=np.random.default_rng(0), ground_z=0.0,
                           confidences=[confianzas[i] for i in elegidas])
error = float(np.linalg.norm(np.asarray(estimacion)))
print(f"  rayos construidos        : {len(rayos)}")
print(f"  vistas que el paper elige: {len(elegidas)}")
print(f"  error contra la verdad   : {error:.3f} m")
assert len(rayos) >= 20, "sin rayos no hay nada que seleccionar ni que triangular"
assert len(elegidas) < len(rayos), \
    "select_best_views tiene que DESCARTAR vistas; si elige todas, no esta decidiendo nada"
assert len(elegidas) >= 2, "con menos de dos vistas no se puede triangular"
assert error < 0.12, "la triangulacion del paper tiene que caer sobre el blanco conocido"

print()
print("TODO OK")
