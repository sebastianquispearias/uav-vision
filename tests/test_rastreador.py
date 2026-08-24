"""Wiring test for the BoT-SORT tracker inside CamaraArduCam.

No hardware: the tracker part of the camera is exercised directly with
synthetic frames -- one static box (the operator standing) and one box
walking across the image. PASS means each keeps ONE stable track_id for
the whole sequence and the two ids are different. That is exactly what
the identity layer consumes.

Needs boxmot installed -- run with the entrenamiento venv:
    entrenamiento/venv/Scripts/python tests/test_rastreador.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import boxmot  # noqa: F401
except ImportError:
    print("boxmot no instalado en este python: test saltado")
    sys.exit(0)

from uav_vision.camera import CamaraArduCam

cam = CamaraArduCam(modelo="no-se-usa.pt", rastreador=True, fps=5.0)
cam._crear_rastreador()

W, H = 640, 480
QUIETA = np.array([300.0, 200.0, 340.0, 280.0])      # operator, static

ids_quieta, ids_movil = [], []
for f in range(30):
    frame = np.full((H, W, 3), 90, dtype=np.uint8)
    movil = np.array([50.0 + 12.0 * f, 220.0, 90.0 + 12.0 * f, 300.0])

    detecciones = [
        {"px": 320.0, "py": 280.0, "conf": 0.8},
        {"px": float((movil[0] + movil[2]) / 2), "py": 300.0, "conf": 0.7},
    ]
    cajas = [QUIETA.copy(), movil]
    cam._rastrear(frame, detecciones, cajas, [])

    if "track_id" in detecciones[0]:
        ids_quieta.append(detecciones[0]["track_id"])
    if "track_id" in detecciones[1]:
        ids_movil.append(detecciones[1]["track_id"])

print(f"frames con id (quieta): {len(ids_quieta)}/30  ids: {sorted(set(ids_quieta))}")
print(f"frames con id (movil) : {len(ids_movil)}/30  ids: {sorted(set(ids_movil))}")

assert len(ids_quieta) >= 25, "la pista quieta perdio identidad"
assert len(ids_movil) >= 25, "la pista movil perdio identidad"
assert len(set(ids_quieta)) == 1, "la pista quieta cambio de id"
assert len(set(ids_movil)) == 1, "la pista movil cambio de id"
assert set(ids_quieta) != set(ids_movil), "dos cosas con el mismo id"

print("\nTODO OK: cada cosa mantiene UN track_id estable, ids distintos")
