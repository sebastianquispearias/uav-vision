"""
Wiring test for the BoT-SORT tracker inside OnboardCamera, without hardware.

The tracker stage is exercised directly with synthetic frames: one static box and one box
walking across the image. Passing means each keeps one stable track_id for the whole sequence
and the two ids differ — which is what the identity layer consumes.

Needs boxmot installed: run with a python environment that has it.
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

from uav_vision.camera import OnboardCamera

cam = OnboardCamera(model="no-se-usa.pt", tracker=True, fps=5.0)
cam._build_tracker()

W, H = 640, 480
QUIETA = np.array([300.0, 200.0, 340.0, 280.0])      # operator, static

ids_quieta, ids_movil = [], []
for f in range(30):
    frame = np.full((H, W, 3), 90, dtype=np.uint8)
    mobile = np.array([50.0 + 12.0 * f, 220.0, 90.0 + 12.0 * f, 300.0])

    detecciones = [
        {"px": 320.0, "py": 280.0, "conf": 0.8},
        {"px": float((mobile[0] + mobile[2]) / 2), "py": 300.0, "conf": 0.7},
    ]
    cajas = [QUIETA.copy(), mobile]
    cam._track(frame, detecciones, cajas, [])

    if "track_id" in detecciones[0]:
        ids_quieta.append(detecciones[0]["track_id"])
    if "track_id" in detecciones[1]:
        ids_movil.append(detecciones[1]["track_id"])

print(f"frames con id (quieta): {len(ids_quieta)}/30  ids: {sorted(set(ids_quieta))}")
print(f"frames con id (mobile) : {len(ids_movil)}/30  ids: {sorted(set(ids_movil))}")

assert len(ids_quieta) >= 25, "la pista quieta perdio identity"
assert len(ids_movil) >= 25, "la pista mobile perdio identity"
assert len(set(ids_quieta)) == 1, "la pista quieta cambio de id"
assert len(set(ids_movil)) == 1, "la pista mobile cambio de id"
assert set(ids_quieta) != set(ids_movil), "dos cosas con el mismo id"

print("\nTODO OK: cada cosa mantiene UN track_id estable, ids distintos")
