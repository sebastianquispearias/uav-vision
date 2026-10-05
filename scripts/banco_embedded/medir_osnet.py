"""How many milliseconds a box costs, on this board, with nothing in front of the camera.

The frame-level measurement is useless when the room is empty: with zero boxes the appearance
model never runs and the two numbers differ by noise. This times the model itself, on a synthetic
frame with N boxes of the size a person has at mission altitude.

CAJA is roughly a pedestrian at 25 m with this camera. The first of the repeated calls is
discarded because it warms the model up rather than measuring it.
"""
import statistics as st
import time

import numpy as np
from boxmot.reid.core.reid import ReID

ANCHO, ALTO = 1920, 1080
CAJA = (60, 150)

print("placa:", open("/proc/device-tree/model").read().strip("\x00"))
t0 = time.time()
reid = ReID("/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt", device="cpu", half=False)
print("  cargar el modelo: %.1f s" % (time.time() - t0))

rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (ALTO, ANCHO, 3), dtype=np.uint8)

for n in (1, 3, 6):
    cajas = np.array([[100 + i * 200, 400, 100 + i * 200 + CAJA[0], 400 + CAJA[1]]
                      for i in range(n)], dtype=float)
    ms = []
    for k in range(6):
        t = time.time()
        out = reid.process({"fallback": True,
                            "boxes": np.asarray(cajas, dtype="float32"),
                            "image": frame})
        dt = (time.time() - t) * 1000.0
        if k:
            ms.append(dt)
    forma = np.asarray(out["_features"], dtype="float32").shape
    print("  %d caja(s): mediana %6.1f ms   -> %5.1f ms por caja   salida %s"
          % (n, st.median(ms), st.median(ms) / n, forma))
