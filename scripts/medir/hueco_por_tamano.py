"""Does the appearance gap survive when the people are small? Measured on the 02-ago flight."""
import sys

import numpy as np

sys.path.insert(0, "scripts")
import personas_encontradas as pe

d = np.load("demo/data/examen_v3_datos.npz", allow_pickle=True)
dets, embs = d["dets"], d["embs"]
embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

letras, alturas = [], []
for i in range(len(dets)):
    L = pe.letra_de(dets[i])
    letras.append(L)
    alturas.append(float(dets[i][5] - dets[i][3]))
letras, alturas = np.array(letras, dtype=object), np.array(alturas)

utiles = np.array([L is not None and L not in ("X", "PERSONA") for L in letras])
print("detecciones con una persona identificada por letra: %d de %d" % (utiles.sum(), len(dets)))
print("altura de caja: mediana %.0f px, rango %.0f a %.0f\n"
      % (np.median(alturas[utiles]), alturas[utiles].min(), alturas[utiles].max()))

TRAMOS = [("CHICAS  (< 35 px)", 0, 35), ("MEDIANAS (35-60)", 35, 60), ("GRANDES (> 60 px)", 60, 1e9)]
print("%-20s %5s   %-22s %-22s %s" % ("tamano", "n", "MISMA persona", "personas DISTINTAS", "hueco"))
for nombre, lo, hi in TRAMOS:
    sel = utiles & (alturas >= lo) & (alturas < hi)
    idx = np.where(sel)[0]
    if len(idx) < 40:
        print("%-20s %5d   (muy pocas para medir)" % (nombre, len(idx)))
        continue
    rng = np.random.default_rng(0)
    if len(idx) > 900:
        idx = rng.choice(idx, 900, replace=False)
    E, L = embs[idx], letras[idx]
    cos = E @ E.T
    misma = L[:, None] == L[None, :]
    tri = np.triu(np.ones_like(cos, dtype=bool), 1)
    a, b = cos[misma & tri], cos[~misma & tri]
    if len(a) < 20 or len(b) < 20:
        print("%-20s %5d   (sin pares suficientes)" % (nombre, len(idx)))
        continue
    p5a, p95b = np.percentile(a, 5), np.percentile(b, 95)
    print("%-20s %5d   p5=%.2f  mediana=%.2f   p95=%.2f  mediana=%.2f   %s"
          % (nombre, len(idx), p5a, np.median(a), p95b, np.median(b),
             ("SE SOLAPAN" if p5a <= p95b else "hueco de %.2f" % (p5a - p95b))))
print("\nel umbral vive en coseno 0.545 (la distancia L2 de 0.95)")
