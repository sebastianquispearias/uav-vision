"""The same gap, but comparing what the system actually compares: averaged vectors.

The window is given in frames: at 3 FPS it is 4 s, the order of a short track.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
import personas_encontradas as pe

d = np.load("demo/data/examen_v3_datos.npz", allow_pickle=True)
dets, embs = d["dets"], d["embs"]
embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

VENTANA = 12
grupos = {}
for i in range(len(dets)):
    L = pe.letra_de(dets[i])
    if L is None or L in ("X", "PERSONA"):
        continue
    clave = (L, int(dets[i][0]) // VENTANA)
    grupos.setdefault(clave, []).append(i)

claves = [k for k, v in grupos.items() if len(v) >= 3]
V = np.array([embs[grupos[k]].mean(0) for k in claves])
V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
L = np.array([k[0] for k in claves], dtype=object)
print("grupos (una pista corta por persona y ventana): %d, de %d personas" % (len(claves), len(set(L))))

cos = V @ V.T
misma = L[:, None] == L[None, :]
tri = np.triu(np.ones_like(cos, dtype=bool), 1)
a, b = cos[misma & tri], cos[~misma & tri]
print("  MISMA persona      n=%4d  p5=%.2f  mediana=%.2f  max=%.2f" % (len(a), np.percentile(a,5), np.median(a), a.max()))
print("  personas DISTINTAS n=%4d  p95=%.2f mediana=%.2f  min=%.2f" % (len(b), np.percentile(b,95), np.median(b), b.min()))
p5, p95 = np.percentile(a, 5), np.percentile(b, 95)
print("  -> %s" % ("SE SOLAPAN" if p5 <= p95 else "hueco de %.2f, y el umbral esta en 0.545" % (p5 - p95)))
mal_a = (a < 0.545).mean() * 100
mal_b = (b > 0.545).mean() * 100
print("  con el umbral puesto: %.1f %% de pares de la MISMA persona quedarian separados," % mal_a)
print("                        %.1f %% de pares de personas DISTINTAS se fundirian." % mal_b)
