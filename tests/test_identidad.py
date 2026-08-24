"""
Empirical gates for IdentidadIncremental, each on a synthetic scene with known ground truth:

  1. Static: one standing person under projection noise yields one candidate near the truth.
  2. Mobile: a walker is classified as mobile and its reported position must beat the lagging
     naive estimate (median of the recent window).
  3. Co-occurrence veto: two people seen in the same frames stay two candidates.
  4. Twin override: duplicate boxes of one person merge into one candidate.

Run with: python tests/test_identidad.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from uav_vision.identity import IdentidadIncremental

RNG = np.random.default_rng(7)
FPS = 5.0
SIGMA = 1.2          # ground projection noise, m


def emb_de(base: int) -> np.ndarray:
    v = np.random.default_rng(base).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def ruido():
    return RNG.normal(0, SIGMA, size=2)


print("=" * 64)
print("1. ESTATICO: una persona parada -> UN candidato cerca de la verdad")
print("=" * 64)
ident = IdentidadIncremental(radio_fusion_m=3.5, fps=FPS)
VERDAD = np.array([2.0, -3.0])
e1 = emb_de(1)
for f in range(300):                                   # 60 s a 5 Hz
    ident.observar(f, 10, VERDAD + ruido(), 0.7, e1 + 0.05 * RNG.normal(size=512))
c = ident.candidatos()
assert len(c) == 1, f"fragmento en {len(c)} candidatos"
err = math.hypot(c[0]["x"] - VERDAD[0], c[0]["y"] - VERDAD[1])
print(f"  1 candidato, error {err:.2f} m, movil={c[0]['movil']}")
assert err < 0.5 and not c[0]["movil"]

print()
print("=" * 64)
print("2. MOVIL: caminante a 1 m/s -> MOVIL y sin retraso")
print("=" * 64)
ident = IdentidadIncremental(radio_fusion_m=3.5, fps=FPS)
e2 = emb_de(2)
pos_final = None
for f in range(300):
    t = f / FPS
    real = np.array([-10.0 + 1.0 * t, 4.0])            # 1 m/s hacia el este
    pos_final = real
    ident.observar(f, 20, real + ruido(), 0.6, e2 + 0.05 * RNG.normal(size=512))
c = ident.candidatos()
assert len(c) == 1 and c[0]["movil"], f"no salio MOVIL: {c}"
err_fit = math.hypot(c[0]["x"] - pos_final[0], c[0]["y"] - pos_final[1])
# el estimador ingenuo que reemplazamos: mediana de la ventana reciente
imps = np.array([( -10.0 + (f / FPS), 4.0) for f in range(300)])
q = max(2, len(imps) // 4)
naive = np.median(imps[-q:], axis=0)
err_naive = math.hypot(naive[0] - pos_final[0], naive[1] - pos_final[1])
print(f"  MOVIL detectado; error ajuste lineal {err_fit:.2f} m "
      f"vs mediana-reciente {err_naive:.2f} m (retraso puro, sin ruido)")
assert err_fit < err_naive, "el ajuste no mejora al estimador ingenuo"
assert err_fit < 1.5

print()
print("=" * 64)
print("3. VETO: dos personas juntas 2 m aparte -> DOS candidatos")
print("=" * 64)
ident = IdentidadIncremental(radio_fusion_m=3.5, fps=FPS)
A, B = np.array([0.0, 0.0]), np.array([2.0, 0.0])
ea, eb = emb_de(3), emb_de(4)
for f in range(300):
    ident.observar(f, 30, A + ruido(), 0.7, ea + 0.05 * RNG.normal(size=512))
    ident.observar(f, 31, B + ruido(), 0.7, eb + 0.05 * RNG.normal(size=512))
c = ident.candidatos()
print(f"  candidatos: {len(c)} (posiciones {[(p['x'], p['y']) for p in c]})")
assert len(c) == 2, "el veto de co-ocurrencia fallo: se fusionaron"

print()
print("=" * 64)
print("4. GEMELO: cajas duplicadas de UNA persona -> UN candidato")
print("=" * 64)
ident = IdentidadIncremental(radio_fusion_m=3.5, fps=FPS)
P = np.array([-1.0, 5.0])
ep = emb_de(5)
for f in range(300):
    ident.observar(f, 40, P + ruido(), 0.7, ep + 0.03 * RNG.normal(size=512))
    if f % 2 == 0:      # el detector duplica la caja la mitad del tiempo
        ident.observar(f, 41, P + ruido(), 0.5, ep + 0.03 * RNG.normal(size=512))
c = ident.candidatos()
print(f"  candidatos: {len(c)} con n_obs={[p['n_obs'] for p in c]}")
assert len(c) == 1, "el gemelo no se fusiono (regla A-1 rota)"

print()
print("TODO OK")
