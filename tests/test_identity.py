"""
Empirical gates for IncrementalIdentity, each on a synthetic scene with known ground truth:

  1. Static: one standing person under projection noise yields one candidate near the truth.
  2. Mobile: a walker is classified as mobile and its reported position must beat the lagging
     naive estimate (median of the recent window).
  3. Co-occurrence veto: two people seen in the same frames stay two candidates.
  4. Twin override: duplicate boxes of one person merge into one candidate.

Run with: python tests/test_identity.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from uav_vision.identity import IncrementalIdentity

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
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
VERDAD = np.array([2.0, -3.0])
e1 = emb_de(1)
for f in range(300):                                   # 60 s a 5 Hz
    ident.observe(f, 10, VERDAD + ruido(), 0.7, e1 + 0.05 * RNG.normal(size=512))
c = ident.candidates()
assert len(c) == 1, f"fragmento en {len(c)} candidates"
err = math.hypot(c[0]["x"] - VERDAD[0], c[0]["y"] - VERDAD[1])
print(f"  1 candidato, error {err:.2f} m, mobile={c[0]['mobile']}")
assert err < 0.5 and not c[0]["mobile"]

print()
print("=" * 64)
print("2. MOVIL: caminante a 1 m/s -> MOVIL y sin retraso")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e2 = emb_de(2)
pos_final = None
for f in range(300):
    t = f / FPS
    real = np.array([-10.0 + 1.0 * t, 4.0])            # 1 m/s hacia el este
    pos_final = real
    ident.observe(f, 20, real + ruido(), 0.6, e2 + 0.05 * RNG.normal(size=512))
c = ident.candidates()
assert len(c) == 1 and c[0]["mobile"], f"no salio MOVIL: {c}"
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
print("3. VETO: dos personas juntas 2 m aparte -> DOS candidates")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
A, B = np.array([0.0, 0.0]), np.array([2.0, 0.0])
ea, eb = emb_de(3), emb_de(4)
for f in range(300):
    ident.observe(f, 30, A + ruido(), 0.7, ea + 0.05 * RNG.normal(size=512))
    ident.observe(f, 31, B + ruido(), 0.7, eb + 0.05 * RNG.normal(size=512))
c = ident.candidates()
print(f"  candidates: {len(c)} (posiciones {[(p['x'], p['y']) for p in c]})")
assert len(c) == 2, "el veto de co-ocurrencia fallo: se fusionaron"

print()
print("=" * 64)
print("4. GEMELO: cajas duplicadas de UNA persona -> UN candidato")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
P = np.array([-1.0, 5.0])
ep = emb_de(5)
for f in range(300):
    ident.observe(f, 40, P + ruido(), 0.7, ep + 0.03 * RNG.normal(size=512))
    if f % 2 == 0:      # el detector duplica la caja la mitad del tiempo
        ident.observe(f, 41, P + ruido(), 0.5, ep + 0.03 * RNG.normal(size=512))
c = ident.candidates()
print(f"  candidates: {len(c)} con n_obs={[p['n_obs'] for p in c]}")
assert len(c) == 1, "el gemelo no se fusiono (regla A-1 rota)"

print()
print("=" * 64)
print("5. PASADA CORTA: nada mature, pero SI un preliminar que verificar")
print("=" * 64)
# The sweep case, measured on flight 3: a pass of 30 s never matures a candidate. A track
# forms and then the drone is gone. Without preliminaries the system says nothing at all
# about a person it tracked perfectly well for half a minute.
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, report_dur_s=36.0)
P = np.array([4.0, -2.0])
ep = emb_de(9)
n_pasada = int(20 * FPS)          # 20 s of pass, well under the 36 s report bar
for f in range(n_pasada):
    ident.observe(f, 70, P + ruido(), 0.6, ep + 0.03 * RNG.normal(size=512))

maduros = ident.candidates()
todos = ident.candidates(preliminary=True)
print(f"  maduros: {len(maduros)}   con preliminary: {len(todos)}")
assert len(maduros) == 0, "una pasada de 20 s no deberia madurar nada"
assert len(todos) == 1, "la pasada corta debe dejar UN preliminar que verificar"
assert todos[0]["mature"] is False, "el preliminar debe venir marcado mature=False"
d = float(np.linalg.norm(np.array([todos[0]["x"], todos[0]["y"]]) - P))
print(f"  preliminar en ({todos[0]['x']}, {todos[0]['y']}), a {d:.2f} m del real, "
      f"n_obs={todos[0]['n_obs']}")
assert d < 1.0, "el preliminar apunta al lugar equivocado"

# And the guarantee that keeps preliminaries honest: once evidence accumulates, the same
# candidate matures, and the two calls agree.
for f in range(n_pasada, int(60 * FPS)):
    ident.observe(f, 70, P + ruido(), 0.6, ep + 0.03 * RNG.normal(size=512))
maduros = ident.candidates()
assert len(maduros) == 1 and maduros[0]["mature"] is True,     "con evidencia suficiente el preliminar tiene que madurar"
print(f"  tras 60 s: mature=True, n_obs={maduros[0]['n_obs']}")

print()
print("TODO OK")
