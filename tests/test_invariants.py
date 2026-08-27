"""
Gates for uav_vision/invariants.py.

Each case is built from the data shape the guard exists to catch, rather than from a tidy
synthetic: a guard that only passes on well-formed input has not been tested.

Run with: python tests/test_invariants.py
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import numpy as np

from uav_vision.invariants import (InvariantError, instantaneous_rate, load_cache,
                                save_cache, check_rate)

FALLOS = []


def revisar(cond, desc, detalle=""):
    print("  [%s] %s%s" % ("ok  " if cond else "FALLA", desc, (" -- " + detalle) if detalle else ""))
    if not cond:
        FALLOS.append(desc)


print("=" * 70)
print("1. La cadencia: huecos largos no pueden mover el resultado")
print("=" * 70)

# Flight 3's shape: a camera at 9.35 Hz with three long spells on the ground. Dividing the
# count by the span says 3.28 Hz for the same recording -- the error made four times.
paso = 1.0 / 9.35
t, ahora = [], 0.0
for tramo in range(3):
    for _ in range(1000):
        t.append(ahora)
        ahora += paso
    ahora += 190.0          # on the ground between passes
t = np.asarray(t)

cad = instantaneous_rate(t)
por_span = len(t) / (t[-1] - t[0])
revisar(abs(cad - 9.35) < 0.1, "devuelve la cadencia real", "%.2f Hz" % cad)
revisar(por_span < 6.0, "   (y el calculo por span habria dado otra cosa)",
        "%.2f Hz, %.1fx mas bajo" % (por_span, cad / por_span))

# Without the gaps the two agree, which is why the wrong version survives so long.
liso = np.arange(0, 100) * paso
revisar(abs(instantaneous_rate(liso, warn=False) - len(liso) / (liso[-1] - liso[0])) < 0.2,
        "sin huecos, ambos calculos coinciden: por eso el error sobrevive")

try:
    instantaneous_rate([1.0])
    revisar(False, "un solo tiempo tiene que reventar")
except InvariantError:
    revisar(True, "un solo tiempo revienta en vez de inventar un numero")

try:
    instantaneous_rate([5.0, 4.0, 3.0])
    revisar(False, "tiempos al reves tienen que reventar")
except InvariantError:
    revisar(True, "tiempos desordenados revientan")


print()
print("=" * 70)
print("2. La tasa declarada tiene que ser la entregada")
print("=" * 70)

try:
    check_rate(3.00, 2.31, what="la tasa del lazo")   # measured on the Pi, 2026-08-25
    revisar(False, "2.31 contra 3.00 tiene que reventar")
except InvariantError as exc:
    revisar("2.31" in str(exc) and "3.00" in str(exc),
            "el caso real de la Pi revienta y dice los dos numeros")

try:
    check_rate(3.00, 2.55, what="el submuestreo")     # the subsampler bug, same night
    revisar(False, "2.55 contra 3.00 tiene que reventar")
except InvariantError:
    revisar(True, "el caso real del submuestreo revienta")

try:
    check_rate(3.00, 2.98)
    revisar(True, "2.98 contra 3.00 pasa: la tolerancia no es paranoia")
except InvariantError:
    revisar(False, "2.98 contra 3.00 no deberia reventar")


print()
print("=" * 70)
print("3. Una cache no se usa si no la produjeron estos parametros")
print("=" * 70)

tmp = tempfile.mkdtemp()
ruta = os.path.join(tmp, "obs.npz")
P1 = {"tasa": 3.0, "modelo": "y960", "buffer_s": 8.0}
P2 = {"tasa": 4.0, "modelo": "y960", "buffer_s": 8.0}

save_cache(ruta, P1, obs=np.arange(10.0), cadencia=np.array(3.11))
d = load_cache(ruta, P1)
revisar(d is not None and len(d["obs"]) == 10, "con los mismos parametros, se carga")
revisar(d is not None and float(d["cadencia"]) == 3.11, "y trae todo lo guardado")

revisar(load_cache(ruta, P2, quiet=True) is None,
        "con OTROS parametros, se ignora en vez de mentir")

# The shape that matters: a stale file with no stamp at all, left by an earlier version.
viejo = os.path.join(tmp, "viejo.npz")
np.savez(viejo, obs=np.arange(3.0))          # written by the old code, no stamp at all
revisar(load_cache(viejo, P1, quiet=True) is None,
        "una cache SIN firma tambien se ignora (la de la corrida zombi)")

revisar(load_cache(os.path.join(tmp, "no_existe.npz"), P1) is None,
        "y si no existe, devuelve None sin drama")

# A float that arrives as int, or a key in another order, must not force a rebuild.
revisar(load_cache(ruta, {"buffer_s": 8.0, "modelo": "y960", "tasa": 3.0}) is not None,
        "el orden de las claves no cuenta como cambio")

print()
if FALLOS:
    print("FALLARON %d:" % len(FALLOS))
    for f in FALLOS:
        print("  -", f)
    sys.exit(1)
print("TODO OK")
