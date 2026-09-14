"""Dos drones, un objetivo: la distancia que se tolera sale del margen de cada reporte.

Cada dron tiene su propio sesgo de GPS y rumbo, asi que el mismo objetivo cae en dos sitios. Con el
radio del 95 % de cada reporte, la tolerancia es sqrt(ra^2 + rb^2); sin radio, el radio de la clase
de siempre. Contrastes que fallan si la regla miente:

  - el mismo objetivo a 6 m, radios de 5 m: se funde (con el 3.5 m fijo de antes, no);
  - el mismo aspecto a 9 m: no se funde (fuera de sqrt(5^2 + 5^2) = 7.07 m);
  - dos personas distintas a 6 m: no se funden (la apariencia decide);
  - dos coches a 3 m con radios de 5 m: no se funden (tope de la plaza de aparcamiento, 2.5 m);
  - sin radio: se comporta como antes.

Run: python tests/test_mismo_objetivo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from uav_vision.flota import distancia_maxima, mismo_objetivo


def emb(semilla, ruido=0.0):
    r = np.random.default_rng(semilla)
    v = r.normal(size=512).astype(np.float32)
    if ruido:
        v = v + ruido * np.random.default_rng(semilla + 1000).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def poi(x, cls='person', r=None, e=None):
    d = {'x': x, 'y': 0.0, 'cls': cls}
    if r is not None:
        d['radius_m'] = r
    if e is not None:
        d['emb'] = e
    return d


casos = [
    ('mismo objetivo a 6 m, radios 5 m', poi(0, r=5.0, e=emb(1)), poi(6, r=5.0, e=emb(1, 0.1)), True),
    ('mismo aspecto a 9 m, radios 5 m', poi(0, r=5.0, e=emb(1)), poi(9, r=5.0, e=emb(1, 0.1)), False),
    ('personas distintas a 6 m', poi(0, r=5.0, e=emb(1)), poi(6, r=5.0, e=emb(2)), False),
    ('dos coches a 3 m, radios 5 m', poi(0, 'car', 5.0, emb(3)), poi(3, 'car', 5.0, emb(3, 0.1)), False),
    ('sin radio, mismo objetivo a 6 m', poi(0, e=emb(1)), poi(6, e=emb(1, 0.1)), False),
    ('sin radio, mismo objetivo a 3 m', poi(0, e=emb(1)), poi(3, e=emb(1, 0.1)), True),
]
for nombre, a, b, esperado in casos:
    got = mismo_objetivo(a, b)
    print('  %-34s tolerancia %5.2f m -> %s' % (nombre, distancia_maxima(a, b), 'SE FUNDE' if got else 'separados'))
    assert got == esperado, '%s: esperaba %s' % (nombre, esperado)

assert abs(distancia_maxima(poi(0, r=5.0), poi(0, r=5.0)) - 7.071) < 0.01
assert distancia_maxima(poi(0, 'car', 5.0), poi(0, 'car', 5.0)) == 2.5

print()
print('TODO OK')
