"""IDF1 and ID switches on cases small enough to count by hand.

Each section is a tracker failure with its numbers worked out on paper beforehand, so the metric
has to reproduce arithmetic that does not depend on the code. The contrasts matter as much as the
values: a switch and a fragment cost the same IDF1 here, a box left without an id costs recall and
not precision, and a false positive labelled 'x' costs nothing at all.

Run: python tests/test_metricas_identidad.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_vision.identity_metrics import identity_scores

T = list(range(8))
VERDAD = ["A", "A", "A", "A", "B", "B", "B", "B"]


def ver(nombre, pred, idf1, idsw, verdad=VERDAD, orden=T, **extra):
    r = identity_scores(verdad, pred, orden)
    print("  %-34s IDF1 %.3f  IDP %.3f  IDR %.3f  IDsw %d"
          % (nombre, r["idf1"], r["idp"], r["idr"], r["id_switches"]))
    assert abs(r["idf1"] - idf1) < 1e-9, (nombre, r)
    assert r["id_switches"] == idsw, (nombre, r)
    for k, v in extra.items():
        assert abs(r[k] - v) < 1e-9, (nombre, k, r)
    return r


print("=" * 70)
print("1. PERFECTO: dos personas, un id cada una")
print("=" * 70)
ver("perfecto", [1, 1, 1, 1, 2, 2, 2, 2], 1.0, 0)
ver("perfecto con otros nombres de id", [7, 7, 7, 7, 3, 3, 3, 3], 1.0, 0)

print("=" * 70)
print("2. CAMBIO DE ID: la mitad de A se lleva el id de B")
print("=" * 70)
# shared: A->1: 2, A->2: 2, B->2: 4. Best one-to-one: A-1 (2) + B-2 (4) = 6 of 8.
# IDF1 = 2*6 / (2*6 + 2 + 2) = 0.75. A goes 1 -> 2: one switch.
ver("A: 1,1,2,2", [1, 1, 2, 2, 2, 2, 2, 2], 0.75, 1, idp=0.75, idr=0.75)

print("=" * 70)
print("3. FRAGMENTO: la mitad de A recibe un id nuevo")
print("=" * 70)
# Same arithmetic as the switch: A-1 (2) + B-2 (4) = 6. The IDF1 cannot tell them apart here.
ver("A: 1,1,3,3", [1, 1, 3, 3, 2, 2, 2, 2], 0.75, 1)

print("=" * 70)
print("4. SIN ID: el tracker no confirmo dos cajas de A")
print("=" * 70)
# 6 boxes with an id, all correct: IDTP 6, IDFP 0, IDFN 2. IDF1 = 12/14. No switch: a gap is
# not a change of id.
ver("A: 1,1,-,-", [1, 1, None, None, 2, 2, 2, 2], 12 / 14, 0, idp=1.0, idr=0.75)

print("=" * 70)
print("5. UNO SOLO PARA TODOS: el tracker funde a A y B")
print("=" * 70)
# shared: A->1: 4, B->1: 4. One-to-one: only one of them keeps id 1. IDTP 4 of 8: IDF1 0.5.
# No switch on either: each identity always carries id 1.
ver("todo id 1", [1] * 8, 0.5, 0)

print("=" * 70)
print("6. LO QUE NO ES IDENTIDAD NO PUNTUA")
print("=" * 70)
r = ver("un falso positivo marcado x", [1, 1, 1, 1, 2, 2, 2, 9], 1.0, 0,
        verdad=VERDAD[:7] + ["x"])
assert r["excluded"] == 1 and r["boxes"] == 7, r
print("  la caja 'x' queda fuera y se cuenta aparte: excluded=%d" % r["excluded"])

print("=" * 70)
print("7. EL ORDEN TEMPORAL MANDA, NO EL ORDEN DE LA LISTA")
print("=" * 70)
# Same boxes, listed out of time order. A in time: 1, 1, 2, 2 -> one switch, not three.
ver("desordenado", [2, 1, 2, 1, 2, 2, 2, 2], 0.75, 1, orden=[3, 0, 2, 1, 4, 5, 6, 7])

print("test_metricas_identidad OK")
