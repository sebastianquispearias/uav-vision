"""The labelling tool's review list flags what a labeller would miss, and nothing a labeller got right.

Each section is a pair: a case that must be flagged and its near twin that must not. A list that
flagged everything would be as useless as one that flagged nothing, so both directions are checked
on synthetic detections whose ground points are set by hand.

Run: python tests/test_etiquetar.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from etiquetar_identidad import problemas


def caja(frame, x1, y1, x2, y2, suelo):
    """One detection row: frame, conf, box, and a vertical ray that lands on the given ground point."""
    return [frame, 0.8, x1, y1, x2, y2, suelo[0], suelo[1], 10.0, 0.0, 0.0, -1.0]


def revisar(filas, etiquetas, tiempos):
    dets = np.array(filas, dtype=float)
    grupos = [[i] for i in range(len(dets))]
    return problemas(dets, lambda f: tiempos[f], grupos, etiquetas)


def tipos(lista, tipo):
    return [p for p in lista if p["tipo"] == tipo]


T = {1: 0.0, 2: 0.5, 3: 1.0}

print("=" * 70)
print("1. MISMA LETRA EN UN FRAME: duplicado del detector frente a dos personas")
print("=" * 70)
dup = revisar([caja(1, 100, 100, 140, 200, (0, 0)), caja(1, 104, 102, 142, 203, (0, 0))],
              {"0": "A", "1": "A"}, T)
dos = revisar([caja(1, 100, 100, 140, 200, (0, 0)), caja(1, 600, 100, 640, 200, (9, 0))],
              {"0": "A", "1": "A"}, T)
assert not tipos(dup, "misma letra dos veces"), dup
assert len(tipos(dos, "misma letra dos veces")) == 1, dos
print("  dos cajas casi encima con A: no se marca | dos cajas lejos con A:", tipos(dos, "misma letra dos veces")[0]["texto"])

print("=" * 70)
print("2. ETIQUETA RARA: AX frente a A, X y ?")
print("=" * 70)
filas = [caja(1, 0, 0, 10, 10, (0, 0)), caja(2, 0, 0, 10, 10, (0, 0)),
         caja(3, 0, 0, 10, 10, (0, 0)), caja(3, 50, 0, 60, 10, (5, 0))]
r = revisar(filas, {"0": "A", "1": "X", "2": "?", "3": "AX"}, T)
raras = tipos(r, "etiqueta rara")
assert [p["caja"] for p in raras] == [3], raras
print("  solo se marca la caja con AX:", raras[0]["texto"])

print("=" * 70)
print("3. SALTO EN EL SUELO: 8 m en medio segundo frente a 1 m")
print("=" * 70)
lejos = revisar([caja(1, 0, 0, 10, 10, (0, 0)), caja(2, 0, 0, 10, 10, (8, 0))], {"0": "B", "1": "B"}, T)
cerca = revisar([caja(1, 0, 0, 10, 10, (0, 0)), caja(2, 0, 0, 10, 10, (1, 0))], {"0": "B", "1": "B"}, T)
assert len(tipos(lejos, "salto en el suelo")) == 1 and not tipos(cerca, "salto en el suelo"), (lejos, cerca)
print("  8 m:", tipos(lejos, "salto en el suelo")[0]["texto"], "| 1 m: no se marca")

print("=" * 70)
print("4. SIN ETIQUETAR y LETRAS MEZCLADAS")
print("=" * 70)
dets = np.array([caja(1, 0, 0, 10, 10, (0, 0)), caja(2, 0, 0, 10, 10, (0, 0))], dtype=float)
mezcla = problemas(dets, lambda f: T[f], [[0, 1]], {"0": "A", "1": "B"})
limpio = problemas(dets, lambda f: T[f], [[0, 1]], {"0": "A", "1": "A"})
falta = problemas(dets, lambda f: T[f], [[0, 1]], {"0": "A"})
assert len(tipos(mezcla, "letras mezcladas")) == 1 and not tipos(limpio, "letras mezcladas")
assert len(tipos(falta, "sin etiquetar")) == 1 and not tipos(limpio, "sin etiquetar")
print("  A y B en un segmento: se marca | A y A: no | una caja sin letra:", tipos(falta, "sin etiquetar")[0]["texto"])

print("test_etiquetar OK")
