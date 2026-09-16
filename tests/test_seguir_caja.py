"""
Gate for the tracker that carries a drawn box across frames.

The feature exists so a person no detector proposed is drawn once instead of once per frame, and the
only way that helps is if the box actually follows the person: a tracker that leaves the box where it
was would be worse than useless, because it would fill the truth with boxes on empty ground. So the
test moves a patch across synthetic frames and demands that the boxes land on it, that nothing is
written where the reviewer already put a person, and that the walk stops instead of inventing when
the patch leaves.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import cv2
import etiquetar_grupos as E


def _frames(carpeta, n, paso):
    """n frames with a dark square sliding to the right by `paso` pixels each time."""
    for k in range(n):
        img = np.full((360, 640, 3), 200, np.uint8)
        x = 80 + k * paso
        cv2.rectangle(img, (x, 150), (x + 40, 230), (30, 30, 30), -1)
        cv2.imwrite(os.path.join(carpeta, "frame_%04d.jpg" % (100 + k)), img)
    return [100 + k for k in range(n)]


def _sesion(carpeta, frames, nombre="salida"):
    # each session gets its own review file: sharing one would let the boxes of a previous walk
    # count as "already there" and the next walk would skip them, which reads like a tracker failure
    cajas = os.path.join(carpeta, "cajas.csv")
    with open(cajas, "w", newline="") as fh:
        fh.write("frame,conf,x1,y1,x2,y2,fuente\n")
        fh.write("%d,0.9,10,10,30,30,vuelo\n" % frames[0])
    embs = np.zeros((1, 512), "float32")
    return E.Sesion(cajas, embs, carpeta, os.path.join(carpeta, nombre + ".json"), 1,
                    None, None, lista_frames=frames)


base = tempfile.mkdtemp()
try:
    frames = _frames(base, 12, 6)
    s = _sesion(base, frames)

    # 1. the box follows the patch instead of staying where it was drawn
    r = s.seguir(frames[0], [80, 150, 120, 230], True, 8)
    assert r["puestas"] >= 5, "el seguidor apenas dejo %d cajas" % r["puestas"]
    lejos = []
    for f, cajas in s.nuevas.items():
        if f == frames[0]:
            continue
        esperado = 80 + (f - frames[0]) * 6 + 20          # centre of the patch in that frame
        cx = (cajas[0][0] + cajas[0][2]) / 2
        lejos.append(abs(cx - esperado))
    assert max(lejos) < 12, "la caja no siguio al objeto: se desvio hasta %.0f px" % max(lejos)
    print("  sigue al objeto: %d cajas, desvio maximo %.1f px" % (r["puestas"], max(lejos)))

    # 2. a frame the reviewer already settled is left alone
    s2 = _sesion(base, frames, "s2")
    s2.nuevas[frames[3]] = [[80 + 3 * 6, 150.0, 80 + 3 * 6 + 40.0, 230.0]]
    antes = list(s2.nuevas[frames[3]])
    s2.seguir(frames[0], [80, 150, 120, 230], True, 8)
    assert s2.nuevas[frames[3]] == antes, "piso una caja que ya estaba en el frame"
    print("  no pisa lo que el revisor ya decidio")

    # 3. walking backwards from the end lands on the patch too
    s3 = _sesion(base, frames, "s3")
    ult = frames[-1]
    x = 80 + (len(frames) - 1) * 6
    r3 = s3.seguir(ult, [x, 150, x + 40, 230], False, 8)
    assert r3["puestas"] >= 5, "hacia atras solo dejo %d" % r3["puestas"]
    assert r3["hasta"] < ult, "hacia atras no retrocedio"
    print("  tambien camina hacia atras: %d cajas hasta el frame %d" % (r3["puestas"], r3["hasta"]))

    # 4. a frame outside the list is refused rather than silently ignored
    try:
        s.seguir(99999, [80, 150, 120, 230], True, 3)
        raise AssertionError("acepto un frame fuera de la lista")
    except ValueError:
        print("  rechaza un frame fuera de la lista")
finally:
    shutil.rmtree(base, ignore_errors=True)

print("TODO OK")
