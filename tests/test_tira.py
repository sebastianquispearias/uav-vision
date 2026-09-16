"""
Gate for the strip view: many frames at once, with the boxes the review holds drawn on them.

A box that slides off a person over twenty frames looks right in each frame on its own, which is how
the drift got into the truth in the first place. The strip exists to make that pattern visible, and it
only does so if every cell shows the SAME piece of ground: a view that recentres on each frame would
hide exactly the drift it is meant to reveal. So the test checks the patch is shared, that the cells
cover the whole run instead of its first few frames, and that a cell really carries the boxes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import cv2
import etiquetar_grupos as E

base = tempfile.mkdtemp()
try:
    frames = list(range(200, 260))
    for f in frames:
        img = np.full((400, 600, 3), 190, np.uint8)
        cv2.rectangle(img, (300, 200), (330, 260), (40, 40, 40), -1)
        cv2.imwrite(os.path.join(base, "frame_%04d.jpg" % f), img)
    cajas = os.path.join(base, "cajas.csv")
    with open(cajas, "w", newline="") as fh:
        fh.write("frame,conf,x1,y1,x2,y2,fuente\n")
        for f in frames[:40]:
            fh.write("%d,0.9,300,200,330,260,vuelo\n" % f)
    embs = np.zeros((40, 512), "float32")
    embs[:, 0] = 1.0
    embs[::2, 1] = 0.5
    salida = os.path.join(base, "etiquetas.json")
    s = E.Sesion(cajas, embs, base, salida, 2, lista_frames=frames)
    for i in range(40):
        s.corregir(i, "persona")

    d = s.tira_datos(200, 259)
    assert d["total"] == 60, "no conto el tramo entero: %s" % d["total"]
    assert len(d["frames"]) > 8, "trajo muy pocas celdas: %d" % len(d["frames"])
    assert d["frames"][-1] > 240, "las celdas se quedan al principio del tramo: terminan en %d" % d["frames"][-1]
    print("  %d celdas repartidas entre %d y %d de %d frames"
          % (len(d["frames"]), d["frames"][0], d["frames"][-1], d["total"]))

    # the patch is the same in every cell: it is one centre and one side, not one per frame
    assert abs(d["centro"][0] - 315) < 12 and abs(d["centro"][1] - 230) < 12, "el centro no cayo en la gente: %s" % d["centro"]
    assert d["lado"] >= 300, "el recorte quedo mas chico que el minimo: %s" % d["lado"]
    print("  un solo recorte para todas: centro (%.0f, %.0f), lado %d" % (d["centro"][0], d["centro"][1], d["lado"]))

    # a cell with a person differs from the same cell without: the boxes are really drawn
    con = s.celda(200, d["centro"][0], d["centro"][1], d["lado"])
    s2 = E.Sesion(cajas, embs, base, os.path.join(base, "otra.json"), 2, lista_frames=frames)
    sin = s2.celda(200, d["centro"][0], d["centro"][1], d["lado"])
    assert con != sin, "la celda salio igual con y sin cajas: no las esta dibujando"
    assert len(con) > 500, "la celda vino vacia"
    print("  la celda dibuja las cajas (%d bytes con, %d sin)" % (len(con), len(sin)))

    # frames with nobody are still shown, flagged by their count, instead of disappearing
    vacios = [f for f in d["frames"] if d["personas"][str(f)] == 0]
    assert vacios, "los frames sin gente desaparecieron de la tira"
    print("  muestra tambien los %d frames sin nadie" % len(vacios))
    # the whole-frame mode shows what the patch hides: somebody standing outside it
    s3 = E.Sesion(cajas, embs, base, os.path.join(base, "lejos.json"), 2, lista_frames=frames)
    for i in range(40):
        s3.corregir(i, "persona")
    s3.nueva(200, [40, 40, 80, 120])                      # a person far from the patch's centre
    recorte = s3.celda(200, d["centro"][0], d["centro"][1], d["lado"], 190)
    entero = s3.celda(200, 0, 0, 0, 320)
    sin_lejos = s.celda(200, 0, 0, 0, 320)                # same frame, without that person
    assert entero != sin_lejos, "el frame entero no muestra a quien esta fuera del recorte"
    assert len(entero) > len(recorte) / 4, "el frame entero vino vacio"
    print("  el modo frame entero muestra a quien el recorte deja afuera")

finally:
    shutil.rmtree(base, ignore_errors=True)

print("TODO OK")
