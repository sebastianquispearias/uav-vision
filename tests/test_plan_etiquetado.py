"""
Gate for the page that decides where to label, by the flight's own altitude.

Where to label was a pair of numbers typed by whoever was looking, and that is how a stretch at three
metres got proposed while the one at twenty-five did not. The page answers it from the flight: it has
to say how much of a stretch is NEW rather than how many frames it holds, and how big a person looks
there, because a stretch full of 180 px people adds nothing to a set that already drowns in them.

The person-size estimate is measured, not derived: the camera looks forward and down, so a person at
four metres is far along the ground and comes out no bigger than one at ten. A formula from the optics
would say otherwise, so the test pins the shape that the labelled flights actually showed.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import cv2
import etiquetar_grupos as E

# 1. the size estimate follows the flights, not the optics: it does not grow as the drone descends
assert E.persona_px(4) >= E.persona_px(10) >= E.persona_px(18) >= E.persona_px(30), \
    "el tamano estimado no decrece con la altura: %s" % [E.persona_px(a) for a in (4, 10, 18, 30)]
assert abs(E.persona_px(25) - 62) <= 8, "a 25 m deberia rondar los 62 px medidos, da %d" % E.persona_px(25)
assert E.persona_px(8) > 140, "a 8 m la persona se ve grande y el estimador dice %d" % E.persona_px(8)
print("  tamano por altura: 4 m %d px, 10 m %d px, 18 m %d px, 25 m %d px, 35 m %d px"
      % tuple(E.persona_px(a) for a in (4, 10, 18, 25, 35)))

base = tempfile.mkdtemp()
try:
    vuelo = os.path.join(base, "vuelo")
    frames_dir = os.path.join(vuelo, "frames")
    os.makedirs(frames_dir)
    # a flight that climbs: low at the start, 25 m at the end
    with open(os.path.join(vuelo, "frames.csv"), "w", newline="") as fh:
        fh.write("frame,alt_agl\n")
        for f in range(100, 200):
            fh.write("%d,%.1f\n" % (f, 1.0 + (f - 100) * 0.3))
    for f in range(100, 200):
        cv2.imwrite(os.path.join(frames_dir, "frame_%04d.jpg" % f), np.full((120, 160, 3), 170, np.uint8))
    cajas = os.path.join(base, "cajas.csv")
    with open(cajas, "w", newline="") as fh:
        fh.write("frame,conf,x1,y1,x2,y2,fuente\n100,0.9,10,10,40,80,vuelo\n101,0.8,12,10,42,80,vuelo\n")
    embs = np.zeros((2, 512), "float32")
    embs[0, 0], embs[1, 1] = 1.0, 1.0
    s = E.Sesion(cajas, embs, frames_dir, os.path.join(base, "et.json"), 2,
                 lista_frames=list(range(100, 140)))
    s.corregir(0, "persona")          # a frame cannot be called reviewed with a box still undecided
    s.marcar_revisado(100, True)

    d = s.alturas(cada=1)
    assert d["hay"] and len(d["frames"]) == 100, "no leyo el perfil entero: %s" % d.get("motivo")
    assert d["revisado"][0] and not d["revisado"][5], "no marca lo revisado sobre el perfil"
    assert d["en_lista"][10] and not d["en_lista"][60], "no distingue lo que esta en la lista de lo que nunca se propuso"
    print("  perfil: %d puntos, altura maxima %.1f m, marca revisados y lista" % (len(d["frames"]), d["max"]))

    # 2. a stretch reports what is NEW, which is the number the decision hangs on
    bajo = s.tramo(100, 120)
    alto = s.tramo(180, 199)
    assert bajo["ya_revisados"] == 1 and bajo["frames"] == 21, "el tramo bajo no conto bien: %s" % bajo
    assert alto["ya_revisados"] == 0 and alto["ya_en_lista"] == 0, "el tramo alto no esta virgen: %s" % alto
    assert alto["alt_mediana"] > bajo["alt_mediana"], "no distingue la altura de los dos tramos"
    assert alto["persona_px"] < bajo["persona_px"], \
        "el tramo alto deberia dar una persona mas chica: %d contra %d" % (alto["persona_px"], bajo["persona_px"])
    print("  tramo bajo: %.1f m, persona %d px, %d ya revisados | tramo alto: %.1f m, persona %d px, %d nuevos"
          % (bajo["alt_mediana"], bajo["persona_px"], bajo["ya_revisados"],
             alto["alt_mediana"], alto["persona_px"], alto["frames"] - alto["ya_revisados"]))

    # 3. a flight without frames.csv says so instead of pretending it has a profile
    solo = os.path.join(base, "solo", "frames")
    os.makedirs(solo)
    cv2.imwrite(os.path.join(solo, "frame_0100.jpg"), np.full((120, 160, 3), 170, np.uint8))
    s2 = E.Sesion(cajas, embs, solo, os.path.join(base, "et2.json"), 2, lista_frames=[100, 101])
    assert s2.alturas()["hay"] is False, "invento un perfil sin frames.csv"
    print("  un vuelo sin frames.csv lo dice en vez de inventar el perfil")
finally:
    shutil.rmtree(base, ignore_errors=True)

print("TODO OK")
