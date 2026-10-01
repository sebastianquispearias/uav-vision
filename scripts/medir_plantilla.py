"""What the operator's click buys, scored in people and phantoms instead of in boxes.

Three runs of the same flight, judged with the same scoreboard:

    baseline            nothing pointed at, which is what flies today
    window              the click lowers the threshold inside the projected square
    window + template   the click also says what the target LOOKS like

The box-level measurement that motivated the third already exists: on 773 frames of the mission
it takes recall on the target from 91.1 % to 94.4 % at no computing cost, because the detector had
already scored those boxes and was discarding them. What that measurement cannot say is whether
the extra boxes reach the map as people or as phantoms, and that is the only question the product
asks. The window was measured this way once before and the answer was that seventeen points of
recall bought nothing: the same five people and the same six phantoms.

The template is built from the hand labels of the flight, which is what makes it the appearance of
a person somebody actually pointed at rather than of whatever the detector was most sure about.
The click itself is placed where the BASELINE run put that person, not on the surveyed truth:
fixing the true position would measure a mode nobody can use.

Run with:
    python scripts/medir_plantilla.py                 # the operator, letter A
    python scripts/medir_plantilla.py --letra=G       # a walker instead
    python scripts/medir_plantilla.py --emb-dist=1.0  # a looser appearance gate
"""
import contextlib
import io as _io
import json
import os
import runpy
import sys
import tempfile

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, AQUI)
os.environ.setdefault("UAV_VISION_DATOS", os.path.join(RAIZ, "demo", "data"))
os.environ.pop("UAV_VISION_GS", None)

DATOS = os.environ["UAV_VISION_DATOS"]
PISTAS = os.path.join(DATOS, "pistas_bot_cmc_sof.npz")
REPLAY = os.path.join(AQUI, "replay_vuelo3.py")
ETIQUETAS = os.environ.get(
    "UAV_VISION_ETIQUETAS",
    os.path.join(os.path.dirname(RAIZ), "drone-geolocation", "entrenamiento"))

LETRA = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--letra=")), "A")
EMB_DIST = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--emb-dist=")), None)

with contextlib.redirect_stdout(_io.StringIO()):
    import personas_encontradas as P

TMP = tempfile.mkdtemp()


def corre(extra):
    """One replay, its candidates on disk, and the scoreboard's verdict on them."""
    cand = os.path.join(TMP, "cand_%d.json" % abs(hash(tuple(extra))))
    sys.argv = [REPLAY, "--pistas=" + PISTAS, "--candidatos=" + cand] + list(extra)
    with contextlib.redirect_stdout(_io.StringIO()):
        runpy.run_path(REPLAY, run_name="__main__")
    with contextlib.redirect_stdout(_io.StringIO()):
        encontradas, fantasmas = P.evaluar("", DATOS, PISTAS, cand)
    with open(cand) as f:
        cands = json.load(f)
    return encontradas, fantasmas, cands


def plantilla_de(letra):
    """The mean appearance of the boxes a human put this letter on, normalised.

    A mean and not one crop on purpose: a single sighting carries whatever the lighting and the
    pose were in that frame, and the operator is pointing at a person, not at a frame.
    """
    gt = json.load(open(os.path.join(ETIQUETAS, "identidad_gt_02ago.json")))["etiquetas"]
    datos = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))
    dets, embs = datos["dets"], datos["embs"].astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    # The ground truth is indexed over the detections above the scoreboard's floor, the same
    # 0.25 cut personas_encontradas.py explains it cannot be read against any other.
    idx = np.nonzero(dets[:, 1] >= 0.25)[0]
    filas = [idx[int(k)] for k, v in gt.items() if v == letra and int(k) < len(idx)]
    if not filas:
        raise SystemExit("la letra %r no tiene cajas etiquetadas" % letra)
    v = embs[filas].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9), len(filas)


def quien(cands, encontradas):
    """Position of the candidate that holds the target letter, as the BASELINE reported it."""
    for etiqueta in encontradas.get(LETRA, []):
        for c in cands:
            if "(%.1f, %.1f)" % (c["x"], c["y"]) in etiqueta:
                return c["x"], c["y"]
    return None


def fila(nombre, encontradas, fantasmas, cands):
    letras = "".join(sorted(k for k in encontradas if len(k) == 1))
    maduros = sum(1 for c in cands if c.get("mature"))
    print("  %-22s %d de 7 (%-7s) %3d fantasmas %4d candidatos %3d maduros"
          % (nombre, len(letras), letras, len(fantasmas), len(cands), maduros))
    return letras, len(fantasmas)


print()
print("=" * 86)
print("QUE COMPRA EL CLICK DEL OPERADOR, EN PERSONAS Y FANTASMAS   (objetivo: letra %s)" % LETRA)
print("=" * 86)

plant, n_cajas = plantilla_de(LETRA)
ruta = os.path.join(TMP, "plantilla.npy")
np.save(ruta, plant)
print("  plantilla de %s construida con %d cajas etiquetadas a mano" % (LETRA, n_cajas))
print()

base_e, base_f, base_c = corre([])
print("  %-22s %s" % ("", "personas          fantasmas   candidatos   maduros"))
fila("base, sin click", base_e, base_f, base_c)

pos = quien(base_c, base_e)
if pos is None:
    raise SystemExit("la base no reporto a %s, no hay donde hacer click" % LETRA)
foco = "--foco=%.2f,%.2f" % pos
print("  el operador hace click donde la base puso a %s: (%.2f, %.2f)" % (LETRA, pos[0], pos[1]))
print()

vent_e, vent_f, vent_c = corre([foco])
fila("solo ventana", vent_e, vent_f, vent_c)

extra = [foco, "--plantilla=" + ruta] + (["--emb-dist=" + EMB_DIST] if EMB_DIST else [])
pl_e, pl_f, pl_c = corre(extra)
fila("ventana + plantilla", pl_e, pl_f, pl_c)

print()
print("  LECTURA")
base_l, base_fn = "".join(sorted(k for k in base_e if len(k) == 1)), len(base_f)
pl_l, pl_fn = "".join(sorted(k for k in pl_e if len(k) == 1)), len(pl_f)
if pl_l == base_l and pl_fn == base_fn:
    print("  La plantilla NO cambia el marcador de producto: las mismas personas (%s) y los" % base_l)
    print("  mismos %d fantasmas. Gana cajas del objetivo y ninguna de esas cajas se convierte" % base_fn)
    print("  en una persona que antes no llegaba, igual que paso con la ventana sola.")
else:
    print("  La plantilla SI mueve el marcador: personas %s -> %s, fantasmas %d -> %d."
          % (base_l, pl_l, base_fn, pl_fn))
print()
print("  LIMITES, para que el numero no se lea de mas: un vuelo, una escena, y la distancia de")
print("  apariencia se eligio mirando este mismo tramo. Re-elegirla en los 01ago es lo que falta")
print("  antes de poder defenderla. Y el marcador no puede ver lo que el click le ahorra al")
print("  operador, porque corre sobre un vuelo grabado sin nadie clickeando.")
