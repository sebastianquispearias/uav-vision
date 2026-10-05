"""
Scores the chain by person, not by box: how many of the people who were there it reported, and how
much it invented, using the letters a human assigned to every detection of the test flight.

The number that matters to an operator is not recall: it is how many real people reach the map and
how many false points have to be walked to. Those two do not move together, and this repo has the
case that proves it, kept in docs/DESCARTADO.md: a retrained detector that won on boxes and lost a
person. Box metrics cannot see that, so this is the scoreboard the product is judged by.

No surveyed position is needed. The letters already say which boxes are the same person, so a
candidate is traced back through its tracks to the boxes that fed it, and those boxes to a letter.
A candidate whose boxes are labelled X is a phantom; one whose boxes are labelled G is the walking
woman, wherever she happened to be standing.

A detection in a frame nobody labelled cannot be judged, and is counted apart rather than called a
phantom: the flight is longer than the labelled windows, and blaming the chain for what was never
labelled would manufacture a result.

The candidates are not kept on disk. They are produced by the replay of the same flight, so the
score always describes the identity layer as it stands today:

    UAV_VISION_DATOS=demo/data python scripts/replay_vuelo3.py \
        --pistas=demo/data/pistas_bot_cmc_sof.npz \
        --candidatos=demo/data/candidatos_vuelo_02ago.json
    python scripts/personas_encontradas.py

The labels live in the flight archive next to this repo (../drone-geolocation/entrenamiento), the
same place the rest of the flight's ground truth lives.

TWO DIFFERENT THINGS LIVE IN THE LABELS, AND CONFUSING THEM WAS AN ERROR WORTH SPELLING OUT.
The REVIEW says which boxes are people: it is complete, it includes the ones the user drew, and
it is what the detector is scored against. The LETTERS say WHICH person a box is, and they were
assigned earlier, only over the boxes the detectors had proposed, so two thirds of the balcony's
people carry no letter. Judging "is this anybody" by the LETTERS therefore turns real people
into phantoms. So the review decides personhood here, and the letters only put a name on it when
they can.

The letters are keyed BY POSITION in the flight's original detection cache, not by frame and
box, so they are always read against demo/data even when another cache is being scored. Reading
them against a cache holding a different number of detections shifts every letter and hands the
score to the wrong person: it once reported the walking woman's candidate as the operator.

THE LETTERS WERE ASSIGNED WINDOW BY WINDOW. Between the windows there are frames whose boxes
carry no letter simply because nobody looked, and scoring a candidate there marks a real person
as a phantom. So only the windows are judged, and everything else is counted as not judgeable.

'piso' is the confidence the TRACKS were computed at, not a taste: the track array has one row
per detection above it, and reading it against a different cut shifts every id by one. The
fixed-target mode is the case that needs another value, because there the tracker was also shown
the band the detector was discarding.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAC = os.path.dirname(_HERE)
_ETIQUETAS_EN_EL_REPO = os.path.join(_HERE, "demo", "etiquetas")
ENT = os.environ.get(
    "UAV_VISION_ETIQUETAS",
    _ETIQUETAS_EN_EL_REPO if os.path.isdir(_ETIQUETAS_EN_EL_REPO)
    else os.path.join(_LAC, "drone-geolocation", "entrenamiento"))
DATOS = os.environ.get("UAV_VISION_DATOS", os.path.join(_HERE, "demo", "data"))
REALES = set("ABCDEGH")

_filas = list(csv.DictReader(open(os.path.join(ENT, "candidatas_02ago.csv"), encoding="utf-8")))
_g = json.load(open(os.path.join(ENT, "etiquetas_detector_02ago.json"), encoding="utf-8"))["etiquetas"]
_r = json.load(open(os.path.join(ENT, "etiquetas_detector_02ago_frames.json"), encoding="utf-8"))
def _final(i):
    return _r["correcciones"].get(str(i), _g.get(str(i)))
_aj = {int(i): c for i, c in _r["ajustes"].items()}
REVISADOS = {int(x) for x in _r["revisados"]}
personas_de_frame = defaultdict(list)
for _i, _fila in enumerate(_filas):
    if _final(_i) == "persona":
        personas_de_frame[int(_fila["frame"])].append(
            _aj.get(_i) or [float(_fila[c]) for c in ("x1", "y1", "x2", "y2")])
for _k, _cs in _r.get("nuevas", {}).items():
    for _c in _cs:
        if not (len(_c) > 4 and _c[4] == "ignorar"):
            personas_de_frame[int(_k)].append([float(x) for x in _c[:4]])

g = json.load(open(os.path.join(ENT, "identidad_gt_02ago.json"), encoding="utf-8"))["etiquetas"]
base = np.load(os.path.join(_HERE, "demo", "data", "examen_v3_datos.npz"))["dets"]
base = base[base[:, 1] >= 0.25]
letra_de_caja = defaultdict(list)
for i, l in g.items():
    i = int(i)
    if i < len(base):
        letra_de_caja[int(base[i, 0])].append((base[i, 2:6], str(l).upper()))
print("frames revisados: %d | personas en la verdad: %d | de esas, con letra: %d"
      % (len(REVISADOS), sum(len(v) for v in personas_de_frame.values()), len(g)))


def iou(a, B):
    B = np.asarray(B, float)
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)


VENTANAS = [(2551, 2641), (2746, 2952), (3000, 3700)]


def letra_de(det):
    """What the review says a detection is: a person's letter, PERSONA when nobody named them, X when
    the review says nobody is there, and None when the frame was never reviewed and cannot be judged."""
    f = int(det[0])
    if f not in REVISADOS:
        return None
    cajas = personas_de_frame.get(f, [])
    if cajas:
        v = iou(det[2:6], cajas)
        j = int(np.argmax(v))
        if v[j] >= 0.3:
            con_letra = letra_de_caja.get(f, [])
            if con_letra:
                w = iou(det[2:6], [c for c, _ in con_letra])
                m = int(np.argmax(w))
                if w[m] >= 0.3 and con_letra[m][1] in REALES:
                    return con_letra[m][1]
            return "PERSONA"
    return "X"


def evaluar(nombre, datos, pistas, candidatos, piso=0.25):
    """Scores one run of the chain by PERSON, against the hand labels of the flight.

    'candidatos' is a file that replay_vuelo3.py --candidatos wrote. It is NOT recomputed here,
    so a file left over from an older version of the chain would be scored as though it were
    today's. tests/test_personas_encontradas.py is the gate that regenerates it from the live
    code and asserts these numbers; this entry point scores whatever it is pointed at, and says
    which file that was.
    """
    dets = np.load(os.path.join(datos, "examen_v3_datos.npz"))["dets"]
    dets = dets[dets[:, 1] >= piso]
    track = np.load(pistas)["track"]
    if not os.path.exists(candidatos):
        raise SystemExit(
            "falta %s, que lo escribe el replay. Regeneralo con:\n"
            "    UAV_VISION_DATOS=demo/data python scripts/replay_vuelo3.py "
            "--pistas=demo/data/pistas_bot_cmc_sof.npz --candidatos=%s\n"
            "Eso necesita gradys_embedded:  pip install -r requirements.txt"
            % (candidatos, candidatos))
    cands = json.load(open(candidatos))
    idx_de_pista = defaultdict(list)
    for i, t in enumerate(track):
        if t >= 0:
            idx_de_pista[int(t)].append(i)

    print("\n===== %s =====" % nombre)
    print("%-28s %6s %7s %7s  %s" % ("candidato", "n_obs", "maduro", "juzgab.", "de quien son sus cajas"))
    encontradas, fantasmas, sin_juzgar = defaultdict(list), [], []
    for k, c in enumerate(cands):
        idx = [i for t in c.get("tracks", []) for i in idx_de_pista.get(int(t), [])]
        letras = [letra_de(dets[i]) for i in idx]
        juzgables = [l for l in letras if l is not None]
        cuenta = Counter(l for l in juzgables if l in REALES)
        gente = sum(1 for l in juzgables if l in REALES or l == "PERSONA")
        etq = "#%d (%.1f, %.1f)" % (k, c.get("x", 0), c.get("y", 0))
        maduro = "si" if c.get("mature") else "no"
        if not juzgables:
            sin_juzgar.append(etq)
            quien = "(fuera de los frames etiquetados)"
        elif gente >= 0.3 * len(juzgables):
            l = cuenta.most_common(1)[0][0] if cuenta else "PERSONA sin letra"
            encontradas[l].append(etq)
            quien = "%s (%d de %d cajas juzgables son de una persona)" % (l, gente, len(juzgables))
        else:
            fantasmas.append(etq)
            quien = "NADA: %s" % dict(Counter(juzgables).most_common(3))
        print("%-28s %6s %7s %7d  %s" % (etq, c.get("n_obs", "?"), maduro, len(juzgables), quien))

    print("\n  personas reales reportadas: %d de %d  ->  %s"
          % (len(encontradas), len(REALES), " ".join("%s(%d)" % (l, len(v)) for l, v in sorted(encontradas.items()))))
    print("  no reportadas: %s" % (" ".join(sorted(REALES - set(encontradas))) or "ninguna"))
    print("  FANTASMAS (candidatos que no son nadie): %d" % len(fantasmas))
    print("  sin juzgar (fuera de los frames etiquetados): %d" % len(sin_juzgar))
    return encontradas, fantasmas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--candidatos", default=os.path.join(DATOS, "candidatos_vuelo_02ago.json"),
                    help="JSON written by replay_vuelo3.py --candidatos")
    ap.add_argument("--pistas", default=os.path.join(DATOS, "pistas_bot_cmc_sof.npz"),
                    help="the same track file the replay was fed")
    ap.add_argument("--datos", default=DATOS, help="directory holding examen_v3_datos.npz")
    ap.add_argument("--nombre", default="DETECTOR QUE VUELA")
    ap.add_argument("--piso", type=float, default=0.25,
                    help="la confianza a la que se calcularon las pistas")
    a = ap.parse_args()
    evaluar(a.nombre, a.datos, a.pistas, a.candidatos, a.piso)


if __name__ == "__main__":
    main()
