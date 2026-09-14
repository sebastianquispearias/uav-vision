"""Identity scores of the chain on the 02-ago flight, against the hand labels.

Four levels, all over the same boxes and the same labels, so the differences between them are the
contribution of each stage and nothing else:

    stand-in tracker     the replay's nearest-centre continuity, track ids as they come
    BoT-SORT             the flight's tracker (scripts/botsort_pistas.py), track ids as they come
    identity / stand-in  the identity layer's candidates, fed by the stand-in's tracks
    identity / BoT-SORT  the identity layer's candidates, fed by BoT-SORT's tracks -- the chain
                         as it flies

A box gets the id of its track, or of the candidate its track was merged into. A track that never
earned a place in a candidate leaves its boxes without an id, which the metric counts as missed.
Only labelled boxes are scored; the scores are conditioned on detections (see
uav_vision/identity_metrics.py) and say so in their output.

    python scripts/etiquetar_identidad.py     # first: the labels
    python scripts/medir_identidad.py
    python scripts/medir_identidad.py --desde 3000 --hasta 3700   # only boxes in a reviewed frame window

The last two levels can be pointed at another tracker output and another evidence floor, so a
variant of the chain is scored exactly like the one that flies:

    python scripts/medir_identidad.py --desde 3000 --hasta 3700         --pistas ../drone-geolocation/entrenamiento/botsort_pistas_02ago_calibrado.npz         --evidencia-min 0.40 --salida metricas_a1.json
"""
import argparse
import contextlib
import io
import json
import os
import re
import runpy
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAC = os.path.dirname(RAIZ)
sys.path.insert(0, RAIZ)

from uav_vision.identity_metrics import identity_scores

ENTRENAMIENTO = os.path.join(LAC, "drone-geolocation", "entrenamiento")
ETIQUETAS = os.path.join(ENTRENAMIENTO, "identidad_gt_02ago.json")
BOTSORT = os.path.join(ENTRENAMIENTO, "botsort_pistas_02ago.npz")
SALIDA = os.path.join(ENTRENAMIENTO, "metricas_identidad_02ago.json")
REPLAY = os.path.join(RAIZ, "scripts", "replay_vuelo3.py")


def correr_replay(*flags):
    """Runs the replay in-process and returns its globals and its printed output."""
    os.environ.setdefault("UAV_VISION_DATOS", os.path.join(RAIZ, "demo", "data"))
    os.environ.pop("UAV_VISION_GS", None)
    sys.argv = [REPLAY, *flags]
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        g = runpy.run_path(REPLAY, run_name="__main__")
    return g, salida.getvalue()


def por_candidato(g, track_de):
    """Candidate index per box, through the track each box belongs to."""
    cands = g["protocol"].identity.candidates(preliminary=True, with_tracks=True)
    de_pista = {tid: k for k, c in enumerate(cands) for tid in c["tracks"]}
    return [de_pista.get(int(t)) if t >= 0 else None for t in track_de], len(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desde", type=int, default=0)
    ap.add_argument("--hasta", type=int, default=10 ** 9)
    ap.add_argument("--pistas", default=BOTSORT,
                    help="tracker output for the BoT-SORT levels (default: the flight's)")
    ap.add_argument("--evidencia-min", type=float, default=None,
                    help="only boxes at or above this confidence reach the identity layer")
    ap.add_argument("--refuerzo", action="store_true",
                    help="short tracks may reinforce an existing candidate (never open one)")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()
    if not os.path.exists(ETIQUETAS):
        sys.exit("faltan las etiquetas: corre antes scripts/etiquetar_identidad.py")
    etiquetas = json.load(open(ETIQUETAS, encoding="utf-8"))["etiquetas"]

    g_sus, _ = correr_replay()
    flags_bot = ["--pistas=" + args.pistas]
    if args.evidencia_min is not None:
        flags_bot.append("--evidencia-min=%g" % args.evidencia_min)
    if args.refuerzo:
        flags_bot.append("--refuerzo")
    g_bot, texto_bot = correr_replay(*flags_bot)
    dets = g_sus["dets"]
    verdad = [etiquetas.get(str(i)) for i in range(len(dets))]
    orden = [float(d[0]) for d in dets]
    # Scoring a reviewed window only, while labels elsewhere are still in progress. The trackers and the
    # identity layer still run over the whole flight; only which boxes are scored changes.
    verdad = [v if args.desde <= dets[i][0] <= args.hasta else None for i, v in enumerate(verdad)]

    pistas_sus = g_sus["track_de"]
    pistas_bot = np.load(args.pistas)["track"]
    cand_sus, n_sus = por_candidato(g_sus, pistas_sus)
    cand_bot, n_bot = por_candidato(g_bot, pistas_bot)
    niveles = {
        "sustituto del replay": [int(t) for t in pistas_sus],
        "BoT-SORT": [int(t) if t >= 0 else None for t in pistas_bot],
        "identidad sobre sustituto": cand_sus,
        "identidad sobre BoT-SORT": cand_bot,
    }

    etiquetadas = sum(1 for v in verdad if v is not None)
    print("vuelo 02ago: %d cajas, %d etiquetadas en frames %d-%d | metricas CONDICIONADAS A LAS DETECCIONES"
          % (len(dets), etiquetadas, args.desde, min(args.hasta, int(dets[:, 0].max()))))
    print("%-27s %6s %6s %6s %5s %7s %8s %6s" % ("nivel", "IDF1", "IDP", "IDR", "IDsw",
                                                  "con id", "ids", "reales"))
    resultado = {}
    for nombre, pred in niveles.items():
        r = identity_scores(verdad, pred, orden)
        resultado[nombre] = r
        print("%-27s %6.3f %6.3f %6.3f %5d %7d %8d %6d" % (
            nombre, r["idf1"], r["idp"], r["idr"], r["id_switches"], r["boxes_with_id"],
            r["predicted_ids"], r["identities"]))
    geo = re.search(r"mejor POI respecto al operador: ([0-9.]+) m", texto_bot)
    print("pistas: %s | evidencia-min: %s | refuerzo: %s"
          % (os.path.basename(args.pistas), args.evidencia_min, args.refuerzo))
    print("candidatos: %d sobre sustituto, %d sobre BoT-SORT | geolocalizacion con BoT-SORT: %s m"
          % (n_sus, n_bot, geo.group(1) if geo else "sin POI"))
    json.dump({"condicionado_a_detecciones": True, "cajas": len(dets), "etiquetadas": etiquetadas,
               "pistas": os.path.basename(args.pistas), "evidencia_min": args.evidencia_min,
               "refuerzo": args.refuerzo,
               "niveles": resultado, "geolocalizacion_botsort_m": float(geo.group(1)) if geo else None},
              open(args.salida, "w", encoding="utf-8"), indent=1)
    print("guardado", args.salida)


if __name__ == "__main__":
    main()
