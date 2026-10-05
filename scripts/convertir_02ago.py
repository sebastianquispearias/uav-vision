"""
Puts the labels of flight 3 (02ago) into the format the labelling tool reads, so the test set can be
reviewed and corrected with the same pages as the training flights.

Flight 3 was labelled before the tool existed, in two separate files: identidad_gt_02ago.json, which
gives a letter to each flight detection (a person's letter, X for not a person, ? for cannot tell), and
perdidas_02ago_etiquetas.json, which says person or not for the boxes an offline detector proposed where
the flight saw nothing. That split is why the test could not be reviewed frame by frame, and reviewing it
matters more than any training flight: the test is what every later number is compared against, and an
error in it moves every result without showing itself.

The conversion keeps the boxes and their OSNet embeddings as they are -- both files are already one row
per box -- and only rewrites the labels:

    letter (A-Z, not X) -> persona        X, x, ?, AX -> no
    lost box "persona"  -> persona        lost box "no"  -> no

Nothing is marked as a duplicate here on purpose. The 104 boxes an earlier review found to be shifted or
doubled copies are left as "persona", so the tool flags them in orange and the person who labelled them
decides, instead of inheriting someone else's decision.

    python scripts/convertir_02ago.py
    python scripts/etiquetar_grupos.py --vuelo 02ago

Only the windows that were labelled by hand are converted.
"""
import argparse
import csv
import json
import os

import numpy as np

_ENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                     "drone-geolocation", "entrenamiento"))
_FRAMES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                        "drone-geolocation", "data", "flight_02ago", "20260802_133309", "frames"))
VENTANAS = [(2551, 2641), (2746, 2952), (3000, 3700)]


def en_ventana(f, ventanas):
    return any(a <= f <= b for a, b in ventanas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrenamiento", default=_ENT)
    ap.add_argument("--frames", default=_FRAMES)
    ap.add_argument("--salida", default=None, help="prefijo (por defecto <entrenamiento>/candidatas_02ago)")
    args = ap.parse_args()
    ent = args.entrenamiento
    salida = args.salida or os.path.join(ent, "candidatas_02ago")

    vuelo = list(csv.DictReader(open(os.path.join(ent, "identidad_cajas_02ago.csv"), encoding="utf-8")))
    emb_vuelo = np.load(os.path.join(ent, "identidad_embs_02ago.npy"))
    letras = json.load(open(os.path.join(ent, "identidad_gt_02ago.json"), encoding="utf-8"))["etiquetas"]
    perdidas = list(csv.DictReader(open(os.path.join(ent, "perdidas_02ago_cajas.csv"), encoding="utf-8")))
    emb_perdidas = np.load(os.path.join(ent, "perdidas_02ago_embs.npy"))
    et_perdidas = json.load(open(os.path.join(ent, "perdidas_02ago_etiquetas.json"), encoding="utf-8"))["etiquetas"]
    if len(vuelo) != len(emb_vuelo) or len(perdidas) != len(emb_perdidas):
        raise SystemExit("cajas y embeddings no coinciden: %d/%d y %d/%d"
                         % (len(vuelo), len(emb_vuelo), len(perdidas), len(emb_perdidas)))

    filas, embs, etiquetas, cuenta = [], [], {}, {"persona": 0, "no": 0, "sin etiqueta": 0}
    for fuente, cajas, emb, traduce in (
            ("vuelo", vuelo, emb_vuelo,
             lambda j: ("persona" if letras[str(j)].isalpha() and letras[str(j)].isupper()
                        and letras[str(j)] not in ("X", "AX") else "no") if str(j) in letras else None),
            ("offline", perdidas, emb_perdidas,
             lambda j: {"persona": "persona", "no": "no"}.get(et_perdidas.get(str(j))))):
        for j, r in enumerate(cajas):
            f = int(r["frame"])
            if not en_ventana(f, VENTANAS):
                continue
            v = traduce(j)
            i = len(filas)
            filas.append([f, round(float(r["conf"]), 4)] + [round(float(r[c]), 1) for c in ("x1", "y1", "x2", "y2")] + [fuente])
            embs.append(emb[j])
            if v:
                etiquetas[str(i)] = v
                cuenta[v] += 1
            else:
                cuenta["sin etiqueta"] += 1

    with open(salida + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "conf", "x1", "y1", "x2", "y2", "fuentes"])
        w.writerows(filas)
    np.save(salida + "_embs.npy", np.asarray(embs, "float32"))
    frames = [f for a, b in VENTANAS for f in range(a, b + 1)
              if os.path.exists(os.path.join(args.frames, "frame_%04d.jpg" % f))]
    with open(salida + "_frames.txt", "w", encoding="utf-8") as fh:
        fh.writelines("%d\n" % f for f in frames)
    destino = os.path.join(ent, "etiquetas_detector_02ago.json")
    if os.path.exists(destino):
        raise SystemExit("%s ya existe: no se pisa (borralo a mano si querés rehacer la conversión)" % destino)
    json.dump({"cajas": os.path.abspath(salida + ".csv"), "clave": "fila del CSV de cajas", "etiquetas": etiquetas},
              open(destino, "w", encoding="utf-8"), indent=1)
    print("%d cajas (%d del vuelo, %d offline) en %d frames -> %s.csv"
          % (len(filas), sum(1 for f in filas if f[6] == "vuelo"), sum(1 for f in filas if f[6] == "offline"),
             len(frames), salida))
    print("etiquetas heredadas: %s -> %s" % (cuenta, destino))


if __name__ == "__main__":
    main()
