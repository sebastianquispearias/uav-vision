"""
Decides by itself the boxes whose answer the proposers already agree on, and leaves the rest for a person.

Labelling every proposed box one by one is wasted effort when several detectors agree. Measured against
the 985 boxes the user labelled by hand on 26jul and 01ago:

    coco+rfdetr+vuelo, conf >= 0.7   204 boxes -> 203 were a person   (100 %)
    coco+rfdetr,       conf >= 0.7    56 boxes ->  55 were a person   ( 98 %)
    only vuelo,        conf >= 0.3    45 boxes ->   1 was a person    (  2 %)
    only coco,         any conf       59 boxes ->   1 was a person    (  2 %)
    coco+vuelo,        conf >= 0.3    26 boxes ->   0 were a person   (  0 %)

So the rule below settles 41 % of the boxes with 1.2 % of error, and the person only sees the 59 % that
are genuinely in doubt. It never invents a box: it only labels boxes a detector already proposed, and it
writes them to the same groups file the tool reads, so every one of them can still be changed by hand.

    python scripts/preetiquetar.py --vuelo 02ago_alto

The two defaults come from the measurement that fixed them: 258 of 260 and 145 of 148.
"""
import argparse
import csv
import json
import os

_ENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                     "drone-geolocation", "entrenamiento"))


def decidir(fuentes, conf):
    """The label the proposers already agree on, or None when a person has to look."""
    f = set(fuentes.split("+"))
    if conf >= 0.7 and {"coco", "rfdetr"} <= f:
        return "persona"
    if f in ({"coco"}, {"vuelo"}, {"coco", "vuelo"}) and (conf >= 0.3 or f == {"coco"}):
        return "no"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vuelo", required=True)
    ap.add_argument("--entrenamiento", default=_ENT)
    ap.add_argument("--forzar", action="store_true", help="rehacer aunque ya existan etiquetas")
    args = ap.parse_args()
    cajas = os.path.join(args.entrenamiento, "candidatas_%s.csv" % args.vuelo)
    salida = os.path.join(args.entrenamiento, "etiquetas_detector_%s.json" % args.vuelo)
    if os.path.exists(salida) and not args.forzar:
        raise SystemExit("%s ya existe: no se pisa (usá --forzar si querés rehacerlo)" % salida)
    filas = list(csv.DictReader(open(cajas, encoding="utf-8")))
    etiquetas, cuenta = {}, {"persona": 0, "no": 0, "para vos": 0}
    for i, r in enumerate(filas):
        v = decidir(r.get("fuentes", ""), float(r["conf"]))
        if v is None:
            cuenta["para vos"] += 1
        else:
            etiquetas[str(i)] = v
            cuenta[v] += 1
    json.dump({"cajas": os.path.abspath(cajas), "clave": "fila del CSV de cajas", "etiquetas": etiquetas},
              open(salida, "w", encoding="utf-8"), indent=1)
    tot = len(filas)
    print("%d cajas: %d decididas solas (%d persona, %d no) y %d quedan para vos (%.0f %% menos trabajo)"
          % (tot, cuenta["persona"] + cuenta["no"], cuenta["persona"], cuenta["no"], cuenta["para vos"],
             100 * (cuenta["persona"] + cuenta["no"]) / max(1, tot)))
    print("escrito %s" % salida)


if __name__ == "__main__":
    main()
