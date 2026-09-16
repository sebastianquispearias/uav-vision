"""
Turns the reviewed labels of the flights into a YOLO dataset, split by day.

Only reviewed frames are exported: a frame nobody confirmed cannot be trusted to be complete, and an
incomplete frame teaches the detector that the people it left out are background, which is the very
error the training is meant to remove.

What each label becomes:

    persona    -> a box of class 0
    duplicado  -> nothing: it is the same person already boxed, not a second one
    ignorar    -> the pixels are painted over, so the detector is neither rewarded nor punished there
    no         -> nothing: it stays as background, which is what it is
    a frame with no person at all -> an empty .txt, a negative the detector needs as much as a positive

The split is by DAY, never by frame. Two frames a fifth of a second apart are nearly the same picture
(grey 96x54 mean absolute difference: 14 levels one frame apart on flight 3, 48 between two days), so a
random split would put a picture in train and its twin in validation and report a score that memory
earned. The manifest written next to the dataset records which flight went where, so the split is a fact
on disk and not a decision remembered wrong later.

    python scripts/exportar_yolo.py --salida ../drone-geolocation/entrenamiento/dataset_personas
"""
import argparse
import csv
import json
import os
import shutil

VUELOS = {
    "26jul": ("20260726_195524", os.path.join("data", "20260726_195524", "frames")),
    "01ago_2a": ("20260801_184259", os.path.join("data", "flight_01ago", "20260801_184259", "frames")),
    "01ago_2b": ("20260801_185326", os.path.join("data", "flight_01ago", "20260801_185326", "frames")),
    "02ago": ("20260802_133309", os.path.join("data", "flight_02ago", "20260802_133309", "frames")),
    "02ago_alto": ("20260802_133309", os.path.join("data", "flight_02ago", "20260802_133309", "frames")),
}
# Split by day: the test is the only flight with the drone high, which is where the detector fails most.
# 02ago_alto are frames 9315-9865 of the test flight, ten minutes after the test windows and 25 m up:
# the only material at the height where the detector fails. Training on it makes the test score optimistic,
# because it shares the day, the place and the people with the test; that has to be said with every number.
REPARTO = {"26jul": "train", "02ago_alto": "train", "01ago_2a": "val", "01ago_2b": "val", "02ago": "test"}
_RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "drone-geolocation"))


def etiqueta_nueva(c):
    return c[4] if len(c) > 4 else "persona"


def cajas_del_frame(filas, g, rev, f, por_frame):
    """The boxes of frame f as the review left them: the people, and the ones to paint over."""
    final = lambda i: rev["correcciones"].get(str(i), g.get(str(i)))
    ajustes = rev.get("ajustes", {})
    personas, ignorar = [], []
    for i in por_frame.get(f, []):
        caja = ajustes.get(str(i)) or [float(filas[i][c]) for c in ("x1", "y1", "x2", "y2")]
        v = final(i)
        if v == "persona":
            personas.append([float(x) for x in caja])
        elif v == "ignorar":
            ignorar.append([float(x) for x in caja])
    for c in rev.get("nuevas", {}).get(str(f), []):
        (ignorar if etiqueta_nueva(c) == "ignorar" else personas).append([float(x) for x in c[:4]])
    return personas, ignorar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrenamiento", default=os.path.join(_RAIZ, "entrenamiento"))
    ap.add_argument("--datos", default=_RAIZ)
    ap.add_argument("--salida", default=os.path.join(_RAIZ, "entrenamiento", "dataset_personas"))
    ap.add_argument("--solo", nargs="*", help="exportar solo estos vuelos")
    args = ap.parse_args()

    import cv2
    resumen, manifiesto = {}, {"reparto": {}, "regla": "por dia, nunca por frame", "vuelos": {}}
    for parte in ("train", "val", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.salida, sub, parte), exist_ok=True)

    for vuelo, (_, rel) in VUELOS.items():
        if args.solo and vuelo not in args.solo:
            continue
        cajas_csv = os.path.join(args.entrenamiento, "candidatas_%s.csv" % vuelo)
        etiquetas = os.path.join(args.entrenamiento, "etiquetas_detector_%s.json" % vuelo)
        revision = os.path.join(args.entrenamiento, "etiquetas_detector_%s_frames.json" % vuelo)
        if not all(os.path.exists(p) for p in (cajas_csv, etiquetas, revision)):
            print("%-10s sin etiquetas: se saltea" % vuelo)
            continue
        filas = list(csv.DictReader(open(cajas_csv, encoding="utf-8")))
        g = json.load(open(etiquetas, encoding="utf-8"))["etiquetas"]
        rev = json.load(open(revision, encoding="utf-8"))
        por_frame = {}
        for i, r in enumerate(filas):
            por_frame.setdefault(int(r["frame"]), []).append(i)
        carpeta = os.path.join(args.datos, rel)
        parte = REPARTO[vuelo]
        n_img = n_caja = n_vacios = n_tapados = 0
        for f in sorted(int(x) for x in rev["revisados"]):
            origen = os.path.join(carpeta, "frame_%04d.jpg" % f)
            if not os.path.exists(origen):
                continue
            personas, ignorar = cajas_del_frame(filas, g, rev, f, por_frame)
            nombre = "%s_%05d" % (vuelo, f)
            destino = os.path.join(args.salida, "images", parte, nombre + ".jpg")
            if ignorar:
                img = cv2.imread(origen)
                for x1, y1, x2, y2 in ignorar:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (114, 114, 114), -1)
                cv2.imwrite(destino, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_tapados += len(ignorar)
                alto, ancho = img.shape[:2]
            else:
                shutil.copyfile(origen, destino)
                alto, ancho = cv2.imread(origen).shape[:2]
            with open(os.path.join(args.salida, "labels", parte, nombre + ".txt"), "w") as fh:
                for x1, y1, x2, y2 in personas:
                    cx, cy = (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto
                    w, h = (x2 - x1) / ancho, (y2 - y1) / alto
                    fh.write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, w, h))
            n_img += 1
            n_caja += len(personas)
            n_vacios += not personas
        resumen[vuelo] = (parte, n_img, n_caja, n_vacios, n_tapados)
        manifiesto["vuelos"][vuelo] = {"parte": parte, "imagenes": n_img, "cajas": n_caja,
                                       "sin_personas": n_vacios, "tapadas": n_tapados, "frames": rel}
        manifiesto["reparto"].setdefault(parte, []).append(vuelo)
        print("%-10s -> %-5s  %4d imagenes, %5d cajas, %4d sin personas, %d zonas tapadas"
              % (vuelo, parte, n_img, n_caja, n_vacios, n_tapados))

    yaml = os.path.join(args.salida, "personas.yaml")
    with open(yaml, "w", encoding="utf-8") as fh:
        fh.write("path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: persona\n"
                 % os.path.abspath(args.salida).replace("\\", "/"))
    json.dump(manifiesto, open(os.path.join(args.salida, "manifiesto.json"), "w", encoding="utf-8"), indent=1)
    print("\n%s\nmanifiesto y personas.yaml escritos en %s" % (
        " | ".join("%s: %s" % (p, ", ".join(v)) for p, v in manifiesto["reparto"].items()), args.salida))


if __name__ == "__main__":
    main()
