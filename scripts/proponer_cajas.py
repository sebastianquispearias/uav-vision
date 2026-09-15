"""
Candidate person boxes for a recorded flight, from several detectors, ready to be labelled.

A detector can only be trained on frames where every person has a box: a person left without one is
taught as background, which is the very error the training should remove. The flight's own detections
are not enough for that -- on flight 3 (02ago) the flight detector missed 660 person boxes against 835
it found, most of them on a balcony seen at a steep angle. So the candidates of a frame are the union
of several sources, merged, and a person decides which are people:

- "vuelo": the weights that flew, at the size that flew, down to a low confidence (0.10), so that what
  the flight saw but its threshold (0.25) dropped is also a candidate.
- "rfdetr": RF-DETR base on the whole frame and on four overlapping tiles. On flight 3 it took part in
  709 of the 764 boxes labelled as lost people.
- "coco": YOLO11m trained on COCO, at 1920. Few people alone (5 of 403 on flight 3), but it agrees
  with the others on 122 of the 764.

VisDrone at 1920 was a fourth source on flight 3 and is left out: alone it gave 48 people in 1268 boxes.
Sources are merged by non-maximum suppression at IoU 0.5; each merged box keeps the highest confidence
and the list of sources that agreed, in the "fuentes" column.

Frames are taken every --paso frames. Consecutive frames are nearly the same image (grey 96x54 mean
absolute difference: 14 levels one frame apart on flight 3, 33 ten apart, 48 between two different
days), so labelling each one buys little that the next one does not already have.

The output CSV has the columns etiquetar_grupos.py reads (frame, conf, x1, y1, x2, y2) plus fuentes, and
an .npy of OSNet embeddings, one row per box, the ones the grouping uses, and a _frames.txt with every
frame looked at, so the frame review also shows the frames where no detector proposed anything.

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/proponer_cajas.py \\
        --frames ../drone-geolocation/data/20260726_195524/frames --paso 10 \\
        --salida ../drone-geolocation/entrenamiento/candidatas_26jul
    -> candidatas_26jul.csv, candidatas_26jul_embs.npy
"""
import argparse
import csv
import glob
import os
import re
import time

import numpy as np

_ENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "drone-geolocation", "entrenamiento"))
PESOS_VUELO = os.path.join(_ENT, "runs", "detect", "runs", "y26n_visdrone_1280", "weights", "best.pt")
PESOS_COCO = os.path.join(_ENT, "modelos_propuesta", "yolo11m.pt")
PESOS_REID = os.path.join(_ENT, "venv", "Lib", "site-packages", "models", "osnet_x0_25_msmt17.pt")
TILES = [(0, 0, 1920, 1080)] + [(x, y, x + 1100, y + 640) for x in (0, 820) for y in (0, 440)]
LADO_MIN = 6          # px: a box thinner than this cannot be judged by a person either


def iou(b, B):
    """IoU of box b against every row of B, boxes as x1, y1, x2, y2."""
    if len(B) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)


def fusionar(cajas, umbral=0.5):
    """Non-maximum suppression across sources.

    cajas is a list of (conf, x1, y1, x2, y2, source). Returns (conf, x1, y1, x2, y2, sources) with the
    box of highest confidence of each cluster and the sorted, deduplicated sources that fell in it.
    """
    if not cajas:
        return []
    orden = sorted(cajas, key=lambda c: -c[0])
    B = np.array([c[1:5] for c in orden], dtype=float)
    usada = np.zeros(len(orden), bool)
    out = []
    for i in range(len(orden)):
        if usada[i]:
            continue
        grupo = (iou(B[i], B) >= umbral) & ~usada
        usada |= grupo
        fuentes = sorted({orden[j][5] for j in np.where(grupo)[0]})
        out.append((orden[i][0],) + tuple(orden[i][1:5]) + ("+".join(fuentes),))
    return out


def frames_de(carpeta, paso, desde=None, hasta=None, alturas=None, alt_min=None):
    """Frame numbers of frame_NNNN.jpg in a folder, every paso-th, inside [desde, hasta].

    With alt_min, only frames whose height above ground in alturas ({frame: metres}) reaches it: the
    step is taken over those, so a long stretch on the ground does not eat the sample.
    """
    nums = sorted(int(re.search(r"(\d+)\.jpg$", p).group(1)) for p in glob.glob(os.path.join(carpeta, "frame_*.jpg")))
    nums = [n for n in nums if (desde is None or n >= desde) and (hasta is None or n <= hasta)]
    if alt_min is not None:
        nums = [n for n in nums if n in alturas and alturas[n] >= alt_min]
    return nums[::paso]


def leer_alturas(carpeta):
    """alt_agl per frame from the frames.csv the flight recorder writes next to the frames folder."""
    ruta = os.path.join(os.path.dirname(os.path.normpath(carpeta)), "frames.csv")
    return {int(r["frame"]): float(r["alt_agl"]) for r in csv.DictReader(open(ruta)) if r.get("alt_agl")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="carpeta con frame_NNNN.jpg")
    ap.add_argument("--salida", required=True, help="prefijo: escribe <salida>.csv y <salida>_embs.npy")
    ap.add_argument("--paso", type=int, default=10)
    ap.add_argument("--desde", type=int)
    ap.add_argument("--hasta", type=int)
    ap.add_argument("--sin-coco", action="store_true")
    ap.add_argument("--alt-min", type=float, help="solo frames con alt_agl >= esto (frames.csv junto a la carpeta)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from rfdetr import RFDETRBase
    from boxmot.reid.core.reid import ReID

    alturas = leer_alturas(args.frames) if args.alt_min is not None else None
    frames = frames_de(args.frames, args.paso, args.desde, args.hasta, alturas, args.alt_min)
    print("%d frames (paso %d)" % (len(frames), args.paso), flush=True)
    vuelo = YOLO(PESOS_VUELO)
    coco = None if args.sin_coco else YOLO(PESOS_COCO)
    rf = RFDETRBase()
    reid = ReID(PESOS_REID, device=0, half=False)

    filas, embs, cuenta = [], [], {"vuelo": 0, "rfdetr": 0, "coco": 0}
    t0 = time.time()
    for k, f in enumerate(frames):
        img = cv2.imread(os.path.join(args.frames, "frame_%04d.jpg" % f))
        cajas = []
        # VisDrone classes 0 and 1 are pedestrian and people, the two the flight reports as a person.
        r = vuelo(img, imgsz=960, conf=0.10, classes=[0, 1], verbose=False)[0]
        cajas += [(c,) + tuple(b) + ("vuelo",) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        if coco is not None:
            r = coco(img, imgsz=1920, conf=0.15, classes=[0], verbose=False)[0]
            cajas += [(c,) + tuple(b) + ("coco",) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for x1, y1, x2, y2 in TILES:
            det = rf.predict(np.ascontiguousarray(rgb[y1:y2, x1:x2]), threshold=0.30)
            for b, cls, c in zip(det.xyxy, det.class_id, det.confidence):
                if int(cls) == 1:      # RF-DETR uses the COCO ids with background at 0: 1 is person
                    cajas.append((float(c), b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1, "rfdetr"))
        cajas = [c for c in cajas if min(c[3] - c[1], c[4] - c[2]) >= LADO_MIN]
        for c in cajas:
            cuenta[c[5]] += 1
        fusion = fusionar(cajas)
        if fusion:
            bx = np.array([c[1:5] for c in fusion], "float32")
            v = np.asarray(reid.process({"fallback": True, "boxes": bx, "image": img})["_features"], "float32")
            embs.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9))
            filas += [[f, round(float(c[0]), 4)] + [round(float(x), 1) for x in c[1:5]] + [c[5]] for c in fusion]
        if (k + 1) % 50 == 0:
            print("%d/%d frames, %d candidatas, fuentes %s, %.0f s" % (k + 1, len(frames), len(filas), cuenta, time.time() - t0), flush=True)

    with open(args.salida + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "conf", "x1", "y1", "x2", "y2", "fuentes"])
        w.writerows(filas)
    np.save(args.salida + "_embs.npy", np.concatenate(embs) if embs else np.zeros((0, 512), "float32"))
    # Every frame looked at, with or without candidates: the review shows them all.
    with open(args.salida + "_frames.txt", "w", encoding="utf-8") as fh:
        fh.writelines("%d\n" % f for f in frames)
    print("hecho: %d frames, %d candidatas, fuentes antes de fusionar %s, %.0f s -> %s.csv"
          % (len(frames), len(filas), cuenta, time.time() - t0, args.salida), flush=True)


if __name__ == "__main__":
    main()
