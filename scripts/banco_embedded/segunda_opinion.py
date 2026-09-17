"""
The ground's second opinion on a frame the aircraft sent, using a detector that could never fly.

What flies is what fits in the power budget, and it finds fewer people than a detector that does not
have to fit: on the 02ago flight, over the window where the drone is high, the aircraft's YOLO26 found
46.2 % of the people and RF-DETR with tiles found 90.5 %, with better precision as well. RF-DETR takes
about 1.4 s per frame against 35 ms, so it will never run on the aircraft; nothing stops it running on
the laptop when an operator asks about a spot.

The frame is cut into overlapping tiles at native resolution rather than shrunk, because shrinking is
what makes a distant person disappear, and the tiles are merged back with non-maximum suppression: the
same person straddling two tiles comes back as two boxes otherwise.

Needs the training venv, which has rfdetr:
    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/banco_embedded/segunda_opinion.py --imagen f.jpg
"""
import argparse
import base64
import json
import os
import sys
import time
from typing import List, Optional, Sequence

import numpy as np

# Five tiles: the whole frame plus its four quadrants, generously overlapped so nobody falls on a seam.
# The same set proponer_cajas.py uses, so what the ground sees here is what the offline proposals saw.
FICHAS = ((0, 0, 1920, 1080), (0, 0, 1100, 640), (820, 0, 1920, 640),
          (0, 440, 1100, 1080), (820, 440, 1920, 1080))
UMBRAL = 0.30
_MODELO = None


def _cargar():
    global _MODELO
    if _MODELO is None:
        from rfdetr import RFDETRBase
        _MODELO = RFDETRBase()
    return _MODELO


def _iou(a, B):
    if not len(B):
        return np.zeros(0)
    B = np.asarray(B, float)
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)


def fusionar(cajas: Sequence, confs: Sequence, umbral: float = 0.5) -> List[dict]:
    """One box per person: a tile and its neighbour both see whoever stands on the seam."""
    if not len(cajas):
        return []
    cajas, confs = np.asarray(cajas, float), np.asarray(confs, float)
    orden, salida = np.argsort(-confs), []
    while len(orden):
        i = int(orden[0])
        salida.append({"caja": [round(float(v), 1) for v in cajas[i]], "conf": round(float(confs[i]), 3)})
        if len(orden) == 1:
            break
        orden = orden[1:][_iou(cajas[i], cajas[orden[1:]]) < umbral]
    return salida


def mirar(imagen, fichas: Sequence = FICHAS, umbral: float = UMBRAL) -> dict:
    """Every person RF-DETR finds in a frame, with how long the ground took to answer."""
    import cv2
    t0 = time.time()
    rf = _cargar()
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    alto, ancho = rgb.shape[:2]
    cajas, confs = [], []
    for x1, y1, x2, y2 in fichas:
        x1, y1 = max(0, min(x1, ancho - 1)), max(0, min(y1, alto - 1))
        x2, y2 = max(x1 + 1, min(x2, ancho)), max(y1 + 1, min(y2, alto))
        d = rf.predict(np.ascontiguousarray(rgb[y1:y2, x1:x2]), threshold=umbral)
        for b, cls, c in zip(d.xyxy, d.class_id, d.confidence):
            if int(cls) == 1:              # RF-DETR keeps the COCO ids with background at 0: 1 is person
                cajas.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
                confs.append(float(c))
    personas = fusionar(cajas, confs)
    return {"personas": personas, "n": len(personas), "fichas": len(fichas),
            "segundos": round(time.time() - t0, 2)}


def mirar_jpeg(datos: bytes, **kw) -> dict:
    """Same, from the bytes the aircraft put on the link."""
    import cv2
    imagen = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        return {"personas": [], "n": 0, "error": "no se pudo decodificar el jpeg"}
    return mirar(imagen, **kw)


def mirar_mensaje(mensaje: dict, **kw) -> dict:
    """Same, from a vision_marco packet, which carries the frame in base64."""
    if mensaje.get("type") != "vision_marco" or not mensaje.get("jpeg"):
        return {"personas": [], "n": 0, "error": "el mensaje no trae un marco"}
    r = mirar_jpeg(base64.b64decode(mensaje["jpeg"]), **kw)
    r["sender"] = mensaje.get("sender")
    r["t"] = mensaje.get("t")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagen", help="un jpeg en disco")
    ap.add_argument("--mensaje", help="un vision_marco guardado como json")
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--dibujar", help="escribe una copia con las cajas encima")
    args = ap.parse_args()
    import cv2
    if args.mensaje:
        r = mirar_mensaje(json.load(open(args.mensaje, encoding="utf-8")), umbral=args.umbral)
        imagen = None
    elif args.imagen:
        imagen = cv2.imread(args.imagen)
        r = mirar(imagen, umbral=args.umbral)
    else:
        ap.error("hace falta --imagen o --mensaje")
    print("%d personas en %.2f s con %d fichas" % (r["n"], r.get("segundos", 0), r.get("fichas", 0)))
    for p in r["personas"]:
        print("  conf %.2f  caja %s" % (p["conf"], p["caja"]))
    if args.dibujar and imagen is not None:
        for p in r["personas"]:
            x1, y1, x2, y2 = (int(v) for v in p["caja"])
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (250, 180, 80), 3)
        cv2.imwrite(args.dibujar, imagen)
        print("escrito %s" % args.dibujar)


if __name__ == "__main__":
    sys.exit(main())
