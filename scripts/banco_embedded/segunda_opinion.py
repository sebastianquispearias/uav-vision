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


def dibujar(imagen, personas, destino) -> None:
    """Writes a copy of the frame with a box around everyone the ground found."""
    import cv2

    for p in personas:
        x1, y1, x2, y2 = (int(v) for v in p["caja"])
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (250, 180, 80), 3)
        cv2.putText(imagen, "%.2f" % p["conf"], (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 180, 80), 1, cv2.LINE_AA)
    cv2.imwrite(destino, imagen)


def servir(entrada=None, salida=None, umbral: float = UMBRAL) -> None:
    """Answers one frame per line, keeping the detector loaded between questions.

    Measured on this laptop: loading RF-DETR takes 17.3 s, the first frame 3.6 s while CUDA warms
    up, and every frame after that 0.26 s. A process started per request would therefore take
    twenty seconds to answer a click, which is not a second opinion, it is a coffee break. Loading
    once and staying is what makes the operator's question cheap.

    The protocol is one JSON object per line in and one per line out, because the station runs on
    the plain interpreter and this has to run on the training venv, which is the only one that has
    rfdetr. A line in carries {"id", "archivo"} and optionally "dibujar"; the line out carries the
    same id with the people found, or with "error". The first line out is {"listo": true} once the
    model is in memory, so the station can say it is still loading instead of looking hung.
    """
    import cv2

    entrada = entrada or sys.stdin
    if salida is None:
        # rfdetr and torch print their own progress on standard output, which would land in the
        # middle of a protocol line and make the station read half a JSON object. The real stdout
        # is duplicated onto a private descriptor and the number 1 is pointed at stderr, so
        # anything anyone prints -- this module, the library, or C code underneath it -- goes to
        # the log and only the answers go to the station.
        salida = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    t0 = time.time()
    _cargar()
    salida.write(json.dumps({"listo": True, "segundos": round(time.time() - t0, 2)}) + "\n")
    salida.flush()
    for linea in entrada:
        linea = linea.strip()
        if not linea:
            continue
        try:
            pedido = json.loads(linea)
            imagen = cv2.imread(pedido["archivo"])
            if imagen is None:
                raise ValueError("no se pudo leer %s" % pedido.get("archivo"))
            r = mirar(imagen, umbral=pedido.get("umbral", umbral))
            if pedido.get("dibujar"):
                dibujar(imagen, r["personas"], pedido["dibujar"])
                r["dibujado"] = pedido["dibujar"]
            r["id"] = pedido.get("id")
        except Exception as e:                      # noqa: BLE001 -- a bad line must not kill the worker
            r = {"id": (pedido.get("id") if isinstance(locals().get("pedido"), dict) else None),
                 "error": str(e), "personas": [], "n": 0}
        salida.write(json.dumps(r) + "\n")
        salida.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagen", help="un jpeg en disco")
    ap.add_argument("--mensaje", help="un vision_marco guardado como json")
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--dibujar", help="escribe una copia con las cajas encima")
    ap.add_argument("--servidor", action="store_true",
                    help="queda vivo y contesta un cuadro por linea: lo que usa la estacion")
    ap.add_argument("--json", action="store_true", help="imprime el resultado como JSON")
    args = ap.parse_args()
    if args.servidor:
        return servir(umbral=args.umbral)
    import cv2
    if args.mensaje:
        r = mirar_mensaje(json.load(open(args.mensaje, encoding="utf-8")), umbral=args.umbral)
        imagen = None
    elif args.imagen:
        imagen = cv2.imread(args.imagen)
        r = mirar(imagen, umbral=args.umbral)
    else:
        ap.error("hace falta --imagen o --mensaje")
    if args.json:
        if args.dibujar and imagen is not None:
            dibujar(imagen, r["personas"], args.dibujar)
            r["dibujado"] = args.dibujar
        print(json.dumps(r))
        return
    print("%d personas en %.2f s con %d fichas" % (r["n"], r.get("segundos", 0), r.get("fichas", 0)))
    for p in r["personas"]:
        print("  conf %.2f  caja %s" % (p["conf"], p["caja"]))
    if args.dibujar and imagen is not None:
        dibujar(imagen, r["personas"], args.dibujar)
        print("escrito %s" % args.dibujar)


if __name__ == "__main__":
    sys.exit(main())
