"""Renders a clip of the real flight with the detector's boxes drawn on it.

The ground station shows one frame at a time, sampled at the rate the protocol runs. This is
the other half of the story: every frame, at the speed it was flown, so an audience can see
the detector hold a target across the gaps instead of being told that it does.

    python scripts/render_clip.py --desde 3000 --hasta 3700

Writes docs/clip_vuelo.mp4. Needs the recording in the sibling repo; nothing here ships it.
"""
import argparse
import os
import sys

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
VUELO = os.path.join(RAIZ, "..", "drone-geolocation", "data", "flight_02ago", "20260802_133309")
FRAMES = os.path.join(VUELO, "frames")
PERSONAS = os.path.join(RAIZ, "demo", "data", "examen_v3_datos.npz")
VEHICULOS = os.path.join(RAIZ, "demo", "data", "vehiculos.npz")

# BGR. The same pair the ground station uses for the two classes, so a viewer who sees both
# does not have to learn two colour schemes.
COLOR = {"person": (201, 95, 128), "car": (62, 155, 190), "van": (62, 155, 190),
         "truck": (62, 155, 190), "bus": (62, 155, 190)}


def cajas_por_frame(conf_min):
    """box lists keyed by frame number, from the cached detections of both classes."""
    out = {}
    d = np.load(PERSONAS)["dets"]
    for fila in d:
        if fila[1] < conf_min:
            continue
        out.setdefault(int(fila[0]), []).append((fila[2:6], float(fila[1]), "person"))
    v = np.load(VEHICULOS, allow_pickle=True)
    for fila, cls in zip(v["dets"], v["clases"]):
        if fila[1] < conf_min:
            continue
        out.setdefault(int(fila[0]), []).append((fila[2:6], float(fila[1]), str(cls)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=3000)
    ap.add_argument("--hasta", type=int, default=3700)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "clip_vuelo.mp4"))
    args = ap.parse_args()

    if not os.path.isdir(FRAMES):
        sys.exit("no estan los frames del vuelo: %s" % FRAMES)

    cajas = cajas_por_frame(args.conf)
    escritor, alto = None, None
    puestos = 0

    for n in range(args.desde, args.hasta + 1):
        ruta = os.path.join(FRAMES, "frame_%04d.jpg" % n)
        img = cv2.imread(ruta)
        if img is None:
            continue
        h, w = img.shape[:2]
        escala = args.ancho / float(w)
        img = cv2.resize(img, (args.ancho, int(h * escala)))
        if escritor is None:
            alto = img.shape[0]
            escritor = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"mp4v"),
                                       args.fps, (args.ancho, alto))
        for (x1, y1, x2, y2), conf, cls in cajas.get(n, []):
            c = COLOR.get(cls, (200, 200, 200))
            p1 = (int(x1 * escala), int(y1 * escala))
            p2 = (int(x2 * escala), int(y2 * escala))
            cv2.rectangle(img, p1, p2, c, 2)
            cv2.putText(img, "%s %.2f" % (cls, conf), (p1[0], max(14, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            puestos += 1
        cv2.putText(img, "frame %d" % n, (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        escritor.write(img)

    if escritor is None:
        sys.exit("no se leyo ningun frame en ese rango")
    escritor.release()
    print("%s  (%d cajas dibujadas, %d a %d)" % (args.salida, puestos, args.desde, args.hasta))


if __name__ == "__main__":
    main()
