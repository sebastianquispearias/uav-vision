"""One video: the camera, the map, and what the chain concluded, side by side.

Two halves of the same story were living in two files. One showed identity and a map at six
frames a second, which reads as a slideshow; the other ran smoothly at twenty-five but was
boxes on frames and nothing else. This is both: every frame, at flight speed, with the map
beside it and the drone moving on it.

    python scripts/render_compuesto.py --desde 3000 --hasta 3700

Writes docs/clip_compuesto.mp4. Needs the recording in the sibling repo.
"""
import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
HNO = os.path.join(RAIZ, "..", "drone-geolocation")
VUELO = os.path.join(HNO, "data", "flight_02ago", "20260802_133309")
FRAMES = os.path.join(VUELO, "frames")
SAT = os.path.join(HNO, "entrenamiento")

LAT0, LNG0, R = -22.978029946, -43.23214256266666, 6378137.0
# What the chain reported for this flight, in metres. These are its output, not ground truth.
POIS = [("person", -0.19, 6.69, (201, 95, 128)),
        ("car", -12.24, 13.06, (62, 155, 190))]
COLOR = {"person": (201, 95, 128), "car": (62, 155, 190), "van": (62, 155, 190),
         "truck": (62, 155, 190), "bus": (62, 155, 190)}
# Map window in metres, chosen to hold both targets and the flight path with room to spare.
E0, E1, N0, N1 = -22.0, 18.0, -14.0, 26.0
LADO = 620


def enu(la, ln):
    return (math.radians(ln - LNG0) * R * math.cos(math.radians(LAT0)),
            math.radians(la - LAT0) * R)


def recorte_satelite():
    img = cv2.imread(os.path.join(SAT, "satelite_zona.png"))
    if img is None:
        return None
    lat0, lon0, lat1, lon1, _z = [float(x) for x in
                                  open(os.path.join(SAT, "satelite_georef.txt")).read().split(",")]
    h, w = img.shape[:2]

    def pix(e, n):
        lat = LAT0 + n / R * 180 / math.pi
        lng = LNG0 + e / (R * math.cos(math.radians(LAT0))) * 180 / math.pi
        return ((lng - lon0) / (lon1 - lon0) * w, (lat - lat0) / (lat1 - lat0) * h)

    xa, yb = pix(E0, N0)
    xb, ya = pix(E1, N1)
    x0, x1 = int(min(xa, xb)), int(max(xa, xb))
    y0, y1 = int(min(ya, yb)), int(max(ya, yb))
    return cv2.resize(img[y0:y1, x0:x1], (LADO, LADO), interpolation=cv2.INTER_CUBIC)


def a_pix(e, n):
    return (int((e - E0) / (E1 - E0) * LADO), int((1 - (n - N0) / (N1 - N0)) * LADO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=3000)
    ap.add_argument("--hasta", type=int, default=3700)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "clip_compuesto.mp4"))
    args = ap.parse_args()

    if not os.path.isdir(FRAMES):
        sys.exit("no estan los frames del vuelo: %s" % FRAMES)

    poses = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(VUELO, "frames.csv")))}

    cajas = {}
    d = np.load(os.path.join(RAIZ, "demo", "data", "examen_v3_datos.npz"))["dets"]
    for f in d:
        if f[1] >= args.conf:
            cajas.setdefault(int(f[0]), []).append((f[2:6], float(f[1]), "person"))
    v = np.load(os.path.join(RAIZ, "demo", "data", "vehiculos.npz"), allow_pickle=True)
    for f, c in zip(v["dets"], v["clases"]):
        if f[1] >= args.conf:
            cajas.setdefault(int(f[0]), []).append((f[2:6], float(f[1]), str(c)))

    fondo = recorte_satelite()
    if fondo is None:
        fondo = np.full((LADO, LADO, 3), 22, np.uint8)

    ANCHO_CAM, ALTO_CAM = 1100, LADO
    W, H = ANCHO_CAM + LADO, ALTO_CAM + 120
    vw = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    rastro, vistos = [], 0
    F = cv2.FONT_HERSHEY_SIMPLEX

    for n in range(args.desde, args.hasta + 1):
        img = cv2.imread(os.path.join(FRAMES, "frame_%04d.jpg" % n))
        p = poses.get(n)
        if img is None or p is None:
            continue
        h0, w0 = img.shape[:2]
        esc = ANCHO_CAM / float(w0)
        cam = cv2.resize(img, (ANCHO_CAM, int(h0 * esc)))
        cam = cam[:ALTO_CAM] if cam.shape[0] >= ALTO_CAM else cv2.copyMakeBorder(
            cam, 0, ALTO_CAM - cam.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 22))

        n_cajas = 0
        for (x1, y1, x2, y2), conf, cls in cajas.get(n, []):
            c = COLOR.get(cls, (200, 200, 200))
            a = (int(x1 * esc), int(y1 * esc))
            b = (int(x2 * esc), int(y2 * esc))
            if a[1] < ALTO_CAM:
                cv2.rectangle(cam, a, b, c, 2)
                cv2.putText(cam, "%s %.2f" % (cls, conf), (a[0], max(14, a[1] - 6)),
                            F, 0.5, c, 1, cv2.LINE_AA)
                n_cajas += 1
        vistos += n_cajas

        mapa = fondo.copy()
        x, y = enu(float(p["lat"]), float(p["lng"]))
        if not rastro or abs(rastro[-1][0] - x) + abs(rastro[-1][1] - y) > 0.3:
            rastro.append((x, y))
        if len(rastro) > 1:
            cv2.polylines(mapa, [np.array([a_pix(*q) for q in rastro], np.int32)],
                          False, (255, 190, 120), 2, cv2.LINE_AA)
        for cls, ex, ny, c in POIS:
            q = a_pix(ex, ny)
            cv2.circle(mapa, q, 11, c, 2, cv2.LINE_AA)
            cv2.circle(mapa, q, 3, c, -1, cv2.LINE_AA)
            cv2.putText(mapa, cls, (q[0] + 15, q[1] + 4), F, 0.5, c, 1, cv2.LINE_AA)
        dq = a_pix(x, y)
        cv2.circle(mapa, dq, 7, (255, 190, 120), -1, cv2.LINE_AA)
        cv2.putText(mapa, "dron", (dq[0] + 11, dq[1] - 8), F, 0.5, (255, 190, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(mapa, "MAPA  (objetivos reportados y trayectoria)", (10, 20),
                    F, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

        lienzo = np.full((H, W, 3), 16, np.uint8)
        lienzo[:ALTO_CAM, :ANCHO_CAM] = cam
        lienzo[:LADO, ANCHO_CAM:] = mapa
        cv2.putText(lienzo, "CAMARA  frame %d   altura %.1f m   %d detecciones en este frame"
                    % (n, float(p["alt_agl"]), n_cajas), (12, 24), F, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        y0 = ALTO_CAM + 30
        cv2.putText(lienzo, "LA MISMA CADENA, DOS CLASES  -  sin reentrenar y sin un segundo modelo",
                    (14, y0), F, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
        for i, (cls, ex, ny, c) in enumerate(POIS):
            cv2.putText(lienzo, "%-8s reportado en  %7.2f m E, %7.2f m N" % (cls, ex, ny),
                        (14, y0 + 32 + i * 26), F, 0.55, c, 1, cv2.LINE_AA)
        vw.write(lienzo)

    vw.release()
    print("%s  (%d cajas, %dx%d)" % (args.salida, vistos, W, H))


if __name__ == "__main__":
    main()
