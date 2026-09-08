"""
Detects vehicles on the recorded 02ago flight and caches them for the replay.

The flight archive holds the JPEGs; the demo holds only the cached detections of
people that the original run produced. This produces the equivalent cache for
vehicles, so the replay can report both classes without a drone and without
retraining anything: same weights, same flight, other classes.

Three decisions are measured, not assumed:

    Altitude floor. Below it the VisDrone weights call things cars that are not.
    Measured on this same flight: at 7.9 m a stack of traffic cones scores
    car 0.76, and the 26jul recording -- which never rose above 6.15 m -- scores
    car 0.53 on the pilot's torso at arm's length. Above 15 m every sampled
    vehicle checked by eye was a vehicle.

    Weights and input size. The ones the aircraft flew: a YOLO fine-tuned on
    VisDrone, run at 960, which is the size of the NCNN export on the Pi.

    Confidence cut. 0.25, the threshold mision_barrido.py flew with. Saving at
    the flown threshold keeps the choice of a stricter cut in the replay.

Appearance embeddings come from the same boxmot ReID wrapper OnboardCamera uses,
so a cached vector means what a live one means. Written by the training venv,
which is where boxmot is installed:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe \
        scripts/generar_vehiculos.py
"""
import csv
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
LAC = os.path.dirname(RAIZ)

VUELO = os.path.join(LAC, "drone-geolocation", "data", "flight_02ago", "20260802_133309")
PESOS = os.path.join(LAC, "drone-geolocation", "entrenamiento", "runs", "detect",
                     "runs", "y26n_visdrone_1280", "weights", "best.pt")
OSNET = os.path.join(LAC, "drone-geolocation", "entrenamiento", "venv", "Lib",
                     "site-packages", "models", "osnet_x0_25_msmt17.pt")
SALIDA = os.path.join(RAIZ, "demo", "data", "vehiculos.npz")

ALT_MIN = 15.0
IMGSZ = 960
CONF_MIN = 0.25
VEHICULOS = {"car", "van", "truck", "bus"}


def main():
    import cv2
    from boxmot.reid.core.reid import ReID
    from ultralytics import YOLO

    alturas = {}
    with open(os.path.join(VUELO, "frames.csv")) as f:
        for r in csv.DictReader(f):
            try:
                alturas[int(r["frame"])] = float(r["alt_agl"])
            except (TypeError, ValueError):
                pass

    frames = []
    for n, a in sorted(alturas.items()):
        ruta = os.path.join(VUELO, "frames", "frame_%04d.jpg" % n)
        if a > ALT_MIN and os.path.exists(ruta):
            frames.append((n, a, ruta))
    print("%d frames con alt_agl > %.0f m y JPEG en disco (de %d filas de telemetria)"
          % (len(frames), ALT_MIN, len(alturas)))
    print("altura: min %.1f m  max %.1f m"
          % (min(a for _, a, _ in frames), max(a for _, a, _ in frames)))

    yolo = YOLO(PESOS)
    reid = ReID(OSNET, device=0 if _hay_gpu() else "cpu", half=False)
    print("clases del detector:", sorted(yolo.names.values()))

    dets, embs, clases = [], [], []
    t0 = time.time()
    for i, (n, _alt, ruta) in enumerate(frames):
        r = yolo(ruta, imgsz=IMGSZ, conf=CONF_MIN, verbose=False)[0]
        cajas, filas, nombres = [], [], []
        for caja in r.boxes:
            nombre = yolo.names[int(caja.cls[0])]
            if nombre not in VEHICULOS:
                continue
            x1, y1, x2, y2 = caja.xyxy[0].tolist()
            cajas.append([x1, y1, x2, y2])
            filas.append([n, float(caja.conf[0]), x1, y1, x2, y2])
            nombres.append(nombre)
        if not cajas:
            continue
        frame = cv2.imread(ruta)
        salida = reid.process({"fallback": True,
                               "boxes": np.asarray(cajas, dtype="float32"),
                               "image": frame})
        feats = np.asarray(salida["_features"], dtype="float32")
        for fila, nombre, v in zip(filas, nombres, feats):
            norma = float(np.linalg.norm(v))
            dets.append(fila)
            embs.append(v / norma if norma > 0 else v)
            clases.append(nombre)
        if (i + 1) % 100 == 0:
            print("  %4d/%d frames  %5d detecciones  %.0f s"
                  % (i + 1, len(frames), len(dets), time.time() - t0))

    dets = np.asarray(dets, dtype="float64")
    embs = np.asarray(embs, dtype="float32")
    clases = np.asarray(clases)
    np.savez_compressed(SALIDA, dets=dets, embs=embs, clases=clases)

    print("\n%d detecciones de vehiculo en %d frames distintos, %.0f s"
          % (len(dets), len(set(dets[:, 0].astype(int))) if len(dets) else 0,
             time.time() - t0))
    unicos, cuentas = np.unique(clases, return_counts=True)
    for u, c in sorted(zip(unicos, cuentas), key=lambda p: -p[1]):
        print("  %-8s %5d" % (u, c))
    print("conf: min %.3f  mediana %.3f  max %.3f"
          % (dets[:, 1].min(), float(np.median(dets[:, 1])), dets[:, 1].max()))
    print("escrito %s  (%.1f MB)" % (SALIDA, os.path.getsize(SALIDA) / 1e6))


def _hay_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
