"""What a second class costs on the Raspberry Pi 5, measured on the flight's own frames.

Asking the detector for cars as well as people adds no inference: the model scores every class on
every frame either way, and the class list only filters its output. What it can add is everything
downstream -- an OSNet embedding for every extra box, more work for the tracker. So this runs the
flight camera end to end, ``OnboardCamera.detect`` with the flight model, OSNet and BoT-SORT, on
recorded frames where cars and people share the scene: once looking for people, once for people
and cars. Per frame it records latency, boxes and embeddings; once a second, CPU, temperature, CPU
clock and throttling flags.

The camera is the real class with its sensor swapped for a player of recorded frames, so the code
timed is the code that flies. Reading a JPEG from disk is not part of a flight frame, so the time
spent inside the player is measured and subtracted. Runs alternate between the two conditions, so
a board that warms up over the session penalises both alike rather than whichever ran last.

On the Pi, from the folder the frames and the uav_vision package were copied to:
    python3 medir_multiclase_pi.py --frames frames --repeticiones 2
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import cv2
import numpy as np

from uav_vision.camera import OnboardCamera

MODELO = os.path.expanduser("~/modelos_visdrone/y960_ncnn_model")
REID = os.path.expanduser("~/modelos_visdrone/osnet_x0_25_msmt17.pt")
CALENTAMIENTO = 10
CONDICIONES = {"personas": None, "personas+coches": ["person", "car"]}


class Reproductor:
    """Stands in for Picamera2: hands out recorded frames in RGB, and times itself."""

    def __init__(self, rutas):
        self.rutas = rutas
        self.k = 0
        self.ultimo_s = 0.0

    def capture_array(self):
        t = time.perf_counter()
        img = cv2.cvtColor(cv2.imread(self.rutas[self.k % len(self.rutas)]), cv2.COLOR_BGR2RGB)
        self.k += 1
        self.ultimo_s = time.perf_counter() - t
        return img

    def stop(self):
        pass

    def close(self):
        pass


class CamaraGrabada(OnboardCamera):
    """The flight camera, loading the same models, reading frames from disk instead of a sensor."""

    rutas = []

    def _power_on(self):
        if self._picam is not None:
            return
        from ultralytics import YOLO
        self._picam = Reproductor(self.rutas)
        self._yolo = YOLO(self.model, task="detect")
        if self.reid_model is not None:
            from boxmot.reid.core.reid import ReID
            self._reid = ReID(self.reid_model, device="cpu", half=False)
        if self.rastreador_habilitado:
            self._build_tracker()


def vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True,
                              timeout=3).stdout.strip()
    except Exception:
        return ""


def telemetria(ruta, parar, etiqueta):
    """Samples the board once a second until told to stop."""
    import psutil
    psutil.cpu_percent(percpu=True)
    with open(ruta, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "condicion", "temp_c", "cpu_total", "cpu_max_nucleo", "clock_mhz",
                    "throttled"])
        while not parar.wait(1.0):
            nucleos = psutil.cpu_percent(percpu=True)
            temp = vcgencmd("measure_temp").replace("temp=", "").replace("'C", "")
            clock = vcgencmd("measure_clock", "arm").split("=")[-1]
            w.writerow([round(time.time(), 1), etiqueta[0], temp,
                        round(sum(nucleos) / len(nucleos), 1), max(nucleos),
                        round(int(clock) / 1e6) if clock.isdigit() else "",
                        vcgencmd("get_throttled").split("=")[-1]])
            fh.flush()


def correr(condicion, clases, rutas, escritor, rep):
    CamaraGrabada.rutas = rutas
    cam = CamaraGrabada(model=MODELO, threshold=0.25, tracker=True, reid_model=REID, fps=3.0)
    cam._power_on()
    cam.set_classes(clases)
    lat, cajas, embs = [], [], []
    for k in range(len(rutas) + CALENTAMIENTO):
        t = time.perf_counter()
        dets = cam.detect((0.0, 0.0, 30.0), 0.0)
        ms = (time.perf_counter() - t - cam._picam.ultimo_s) * 1000.0
        if k < CALENTAMIENTO:
            continue
        n_emb = sum(1 for d in dets if d.get("emb") is not None)
        n_car = sum(1 for d in dets if d.get("cls") == "car")
        lat.append(ms)
        cajas.append(len(dets))
        embs.append(n_emb)
        escritor.writerow([condicion, rep, k - CALENTAMIENTO, round(ms, 1), len(dets), n_emb,
                           len(dets) - n_car, n_car])
    return {"condicion": condicion, "repeticion": rep, "clases": sorted(cam.classes),
            "frames": len(lat), "lat_mediana_ms": round(float(np.median(lat)), 1),
            "lat_p90_ms": round(float(np.percentile(lat, 90)), 1),
            "fps_capacidad": round(1000.0 / float(np.median(lat)), 2),
            "cajas_media": round(float(np.mean(cajas)), 2),
            "embeddings_media": round(float(np.mean(embs)), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=os.path.join(AQUI, "frames"))
    ap.add_argument("--repeticiones", type=int, default=2)
    args = ap.parse_args()

    rutas = sorted(os.path.join(args.frames, f) for f in os.listdir(args.frames)
                   if f.endswith(".jpg"))
    salida = os.path.join(AQUI, "resultados_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(salida)
    print("%d frames | salida en %s" % (len(rutas), salida), flush=True)

    etiqueta = ["reposo"]
    parar = threading.Event()
    hilo = threading.Thread(target=telemetria,
                            args=(os.path.join(salida, "telemetria.csv"), parar, etiqueta),
                            daemon=True)
    hilo.start()
    resumen = []
    with open(os.path.join(salida, "frames.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condicion", "repeticion", "frame", "lat_ms", "cajas", "embeddings",
                    "personas", "coches"])
        for rep in range(1, args.repeticiones + 1):
            for condicion, clases in CONDICIONES.items():
                etiqueta[0] = "%s#%d" % (condicion, rep)
                print("corriendo %s ..." % etiqueta[0], flush=True)
                r = correr(condicion, clases, rutas, w, rep)
                fh.flush()
                resumen.append(r)
                print("  %s" % json.dumps(r), flush=True)
    parar.set()
    hilo.join()

    temps = []
    for fila in csv.DictReader(open(os.path.join(salida, "telemetria.csv"))):
        try:
            temps.append((fila["condicion"], float(fila["temp_c"]), fila["throttled"]))
        except ValueError:
            pass
    json.dump({"resumen": resumen,
               "temp_max_c": max(t for _, t, _ in temps) if temps else None,
               "throttled_visto": sorted({th for _, _, th in temps})},
              open(os.path.join(salida, "resumen.json"), "w"), indent=1)
    print("\n%-16s %4s %12s %10s %8s %8s %6s" % ("condicion", "rep", "lat mediana", "lat p90",
                                                 "FPS", "cajas", "embs"))
    for r in resumen:
        print("%-16s %4d %9.1f ms %7.1f ms %8.2f %8.2f %6.2f" % (
            r["condicion"], r["repeticion"], r["lat_mediana_ms"], r["lat_p90_ms"],
            r["fps_capacidad"], r["cajas_media"], r["embeddings_media"]))
    if temps:
        print("temperatura maxima %.1f C | get_throttled visto: %s"
              % (max(t for _, t, _ in temps), sorted({th for _, _, th in temps})))


if __name__ == "__main__":
    main()
