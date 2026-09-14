"""
What the chain does with a target that moves: flight 3 replayed with a synthetic patrolling target.

The replay's --sintetico=V puts a target with known ground truth into the real flight, moving
east-west at V m/s, detected only when in frame and only as often as a real target in view was.
For each speed, from a standing person to a boat:

    appears     first report carrying a POI near where the target is at that moment
    confirmed   first report in which that POI is mature
    mobile      whether the POI is classified as moving
    error       distance from the reported POI to where the target is at the report time
    covered     share of confirmed reports whose 95 % radius actually contains the target: a margin
                that holds the truth less often than 95 % misleads whoever is sent to look

Only POIs of class 'boat' are scored: the target is the only thing named that, so the flight's own
people near its path cannot be mistaken for it.

    python scripts/medir_blanco_movil.py
"""
import contextlib
import io
import math
import os
import runpy
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(RAIZ, "scripts", "replay_vuelo3.py")
CERCA_M = 12.0


def correr(v):
    os.environ["UAV_VISION_DATOS"] = os.path.join(RAIZ, "demo", "data")
    os.environ.pop("UAV_VISION_GS", None)
    sys.argv = [REPLAY, "--preliminares", "--sintetico=%g" % v]
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        g = runpy.run_path(REPLAY, run_name="__main__")
    linea = next((l for l in salida.getvalue().splitlines() if l.startswith("blanco sintetico")), "")
    return g, linea


def main():
    print("Vuelo 3 con un blanco sintetico que patrulla de x=-25 a x=+25 m sobre y=25 m\n")
    print("%-9s | %-42s | %9s | %10s | %5s | %s"
          % ("velocidad", "en cuadro / detectado", "aparece", "confirmado", "movil", "error (mediana / p90) | verdad dentro del radio"))
    for v in (0.0, 1.5, 4.0, 8.0):
        g, linea = correr(v)
        reps, verdad = g["reportes"], g["posicion_sintetica"]
        t_base = float(g["poses"][g["frames_aire"][0]]["t_mono"])
        aparece = confirmado = None
        errores, moviles, dentro = [], [], []
        for r in reps:
            T = float(r["time"])
            vx, vy = verdad(t_base + T)
            barco = [p for p in r.get("pois", []) if "x" in p and p.get("cls") == "boat"]
            if not barco:
                continue
            p = min(barco, key=lambda p: math.hypot(p["x"] - vx, p["y"] - vy))
            if aparece is None:
                aparece = T
            if p.get("mature"):
                if confirmado is None:
                    confirmado = T
                errores.append(math.hypot(p["x"] - vx, p["y"] - vy))
                if p.get("radius_m") is not None:
                    dentro.append(errores[-1] <= p["radius_m"])
                moviles.append(bool(p.get("mobile")))
        detalle = linea.replace("blanco sintetico a ", "").split(": ", 1)[-1]
        err = ("%.2f / %.2f m" % (np.median(errores), np.percentile(errores, 90))) if errores else "sin confirmar"
        if dentro:
            err += " | %d %%" % round(100 * np.mean(dentro))
        print("%-9s | %-42s | %9s | %10s | %5s | %s"
              % ("%.1f m/s" % v, detalle, "%.0f s" % aparece if aparece is not None else "nunca",
                 "%.0f s" % confirmado if confirmado is not None else "nunca",
                 ("%d%%" % round(100 * np.mean(moviles))) if moviles else "-", err))


if __name__ == "__main__":
    main()
