"""
Run the whole system on a real flight, without a drone.

    python demo/demo.py

Replays the recording of a real flight (2026-08-02) through the same protocol
that runs on the aircraft: the poses are the ones the Pixhawk logged, the
detections are the ones the detector produced in the air. Nothing here is
simulated except the clock.

It opens the ground station in a browser and fills the map with the points the
protocol reports, then prints how far the best one landed from a surveyed
ground truth.

    --sin-mapa    skip the ground station, print the numbers only
"""

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DATOS = os.path.join(AQUI, "data")
REPLAY = os.path.join(RAIZ, "scripts", "replay_vuelo3.py")
GS = os.path.join(RAIZ, "scripts", "banco_embedded", "gs_mapa.py")

PUERTO = 8300
# The surveyed reference post of that flight. The map is drawn around it.
ORIGEN = "-22.978029946,-43.23214256266666"


HOST = "127.0.0.1"   # not "localhost": IPv6-first resolution costs ~1 s per POST on Windows


def esperar(url, intentos=25):
    """Polls until the ground station answers, so the browser never opens early."""
    for _ in range(intentos):
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.4)
    return False


def main():
    con_mapa = "--sin-mapa" not in sys.argv
    entorno = dict(os.environ, UAV_VISION_DATOS=DATOS)
    estacion = None

    if con_mapa:
        print("levantando la estacion de tierra en el puerto %d..." % PUERTO)
        estacion = subprocess.Popen(
            [sys.executable, GS, "--puerto", str(PUERTO), "--origen=" + ORIGEN],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if esperar("http://%s:%d/estado" % (HOST, PUERTO)):
            entorno["UAV_VISION_GS"] = "http://%s:%d/" % (HOST, PUERTO)
            webbrowser.open("http://%s:%d/" % (HOST, PUERTO))
        else:
            print("  no arranco; sigo sin mapa")

    print("reproduciendo el vuelo del 2026-08-02...\n")
    try:
        subprocess.run([sys.executable, REPLAY], env=entorno, check=True)
    finally:
        if con_mapa and estacion is not None:
            print("\nel mapa sigue en http://%s:%d/  (Ctrl+C para cerrar)"
                  % (HOST, PUERTO))
            try:
                estacion.wait()
            except KeyboardInterrupt:
                estacion.terminate()


if __name__ == "__main__":
    main()
