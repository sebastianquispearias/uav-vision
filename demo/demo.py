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
    --vehiculos   also replay the cached vehicle detections, so the map carries
                  two classes. The people-only run is the equivalence gate of
                  this repo and must keep printing 2.39 m, so it is opt-in.
"""

import os
import json
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
# Satellite imagery for the map background. It lives in the sibling repo that
# holds the flight data, so a clone of this one alone simply gets the metric
# grid: the pins are in the right place either way, and the station hides the
# switch when there is nothing to switch to.
_SAT = os.path.join(RAIZ, "..", "drone-geolocation", "entrenamiento")
FONDO = os.path.join(_SAT, "satelite_zona.png")
GEOREF = os.path.join(_SAT, "satelite_georef.txt")

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
    # Flags the replay understands are forwarded rather than reimplemented here.
    extra = [a for a in sys.argv[1:]
             if a in ("--vehiculos", "--preliminares", "--vivo")
             or a.startswith("--velocidad=")]
    # --vivo is the operator's demo: the flight plays against the wall clock
    # and the drone asks the map what to look for. It starts on vehicles and
    # reports unconfirmed candidates, because a class asked for mid-flight
    # arrives with half a pass behind it -- which the map shows as POR
    # VERIFICAR rather than hiding. Asking early is what earns a CONFIRMADO.
    vivo = "--vivo" in extra
    if vivo and "--preliminares" not in extra:
        extra.append("--preliminares")
    entorno = dict(os.environ, UAV_VISION_DATOS=DATOS)
    estacion = None

    if con_mapa:
        print("levantando la estacion de tierra en el puerto %d..." % PUERTO)
        orden_gs = [sys.executable, GS, "--puerto", str(PUERTO), "--origen=" + ORIGEN]
        if os.path.exists(FONDO) and os.path.exists(GEOREF):
            orden_gs += ["--fondo", FONDO, "--georef", GEOREF]
        # The flight frames, when the sibling repo with the recording is there. Bench only:
        # what the camera saw, shown beside what the map made of it.
        cuadros = os.path.join(_SAT, "..", "data", "flight_02ago", "20260802_133309", "frames")
        if vivo and os.path.isdir(cuadros):
            orden_gs += ["--frames", cuadros]
        else:
            print("  sin imagen de satelite: el mapa usa la cuadricula metrica")
        estacion = subprocess.Popen(
            orden_gs,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if esperar("http://%s:%d/estado" % (HOST, PUERTO)):
            entorno["UAV_VISION_GS"] = "http://%s:%d/" % (HOST, PUERTO)
            if vivo:
                # Seed the order so the flight starts on vehicles: the operator's
                # click has to add something, not merely confirm what was already
                # being reported.
                try:
                    urllib.request.urlopen(urllib.request.Request(
                        "http://%s:%d/buscar" % (HOST, PUERTO),
                        data=json.dumps({"clases": ["car"]}).encode(),
                        headers={"Content-Type": "application/json"}), timeout=2).read()
                except Exception as e:
                    print("  no se pudo fijar la clase inicial: %s" % e)
            webbrowser.open("http://%s:%d/" % (HOST, PUERTO))
        else:
            print("  no arranco; sigo sin mapa")

    if vivo:
        print("")
        print("  EN VIVO. El dron arranca buscando VEHICULOS.")
        print("  Pulsa 'person' en la fila BUSCANDO del mapa, en el primer")
        print("  tercio del vuelo, y mira aparecer al operador.")
    print("reproduciendo el vuelo del 2026-08-02...\n")
    try:
        subprocess.run([sys.executable, REPLAY] + extra, env=entorno, check=True)
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
