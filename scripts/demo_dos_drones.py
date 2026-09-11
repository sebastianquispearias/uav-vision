"""Two drones on one map, from a single flight.

The flight crossed the same ground twice, minutes apart, and the paper measures that the GPS
bias between the two passes is independent: pass 1 and pass 2 behave like two aircraft. That
is the pseudo-swarm protocol, and it is what lets a fleet be exercised with one camera.

Each pass is replayed as its own drone, reporting to the same station. What the map then shows
is the part that cannot be seen with one drone: a target both of them saw appears once, not
twice, and one only a single drone saw is offered for a second look.

    python scripts/demo_dos_drones.py

The station stays up when the flight ends. Ctrl+C closes it.
"""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
GS = os.path.join(AQUI, "banco_embedded", "gs_mapa.py")
REPLAY = os.path.join(AQUI, "replay_vuelo3.py")
DATOS = os.path.join(RAIZ, "demo", "data")
SAT = os.path.join(RAIZ, "..", "drone-geolocation", "entrenamiento")
ORIGEN = "-22.978029946,-43.23214256266666"
PUERTO = 8300
BASE = "http://127.0.0.1:%d" % PUERTO


def espera(url, intentos=40):
    for _ in range(intentos):
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    orden = [sys.executable, GS, "--puerto", str(PUERTO), "--origen=" + ORIGEN]
    fondo = os.path.join(SAT, "satelite_zona.png")
    georef = os.path.join(SAT, "satelite_georef.txt")
    if os.path.exists(fondo) and os.path.exists(georef):
        orden += ["--fondo", fondo, "--georef", georef]

    estacion = subprocess.Popen(orden, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not espera(BASE + "/estado"):
        estacion.terminate()
        sys.exit("la estacion no arranco: mira si ya hay una en el puerto %d" % PUERTO)
    webbrowser.open(BASE + "/")

    entorno = dict(os.environ, UAV_VISION_DATOS=DATOS, UAV_VISION_GS=BASE + "/")
    print("")
    print("  DOS DRONES. Cada pasada del vuelo reporta como un dron distinto.")
    print("  En el mapa: un objetivo que vieron los dos sale UNA vez, con dron '1+2'.")
    print("  El que vio uno solo aparece como VERIFICACION SUGERIDA.")
    print("")

    vuelos = []
    for dron, pasada in ((1, 1), (2, 2)):
        vuelos.append(subprocess.Popen(
            [sys.executable, REPLAY, "--vehiculos", "--preliminares",
             "--dron=%d" % dron, "--pasada=%d" % pasada],
            env=entorno))
        # Staggered so the map fills the way a real fleet would, one drone reporting first.
        time.sleep(2)

    try:
        for v in vuelos:
            v.wait()
        print("")
        print("  el mapa sigue en %s/  (Ctrl+C para cerrar)" % BASE)
        estacion.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for v in vuelos:
            v.terminate()
        estacion.terminate()


if __name__ == "__main__":
    main()
