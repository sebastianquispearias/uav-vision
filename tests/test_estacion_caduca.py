"""A drone that goes silent stops corroborating on the station's map.

Two drones report the same person and the station shows one pin, seen by 1+2. Before this test
existed, that pin kept saying 1+2 after one of the two stopped reporting, for as long as the
station ran: measured on the two-Raspberry bench, fifteen seconds after drone 1's runner was
killed. Now a silent drone's targets leave the fusion once it has been quiet longer than the
window, and come back when it reports again. The window is 1.5 s here so the test does not wait 15.

Run: python tests/test_estacion_caduca.py
"""
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(AQUI, "..", "scripts", "banco_embedded", "gs_mapa.py")


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pedir(base, ruta, cuerpo=None):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(base + ruta, data=datos, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


v = np.random.default_rng(1).normal(size=512).astype(np.float32)
EMB = base64.b64encode((v / np.linalg.norm(v)).astype(np.float16).tobytes()).decode()


def reporte(base, dron, x, y):
    msg = {"type": "vision_poi", "sender": dron,
           "pois": [{"x": x, "y": y, "n_obs": 50, "cls": "person", "mature": True, "conf": 0.7, "emb": EMB}]}
    pedir(base, "/", {"message": json.dumps(msg), "source": dron})


def pines(base):
    return sorted((str(p.get("dron")), p["n_obs"]) for p in pedir(base, "/estado")["pois"])


puerto = puerto_libre()
base = "http://127.0.0.1:%d" % puerto
estacion = subprocess.Popen([sys.executable, GS, "--puerto", str(puerto), "--origen=-22.978,-43.232",
                             "--callado-s", "1.5"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            pedir(base, "/buscar")
            break
        except Exception:
            time.sleep(0.25)

    reporte(base, 1, 0.0, 6.0)
    reporte(base, 2, 0.4, 6.2)
    assert pines(base) == [("1+2", 100)], pines(base)
    print("  los dos hablan: un pin visto por 1+2 ->", pines(base))

    time.sleep(2.0)
    reporte(base, 2, 0.4, 6.2)
    assert pines(base) == [("2", 50)], pines(base)
    print("  el 1 se calla y el 2 sigue: el pin es solo del 2 ->", pines(base))

    time.sleep(2.0)
    assert pines(base) == [], pines(base)
    print("  se callan los dos, sin reportes nuevos: el mapa no afirma nada ->", pines(base))

    reporte(base, 1, 0.0, 6.0)
    assert pines(base) == [("1", 50)], pines(base)
    print("  el 1 vuelve: su objetivo reaparece ->", pines(base))
    print("test_estacion_caduca OK")
finally:
    estacion.terminate()
