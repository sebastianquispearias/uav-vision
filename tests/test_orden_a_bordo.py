"""A search order reaches the flying protocol by both paths, and the camera obeys it.

The map's class button only ever worked in the replay: the protocol that flies did not listen
for the order. It now takes it two ways -- as a packet on the fleet's data plane, which the
station pushes, or by polling the station -- and both end in one handler. That handler ignores
repeats and stale orders, obeys a restarted station, refuses without crashing a class the model
cannot emit, and says in the next report what the camera is really searching for.

The camera is the real OnboardCamera with a stand-in model, so the class names are checked by
the same code that runs on the drone: 'person' on a VisDrone model has to become the names that
model uses.

Run: python tests/test_orden_a_bordo.py

The station also stores what the drone says it is doing, which the page shows per drone.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http import server

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
_GRADYS = os.path.join(os.path.dirname(RAIZ), "gradys-embedded")
if os.path.isdir(_GRADYS):
    sys.path.insert(0, _GRADYS)

from uav_vision.camera import OnboardCamera
from uav_vision.vision_protocol import TIMER_REPORT, VisionProtocol

GS = os.path.join(RAIZ, "scripts", "banco_embedded", "gs_mapa.py")
VISDRONE = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
            "awning-tricycle", "bus", "motor"]


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ModeloFalso:
    names = dict(enumerate(VISDRONE))


class Proveedor:
    def __init__(self):
        self.t = 100.0
        self.enviados = []

    def current_time(self):
        return self.t

    def get_id(self):
        return 1

    def schedule_timer(self, timer, timestamp):
        pass

    def cancel_timer(self, timer):
        pass

    def send_communication_command(self, command):
        self.enviados.append(command)

    tracked_variables = {}


def dron(station_url=None):
    camara = OnboardCamera(model="stand-in")
    camara._yolo = ModeloFalso()
    llamadas = []
    original = camara.set_classes

    def contar(classes=None):
        llamadas.append(classes)
        return original(classes)

    camara.set_classes = contar
    prov = Proveedor()
    clase = VisionProtocol.with_config(camera=camara, pitch_deg=-55.0, yaw_source=lambda: 0.0,
                                       station_url=station_url)
    p = clase.instantiate(prov)
    p.initialize()
    return p, camara, prov, llamadas


def reporte(p, prov):
    p.handle_timer(TIMER_REPORT)
    return json.loads(prov.enviados[-1].message)


def pedir(base, ruta, cuerpo=None):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(base + ruta, data=datos,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def estacion(puerto, *extra):
    proc = subprocess.Popen(
        [sys.executable, GS, "--puerto", str(puerto), "--origen=-22.978,-43.232", *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = "http://127.0.0.1:%d" % puerto
    for _ in range(40):
        try:
            pedir(base, "/buscar")
            return proc, base
        except Exception:
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("la estacion no arranco")


def paquete(clases, v, epoca):
    return json.dumps({"type": "vision_buscar", "clases": clases, "v": v, "epoca": epoca})


print("=" * 70)
print("1. EL PAQUETE: el dron toma la orden del plano de datos")
print("=" * 70)
p, cam, prov, llamadas = dron()
p.handle_packet(paquete(["car"], 1, "e1"))
assert cam.classes == {"car"}, cam.classes
b = reporte(p, prov)["buscando"]
assert b["clases"] == ["car"] and b["v"] == 1 and b["rechazo"] is None, b
assert "car" in b["conocidas"] and "pedestrian" in b["conocidas"], b
print("  orden v1 car -> la camara busca", sorted(cam.classes), "| el reporte lo dice:", b["clases"])

print("=" * 70)
print("2. REPETIDA, VIEJA, Y DE UNA ESTACION REINICIADA")
print("=" * 70)
n = len(llamadas)
assert p.apply_search_order(["car"], 1, "e1") is False
assert p.apply_search_order(["truck"], 0, "e1") is False
assert len(llamadas) == n and cam.classes == {"car"}, "una orden repetida o vieja toco la camara"
print("  la misma v1 y una v0 vieja: ignoradas, la camara no se toca")
assert p.apply_search_order(["truck"], 1, "e2") is True and cam.classes == {"truck"}
print("  estacion reiniciada (epoca nueva, v1 otra vez): obedecida ->", sorted(cam.classes))

print("=" * 70)
print("3. NOMBRES: 'person' en un modelo VisDrone, y una clase que el modelo no tiene")
print("=" * 70)
p.handle_packet(paquete(["person"], 2, "e2"))
assert cam.classes == {"pedestrian", "people"}, cam.classes
print("  'person' ->", sorted(cam.classes))
p.handle_packet(paquete(["boat"], 3, "e2"))
assert cam.classes == {"pedestrian", "people"}, "un rechazo no puede cambiar lo que se busca"
b = reporte(p, prov)["buscando"]
assert b["v"] == 3 and b["rechazo"] and "boat" in b["rechazo"], b
print("  'boat' -> rechazada sin caerse; el reporte avisa:", b["rechazo"][:45], "...")
p.handle_packet(paquete(None, 4, "e2"))
assert cam.classes == OnboardCamera.CLASES_PERSONA and reporte(p, prov)["buscando"]["rechazo"] is None
print("  pedir nada -> vuelve a lo de siempre, y el rechazo anterior se borra")

print("=" * 70)
print("4. EL SONDEO: el dron pregunta a la estacion en cada reporte")
print("=" * 70)
proc, base = estacion(puerto_libre())
try:
    orden = pedir(base, "/buscar", {"clases": ["bus"]})
    p, cam, prov, _ = dron(station_url=base)
    b = reporte(p, prov)["buscando"]
    assert cam.classes == {"bus"}, cam.classes
    assert b["v"] == orden["v"] and b["epoca"] == orden["epoca"], (b, orden)
    print("  la estacion pide bus (v%d, epoca %s) -> el dron busca %s"
          % (orden["v"], orden["epoca"], sorted(cam.classes)))

    pedir(base, "/", {"message": prov.enviados[-1].message, "source": 1})
    ficha = pedir(base, "/estado")["drones"]["1"]
    assert ficha["buscando"]["clases"] == ["bus"], ficha
    print("  y la estacion guarda lo que el dron dice que busca:", ficha["buscando"]["clases"])
finally:
    proc.terminate()

p, cam, prov, _ = dron(station_url="http://127.0.0.1:%d" % puerto_libre())
t0 = time.time()
reporte(p, prov)
reporte(p, prov)
dt = time.time() - t0
assert p._proxima_consulta > prov.t, "tras un fallo no se aplazo la siguiente consulta"
print("  estacion caida: dos reportes en %.2f s, la siguiente consulta aplazada %.0f s"
      % (dt, p._proxima_consulta - prov.t))

print("=" * 70)
print("5. EL EMPUJE: la estacion manda la orden a cada dron de --nodos")
print("=" * 70)
recibidos = []


class Dron(server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        recibidos.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')


oyente = server.HTTPServer(("127.0.0.1", 0), Dron)
threading.Thread(target=oyente.serve_forever, daemon=True).start()
proc, base = estacion(puerto_libre(), "--nodos", "1=127.0.0.1:%d" % oyente.server_address[1])
try:
    orden = pedir(base, "/buscar", {"clases": ["car", "person"]})
    for _ in range(40):
        if recibidos:
            break
        time.sleep(0.1)
    assert recibidos, "la orden no llego al dron"
    ruta, cuerpo = recibidos[0]
    assert ruta == "/message" and isinstance(cuerpo["source"], int), (ruta, cuerpo)
    p, cam, prov, _ = dron()
    p.handle_packet(cuerpo["message"])
    assert cam.classes == {"car", "pedestrian", "people"}, cam.classes
    print("  POST %s con source=%d -> el dron busca %s" % (ruta, cuerpo["source"], sorted(cam.classes)))
finally:
    proc.terminate()
    oyente.shutdown()

print("test_orden_a_bordo OK")
