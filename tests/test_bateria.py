"""
Gates for the pack voltage, after it turned out the battery was never missing, only unasked.

On 2026-10-07 the drone's own telemetry service answered
{"detail": "GET_BATTERY_INFO FAIL: 'SYS_STATUS'"} while a measurement on the same link counted
30 BATTERY_STATUS packets in 15 seconds. The volts were on the wire the whole time. uav_api
reads the pack out of SYS_STATUS, which belongs to ArduPilot's EXTENDED_STATUS stream group,
and that was the one group of the six on the live channel left at rate zero -- BATTERY_STATUS
rides in EXTRA3, which ran at 2 Hz. Nothing was broken; one message was never requested.

Two things in that story are what this file guards.

The UNIT, because the fix moves a number across a boundary where it changes name. MAVLink says
millivolts, the station paints volts, and the two differ by a factor of a thousand with no type
to stop a mistake. 15013 is a plausible-looking voltage and 15.013 is a plausible-looking
voltage, so a test that only checks "a number came back" proves nothing at all.

The BOUNDARY, because the whole reason this is a separate class is that vision_protocol.py has
to import on a laptop. The replay that guards the chain -- demo.py, which must keep printing
2.39 m -- runs where there is no autopilot, no serial port and no pymavlink. A protocol that
reached for MAVLink itself would not be testable at all.

Run with: python tests/test_bateria.py

WHAT EACH SECTION PROVES
    1. Millivolts in, volts out, measured against the exact payload the real service returned.
    2. The failure the real service actually produced returns None and NOT zero. A zero would
       paint a flat pack on the station, which is the one answer worse than no answer.
    3. A source that is not there at all costs one report, not the mission.
    4. The protocol's default is unchanged: no source, no field, same POIs.
    5. An injected source that raises cannot take the report down with it.
    6. The protocol does not import MAVLink. The architectural claim, as a test.
    7. The bench stub and the real adapter agree, over real HTTP, and the voltage MOVES.
    8. The number survives the trip to the ground station.
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "tests"))
_GRADYS = os.path.join(os.path.dirname(_HERE), "gradys-embedded")
if os.path.isdir(_GRADYS):
    sys.path.insert(0, _GRADYS)

from gradys_embedded.protocol.messages.telemetry import Telemetry
from test_vision_protocol import FakeProvider

from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import UavApiBattery, VisionProtocol

GS = os.path.join(_HERE, "scripts", "banco_embedded", "gs_mapa.py")
STUB = os.path.join(_HERE, "scripts", "banco_embedded", "uav_api_stub.py")
PUERTO_GS = 8391

FALLOS = []


def revisar(condicion, descripcion, detalle=""):
    print("  [%s] %s%s" % ("ok  " if condicion else "FALLA", descripcion,
                           (" -- " + detalle) if detalle else ""))
    if not condicion:
        FALLOS.append(descripcion)


def puerto_libre():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class ServidorFalso:
    """An HTTP server that answers /telemetry/battery_info with whatever is handed to it.

    A real socket and not a monkeypatch of urllib: the adapter's job is to survive a service
    that answers badly, and a patched opener cannot return a 500 with a body.
    """

    def __init__(self, cuerpo, estado=200):
        self.cuerpo = cuerpo
        self.estado = estado
        prueba = self

        class H(server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                datos = json.dumps(prueba.cuerpo).encode()
                self.send_response(prueba.estado)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

        self.srv = server.HTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        self.hilo = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.hilo.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


DETECCION = {
    "px": ARDUCAM_MODULE_3.image_width / 2.0,
    "py": ARDUCAM_MODULE_3.image_height * 0.62,
    "conf": 0.88,
    "track_id": 1,
}


class CamaraFija:
    """One detection per frame, always the same pixel, so the POIs do not depend on the run."""

    def __init__(self):
        self.camera = ARDUCAM_MODULE_3

    def detect(self, pos, yaw):
        return [dict(DETECCION)]


def correr(battery_source, pasos=400):
    """Drives a mission on a fake clock and returns every report it broadcast."""
    provider = FakeProvider()
    Protocolo = VisionProtocol.with_config(
        camera=CamaraFija(), pitch_deg=-55.0, yaw_source=lambda: 0.0,
        see_period_s=0.25, report_period_s=2.0,
        identity=IncrementalIdentity(fusion_radius_m=3.5, fps=4.0, maturity="span"),
        report_preliminary=True, battery_source=battery_source)
    protocolo = Protocolo.instantiate(provider)
    protocolo.initialize()
    protocolo.handle_telemetry(Telemetry(current_position=(0.0, 0.0, 35.0)))
    for _ in range(pasos):
        if not provider.timers:
            break
        cuando, nombre = min(provider.timers)
        provider.timers.remove((cuando, nombre))
        provider.time = max(provider.time, cuando)
        protocolo.handle_timer(nombre)
    return [json.loads(c.message) for c in provider.sent]


print("=" * 72)
print("1. MILIVOLTIOS EN EL CABLE, VOLTIOS EN LA SALIDA")
print("=" * 72)
print("  el cuerpo es el que devolvio el servicio real en la Pi 5 el 2026-10-07")

REAL = {"device": "uav", "id": "1", "result": "success",
        "info": {"voltage": 15013, "current": 50, "battery_remaining": 75}}

with ServidorFalso(REAL) as s:
    v = UavApiBattery(s.url)()
print("  info.voltage servido      = %r  (milivoltios, como manda MAVLink)" % REAL["info"]["voltage"])
print("  UavApiBattery devuelve    = %r" % v)
print("  sin convertir habria dado = %r      <- 15 kV, una celda de alta tension" % 15013.0)
print("  convirtiendo dos veces    = %r   <- 15 mV, una pila descargada" % 0.015013)
revisar(v == 15.013, "15013 mV se leen como 15.013 V", "4S a 3.75 V por celda")
revisar(1.0 < v < 100.0,
        "el valor cae en el rango de un pack de verdad, no de uno de los dos errores",
        "15013.0 y 0.015013 quedan los dos fuera")

print()
print("  y el mismo adaptador acepta la forma plana que sirve un mock sin 'info':")
with ServidorFalso({"voltage": 14800}) as s:
    plano = UavApiBattery(s.url)()
print("  {'voltage': 14800} -> %r" % plano)
revisar(plano == 14.8, "la forma plana da el mismo voltaje", "14.8 V, nominal de un 4S")

print()
print("=" * 72)
print("2. EL FALLO REAL DEVUELVE None, Y None NO ES CERO")
print("=" * 72)

FALLO_REAL = {"detail": "GET_BATTERY_INFO FAIL: 'SYS_STATUS'"}
with ServidorFalso(FALLO_REAL, estado=500) as s:
    v = UavApiBattery(s.url)()
print("  cuerpo: %s" % json.dumps(FALLO_REAL))
print("  UavApiBattery devuelve: %r" % v)
revisar(v is None, "el cuerpo que daba la Pi 5 ayer devuelve None")
revisar(not isinstance(v, (int, float)),
        "NO devuelve un numero, y menos 0.0, que la estacion pintaria como pack muerto",
        "tipo devuelto: %s" % type(v).__name__)

with ServidorFalso({"info": {"current": 50}}) as s:
    sin_campo = UavApiBattery(s.url)()
print("  una respuesta 200 pero SIN el campo 'voltage' -> %r" % sin_campo)
revisar(sin_campo is None, "un 200 sin el campo tampoco inventa un numero")

print()
print("=" * 72)
print("3. SIN SERVICIO DETRAS, UN REPORTE Y NO LA MISION")
print("=" * 72)

muerto = "http://127.0.0.1:%d" % puerto_libre()
t0 = time.monotonic()
v = UavApiBattery(muerto, timeout_s=0.5)()
dt = time.monotonic() - t0
print("  nada escuchando en %s" % muerto)
print("  devuelve %r en %.3f s (presupuesto 0.5 s)" % (v, dt))
revisar(v is None, "un puerto vacio devuelve None en vez de levantar")
revisar(dt < 2.0, "y no bloquea el lazo de vuelo", "%.3f s" % dt)

print()
print("=" * 72)
print("4. EL DEFAULT NO CAMBIO: SIN FUENTE, NO HAY CAMPO, Y LOS POIs SON LOS MISMOS")
print("=" * 72)

sin = correr(None)
con = correr(lambda: 15.013)
print("  reportes emitidos: sin fuente %d, con fuente %d" % (len(sin), len(con)))
revisar(len(sin) > 0, "la mision emitio reportes", "%d" % len(sin))
revisar(all(m["bateria_v"] is None for m in sin),
        "sin battery_source, bateria_v es None en TODOS los reportes",
        "%d reportes" % len(sin))
revisar(all(m["bateria_v"] == 15.013 for m in con),
        "con battery_source, el voltaje viaja en TODOS los reportes",
        "%d reportes" % len(con))

pois_sin = [[(p["x"], p["y"]) for p in m["pois"]] for m in sin]
pois_con = [[(p["x"], p["y"]) for p in m["pois"]] for m in con]
print("  POIs del ultimo reporte sin fuente: %s" % (pois_sin[-1],))
print("  POIs del ultimo reporte con fuente: %s" % (pois_con[-1],))
revisar(len(pois_sin[-1]) > 0,
        "la mision llego a reportar POIs, asi que compararlos significa algo",
        "%d POIs en el ultimo reporte de %d" % (len(pois_sin[-1]), len(sin)))
revisar(pois_sin == pois_con,
        "los POIs son IDENTICOS con y sin bateria: la geolocalizacion no se toco",
        "%d reportes comparados uno a uno" % len(sin))

print()
print("=" * 72)
print("5. UNA FUENTE QUE EXPLOTA NO SE LLEVA EL REPORTE")
print("=" * 72)


def fuente_que_explota():
    raise RuntimeError("el serial se desconecto a mitad de la lectura")


roto = correr(fuente_que_explota)
print("  reportes emitidos con la fuente rota: %d" % len(roto))
print("  bateria_v en el ultimo: %r" % roto[-1]["bateria_v"])
print("  POIs en el ultimo: %s" % ([(p["x"], p["y"]) for p in roto[-1]["pois"]],))
revisar(len(roto) == len(sin),
        "se emitieron los mismos reportes que sin fuente ninguna",
        "%d contra %d" % (len(roto), len(sin)))
revisar(roto[-1]["bateria_v"] is None, "el campo cae a None")
revisar([[(p["x"], p["y"]) for p in m["pois"]] for m in roto] == pois_sin,
        "y los POIs llegan intactos: se perdio el voltaje, no el hallazgo")

print()
print("=" * 72)
print("6. EL PROTOCOLO NO IMPORTA MAVLink. LA FRONTERA, COMO PRUEBA")
print("=" * 72)

mav = sorted(m for m in sys.modules if "mavlink" in m.lower() or m.startswith("pymavlink"))
print("  modulos MAVLink cargados tras importar vision_protocol: %s" % (mav or "ninguno"))
revisar(not mav, "nada de MAVLink entro en el proceso", "%s" % (mav,))

fuente = open(os.path.join(_HERE, "uav_vision", "vision_protocol.py"), encoding="utf-8").read()
# Only real import statements. The prose of this file's own docstrings talks about MAVLink and
# serial ports on purpose, so a check that scanned every line would fire on the rule being
# written down rather than on the rule being broken. The claim is about what the module LOADS.
imports = [linea.strip() for linea in fuente.splitlines()
           if linea.strip().startswith(("import ", "from "))]
prohibidos = [linea for linea in imports
              if any(m in linea.lower() for m in ("mavlink", "serial", "pyserial"))]
print("  sentencias import en el archivo: %d" % len(imports))
print("  de ellas, de MAVLink o de puerto serie: %s" % (prohibidos or "ninguna"))
revisar(not prohibidos, "ni un import de MAVLink ni de serie en todo el archivo",
        "el mismo archivo corre en la laptop y en el dron")

print()
print("=" * 72)
print("7. EL STUB DEL BANCO Y EL ADAPTADOR SE ENTIENDEN, Y EL VOLTAJE SE MUEVE")
print("=" * 72)

puerto_stub = puerto_libre()
entorno = dict(os.environ, UAV_API_ORIGEN="-22.978,-43.232")
stub = subprocess.Popen([sys.executable, STUB, "--puerto", str(puerto_stub)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=entorno)
try:
    base = "http://127.0.0.1:%d" % puerto_stub
    fuente_stub = UavApiBattery(base, timeout_s=2.0)
    primera = None
    for _ in range(50):
        primera = fuente_stub()
        if primera is not None:
            break
        time.sleep(0.1)
    crudo = json.loads(urllib.request.urlopen(base + "/telemetry/battery_info", timeout=2).read())
    print("  el stub sirve: %s" % json.dumps(crudo["info"]))
    print("  el adaptador lee: %r V" % primera)
    revisar(primera is not None, "el stub del banco contesta el endpoint")
    revisar(primera is not None and 13.0 < primera < 17.5,
            "y el adaptador lo lee como un 4S plausible", "%r V" % primera)
    revisar(primera is not None and abs(crudo["info"]["voltage"] / 1000.0 - primera) < 0.05,
            "el milivoltio del stub y el voltio del adaptador son el mismo numero",
            "%d mV contra %r V" % (crudo["info"]["voltage"], primera))

    time.sleep(2.5)
    segunda = fuente_stub()
    print("  2.5 s mas tarde: %r V  (diferencia %+.4f V)" % (segunda, segunda - primera))
    revisar(segunda < primera,
            "el pack BAJA: el campo se actualiza, no es una constante disfrazada",
            "un valor fijo y un campo nunca escrito se ven igual")
finally:
    stub.terminate()
    stub.wait(timeout=5)

print()
print("=" * 72)
print("8. EL NUMERO LLEGA A LA ESTACION")
print("=" * 72)

estacion = subprocess.Popen(
    [sys.executable, GS, "--puerto", str(PUERTO_GS), "--origen=-22.978,-43.232"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    base = "http://127.0.0.1:%d" % PUERTO_GS
    for _ in range(80):
        try:
            urllib.request.urlopen(base + "/estado", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)

    def postear(cuerpo):
        req = urllib.request.Request(base + "/", data=json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()

    postear({"message": json.dumps({"type": "vision_poi", "bateria_v": 15.013, "pois": []}),
             "source": 11})
    postear({"message": json.dumps({"type": "vision_poi", "pois": []}), "source": 12})
    estado = json.loads(urllib.request.urlopen(base + "/estado", timeout=2).read())
    drones = estado.get("drones", estado)
    print("  dron 11 reporto bateria_v=15.013 -> la estacion guarda %r"
          % drones.get("11", {}).get("bateria_v"))
    print("  dron 12 no reporto el campo       -> la estacion guarda %r"
          % drones.get("12", {}).get("bateria_v"))
    revisar(drones.get("11", {}).get("bateria_v") == 15.013,
            "el voltaje sobrevive el viaje dron -> estacion")
    revisar(drones.get("12", {}).get("bateria_v") is None,
            "y un dron sin pack no finge tener uno", "None, no 0.0 ni ausente")
finally:
    estacion.terminate()
    estacion.wait(timeout=5)

print()
print("=" * 72)
if FALLOS:
    print("FALLARON %d:" % len(FALLOS))
    for f in FALLOS:
        print("  - %s" % f)
    sys.exit(1)
print("TODO OK")
