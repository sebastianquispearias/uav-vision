"""The flight-day command, and the one thing that makes it worth having: it refuses.

A script that starts things is not what a flight day needs. The question at the field is not
"did it start" but "can I take off", and on 2026-10-03 those came apart in three places:

  the code does not travel when a mission starts, so a board can hold the new file and keep
  running the old module, with a new field arriving as null as the only sign;

  the mission whose name sounds like the flight one, mision_vision, is the desk bench;

  and "started" is not "reporting": the station prints a line only for reports that carry a
  find, so a healthy aircraft that has not seen anybody yet looks exactly like a dead one.

That last one is why this gate exists. volar.sh must not print LISTO on a drone that is silent,
and it must print it when they all speak. Everything else it does is orchestration that
test_levantar_banco.py already covers.

Run: python tests/test_volar.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
TMP = tempfile.mkdtemp(prefix="volar_")
REGISTRO = os.path.join(TMP, "ssh.txt").replace("\\", "/")
PUERTO = 8399

# Cuanto hace que hablo cada dron, en segundos. La prueba lo mueve entre secciones.
EDADES = {"1": 1.0, "2": 1.0}


class Estacion(BaseHTTPRequestHandler):
    """The two routes volar.sh touches, and nothing else."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        ahora = 1000.0
        cuerpo = {"ahora": ahora,
                  "drones": {k: {"t": ahora - v} for k, v in EDADES.items()},
                  "pois": []}
        datos = json.dumps(cuerpo).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)


servidor = HTTPServer(("127.0.0.1", PUERTO), Estacion)
threading.Thread(target=servidor.serve_forever, daemon=True).start()

# Un ssh falso que no hace nada y no lee stdin: el guion le pasa el script de la placa por ahi, y
# un cat aqui se quedaria esperando para siempre en las llamadas que llevan -n.
STUB = os.path.join(TMP, "ssh_falso.sh").replace("\\", "/")
with open(STUB, "w", newline="\n") as f:
    f.write('echo "ssh $*" >> %s\n' % REGISTRO)
    f.write('echo "   setup: {\\"state\\": \\"ready\\"}"\n')

# El bash de Git y no el que gane el PATH: con WSL instalado, "bash" encuentra el de System32,
# que vive en otro sistema de archivos y no ve las rutas de Windows que esta prueba escribe.
BASH = r"C:\Program Files\Git\bin\bash.exe"
if not os.path.exists(BASH):
    BASH = "bash"


def correr(edades, extra=None):
    EDADES.clear()
    EDADES.update(edades)
    entorno = dict(os.environ)
    entorno["SSH"] = "bash " + STUB
    entorno["VOLAR_SIN_PREPARAR"] = "1"     # la provision tiene su propia prueba
    entorno["VOLAR_SIN_PING"] = "1"        # las direcciones de la prueba no existen
    entorno["VOLAR_FRESCO_S"] = "10"
    entorno.update(extra or {})
    return subprocess.run(
        [BASH, "scripts/banco_embedded/volar.sh", "127.0.0.1:%d" % PUERTO,
         "pi@10.0.0.11", "pi@10.0.0.12"],
        cwd=RAIZ, capture_output=True, text=True, env=entorno, timeout=600)


print("=" * 74)
print("1. CON LAS DOS AERONAVES HABLANDO, DICE QUE ESTA LISTO")
print("=" * 74)
r = correr({"1": 1.0, "2": 2.0})
print("  salida: %d" % r.returncode)
for l in r.stdout.splitlines():
    if "hablo hace" in l or "LISTO" in l:
        print("  %s" % l.strip()[:90])
assert r.returncode == 0, "se nego con las dos aeronaves sanas:\n%s" % r.stdout[-700:]
assert "LISTO PARA VOLAR" in r.stdout, "no dijo que estaba listo:\n%s" % r.stdout[-700:]
# Y dice que hacer, porque un comando que termina en silencio deja al operador adivinando.
assert "SEARCHING" in r.stdout, "no recuerda mandar la orden de busqueda antes de despegar"

print()
print("=" * 74)
print("2. SI UNA CALLA, SE NIEGA. ESTA ES LA RAZON DE SER DEL GUION")
print("=" * 74)
# La segunda lleva un minuto sin hablar. Arranco igual, asi que un guion que solo mire el
# arranque diria que si. La estacion tampoco ayuda: su log calla cuando no hay hallazgos.
r = correr({"1": 1.0, "2": 60.0}, {"VOLAR_FRESCO_S": "10"})
print("  salida: %d" % r.returncode)
for l in r.stdout.splitlines():
    if "NO ESTA LISTO" in l or "NO DESPEGAR" in l:
        print("  %s" % l.strip()[:90])
assert r.returncode != 0, ("dijo que estaba listo con una aeronave callada hace 60 s:\n%s"
                           % r.stdout[-700:])
assert "NO DESPEGAR" in r.stdout, "no lo dice con todas las letras:\n%s" % r.stdout[-700:]
# Y manda a mirar donde sirve, no al log de la estacion.
assert "runner.log" in r.stdout, "no dice donde mirar"

print()
print("=" * 74)
print("3. NI SIQUIERA LO INTENTA CON ARGUMENTOS QUE NO SON PLACAS")
print("=" * 74)
# Pegar la linea dos veces en un terminal la une, y el resultado es una lista de seis "placas"
# de las que tres son un nombre de archivo y una direccion. El lanzador viejo les daba node_id.
entorno = dict(os.environ, SSH="bash " + STUB, VOLAR_SIN_PREPARAR="1",
               VOLAR_SIN_PING="1")
r = subprocess.run([BASH, "scripts/banco_embedded/volar.sh", "127.0.0.1:%d" % PUERTO,
                    "pi@10.0.0.11", "volar.sh"],
                   cwd=RAIZ, capture_output=True, text=True, env=entorno, timeout=120)
print("  con un argumento que no es una placa -> salida %d" % r.returncode)
assert r.returncode != 0 and "NO ARRANCO" in r.stdout, "acepto algo que no es una placa"
r = subprocess.run([BASH, "scripts/banco_embedded/volar.sh", "127.0.0.1:%d" % PUERTO,
                    "pi@10.0.0.11", "pi@10.0.0.11"],
                   cwd=RAIZ, capture_output=True, text=True, env=entorno, timeout=120)
print("  con la misma placa dos veces        -> salida %d" % r.returncode)
assert r.returncode != 0, "acepto la misma placa dos veces, que son dos node_id para una aeronave"

print()
print("=" * 74)
print("4. LA MISION POR OMISION ES LA DE VUELO, NO LA DEL ESCRITORIO")
print("=" * 74)
# mision_vision.py suena a la de vuelo y es la bancada: sin recortes, sin apariencia, sin
# preliminares y con el detector de interior. Volar eso es volar sin lazo de operador.
fuente = open(os.path.join(AQUI, "..", "scripts", "banco_embedded", "volar.sh"),
              encoding="utf-8").read()
import re
m = re.search(r'MISION="\$\{VOLAR_MISION:-([^}]+)\}"', fuente)
assert m, "no encuentro la mision por omision en volar.sh"
print("  por omision carga: %s" % m.group(1))
assert m.group(1).startswith("mision_barrido"), (
    "la mision por omision es %s, y la de vuelo es mision_barrido" % m.group(1))

print()
print("test_volar OK")
servidor.shutdown()
