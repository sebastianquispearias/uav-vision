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

WHY THE FAKE ssh IS BUILT THE WAY IT IS
    It does nothing and does NOT read stdin: the script hands it the board script that way, and
    a cat here would wait forever on the calls that carry -n. It is GIT's bash and not whatever
    wins the PATH: with WSL installed, "bash" finds the one in System32, which lives in another
    filesystem and cannot see the Windows paths this test writes. Provisioning has its own test,
    and the addresses used here do not exist.

WHAT EACH SECTION PROVES
    The command SAYS WHAT TO DO, because one that ends in silence leaves the operator guessing.

    A BOARD THAT STARTED AND THEN WENT QUIET MUST NOT BE CALLED READY. The second board booted
    fine and has not spoken for a minute, so a script that only looked at the boot would say
    yes. The station does not help either: its log is silent when there are no finds. So the
    command sends the operator to look WHERE IT HELPS, and not at the station's log.

    PASTING THE LINE TWICE IN A TERMINAL JOINS IT, and the result is a list of six "boards" of
    which three are a filename and an address. The old launcher gave those node ids.

    A MISSION THAT SOUNDS LIKE THE FLYING ONE AND IS THE BENCH ONE IS REFUSED: no crops, no
    appearance, no preliminaries and the indoor detector. Flying that is flying with no operator
    loop at all.
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

STUB = os.path.join(TMP, "ssh_falso.sh").replace("\\", "/")
with open(STUB, "w", newline="\n") as f:
    f.write('echo "ssh $*" >> %s\n' % REGISTRO)
    f.write('echo "   setup: {\\"state\\": \\"ready\\"}"\n')

BASH = r"C:\Program Files\Git\bin\bash.exe"
if not os.path.exists(BASH):
    BASH = "bash"


def correr(edades, extra=None):
    EDADES.clear()
    EDADES.update(edades)
    entorno = dict(os.environ)
    entorno["SSH"] = "bash " + STUB
    entorno["VOLAR_SIN_PREPARAR"] = "1"
    entorno["VOLAR_SIN_PING"] = "1"
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
assert "SEARCHING" in r.stdout, "no recuerda mandar la orden de busqueda antes de despegar"

print()
print("=" * 74)
print("2. SI UNA CALLA, SE NIEGA. ESTA ES LA RAZON DE SER DEL GUION")
print("=" * 74)
r = correr({"1": 1.0, "2": 60.0}, {"VOLAR_FRESCO_S": "10"})
print("  salida: %d" % r.returncode)
for l in r.stdout.splitlines():
    if "NO ESTA LISTO" in l or "NO DESPEGAR" in l:
        print("  %s" % l.strip()[:90])
assert r.returncode != 0, ("dijo que estaba listo con una aeronave callada hace 60 s:\n%s"
                           % r.stdout[-700:])
assert "NO DESPEGAR" in r.stdout, "no lo dice con todas las letras:\n%s" % r.stdout[-700:]
assert "runner.log" in r.stdout, "no dice donde mirar"

print()
print("=" * 74)
print("3. NI SIQUIERA LO INTENTA CON ARGUMENTOS QUE NO SON PLACAS")
print("=" * 74)
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
