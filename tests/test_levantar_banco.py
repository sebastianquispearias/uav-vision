"""The bench comes up from one command, and the three things that broke it before cannot happen.

Bringing up three boards by hand is where this bench went wrong, and none of the three failures
announce themselves. Autodiscovery does not work on this LAN, so a board that was not told the
others' addresses simply reports alone and nothing fuses. A Raspberry comes up with the wrong clock
after a cold boot, and then the three logs cannot be read against each other. And boards started
one at a time are each replaying a different moment of the recording, so there is nothing to fuse
even when the addresses are right.

So this gate runs the real launcher with a fake ssh on the PATH and checks the orchestration: every
board gets the same dictionary of addresses with the station last, its own node id, its own replay
offset, the clock is set before any of it, and every board is set up before any board is started.

No Raspberry is needed and none is contacted.

Run with: python tests/test_levantar_banco.py

WHY THE FAKE ssh IS BUILT THE WAY IT IS
    It is run through bash BY NAME rather than installed as an executable, because the x bit of
    a file Python creates on Windows is not something Git Bash can be relied on to honour. It is
    handed in by name and not won on the PATH, because on Windows the PATH separator is not the
    one bash reads, so a test that tried to shadow the system ssh would lose the race and the
    real client would sit waiting on a TCP connection to an address that does not exist. And it
    is GIT's bash, not whatever "bash" resolves to: on a machine with WSL installed, PATH finds
    System32's bash.exe first, and that one lives in another filesystem, so it cannot see the
    Windows paths this test writes its stub and its log to and nothing is recorded or explained.

    It does not read stdin on purpose: the launcher hands it the board script that way, and a
    cat here would wait forever on the calls that carry -n.

    There are TWO logs, and the second one is the point: the first joins the arguments with
    spaces, so an EMPTY argument is invisible in it. That is not a formatting detail, which the
    next section is about.

WHAT EACH SECTION PROVES
    Autodiscovery does not work on this LAN, so a board that was not handed the others'
    addresses reports alone, and the dictionary has to be IDENTICAL on all three or they
    disagree about who exists. The node id is what a drone signs its reports with, so two boards
    sharing one are one drone as far as the station is concerned, and the replay offsets are
    what put two of them on the same take-off. If one board starts before another, each is
    replaying a different moment of the flight and there is nothing to fuse; that is why one
    board's script waits before starting, and it would be invisible, because everything "works"
    and the fusion simply never happens.

    AN EMPTY ARGUMENT DOES NOT SURVIVE ssh. ssh does not deliver a list of arguments: it joins
    them into ONE command and the board's shell parses it again, and an empty string has no
    representation there, so it disappears and everything behind it shifts one place. Verified
    against a real board: seven arguments sent, six received. The symptom does not name the
    cause -- it reads as a launcher error and is in fact an address sitting where the
    environment should be. That is why the launcher sends a placeholder when there is no extra
    environment, and why this section reads the RAW log: in the other one the failure is
    literally invisible.

    NOR DO SPACES. The board re-parses the command, so what was ONE argument arrives as
    SEVERAL. With two environment variables that shifted the addresses one place and left the
    STATION out of the node dictionary, so the drone broadcast to its neighbours and never to
    the ground. That reads as "the protocol does not report" and is a badly delivered argument.
    That is why the assignments travel comma-separated.

    WRITING CODE THROUGH LAYERS THAT INTERPRET ESCAPES turns a backslash-b into a literal
    BACKSPACE, and the result looks right on screen. It happened three times in one month: a
    test regex whose lookahead no longer matched anything, so THE TEST COULD NOT FAIL; a path
    that lost two characters in the state file; and the port-wait pattern of
    lanzar_banco_nodo.sh, which, never matching, made the loop exit on its first turn so the
    wait did not wait. All three are SILENT: nothing fails, it just stops protecting. That is
    why this sweep covers the whole repository.
"""
import os
import re
import subprocess
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
LANZADOR = "scripts/banco_embedded/levantar_banco.sh"

TMP = tempfile.mkdtemp()
REGISTRO = os.path.join(TMP, "llamadas.txt").replace("\\", "/")

CRUDO = os.path.join(TMP, "ssh_crudo.txt").replace("\\", "/")
STUB = os.path.join(TMP, "ssh_falso.sh").replace("\\", "/")
with open(STUB, "w", newline="\n") as f:
    f.write('echo "ssh $*" >> %s\n' % REGISTRO)
    f.write('{ for a in "$@"; do printf "[%s]" "$a"; done; echo; } >> ' + CRUDO + '\n')

entorno = dict(os.environ)
entorno["SSH"] = "bash " + STUB
entorno["BANCO_DESDE_S"] = "0 700 700"

BASH = next((r for r in (r"C:\Program Files\Git\bin\bash.exe",
                         r"C:\Program Files (x86)\Git\bin\bash.exe",
                         os.path.join(os.environ.get("ProgramW6432", ""), "Git", "bin", "bash.exe"))
             if r and os.path.exists(r)), None)
if BASH is None:
    import shutil
    BASH = shutil.which("bash")
assert BASH, "no se encontro un bash con el que correr el lanzador"

salida = subprocess.run(
    [BASH, LANZADOR, "10.0.0.9:8300",
     "pi@10.0.0.11", "pi@10.0.0.12", "pi@10.0.0.13"],
    capture_output=True, text=True, env=entorno, cwd=RAIZ, timeout=120)

llamadas = []
if os.path.exists(REGISTRO):
    with open(REGISTRO) as f:
        llamadas = [l.strip() for l in f if l.strip()]

print()
print("=" * 78)
print("1. UNA SOLA ORDEN LEVANTA LAS TRES PLACAS, SIN TOCAR NINGUNA PLACA")
print("=" * 78)
print("  codigo de salida del lanzador: %d" % salida.returncode)
print("  llamadas a ssh registradas   : %d" % sum(1 for c in llamadas if c.startswith("ssh ")))
assert salida.returncode == 0, "el lanzador tiene que terminar bien:\n%s" % salida.stderr[-600:]
assert llamadas, "no se registro una sola llamada: el lanzador no contacto a nadie"

relojes = [c for c in llamadas if "date" in c and "sudo" in c]
preparar = [c for c in llamadas if "8200" in c]
arrancar = [c for c in llamadas if "mission/start" in c]
print("  relojes puestos              : %d" % len(relojes))
print("  placas preparadas            : %d" % len(preparar))
print("  placas arrancadas            : %d" % len(arrancar))
assert len(relojes) == 3, "las tres placas necesitan la hora, o los logs no se pueden cruzar"
assert len(preparar) == 3, "las tres placas tienen que prepararse"
assert len(arrancar) == 3, "las tres placas tienen que arrancar"

print()
print("=" * 78)
print("2. CADA PLACA RECIBE EL MISMO DICCIONARIO, CON LA ESTACION AL FINAL")
print("=" * 78)
esperado = "10.0.0.11:8200 10.0.0.12:8200 10.0.0.13:8200 10.0.0.9:8300"
for c in preparar:
    dirs = " ".join(re.findall(r"\d+\.\d+\.\d+\.\d+:\d+", c)[-4:])
    print("  %s" % dirs)
    assert dirs == esperado, "el diccionario llego distinto o la estacion no quedo ultima"

print()
print("=" * 78)
print("3. CADA PLACA ES UN NODO DISTINTO Y REPRODUCE SU PROPIO TRAMO")
print("=" * 78)
ids, desdes = [], []
for c in preparar:
    partes = c.split()
    i = partes.index("--")
    ids.append(partes[i + 1])
    desdes.append(partes[i + 2])
print("  node_id por placa : %s" % " ".join(ids))
print("  tramo por placa   : %s s" % " s ".join(desdes))
assert ids == ["1", "2", "3"], "dos placas con el mismo node_id son un solo dron para la estacion"
assert desdes == ["0", "700", "700"], \
    "los tramos son lo que pone a dos placas sobre el mismo despegue a la vez"

print()
print("=" * 78)
print("4. NINGUNA ARRANCA ANTES DE QUE TODAS ESTEN PREPARADAS")
print("=" * 78)
ultimo_preparar = max(i for i, c in enumerate(llamadas) if "8200" in c)
primer_arrancar = min(i for i, c in enumerate(llamadas) if "mission/start" in c)
print("  ultima preparacion en la llamada %d, primer arranque en la %d"
      % (ultimo_preparar, primer_arrancar))
assert primer_arrancar > ultimo_preparar, \
    "arrancar una placa antes de preparar las otras deja a cada dron en otro momento del vuelo"

print("=" * 78)
print("5. NINGUN ARGUMENTO VIAJA VACIO, PORQUE UN ARGUMENTO VACIO NO SOBREVIVE A SSH")
print("=" * 78)
llamadas = [l.strip() for l in open(CRUDO, encoding="utf-8") if l.strip()]
preparaciones = [l for l in llamadas if "[bash -s]" in l]
print("  llamadas de preparacion registradas: %d" % len(preparaciones))
assert preparaciones, "el stub no registro ninguna preparacion: %r" % llamadas[:3]
for l in preparaciones:
    print("    %s" % l[:104])
    assert "[]" not in l, (
        "un argumento viaja VACIO y ssh lo va a borrar, corriendo las direcciones un lugar: %s" % l)
    assert l.count("[") >= 8, (
        "la llamada lleva %d argumentos y se esperaban al menos 8: %s" % (l.count("["), l))
    entorno = l.split("][")[6] if l.count("][") > 6 else ""
    assert " " not in entorno, (
        "el entorno viaja con espacios y ssh lo va a partir, corriendo las direcciones: %s" % l)
print("  -> ninguna lleva un argumento vacio")
print("=" * 78)
print("6. NINGUN ARCHIVO LLEVA CARACTERES DE CONTROL")
print("=" * 78)
import glob

sospechosos = []
for patron in ("tests/*.py", "tests/*.js", "uav_vision/*.py",
               "scripts/*.py", "scripts/banco_embedded/*.py", "scripts/banco_embedded/*.sh",
               "demo/*.py"):
    for f in glob.glob(os.path.join(RAIZ, patron)):
        texto = open(f, encoding="utf-8", errors="replace").read()
        malos = sorted({c for c in texto if ord(c) < 32 and c not in "\n\t\r"})
        if malos:
            sospechosos.append((os.path.relpath(f, RAIZ), [hex(ord(c)) for c in malos]))
print("  archivos revisados: %d" % sum(len(glob.glob(os.path.join(RAIZ, p))) for p in
      ("tests/*.py", "tests/*.js", "uav_vision/*.py", "scripts/*.py",
       "scripts/banco_embedded/*.py", "scripts/banco_embedded/*.sh", "demo/*.py")))
for f, malos in sospechosos:
    print("  CONTROL %s en %s" % (malos, f))
assert not sospechosos, (
    "hay caracteres de control en el fuente. Casi siempre es un escape que se convirtio en el "
    "caracter que nombraba, y el sintoma es que algo deja de proteger sin fallar: %r" % sospechosos)
print("  -> limpio")
print()
print("TODO OK")
