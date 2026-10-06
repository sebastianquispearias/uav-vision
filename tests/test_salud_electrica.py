"""A board that browns out does not warn, it disappears. This makes it warn.

On 2026-10-03 a Raspberry Pi 4 powered from the drone's LiPo logged "Undervoltage detected!" in
its own kernel from 21:56 onwards. The station said it was healthy, the operator saw nothing, and
at 01:58 it stopped mid-log-line with no shutdown sequence and never came back. The cause was
found the next day by moving its SD card into another board and reading its journal:

    arranque -2 (en la Pi 4): 10 avisos de bajo voltaje
    arranques -1 y -3:         0

In the air that is an aircraft that disappears with no explanation, and the fleet's own station is
the last place anybody would think to look afterwards, because it never knew.

So the drone now reads the firmware's own complaint every report, the station keeps it beside
fps_real and slots_perdidos, and the page puts it in the alarm bar AHEAD of everything else: it is
the only thing on that screen that predicts a loss rather than describing one.

Run: python tests/test_salud_electrica.py

The function is imported on its own rather than through the class, because the protocol cannot
be imported without the GrADyS runtime, which is not on the laptop. It is deliberately
independent of the class for that reason.

WHAT EACH SECTION PROVES
    The LOW bit is NOW: acting on it means the aircraft is in trouble this second. The HIGH bit
    is EVER SINCE BOOT and does NOT clear when the dip passes. That is the difference between
    seeing the problem and not seeing it, because the dips last an instant and nobody is
    watching in that instant.

    A laptop replaying the recording does not have that file. A station that drew a warning for
    a MISSING file would cry wolf on every desk run, and an alarm that always sounds stops being
    read, which is worse than not having one.

    The message name is checked as text: two copies of a name is how a drone and its station
    stop understanding each other in silence.

    THE WARNING COMES FIRST, and not because of the order things appear in. An origin
    disagreement describes an error that is ALREADY visible on the map; low voltage PREDICTS an
    aircraft that is going to be lost. An operator who only reads the first line has to read
    that one. And it says what to DO, not only what happened.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
sys.path.insert(0, RAIZ)

import importlib.util

ruta = os.path.join(RAIZ, "uav_vision", "vision_protocol.py")
fuente = open(ruta, encoding="utf-8").read()
from typing import Optional

ns = {"Optional": Optional}
inicio = fuente.index("THROTTLED =")
fin = fuente.index("class UavApiYaw:")
exec(compile(fuente[inicio:fin], ruta, "exec"), ns)
salud_electrica = ns["salud_electrica"]

import tempfile

TMP = tempfile.mkdtemp(prefix="salud_")


def con(valor):
    p = os.path.join(TMP, "throttled")
    with open(p, "w") as f:
        f.write(valor)
    return salud_electrica(p)


print("=" * 72)
print("1. LOS DOS BITS SIGNIFICAN COSAS DISTINTAS Y HACEN FALTA LOS DOS")
print("=" * 72)
limpio = con("0")
print("  0x0        -> ahora=%s  alguna_vez=%s" % (limpio["ahora"], limpio["alguna_vez"]))
assert limpio["ahora"] == [] and limpio["alguna_vez"] == [], limpio

ahora = con("0x1")
print("  0x1        -> ahora=%s  alguna_vez=%s" % (ahora["ahora"], ahora["alguna_vez"]))
assert ahora["ahora"] == ["bajo_voltaje"], ahora
assert ahora["alguna_vez"] == [], "un bit de ahora no puede encender el de alguna vez: %r" % ahora

paso = con("0x10000")
print("  0x10000    -> ahora=%s  alguna_vez=%s" % (paso["ahora"], paso["alguna_vez"]))
assert paso["alguna_vez"] == ["bajo_voltaje"], paso
assert paso["ahora"] == [], "la caida ya paso y sigue marcada como actual: %r" % paso
print("  -> ESTE es el caso del 3oct: la caida paso, el aviso tiene que quedar")

real = con("0x50005")
print("  0x50005    -> ahora=%s" % real["ahora"])
print("                alguna_vez=%s" % real["alguna_vez"])
assert set(real["ahora"]) == {"bajo_voltaje", "acelerador"}, real
assert set(real["alguna_vez"]) == {"bajo_voltaje", "acelerador"}, real

print()
print("=" * 72)
print("1b. EL ARCHIVO DEL KERNEL NO LLEVA PREFIJO, Y ASI SE LEIA MAL")
print("=" * 72)
# Las lineas de arriba pasan el valor con '0x' delante, y con prefijo cualquier base acierta.
# El archivo real no lo lleva: leido en la Pi 4 del banco decia '80008' pelado. Esta seccion
# alimenta las dos formas y exige que coincidan, que es el contraste que faltaba.
for sin_prefijo, con_prefijo in (("0", "0x0"), ("80008", "0x80008"), ("50005", "0x50005"),
                                 ("e0008", "0xe0008")):
    a, b = con(sin_prefijo), con(con_prefijo)
    print("  %-8s vs %-9s -> ahora=%-22s alguna_vez=%s"
          % (sin_prefijo, con_prefijo, a["ahora"], a["alguna_vez"]))
    assert a is not None, (
        "'%s' sin prefijo devolvio None, o sea 'esta maquina no tiene firmware'. Un valor con "
        "letra hexadecimal reventaba int(texto, 0) y el except lo convertia en silencio: la "
        "alarma desaparecia en vez de sonar." % sin_prefijo)
    assert a == b, (
        "'%s' y '%s' tienen que decodificar igual y dieron %r contra %r. Si difieren, el valor "
        "se esta leyendo en base 10: '80008' pasa a ser 0x13888, y ahi los bits caen en otro "
        "significado. Medido en el banco: una placa cuyo estado real era limite termico se "
        "reportaba a la estacion como BAJO VOLTAJE." % (sin_prefijo, con_prefijo, a, b))
sin_pref = con("80008")
assert "bajo_voltaje" not in sin_pref["alguna_vez"], (
    "80008 no tiene ningun bit de bajo voltaje (0x80008 = limite termico ahora y alguna vez) "
    "y se reporto uno: %r" % sin_pref)
assert sin_pref["alguna_vez"] == ["limite_termico"], sin_pref
print("  las dos formas coinciden, y 80008 es limite termico y NO bajo voltaje")

print()
print("=" * 72)
print("2. SIN FIRMWARE DE RASPBERRY NO SE INVENTA UNA ALARMA")
print("=" * 72)
sin = salud_electrica(os.path.join(TMP, "no-existe"))
print("  archivo ausente -> %r" % sin)
assert sin is None, "invento una lectura donde no hay firmware: %r" % sin
roto = con("esto no es un numero")
print("  archivo ilegible -> %r" % roto)
assert roto is None, "no trago un archivo con basura: %r" % roto

print()
print("=" * 72)
print("3. LA CADENA ENTERA NOMBRA LO MISMO EN LOS TRES SITIOS")
print("=" * 72)
gs = open(os.path.join(RAIZ, "scripts", "banco_embedded", "gs_mapa.py"), encoding="utf-8").read()
assert '"salud": salud_electrica()' in fuente, "el dron no manda su salud en el reporte"
print("  el dron la manda en cada reporte")
assert "'salud': mensaje.get('salud')" in gs, "la estacion no la guarda en la ficha del dron"
print("  la estacion la guarda en la ficha, junto a fps_real y slots_perdidos")
assert "d.salud" in gs and "alguna_vez" in gs, "la pagina no la mira"
print("  la pagina la lee y la pone en la barra de alarma")

print()
print("=" * 72)
print("4. LA ALARMA DE CORRIENTE VA ANTES QUE LA DEL ORIGEN")
print("=" * 72)
i_corriente = gs.index("POWER:")
i_origen = gs.index("The origin the drone declares")
print("  POWER en la posicion %d, el origen en la %d" % (i_corriente, i_origen))
assert i_corriente < i_origen, (
    "el aviso de origen tapa al de corriente; el de corriente es el unico que predice una perdida")
assert "Check its supply" in gs, "la alarma no dice que hacer"
print("  y la alarma termina en que hacer, no solo en que pasa")

print()
print("test_salud_electrica OK")
