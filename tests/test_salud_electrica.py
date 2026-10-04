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
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
sys.path.insert(0, RAIZ)

# El protocolo no se puede importar sin el runtime de GrADyS, que no esta en la laptop, asi que la
# funcion se trae sola. Es deliberadamente independiente de la clase por esta razon.
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
# El bit bajo es AHORA: actuar sobre el significa que la aeronave esta en problemas este segundo.
# El alto es DESDE QUE ARRANCO y NO se apaga cuando la caida pasa. Esa es la diferencia entre ver
# el problema y no verlo: las caidas duran un instante y nadie esta mirando en ese instante.
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
print("2. SIN FIRMWARE DE RASPBERRY NO SE INVENTA UNA ALARMA")
print("=" * 72)
# Una laptop reproduciendo la grabacion no tiene ese archivo. Una estacion que dibujara una
# advertencia por un archivo ausente gritaria en cada corrida de escritorio, y una alarma que
# suena siempre deja de leerse, que es peor que no tenerla.
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
# Dos copias de un nombre es como un dron y su estacion dejan de entenderse en silencio.
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
# No es orden de aparicion: es que el desacuerdo de origen describe un error que ya esta a la
# vista en el mapa, y el bajo voltaje PREDICE una aeronave que se va a perder. Un operador que
# solo lee la primera linea tiene que leer esa.
i_corriente = gs.index("POWER:")
i_origen = gs.index("The origin the drone declares")
print("  POWER en la posicion %d, el origen en la %d" % (i_corriente, i_origen))
assert i_corriente < i_origen, (
    "el aviso de origen tapa al de corriente; el de corriente es el unico que predice una perdida")
# Y dice que hacer, no solo que pasa.
assert "Check its supply" in gs, "la alarma no dice que hacer"
print("  y la alarma termina en que hacer, no solo en que pasa")

print()
print("test_salud_electrica OK")
