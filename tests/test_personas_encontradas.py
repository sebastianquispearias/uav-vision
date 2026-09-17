"""El marcador de producto: cuantas personas reales llegan al mapa y cuantos puntos falsos.

Por que existe este gate y no alcanza con el de `demo/demo.py`: el gate de la demo fija los METROS
del mejor POI, y los metros no se mueven cuando el sistema pierde una persona. El caso esta medido en
`docs/DESCARTADO.md`: un detector reentrenado gano en recall y precision por caja (57,5 / 76,6 contra
54,3 / 60,9) y perdio dos personas. Una metrica de caja no puede ver eso. Esta si.

El marcador no guarda candidatos en disco: los regenera con el replay del vuelo del 02ago, asi que
puntua la capa de identidad TAL COMO ESTA HOY. Un `candidatos.json` congelado dejaba de seguir al
codigo: el que habia en un scratchpad era anterior al arreglo de moviles del 16sep y daba 16
candidatos donde el codigo de hoy da 15.

Run: python tests/test_personas_encontradas.py
"""
import contextlib
import io
import os
import runpy
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
DATOS = os.path.join(RAIZ, "demo", "data")
os.environ["UAV_VISION_DATOS"] = DATOS
os.environ.pop("UAV_VISION_GS", None)
REPLAY = os.path.join(RAIZ, "scripts", "replay_vuelo3.py")
PISTAS = os.path.join(DATOS, "pistas_bot_cmc_sof.npz")

_tmp = tempfile.mkdtemp()
CAND = os.path.join(_tmp, "candidatos.json")

# ---------------------------------------------------------------- el vuelo, con el codigo de hoy
sys.argv = [REPLAY, "--pistas=" + PISTAS, "--candidatos=" + CAND]
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(REPLAY, run_name="__main__")

import personas_encontradas as P  # noqa: E402  (despues de fijar UAV_VISION_DATOS)

with contextlib.redirect_stdout(io.StringIO()) as salida:
    encontradas, fantasmas = P.evaluar("gate", DATOS, PISTAS, CAND)
print(salida.getvalue().strip())

quienes = set(encontradas)
assert quienes == set("ABCGH"), "cambiaron las personas que el sistema encuentra: %s" % sorted(quienes)
# D y E tienen 2 y 5 cajas en todo el vuelo: el umbral de evidencia las descarta A PROPOSITO, y por
# eso no se encuentran ni alimentando la cadena con cajas perfectas (experimento del oraculo, 16sep).
assert set("DE") - quienes == set("DE"), "D y E no deberian encontrarse con este umbral de evidencia"
assert len(fantasmas) == 6, "cambiaron los fantasmas: %d" % len(fantasmas)
assert len(encontradas["A"]) == 1, "el operador vuelve a partirse en varios puntos del mapa"
assert len(encontradas["G"]) == 3, "cambio la fragmentacion de la mujer que camina"

# ------------------------------------------- contraste: el mismo vuelo con el tracker sin CMC
# Un marcador que devolviera "5 de 7" por construccion no probaria nada: hace falta una corrida que
# salga distinta. Esta es la que volaba antes del 16sep, el mismo detector y las mismas cajas, con la
# unica diferencia de que BoT-SORT no compensaba el movimiento de la camara (use_cmc=False): de 2637
# cajas solo 852 reciben id, contra 1608 con CMC, y la evidencia que nunca llega a la capa de
# identidad no forma candidato.
#
# Lo que hay que mirar es que el mapa SIN CMC se ve MAS LIMPIO y tiene DOS PERSONAS MENOS. Esa es la
# razon de ser de este archivo: juzgando por fantasmas, la version vieja gana.
SIN_CMC = os.path.join(DATOS, "pistas_bot_sin_cmc.npz")
CAND_SIN = os.path.join(_tmp, "candidatos_sin_cmc.json")
sys.argv = [REPLAY, "--pistas=" + SIN_CMC, "--candidatos=" + CAND_SIN]
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(REPLAY, run_name="__main__")
with contextlib.redirect_stdout(io.StringIO()):
    sin_cmc, fantasmas_sin = P.evaluar("sin CMC", DATOS, SIN_CMC, CAND_SIN)

print()
print("  %-26s %-22s %s" % ("", "personas", "fantasmas"))
print("  %-26s %-22s %d" % ("con CMC (lo que vuela)",
                            " ".join("%s(%d)" % (l, len(v)) for l, v in sorted(encontradas.items())),
                            len(fantasmas)))
print("  %-26s %-22s %d" % ("sin CMC (antes del 16sep)",
                            " ".join("%s(%d)" % (l, len(v)) for l, v in sorted(sin_cmc.items())),
                            len(fantasmas_sin)))
assert set(sin_cmc) == set("AGH"), "cambio la corrida sin CMC: %s" % sorted(sin_cmc)
assert len(fantasmas_sin) < len(fantasmas), ("sin CMC el mapa tiene que verse MAS limpio: es lo que "
                                             "hace que contar fantasmas solo sea enganoso")
assert set(encontradas) - set(sin_cmc) == set("BC"), "CMC tiene que aportar a B y C"

print()
print("TODO OK")
