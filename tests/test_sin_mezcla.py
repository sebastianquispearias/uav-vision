"""Con capa de identidad, el dron no reporta un punto que mezcla todo lo que vio.

Antes de que exista un candidato, el protocolo caia a un consenso RANSAC sobre todos los impactos
guardados: en el vuelo 3 eso pone un punto entre el operador y la caja de equipo, la mezcla que la
capa de identidad existe para evitar, reportada justo cuando el dron sabe menos. Ahora, con capa de
identidad y sin candidatos, el dron manda el latido: sigue vivo y no afirma nada.

Sobre el vuelo real (el replay): ningun POI sin marca de madurez, y los reportes que antes llevaban
la mezcla llegan como latidos. El resultado del vuelo no cambia: el operador sigue donde estaba.

Run: python tests/test_sin_mezcla.py
"""
import contextlib, io, os, runpy, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
os.environ["UAV_VISION_DATOS"] = os.path.join(RAIZ, "demo", "data")
os.environ.pop("UAV_VISION_GS", None)
REPLAY = os.path.join(RAIZ, "scripts", "replay_vuelo3.py")

import numpy as np

for modo in ([], ["--span"]):
    sys.argv = [REPLAY, *modo]
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(REPLAY, run_name="__main__")
    reps, PIES = g["reportes"], g["PIES"]
    mezcla = [p for r in reps for p in r.get("pois", []) if "x" in p and "mature" not in p]
    latidos = sum(1 for r in reps if not r.get("pois"))
    fin = min((p for p in reps[-1]["pois"] if "x" in p), key=lambda p: np.hypot(p["x"] - PIES[0], p["y"] - PIES[1]))
    d = float(np.hypot(fin["x"] - PIES[0], fin["y"] - PIES[1]))
    nombre = "span" if modo else "miradas"
    print("  %-8s %3d reportes | %d POI mezclados | %2d latidos | operador al final a %.2f m"
          % (nombre, len(reps), len(mezcla), latidos, d))
    assert not mezcla, "con capa de identidad no puede salir un punto RANSAC: %s" % mezcla[:1]
    assert latidos > 0, "antes del primer candidato el dron tiene que mandar latidos"
    esperado = 2.39 if modo else 2.18
    assert abs(d - esperado) < 0.005, "el resultado del vuelo cambio: %.2f m en vez de %.2f" % (d, esperado)

print()
print("TODO OK")
