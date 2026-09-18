"""Re-unir a quien camina, sale de cuadro y vuelve: funciona, y este test dice a que precio.

Una persona que camina sale del mapa como varios puntos, y eso sobrevive a un detector perfecto
(experimento del oraculo, 16sep): no es culpa de la vision sino del emparejador, que se negaba a
mirar a un candidato movil y comparaba la posicion contra donde la persona ESTABA, no contra donde
estaria. Sobre el vuelo del 02ago la mujer que camina queda partida en tres puntos a 2,9, 7,0 y
9,1 m con un radio de fusion de 3,5 m, mientras OSNet la reconoce con holgura (0,41 a 0,61 contra un
umbral de 0,95). La posicion decia tres personas, la apariencia decia una, y la equivocada era la
posicion.

Las tres secciones son el argumento entero, y la tercera es la que decide el valor por defecto:

  1. APAGADO, que es como viene, el sistema hace lo de siempre. Si esto cambia, la mejora se colo
     en la configuracion que vuela sin que nadie la aprobara.
  2. ENCENDIDO con lo medido, la mujer que camina pasa de tres puntos a uno, con las mismas cinco
     personas, los mismos seis fantasmas y el punto a la misma distancia.
  3. ENCENDIDO un poco mas suelto, el chico del balcon es absorbido por otro y DESAPARECE DEL MAPA.
     Esa es la razon de que venga apagado: la distancia entre unir a quien hay que unir y borrar a
     una persona son 0,07 en una escala donde pedazos de la misma persona llegan a 0,81.

Run: python tests/test_reunir_movil.py
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

import uav_vision.identity as I  # noqa: E402

_ini = I.IncrementalIdentity.__init__
_tmp = tempfile.mkdtemp()


def corrida(nombre, **ajustes):
    """The whole chain over the flight with these settings, scored by person."""
    def _con(self, *a, **k):
        for clave, valor in ajustes.items():
            k.setdefault(clave, valor)
        _ini(self, *a, **k)

    I.IncrementalIdentity.__init__ = _con
    cand = os.path.join(_tmp, nombre + ".json")
    sys.argv = [REPLAY, "--pistas=" + PISTAS, "--candidatos=" + cand]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(REPLAY, run_name="__main__")
    finally:
        I.IncrementalIdentity.__init__ = _ini
    with contextlib.redirect_stdout(io.StringIO()):
        encontradas, fantasmas = P.evaluar(nombre, DATOS, PISTAS, cand)
    print("  %-34s %-26s %d fantasmas" % (
        nombre, " ".join("%s(%d)" % (l, len(v)) for l, v in sorted(encontradas.items())),
        len(fantasmas)))
    return encontradas, fantasmas


import personas_encontradas as P  # noqa: E402

print()
# 1. off is off: the configuration that flies must not have changed
apagado, fant_apagado = corrida("como viene (apagado)")
assert set(apagado) == set("ABCGH"), "cambio lo que encuentra el sistema por defecto"
assert len(apagado["G"]) == 3, "la que camina ya no sale partida en tres sin encender nada"
assert len(fant_apagado) == 6, "cambiaron los fantasmas del sistema por defecto"

# 2. on, with the numbers measured on this flight
unida, fant_unida = corrida("encendido (0.63, 30 s)", rejoin_mobile=True,
                            emb_dist_rejoin=I.EMB_DIST_REUNE, rejoin_max_gap_s=30.0)
assert set(unida) == set(apagado), "encenderlo tiene que conservar las mismas personas: %s" % sorted(unida)
assert len(unida["G"]) == 1, "no la re-unio: sigue en %d puntos" % len(unida["G"])
assert len(fant_unida) == len(fant_apagado), "aparecieron fantasmas nuevos: %d" % len(fant_unida)
assert all(len(unida[l]) == len(apagado[l]) for l in "ABCH"), "movio a alguien que no habia que mover"

# 3. and why it is off: a little looser and somebody stops existing for the operator
suelto, _ = corrida("un poco mas suelto (0.70)", rejoin_mobile=True, emb_dist_rejoin=0.70,
                    rejoin_max_gap_s=30.0)
assert "C" not in suelto, ("con 0.70 el chico del balcon tenia que desaparecer, y no lo hizo: si esto "
                           "cambia, el margen ya no es el que justifica que venga apagado")
assert len(suelto) == len(unida) - 1, "a 0.70 se pierde exactamente una persona, no %d" % (len(unida) - len(suelto))
print()
print("  entre unir a la que camina y borrar al chico del balcon hay 0.07 de distancia OSNet,")
print("  en una escala donde dos pedazos de la misma persona llegan a estar a 0.81.")

print()
print("TODO OK")
