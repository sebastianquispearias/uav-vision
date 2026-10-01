"""The operator's click keeps a box the detector doubted, when the box looks like the target.

Three things can be done with a click. Retraining the weights in flight was measured and does not
work from one click: eight frames with a frozen trunk are six gradient steps, the weights change
and the behaviour does not, and with more passes over those eight frames it collapses. Lowering
the threshold inside the projected window was measured too, and the seventeen points of recall it
buys do not survive the pipeline: the same five people and the same six phantoms (commit e186c50).

This is the third, and it is the one that works. A box the detector scored between the band floor
and the reporting threshold, which today is thrown away, is kept if its appearance matches the
template of the target the operator pointed at. Nothing is retrained and nothing is adapted, so
switching it off returns the system to exactly what it was.

What it is NOT is a second window. It does not ask where the box is, and that is the whole point:
the window is the projection of a ground position and goes wrong exactly when the aircraft's
attitude is least certain, while an appearance match does not care whether the geometry agrees.
The first section below is that contrast, and it would pass trivially if the gate looked at
position.

Run with: python tests/test_plantilla_objetivo.py
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
# gradys-embedded vive al lado de este repositorio, como en el resto de la suite.
sys.path.insert(0, os.path.join(os.path.dirname(RAIZ), "gradys-embedded"))

import numpy as np  # noqa: E402

from uav_vision.camera import (EMB_DIST_OBJETIVO, OnboardCamera,  # noqa: E402
                               solo_confirmadas)
from uav_vision.vision_protocol import VisionProtocol  # noqa: E402

RNG = np.random.default_rng(11)
UMBRAL = 0.25          # the reporting threshold, as it flies
BANDA = 0.10           # the BYTE band floor: below this the detector returns nothing


class CamaraDeFoco:
    """Just enough camera to build the focus dict with the real code that builds it."""

    foco = None
    set_focus = OnboardCamera.set_focus


def unitario(semilla):
    v = np.random.default_rng(semilla).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def a_distancia(v, d):
    """A vector exactly d away from v, in a random direction, so the gate is tested on a number
    and not on a vague resemblance."""
    u = RNG.normal(size=v.size).astype(np.float32)
    return v + np.float32(d) * (u / np.linalg.norm(u))


def esta(c, lista):
    """Membership by identity. These detections carry numpy arrays, so `in` would compare the
    dicts field by field and numpy refuses to say whether two vectors are one value."""
    return any(c is x for x in lista)


def caja(px, conf, emb=None):
    d = {"px": float(px), "py": 500.0, "conf": float(conf)}
    if emb is not None:
        d["emb"] = emb
    return d


PLANTILLA = unitario(7)
# The window sits at x=1000; every doubted box below is at x=100, far outside it.
FOCO_LEJOS = 1000.0
X_LEJOS = 100.0

print()
print("=" * 70)
print("1. UNA CAJA DUDOSA LEJOS DE LA VENTANA SE CONSERVA SI SE PARECE AL OBJETIVO")
print("=" * 70)
cam = CamaraDeFoco()
cam.set_focus(FOCO_LEJOS, 500.0, radio_px=320.0, umbral=BANDA,
              plantilla=PLANTILLA, emb_dist=EMB_DIST_OBJETIVO)
parecida = caja(X_LEJOS, 0.14, a_distancia(PLANTILLA, 0.30))
distinta = caja(X_LEJOS, 0.14, a_distancia(PLANTILLA, 1.30))
firme = caja(X_LEJOS, 0.80)
lista = [parecida, distinta, firme]

con = solo_confirmadas(lista, UMBRAL, cam.foco)
cam_sin = CamaraDeFoco()
cam_sin.set_focus(FOCO_LEJOS, 500.0, radio_px=320.0, umbral=BANDA)   # same window, no template
sin = solo_confirmadas(lista, UMBRAL, cam_sin.foco)
for nom, c in (("dudosa que SE PARECE  (d=0.30)", parecida),
               ("dudosa que NO se parece (d=1.30)", distinta),
               ("firme, conf 0.80              ", firme)):
    print(f"  {nom}  con plantilla: {'pasa' if esta(c, con) else 'se cae':<7} "
          f"sin plantilla: {'pasa' if esta(c, sin) else 'se cae'}")
assert esta(parecida, con), "la caja que se parece al objetivo tiene que conservarse"
assert not esta(parecida, sin), \
    "el contraste: sin plantilla esa misma caja se cae, asi que la ganancia es de la plantilla"
assert not esta(distinta, con), \
    "parecerse es la condicion; si pasara cualquier caja dudosa esto seria bajar el umbral"
assert esta(firme, con) and esta(firme, sin), "una caja firme no depende de nada de esto"

print()
print("=" * 70)
print("2. LA DISTANCIA ES UNA DECISION, Y SE VE QUE DECIDE")
print("=" * 70)
# The same box, three distances. If the number were decoration the three rows would agree.
for tope in (1.40, EMB_DIST_OBJETIVO, 0.20):
    c = CamaraDeFoco()
    c.set_focus(FOCO_LEJOS, 500.0, radio_px=320.0, umbral=BANDA,
                plantilla=PLANTILLA, emb_dist=tope)
    pasan = solo_confirmadas([parecida, distinta], UMBRAL, c.foco)
    print(f"  tope {tope:<5}  conserva {len(pasan)} de 2 dudosas "
          f"(parecida {'si' if esta(parecida, pasan) else 'no'}, "
          f"distinta {'si' if esta(distinta, pasan) else 'no'})")
assert len(solo_confirmadas([parecida, distinta], UMBRAL,
                            dict(cam.foco, emb_dist=1.40))) == 2, \
    "con el tope muy flojo entran las dos, que es como se ve que el numero manda"
assert len(solo_confirmadas([parecida, distinta], UMBRAL,
                            dict(cam.foco, emb_dist=0.20))) == 0, \
    "con el tope muy duro no entra ninguna"

print()
print("=" * 70)
print("3. SIN PLANTILLA, EL COMPORTAMIENTO ES EL DE ANTES, CAJA POR CAJA")
print("=" * 70)
# The gate is off unless a caller asks for it. Not "almost the same": the same list.
mezcla = [caja(X_LEJOS, 0.14, a_distancia(PLANTILLA, 0.10)),
          caja(FOCO_LEJOS, 0.14, a_distancia(PLANTILLA, 0.10)),
          caja(X_LEJOS, 0.05, a_distancia(PLANTILLA, 0.10)),
          caja(X_LEJOS, 0.90)]
apagado = solo_confirmadas(mezcla, UMBRAL, cam_sin.foco)
nada = solo_confirmadas(mezcla, UMBRAL, None)
print(f"  con ventana y sin plantilla: {len(apagado)} de {len(mezcla)}")
print(f"  sin ventana ni plantilla   : {len(nada)} de {len(mezcla)}")
assert [id(c) for c in apagado] == [id(mezcla[1]), id(mezcla[3])], \
    "sin plantilla solo pasan la de la ventana sobre la banda y la firme"
assert [id(c) for c in nada] == [id(mezcla[3])], "sin nada, solo la firme"

print()
print("=" * 70)
print("4. LA PLANTILLA SOBREVIVE EL VIAJE: base64 de float16, ida y vuelta")
print("=" * 70)
# The station does not compute the template, it hands back the embedding it already received,
# which travels as base64 of float16 to keep the field at 1.4 kB instead of 2.7. If the round
# trip changed the decision, the operator's click would mean one thing on the drone and another
# on the ground.
import base64  # noqa: E402

empaquetada = base64.b64encode(
    np.asarray(PLANTILLA, dtype=np.float16).tobytes()).decode("ascii")
vuelta = VisionProtocol._emb_de_mensaje(empaquetada)
desde_lista = VisionProtocol._emb_de_mensaje(PLANTILLA.tolist())
error = float(np.linalg.norm(vuelta - PLANTILLA))
print(f"  tamano en el cable         : {len(empaquetada)} caracteres base64")
print(f"  error que introduce float16: {error:.5f}")
cam_cable = CamaraDeFoco()
cam_cable.set_focus(FOCO_LEJOS, 500.0, radio_px=320.0, umbral=BANDA,
                    plantilla=vuelta, emb_dist=EMB_DIST_OBJETIVO)
pasan = solo_confirmadas([parecida, distinta], UMBRAL, cam_cable.foco)
print(f"  misma decision tras el viaje: parecida {'pasa' if esta(parecida, pasan) else 'se cae'}, "
      f"distinta {'pasa' if esta(distinta, pasan) else 'se cae'}")
assert error < 0.01, "float16 no puede mover la plantilla lo suficiente para cambiar una decision"
assert vuelta is not None and desde_lista is not None, \
    "la plantilla tiene que leerse en las dos formas en que puede llegar"
assert esta(parecida, pasan) and not esta(distinta, pasan), \
    "la decision despues del viaje tiene que ser la misma que antes"
assert VisionProtocol._emb_de_mensaje(None) is None, "sin plantilla no se inventa una"

print()
print("=" * 70)
print("5. LA ESTACION ELIGE DE QUIEN ES LA PLANTILLA: EL CANDIDATO MAS CERCANO AL CLICK")
print("=" * 70)
# The click has the precision of a finger on a map, so the station has to decide which candidate
# was meant. Reaching too far would hand the drone the appearance of somebody standing next to
# the person pointed at, which is the one mistake that turns this feature into a target swap.
sys.path.insert(0, os.path.join(RAIZ, "scripts", "banco_embedded"))
import gs_mapa  # noqa: E402

A, B = "AAAA", "BBBB"          # two different appearances, as they arrive: already encoded
vigentes = [{"x": 0.0, "y": 0.0, "emb": A},
            {"x": 10.0, "y": 0.0, "emb": B},
            {"x": 0.5, "y": 20.0, "emb": None}]      # a candidate with no appearance at all
casos = [((0.4, 0.0), A, "sobre el primero"),
         ((9.0, 0.0), B, "mas cerca del segundo"),
         ((5.0, 0.0), None, "a cinco metros de los dos: nadie"),
         ((0.5, 20.0), None, "sobre el que no trae apariencia")]
print("  radio del click: %.1f m" % gs_mapa.OBJETIVO_RADIO_M)
for (cx, cy), esperado, nota in casos:
    got = gs_mapa.plantilla_para(cx, cy, vigentes)
    print("  click (%5.1f,%5.1f) %-34s -> %s" % (cx, cy, nota, got if got else "sin plantilla"))
    assert got == esperado, "la estacion eligio la plantilla equivocada %s" % nota
assert gs_mapa.plantilla_para(0.4, 0.0, []) is None, "sin candidatos no hay plantilla que mandar"
assert gs_mapa.EMB_DIST_OBJETIVO == EMB_DIST_OBJETIVO,     "la estacion y el dron tienen que estar usando la MISMA distancia, importada y no copiada"

print()
print("TODO OK")
