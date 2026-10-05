"""Modo objetivo fijado: el sistema mira mas fino donde el operador senalo, y solo ahi.

El operador sabe una cosa que el dron no puede deducir: que en ese punto hay una persona. Actuar
sobre eso no cuesta nada de computo, porque el detector YA puntuo esas cajas y las estaba tirando
por estar bajo el umbral de reporte. Medido sobre la ventana del balcon del vuelo del 02ago, bajar
el umbral a 0,10 dentro de la ventana del objetivo lleva el recall sobre el objetivo de 43,9 % a
60,7 %, y la precision de 71,2 % a 53,3 %.

Lo contrario, que era la idea intuitiva, esta medido y es mucho peor: recortar esa ventana y
AMPLIARLA hunde el recall de 43,9 % a 11,2 %, y ampliando mas, a 2,3 %. El modelo esta entrenado con
VisDrone, que es gente diminuta, y solo reconoce personas de unos 28 pixeles: agrandarla deja de
parecerle una persona. Por eso lo que se enciende es el umbral y no el zoom.

Tres cosas pueden fallar sin que nadie se entere, y son las tres secciones:

  1. Que la ventana no filtre nada y el umbral bajo valga para todo el cuadro. Eso son fantasmas
     por toda la escena a cambio de una persona.
  2. Que una ventana vieja se quede pegada cuando el objetivo sale del encuadre. Bajar el umbral
     sobre un pedazo de suelo que nadie avalo es exactamente como la ganancia gratis se convierte
     en puntos inventados.
  3. Que no este apagado mientras nadie senale nada.

Run: python tests/test_objetivo_fijado.py

WHAT EACH SECTION PROVES
    1. The window filters: a weak box inside survives and the same box outside does not. And
       WITHOUT the window the same list behaves as it always did, which is what proves the gain
       comes from the window and not from having loosened the general threshold.
    2. The window FOLLOWS THE AIRCRAFT and is cleared when the target leaves the frame. With
       nothing pointed at, it stays off; the operator fixes a point in front of the aircraft;
       the aircraft moves and the same ground point lands somewhere else in the image; the
       target leaves the frame and the window must be CLEARED, not left where it was.
    3. Releasing it turns everything off.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(os.path.dirname(RAIZ), "gradys-embedded"))

from uav_vision.camera import dentro_del_foco, solo_confirmadas
from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.vision_protocol import FOCO_RADIO_PX, FOCO_UMBRAL, VisionProtocol

print()

FOCO = {"cx": 960.0, "cy": 540.0, "radio": 320.0, "umbral": 0.10}
dentro = {"px": 1000.0, "py": 500.0, "conf": 0.14}
fuera = {"px": 100.0, "py": 500.0, "conf": 0.14}
floja = {"px": 1000.0, "py": 500.0, "conf": 0.05}
firme = {"px": 100.0, "py": 500.0, "conf": 0.80}
sobrevive = solo_confirmadas([dentro, fuera, floja, firme], 0.25, FOCO)
print("  con ventana en (960,540) r=320, umbral 0.25 y foco 0.10:")
for d in (dentro, fuera, floja, firme):
    print("     conf %.2f en (%4d,%3d)  %-6s  %s" % (
        d["conf"], d["px"], d["py"], "DENTRO" if dentro_del_foco(d, FOCO) else "fuera",
        "pasa" if d in sobrevive else "se cae"))
assert dentro in sobrevive, "una caja floja DENTRO de la ventana tenia que pasar"
assert fuera not in sobrevive, "la misma caja floja FUERA de la ventana no puede pasar"
assert floja not in sobrevive, "una caja por debajo del umbral del foco no puede pasar ni dentro"
assert firme in sobrevive, "una caja firme fuera de la ventana no puede perderse"
sin_foco = solo_confirmadas([dentro, fuera, floja, firme], 0.25, None)
assert sin_foco == [firme], "sin ventana solo puede pasar la caja firme, y paso %d" % len(sin_foco)
print("  sin ventana, esa misma lista deja pasar solo la caja firme: la ganancia es de la ventana")


class CamaraDePrueba:
    """Just enough camera to see where the protocol points the window."""

    camera = ARDUCAM_MODULE_3.rotated_180()
    classes = None

    def __init__(self):
        self.foco = None
        self.pedidos = []

    def set_focus(self, cx=None, cy=None, radio_px=320.0, umbral=0.10):
        self.pedidos.append(None if cx is None else (round(cx), round(cy), radio_px, umbral))
        self.foco = None if cx is None else {"cx": cx, "cy": cy, "radio": radio_px, "umbral": umbral}

    def detect(self, pos, yaw):
        return []


p = VisionProtocol.__new__(VisionProtocol)
p.camera = CamaraDePrueba()
p.pitch_deg = -55.0
p.ground_z = 0.0
p._objetivo = None

p._position = (0.0, 0.0, 20.0)
p._apuntar_foco(0.0, 0.0, 0.0)
assert p.camera.foco is None, "sin objetivo no puede haber ventana"
print("  sin objetivo senalado: no hay ventana")

p.fix_target(0.0, 14.0)
p._apuntar_foco(0.0, 0.0, 0.0)
primera = p.camera.foco
assert primera is not None, "el objetivo esta delante de la camara y no se proyecto"
assert primera["umbral"] == FOCO_UMBRAL and primera["radio"] == FOCO_RADIO_PX, "no uso los valores medidos"
print("  objetivo a 14 m al norte, dron a 20 m: la ventana cae en (%d, %d)"
      % (primera["cx"], primera["cy"]))

p._position = (0.0, 4.0, 20.0)
p._apuntar_foco(0.0, 0.0, 0.0)
segunda = p.camera.foco
assert segunda is not None, "el objetivo sigue delante y la ventana desaparecio"
assert abs(segunda["cy"] - primera["cy"]) > 50, (
    "la ventana no se movio al mover el dron: %d contra %d" % (segunda["cy"], primera["cy"]))
print("  el dron avanza 4 m y la misma persona cae en (%d, %d): la ventana lo sigue"
      % (segunda["cx"], segunda["cy"]))

p._position = (400.0, 400.0, 20.0)
p._apuntar_foco(0.0, 0.0, 0.0)
assert p.camera.foco is None, ("el objetivo salio del encuadre y la ventana vieja quedo pegada: "
                               "eso baja el umbral sobre suelo que nadie avalo")
print("  el dron se va a 560 m: el objetivo sale del cuadro y la ventana se APAGA")

p._position = (0.0, 0.0, 20.0)
p.fix_target(None)
assert p._objetivo is None and p.camera.foco is None, "soltar el objetivo no apago la ventana"
p._apuntar_foco(0.0, 0.0, 0.0)
assert p.camera.foco is None, "la ventana volvio sola despues de soltar el objetivo"
print("  el operador lo suelta: la ventana se apaga y no vuelve sola")

print()
print("TODO OK")
