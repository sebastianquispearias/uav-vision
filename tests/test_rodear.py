"""The aircraft flies to look at a target from another side, and only because a human asked.

This is the second half of the mission as it was written down: detect a POI, and then the drones
go, circle it and hold position. Everything before this looked, computed a position and reported
it, and never touched the flight. So this file is the gate on the first thing in the package that
moves an aircraft, and it is strict on purpose.

The place it flies to is not "closer". Closer was measured on the 02ago flight and does not settle
the detection question: within one frame, where altitude and light are identical, the detector
missed people LARGER than ones it found in the same picture. What a single pass lacks is not
pixels, it is a second direction. That is what the paper's own criterion returns, used backwards:
compute_angular_diversity ranks a direction, and it does not care whether the ray exists yet, so
the function that picks the best views already taken also picks where to go and take one.

What the extra view is for, stated exactly because it decides how to judge it: a second geometry
for the same target, and a picture from the side for the person who has to decide. NOT a claim
that the detector does better from there, which depends on the model and is not measured.

Run with: python tests/test_rodear.py
"""
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
# gradys-embedded vive al lado de este repositorio, como en el resto de la suite.
sys.path.insert(0, os.path.join(os.path.dirname(RAIZ), "gradys-embedded"))

import numpy as np  # noqa: E402

from uav_vision.vision_protocol import (RODEO_TOLERANCIA_M,  # noqa: E402
                                        VisionProtocol)
from uav_vision.view_selection import next_best_viewpoint  # noqa: E402

RADIO, ALTURA = 30.0, 25.0
OBJETIVO = (0.0, 0.0)


class ProveedorQueAnota:
    """Records every mobility command, because the point of this file is that there are none
    unless somebody asked, and exactly one when they did."""

    def __init__(self):
        self.ordenes = []

    def send_mobility_command(self, command):
        self.ordenes.append(command)

    def current_time(self):
        return 0.0

    def get_id(self):
        return 1


class CamaraMuda:
    classes = None

    def detect(self, pos, yaw):
        return []


def protocolo(pos=(0.0, -30.0, 25.0)):
    p = VisionProtocol.__new__(VisionProtocol)
    p.provider = ProveedorQueAnota()
    p.camera = CamaraMuda()
    p.ground_z = 0.0
    p._position = pos
    p._rodeo = None
    p._marcos = []
    p.send_frame = lambda para=None: p._marcos.append(para) or True
    return p


def azimut(x, y):
    return math.degrees(math.atan2(y, x)) % 360.0


print()
print("=" * 76)
print("1. SIN ORDEN NO SE MUEVE. ES LA MITAD MAS IMPORTANTE DE ESTE ARCHIVO")
print("=" * 76)
p = protocolo()
for _ in range(5):
    p._llego_al_rodeo()          # telemetry ticking by, nothing asked
print("  cinco llegadas de telemetria sin orden -> %d ordenes de movimiento" % len(p.provider.ordenes))
print("  y %d cuadros mandados" % len(p._marcos))
assert p.provider.ordenes == [], "sin que un humano lo pida, la aeronave no se mueve"
assert p._marcos == [], "ni manda cuadros"

print()
print("=" * 76)
print("2. CON ORDEN: UNA SOLA, AL RADIO Y LA ALTURA PEDIDOS, Y DESDE OTRO LADO")
print("=" * 76)
ok = p.rodear(OBJETIVO[0], OBJETIVO[1], radio_m=RADIO, altura_m=ALTURA)
orden = p.provider.ordenes[0]
ir = p._rodeo["ir_a"]
r = math.hypot(ir[0] - OBJETIVO[0], ir[1] - OBJETIVO[1])
print("  el dron esta en (%.1f, %.1f, %.1f), azimut %.0f deg" % (*p._position, azimut(p._position[0], p._position[1])))
print("  se le ordena ir a (%.1f, %.1f, %.1f), azimut %.0f deg" % (*ir, azimut(ir[0], ir[1])))
print("  radio %.1f m (pedido %.1f), altura %.1f m (pedida %.1f)" % (r, RADIO, ir[2], ALTURA))
print("  diversidad del rayo nuevo: %.1f grados" % p._rodeo["diversidad"])
print("  ordenes de movimiento emitidas: %d" % len(p.provider.ordenes))
assert ok and len(p.provider.ordenes) == 1, "una orden, no ninguna y no dos"
assert abs(r - RADIO) < 0.5 and abs(ir[2] - ALTURA) < 0.01, \
    "el radio y la altura son del operador, no del algoritmo"
assert p._rodeo["diversidad"] > 60.0, \
    "si el rayo nuevo no es MAS DIVERSO, volar hasta ahi no compra geometria"
# Same target, same radius: a drone standing somewhere else is sent somewhere else. If the answer
# did not depend on where the aircraft already is, it would not be using the criterion at all.
q = protocolo(pos=(30.0, 0.0, 25.0))
q.rodear(OBJETIVO[0], OBJETIVO[1], radio_m=RADIO, altura_m=ALTURA)
print("  el contraste: un dron en azimut %.0f deg es mandado a %.0f deg, no al mismo sitio"
      % (azimut(30.0, 0.0), azimut(q._rodeo["ir_a"][0], q._rodeo["ir_a"][1])))
assert math.hypot(q._rodeo["ir_a"][0] - ir[0], q._rodeo["ir_a"][1] - ir[1]) > 10.0, \
    "el destino tiene que depender de donde ESTA el dron, o no se esta usando el criterio"

print()
print("=" * 76)
print("3. EL CUADRO SE MANDA AL LLEGAR, NO EN EL CAMINO, Y UNA SOLA VEZ")
print("=" * 76)
# Sending on the way hands the ground the view it already had, which is the whole point of having
# flown. The tolerance is a radius because an aircraft holding position drifts.
# Fractions of the way, all of them OUTSIDE the tolerance: the closest is 9 m from the
# destination against a tolerance of 5. A point at 0.9 of the way is 3 m away, already arrived.
camino = [(ir[0] * f, ir[1] * f, ir[2]) for f in (0.2, 0.5, 0.7)]
for pos in camino:
    p._position = pos
    p._llego_al_rodeo()
    d = math.hypot(pos[0] - ir[0], pos[1] - ir[1])
    print("  a %5.1f m del destino -> %d cuadros" % (d, len(p._marcos)))
assert p._marcos == [], "en el camino no se manda nada: seria la misma vista de antes"
p._position = (ir[0] + RODEO_TOLERANCIA_M * 0.5, ir[1], ir[2])
p._llego_al_rodeo()
print("  dentro de la tolerancia de %.1f m -> %d cuadro" % (RODEO_TOLERANCIA_M, len(p._marcos)))
for _ in range(4):
    p._llego_al_rodeo()
print("  cuatro telemetrias mas estando ahi -> %d cuadros" % len(p._marcos))
assert len(p._marcos) == 1, "uno al llegar, y no uno por cada telemetria que entra"

print()
print("=" * 76)
print("4. SE PUEDE CANCELAR, Y SIN RADIO NI ALTURA SE NIEGA")
print("=" * 76)
p.rodear(None)
print("  rodear(None) -> rodeo %s" % p._rodeo)
assert p._rodeo is None, "una orden de vuelo tiene que poder cancelarse"
r2 = protocolo()
sin_radio = r2.rodear(0.0, 0.0)
print("  rodear sin radio ni altura -> %s, %d ordenes"
      % ("aceptado" if sin_radio else "rechazado", len(r2.provider.ordenes)))
assert not sin_radio and r2.provider.ordenes == [], \
    "el radio y la altura son decisiones de mision: esta capa no las inventa"
r3 = protocolo()
r3._position = None
print("  rodear sin saber donde esta -> %s" % ("aceptado" if r3.rodear(0.0, 0.0, RADIO, ALTURA) else "rechazado"))
assert not r3.rodear(0.0, 0.0, RADIO, ALTURA), \
    "sin su propia posicion no puede saber desde donde ya vio, ni a donde ir"

print()
print("=" * 76)
print("5. CON DOS DIRECCIONES YA VISTAS, ELIGE LA TERCERA Y NO REPITE NINGUNA")
print("=" * 76)
# The criterion is the minimum angle against EVERY ray already taken, so adding a second seen
# direction has to move the answer. If it did not, the function would be ignoring all but one.
def rayo_desde(az):
    pos = (RADIO * math.cos(math.radians(az)), RADIO * math.sin(math.radians(az)), ALTURA)
    d = np.array([-pos[0], -pos[1], -pos[2]], dtype=float)
    return tuple(d / np.linalg.norm(d))

una = next_best_viewpoint((0.0, 0.0, 0.0), [rayo_desde(0)], RADIO, ALTURA)
dos = next_best_viewpoint((0.0, 0.0, 0.0), [rayo_desde(0), rayo_desde(90)], RADIO, ALTURA)
print("  vista desde 0 deg            -> propone %3.0f deg (diversidad %.1f)"
      % (azimut(una[0][0], una[0][1]), una[1]))
print("  vistas desde 0 y 90 deg      -> propone %3.0f deg (diversidad %.1f)"
      % (azimut(dos[0][0], dos[0][1]), dos[1]))
assert abs(azimut(una[0][0], una[0][1]) - azimut(dos[0][0], dos[0][1])) > 5.0, \
    "una segunda vista ya tomada tiene que cambiar la respuesta"
assert dos[1] <= una[1] + 1e-6, \
    "con mas vistas cubiertas, lo mejor que queda no puede ser MAS diverso que antes"

print()
print("TODO OK")
