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

from uav_vision.vision_protocol import (RODEO_PLAZO_S,  # noqa: E402
                                        RODEO_TOLERANCIA_M, VisionProtocol)
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

    ahora = 0.0

    def current_time(self):
        return self.ahora

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
print("=" * 76)
print("6. LA ESTACION ELIGE QUE AERONAVE VA, Y MANDA A OTRA QUE LA QUE YA MIRO")
print("=" * 76)
# The drone that reported the target is standing in the direction we already have, so sending it
# would fly something and collect the view we had. The card cannot decide this: it knows who
# reported and not who is connected.
sys.path.insert(0, os.path.join(RAIZ, "scripts", "banco_embedded"))
import gs_mapa  # noqa: E402

casos = [({"1": {}, "2": {}, "3": {}}, "1", "2", "hay otros: va el primero de los otros"),
         ({"1": {}}, "1", "1", "uno solo: va ese, que todavia puede moverse"),
         ({"2": {}, "3": {}}, "9", "2", "el que vio ya no esta: va cualquiera"),
         ({}, "1", None, "no hay nadie a quien mandar")]
for nodos, vio, esperado, nota in casos:
    got = gs_mapa.dron_para_rodear(nodos, vio)
    print("  drones %-18s vio el %-2s -> %-6s  %s"
          % (sorted(nodos) or "ninguno", vio, got if got else "ninguno", nota))
    assert got == esperado, "eligio mal: %s" % nota
assert gs_mapa.RODEO_RADIO_M > 0 and gs_mapa.RODEO_ALTURA_M > 0,     "la estacion tiene que traer un radio y una altura, porque la capa que vuela se niega sin ellos"
print("  radio %.0f m y altura %.0f m: decisiones de mision, en el instrumento del operador"
      % (gs_mapa.RODEO_RADIO_M, gs_mapa.RODEO_ALTURA_M))

print()
print("=" * 76)
print("7. UN BLANCO QUE CAMINA SE RECHAZA, Y ES LA LINEA MAS IMPORTANTE DEL ARCHIVO")
print("=" * 76)
# La orden lleva una COORDENADA, no un pixel, y por eso funciona con la camara sin ver nada en el
# momento del click: la posicion sale de la capa de identidad, que nunca olvida un candidato. Pero
# la posicion de quien camina envejece mientras el avion vuela, y esta capa se niega a extrapolar
# un movil mas de extrapolation_max_s = 3 s. Un vuelo de decenas de segundos esta un orden de
# magnitud afuera, asi que llegaria a fotografiar suelo vacio.
m = protocolo()
quieto = protocolo()
print("  blanco quieto -> %s" % ("vuela" if quieto.rodear(0.0, 0.0, RADIO, ALTURA) else "rechazado"))
print("  blanco movil  -> %s" % ("vuela" if m.rodear(0.0, 0.0, RADIO, ALTURA, movil=True) else "rechazado"))
print("  ordenes de movimiento al movil: %d" % len(m.provider.ordenes))
assert not m.rodear(0.0, 0.0, RADIO, ALTURA, movil=True),     "no se manda una aeronave a fotografiar donde alguien ESTUVO"
assert m.provider.ordenes == [], "y no se emite ninguna orden al negarse"
assert quieto._rodeo is not None, "el contraste: el mismo pedido sobre un quieto si vuela"

print()
print("=" * 76)
print("8. LA VUELTA COMPLETA AVANZA PUNTO A PUNTO Y TERMINA")
print("=" * 76)
# El lazo no es un controlador: el piloto automatico ya cierra el lazo de posicion contra su GPS.
# Lo unico nuestro es "llegue? entonces el siguiente", y que la lista se TERMINE.
o = protocolo()
o.rodear(0.0, 0.0, RADIO, ALTURA, puntos=6)
print("  puntos de la vuelta: %d   ordenes tras la primera: %d"
      % (len(o._rodeo["puntos"]), len(o.provider.ordenes)))
assert len(o._rodeo["puntos"]) == 6 and len(o.provider.ordenes) == 1,     "se manda UN punto a la vez, no los seis de golpe"
for paso in range(6):
    o._position = o._rodeo["ir_a"] if o._rodeo else o._position
    o.provider.ahora += 5.0
    o._llego_al_rodeo()
    print("  tramo %d -> %d cuadros, %d ordenes, vuelta %s"
          % (paso + 1, len(o._marcos), len(o.provider.ordenes),
             "en curso" if o._rodeo else "TERMINADA"))
assert len(o._marcos) == 6, "una foto en cada punto de la vuelta"
assert len(o.provider.ordenes) == 6, "seis puntos, seis ordenes, ni una mas"
assert o._rodeo is None, "una vuelta que no termina es una aeronave que nadie mando a parar"

print()
print("=" * 76)
print("9. UN PUNTO AL QUE NO LLEGA NO CUELGA LA SECUENCIA")
print("=" * 76)
# Una aeronave aparcada contra un punto que no puede alcanzar, mientras el operador espera, es
# peor que abandonar la vuelta y decirlo.
z = protocolo()
z.rodear(0.0, 0.0, RADIO, ALTURA, puntos=6)
z._position = (999.0, 999.0, ALTURA)          # nunca llega
z.provider.ahora = RODEO_PLAZO_S - 1.0
z._llego_al_rodeo()
print("  a %.0f s del plazo de %.0f s -> vuelta %s"
      % (z.provider.ahora, RODEO_PLAZO_S, "en curso" if z._rodeo else "abandonada"))
assert z._rodeo is not None, "antes del plazo sigue intentando"
z.provider.ahora = RODEO_PLAZO_S + 1.0
z._llego_al_rodeo()
print("  pasado el plazo                -> vuelta %s" % ("en curso" if z._rodeo else "abandonada"))
assert z._rodeo is None, "pasado el plazo se abandona"
assert len(z._marcos) == 0, "y no se manda una foto de un sitio al que no llego"

print()
print("=" * 76)
print("10. VOLAR Y VER SON DOS POSICIONES, Y EN EL BANCO NO SON LA MISMA")
print("=" * 76)
# Una placa sobre un escritorio lanza sus rayos desde la pose que tenia la GRABACION, porque ahi se
# tomaron las fotos, y su propio piloto automatico falso reporta el escritorio. La maniobra es sobre
# la aeronave, asi que pregunta por la posicion de VUELO. Sin esta separacion el dron recibe la
# orden, el piloto falso lo camina hasta el punto, y nadie se entera de que llego.
b = protocolo()
b.rodear(0.0, 0.0, RADIO, ALTURA)
destino = b._rodeo["ir_a"]
# Donde MIRA se queda donde estaba la grabacion, lejos del destino.
b._position = (0.0, -30.0, ALTURA)
# Donde VUELA es lo que diria el piloto automatico: ya llego.
b._posicion_para_vuelo = lambda: destino
b._llego_al_rodeo()
print("  mira desde (%.0f,%.0f) y vuela en (%.0f,%.0f) -> %d cuadro"
      % (b._position[0], b._position[1], destino[0], destino[1], len(b._marcos)))
assert len(b._marcos) == 1,     "la llegada se mide con la posicion de VUELO, o en el banco la maniobra no cierra nunca"

c = protocolo()
c.rodear(0.0, 0.0, RADIO, ALTURA)
d2 = c._rodeo["ir_a"]
c._position = d2                      # donde mira coincide con el destino
c._posicion_para_vuelo = lambda: (0.0, -30.0, ALTURA)   # pero todavia no llego
c._llego_al_rodeo()
print("  el contraste: mira en el destino pero vuela lejos -> %d cuadros" % len(c._marcos))
assert c._marcos == [],     "si se midiera con la posicion de la camara, una placa quieta creeria haber llegado siempre"

print()
print("TODO OK")
