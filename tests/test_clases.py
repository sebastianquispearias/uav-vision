"""
Empirical gates for the object class, from the detection to the reported POI.

Each section is a contrast, not a description: the same scene twice, changing only the class,
so the rule is visible in the numbers rather than asserted in a comment.

  1. The class travels, and it is a vote: one flipped label does not rename a track.
  2. Class veto: two things in the SAME place with the SAME look stay two candidates when
     their classes differ -- and merge when they agree. Same scene, one field changed.
  3. Silence is not disagreement: with no class at all, association is what it always was.
  4. Per-class fusion radius: two cars 3 m apart merge under the radius tuned for people and
     survive under their own.
  5. End to end through VisionProtocol: the report names the class, and the ground point of a
     car is pushed from the near bumper to the middle of the footprint.

Run with: python tests/test_clases.py
Needs gradys-embedded importable for section 5; the sys.path fallback picks up the sibling clone.
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
_GRADYS = os.path.join(os.path.dirname(_HERE), "gradys-embedded")
if os.path.isdir(_GRADYS):
    sys.path.insert(0, _GRADYS)

import numpy as np

from uav_vision.identity import IncrementalIdentity, radii_by_class

RNG = np.random.default_rng(11)
FPS = 5.0
SIGMA = 0.3          # ground projection noise, m


def emb_de(base):
    v = np.random.default_rng(base).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def ruido():
    return RNG.normal(0, SIGMA, size=2)


def poblar(ident, tid, centro, frames, cls, emb=None):
    """One track: sightings of the same thing, over the given frame range."""
    for f in frames:
        ident.observe(f, tid, np.asarray(centro) + ruido(), 0.8,
                      emb=None if emb is None else emb + 0.02 * RNG.normal(size=512),
                      cls=cls)


print("=" * 70)
print("1. LA CLASE VIAJA, Y ES UN VOTO: una etiqueta suelta no renombra la pista")
print("=" * 70)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
for f in range(200):
    # the detector says 'van' on three frames out of two hundred
    ident.observe(f, 1, np.array([4.0, 1.0]) + ruido(), 0.8,
                  cls="van" if f % 67 == 0 else "car")
c = ident.candidates()
assert len(c) == 1, "deberia haber 1 candidato, hay %d" % len(c)
print("  197 votos 'car' + 3 votos 'van' -> cls=%r" % c[0]["cls"])
assert c[0]["cls"] == "car", "el voto no gano: la clase reportada es la ultima etiqueta"

print()
print("=" * 70)
print("2. VETO DE CLASE: mismo sitio, mismo aspecto, distinta clase")
print("=" * 70)
MISMO = (2.0, -3.0)
E = emb_de(3)


def dos_pistas(cls_a, cls_b):
    """Two tracks at the same point, never seen together, alike in appearance."""
    id2 = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
    poblar(id2, 10, MISMO, range(0, 200), cls_a, E)
    poblar(id2, 11, MISMO, range(200, 400), cls_b, E)
    return id2.candidates()


mezcla = dos_pistas("person", "car")
igual = dos_pistas("car", "car")
print("  person + car -> %d candidates %s" % (len(mezcla), [p["cls"] for p in mezcla]))
print("  car    + car -> %d candidato(s) %s" % (len(igual), [p["cls"] for p in igual]))
assert len(mezcla) == 2, "una persona y un coche se fusionaron en un solo POI"
assert len(igual) == 1, ("el control fallo: con la MISMA clase tambien salen dos, asi que el "
                         "test de arriba no prueba nada sobre la clase")

print()
print("=" * 70)
print("3. SIN CLASE, TODO IGUAL QUE ANTES: None no es desacuerdo")
print("=" * 70)
sin = dos_pistas(None, None)
media = dos_pistas(None, "car")
print("  sin clase + sin clase -> %d candidato(s), cls=%r" % (len(sin), sin[0]["cls"]))
print("  sin clase + car       -> %d candidato(s), cls=%r" % (len(media), media[0]["cls"]))
assert len(sin) == 1 and sin[0]["cls"] is None, "una camera sin clase cambio de comportamiento"
assert len(media) == 1, "callar no puede impedir una fusion: None significa desconocido"

print()
print("=" * 70)
print("4. RADIO POR CLASE: dos coches aparcados a 3.0 m")
print("=" * 70)


def dos_coches(radios):
    id3 = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS,
                              fusion_radius_by_class=radios)
    poblar(id3, 20, (0.0, 0.0), range(0, 200), "car")     # sin emb: decide la distancia
    poblar(id3, 21, (3.0, 0.0), range(200, 400), "car")
    return id3.candidates()


con_radio_persona = dos_coches(None)
con_radio_coche = dos_coches(radii_by_class(3.5))
print("  radio 3.5 m (ajustado a personas) -> %d candidato(s)" % len(con_radio_persona))
print("  radio 2.5 m (ancho de plaza)      -> %d candidatos %s"
      % (len(con_radio_coche), [(p["x"], p["y"]) for p in con_radio_coche]))
assert len(con_radio_persona) == 1, ("el control fallo: con 3.5 m los dos coches deberian "
                                     "fusionarse, que es el bug que esto arregla")
assert len(con_radio_coche) == 2, "con el radio del coche siguen fusionandose"

print()
print("=" * 70)
print("5. DE PUNTA A PUNTA: el reporte nombra la clase y corrige el punto de contacto")
print("=" * 70)

from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.camera import SimulatedCamera
from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.vision_protocol import GROUND_EXTENT_M, VisionProtocol


class FakeProvider:
    """Minimal stand-in for the runner: clock, timers, outbox."""

    def __init__(self):
        self.time = 0.0
        self.timers = []
        self.sent = []

    def schedule_timer(self, timer, timestamp):
        self.timers.append((timestamp, timer))

    def cancel_timer(self, timer):
        self.timers = [t for t in self.timers if t[1] != timer]

    def send_communication_command(self, command):
        self.sent.append(command)

    def current_time(self):
        return self.time

    def get_id(self):
        return 7

    tracked_variables = {}

    def fire_due(self, protocol):
        due = sorted(t for t in self.timers if t[0] <= self.time)
        self.timers = [t for t in self.timers if t[0] > self.time]
        for _, name in due:
            protocol.handle_timer(name)


ALVO = (0.0, 0.0, 0.0)       # the thing on the ground
DRON = (0.0, -20.0, 35.0)    # hovering 20 m south of it
PITCH = -55.0


def volar(cls):
    """The same hover, twice, changing only what the detector calls the target."""
    camera = SimulatedCamera(target=ALVO, pitch_deg=PITCH, camera=ARDUCAM_MODULE_3,
                             pixel_noise=False, cls=cls)
    Protocolo = VisionProtocol.with_config(camera=camera, pitch_deg=PITCH,
                                           yaw_source=lambda: 0.0)
    provider = FakeProvider()
    protocol = Protocolo.instantiate(provider)
    protocol.initialize()
    for paso in range(1, 41):                       # 10 s at 4 Hz
        provider.time = paso * 0.25
        protocol.handle_telemetry(Telemetry(current_position=DRON))
        provider.fire_due(protocol)
    protocol.finish()
    reportes = [json.loads(c.message) for c in provider.sent]
    con_poi = [r for r in reportes if r["pois"]]
    assert con_poi, "no se reporto ningun POI para cls=%r" % cls
    return con_poi[-1]["pois"][0]


persona = volar("person")
coche = volar("car")
d_persona = math.hypot(persona["x"] - DRON[0], persona["y"] - DRON[1])
d_coche = math.hypot(coche["x"] - DRON[0], coche["y"] - DRON[1])
print("  person -> cls=%r  POI (%.2f, %.2f)  a %.2f m del dron"
      % (persona["cls"], persona["x"], persona["y"], d_persona))
print("  car    -> cls=%r     POI (%.2f, %.2f)  a %.2f m del dron"
      % (coche["cls"], coche["x"], coche["y"], d_coche))
esperado = 0.5 * GROUND_EXTENT_M["car"]
print("  desplazamiento: %.2f m   esperado %.2f m (media extension de un coche)"
      % (d_coche - d_persona, esperado))
assert persona["cls"] == "person" and coche["cls"] == "car", "el reporte no lleva la clase"
assert abs((d_coche - d_persona) - esperado) < 0.02, (
    "la correccion del punto de contacto no es media extension a lo largo de la vista")
assert abs(d_persona - math.hypot(DRON[0], DRON[1])) < 0.05, (
    "la persona se movio: la correccion no debe tocar las clases fuera de la tabla")

print()
print("TODO OK")
