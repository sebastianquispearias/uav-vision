"""The operator's "no es" stops the drone from reporting that point again.

On flight 3 the drone reported eleven candidates and six of them were nobody. Neither of the
drone's own filters catches that kind of phantom: CLIP scores the crop and the physical size
check bounds the box, and both pass a thing that is person-shaped and person-sized. A human
looking at the crop settles it in one click, and until now that click only wrote a line to disk.
The drone kept sending the same wrong point every two seconds, over a 4G dongle, for the rest of
the flight, and the next session started knowing nothing.

The dangerous failure of a feature like this is the opposite one: silencing a real person because
she walked to where a refused object stood. Section 2 is that contrast, and it is the reason a
refusal carries an appearance and is rejected without one.

Run with: python tests/test_descarte_operador.py

The scene is a phantom, which is a bin the detector keeps calling a person, and somebody else
entirely.

THE SAME COORDINATES THE OPERATOR REFUSED WITH A DIFFERENT APPEARANCE MUST STILL BE REPORTED.
If that were silenced, the feature would be a way of LOSING people and the refusal would be
worse than no refusal at all.

The station never decodes the embedding: it hands back the base64 of float16 it received. If the
drone could only act on a refusal built from a vector in memory, the feature would work in a
test and not in the air.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(os.path.dirname(RAIZ), "gradys-embedded"))

import base64

import numpy as np

from uav_vision.vision_protocol import VisionProtocol

RNG = np.random.default_rng(23)


def unitario(semilla):
    v = np.random.default_rng(semilla).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def en_cable(v):
    """As the station hands it back: base64 of float16, never decoded on the ground."""
    return base64.b64encode(np.asarray(v, dtype=np.float16).tobytes()).decode("ascii")


def poi(x, y, emb=None, cls="person"):
    d = {"x": float(x), "y": float(y), "cls": cls, "n_obs": 40}
    if emb is not None:
        d["emb"] = emb
    return d


APARIENCIA_TACHO = unitario(3)
APARIENCIA_PERSONA = unitario(9)


def protocolo():
    p = VisionProtocol.__new__(VisionProtocol)
    p._descartados = []
    return p


print()
print("=" * 72)
print("1. UN PUNTO QUE EL OPERADOR RECHAZO DEJA DE REPORTARSE")
print("=" * 72)
p = protocolo()
tacho = poi(5.0, 5.0, APARIENCIA_TACHO)
print("  antes del click  :", "se reporta" if not p._fue_descartado(tacho) else "silenciado")
assert not p._fue_descartado(tacho), "sin veredictos no se puede silenciar nada"
ok = p.descartar(5.0, 5.0, "person", en_cable(APARIENCIA_TACHO))
print("  el operador hace click en 'no es' ->", "aceptado" if ok else "RECHAZADO")
print("  despues del click:", "silenciado" if p._fue_descartado(tacho) else "SE SIGUE REPORTANDO")
assert ok, "un descarte con apariencia tiene que aceptarse"
assert p._fue_descartado(tacho), "lo que el operador rechazo no puede volver a reportarse"

print()
print("=" * 72)
print("2. EL CONTRASTE QUE IMPORTA: UNA PERSONA EN ESE MISMO SITIO SI SE REPORTA")
print("=" * 72)
persona_ahi = poi(5.0, 5.0, APARIENCIA_PERSONA)
sin_huella = poi(5.0, 5.0)
otro_tacho_lejos = poi(40.0, 40.0, APARIENCIA_TACHO)
for nom, c in (("el tacho rechazado, mismo sitio      ", tacho),
               ("una PERSONA en el mismo sitio        ", persona_ahi),
               ("un candidato sin huella de apariencia", sin_huella),
               ("la misma apariencia a 40 m de ahi    ", otro_tacho_lejos)):
    print("  %s -> %s" % (nom, "silenciado" if p._fue_descartado(c) else "se reporta"))
assert not p._fue_descartado(persona_ahi), \
    "silenciar a una persona por estar donde habia un tacho seria perder gente, no filtrarla"
assert not p._fue_descartado(sin_huella), \
    "sin huella propia no se puede comparar apariencia, y entonces no se silencia"
assert not p._fue_descartado(otro_tacho_lejos), \
    "un descarte vale para un punto, no para toda la escena"

print()
print("=" * 72)
print("3. UN DESCARTE SIN APARIENCIA SE RECHAZA, Y NO DEJA NADA ANOTADO")
print("=" * 72)
q = protocolo()
rechazado = q.descartar(5.0, 5.0, "person", None)
print("  descartar sin plantilla  ->", "aceptado" if rechazado else "rechazado")
print("  descartes anotados       :", len(q._descartados))
print("  la persona de ese sitio  :", "silenciada" if q._fue_descartado(persona_ahi) else "se reporta")
assert not rechazado, "una posicion sola no alcanza para dejar de reportar"
assert q._descartados == [], "un descarte rechazado no puede quedar anotado a medias"
assert not q._fue_descartado(persona_ahi), "y entonces no silencia a nadie"

print()
print("=" * 72)
print("4. HAY VUELTA ATRAS: EL OPERADOR PUEDE RETIRAR SU JUICIO")
print("=" * 72)
print("  descartes antes :", len(p._descartados))
p.descartar(None)
print("  descartes despues:", len(p._descartados))
print("  el tacho         :", "silenciado" if p._fue_descartado(tacho) else "vuelve a reportarse")
assert p._descartados == [], "descartar(None) tiene que olvidar todo"
assert not p._fue_descartado(tacho), \
    "un juicio del operador tiene que poder retirarse, o es una trampa sin salida"

print()
print("=" * 72)
print("5. LA APARIENCIA LLEGA POR EL CABLE, NO POR REFERENCIA")
print("=" * 72)
r = protocolo()
r.descartar(5.0, 5.0, "person", en_cable(APARIENCIA_TACHO))
desde_lista = protocolo()
desde_lista.descartar(5.0, 5.0, "person", APARIENCIA_TACHO.tolist())
print("  desde base64 de float16:", "silencia" if r._fue_descartado(tacho) else "NO silencia")
print("  desde la lista de floats:", "silencia" if desde_lista._fue_descartado(tacho) else "NO silencia")
assert r._fue_descartado(tacho), "lo que manda la estacion tiene que bastar para silenciar"
assert desde_lista._fue_descartado(tacho), "y un llamador dentro del proceso tambien"

print()
print("TODO OK")
