"""
Gates for the two holes the first full rehearsal exposed (2026-08-25).

Both were invisible to every existing test, because both only appear when the whole chain
runs at once. The camera computed a crop for every detection and nothing downstream had a
place to put it, so it was encoded and dropped -- and the drone, having found nothing, said
nothing, which from the ground is indistinguishable from a drone that has died.

What is asserted here is deliberately end-of-pipe: not that the crop is stored, but that it
comes out the far end, in the message, decodable. A test on the storage would have passed
happily all along while the field saw nothing.

Run with: python tests/test_recorte.py
"""
import base64
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
_GRADYS = os.path.join(os.path.dirname(_HERE), "gradys-embedded")
if os.path.isdir(_GRADYS):
    sys.path.insert(0, _GRADYS)

import numpy as np

from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.identity import IdentidadIncremental
from uav_vision.vision_protocol import VisionProtocol

from test_vision_protocol import FakeProvider  # noqa: E402  (same directory)

FALLOS = []


def revisar(condicion, descripcion, detalle=""):
    marca = "ok  " if condicion else "FALLA"
    print("  [%s] %s%s" % (marca, descripcion, (" -- " + detalle) if detalle else ""))
    if not condicion:
        FALLOS.append(descripcion)


# ============================================================ 1. la identidad
print("=" * 64)
print("1. El recorte sobrevive la fusion, y gana el de la vista mas clara")
print("=" * 64)

BORROSO, NITIDO = b"\xff\xd8jpeg-borroso", b"\xff\xd8jpeg-nitido"

ident = IdentidadIncremental(radio_fusion_m=3.5, fps=4.0)
# Two tracks on the same spot: BoT-SORT breaks a track and renumbers it all the time, which
# is exactly the case where the two halves carry different-quality photographs.
for f in range(80):
    ident.observar(frame=f, track_id=1, impacto_xy=(10.0, 5.0), conf=0.40, recorte=BORROSO)
for f in range(80, 160):
    ident.observar(frame=f, track_id=2, impacto_xy=(10.2, 5.1), conf=0.91, recorte=NITIDO)

cands = ident.candidatos(preliminares=True)
revisar(len(cands) == 1, "las dos pistas fusionan en un candidato", "n=%d" % len(cands))
if cands:
    c = cands[0]
    revisar("recorte" in c, "el candidato trae el campo 'recorte'")
    revisar(c.get("recorte") == NITIDO,
            "sobrevive el recorte de la deteccion mas confiada",
            "conf 0.91 gana a 0.40")

# A track with no crop must not invent one, and must not crash on the comparison.
ident_sin = IdentidadIncremental(radio_fusion_m=3.5, fps=4.0)
for f in range(80):
    ident_sin.observar(frame=f, track_id=1, impacto_xy=(1.0, 1.0), conf=0.8)
sin = ident_sin.candidatos(preliminares=True)
revisar(bool(sin) and sin[0].get("recorte") is None,
        "sin recorte el campo viaja como None, sin romper nada")


# ============================================================= 2. el mensaje
print()
print("=" * 64)
print("2. El recorte llega al mensaje, y el silencio se convierte en latido")
print("=" * 64)

RECORTE = b"\xff\xd8\xff\xe0" + b"bytes-crudos-de-un-jpeg" * 4


class CamaraDeMentira:
    """Stands in for CamaraArduCam: returns whatever the test tells it to."""

    def __init__(self):
        self.camara = ARDUCAM_MODULE_3
        self.detecciones = []

    def ver_alvo(self, pos, yaw):
        return [dict(d) for d in self.detecciones]


def correr(camara, segundos, reportar_preliminares=True):
    Protocolo = VisionProtocol.with_config(
        camera=camara,
        pitch_deg=-55.0,
        yaw_source=lambda: 0.0,
        identidad=IdentidadIncremental(radio_fusion_m=3.5, fps=4.0),
        reportar_preliminares=reportar_preliminares,
    )
    provider = FakeProvider()
    protocolo = Protocolo.instantiate(provider)
    protocolo.initialize()
    from gradys_embedded.protocol.messages.telemetry import Telemetry

    paso = 0.1
    for k in range(int(segundos / paso)):
        provider.time = k * paso
        protocolo.handle_telemetry(Telemetry(current_position=(0.0, 0.0, 35.0)))
        provider.fire_due(protocolo)
    return [json.loads(c.message) for c in provider.sent]


# -- nothing in frame: the drone must still speak --------------------------
vacia = CamaraDeMentira()
mensajes = correr(vacia, 12.0)
revisar(len(mensajes) > 0, "con la camara vacia igual se emiten mensajes",
        "%d en 12 s" % len(mensajes))
revisar(bool(mensajes) and all(m.get("latido") for m in mensajes),
        "todos vienen marcados como latido")
revisar(bool(mensajes) and all(m.get("pois") == [] for m in mensajes),
        "el latido no inventa POIs")

# -- something in frame, carrying a crop -----------------------------------
vista = CamaraDeMentira()
vista.detecciones = [{
    "px": ARDUCAM_MODULE_3.image_width / 2.0,
    "py": ARDUCAM_MODULE_3.image_height * 0.62,
    "conf": 0.88,
    "track_id": 1,
    "recorte": RECORTE,
}]
mensajes = correr(vista, 20.0)
con_pois = [m for m in mensajes if m.get("pois")]
revisar(bool(con_pois), "con una deteccion sostenida sale al menos un POI",
        "%d de %d mensajes" % (len(con_pois), len(mensajes)))

if con_pois:
    poi = con_pois[-1]["pois"][0]
    revisar(not con_pois[-1].get("latido"), "un mensaje con POIs no se marca como latido")
    revisar("recorte" in poi, "el POI del mensaje trae el recorte")
    crudo = poi.get("recorte")
    revisar(isinstance(crudo, str), "viaja como texto, no como bytes",
            "tipo %s" % type(crudo).__name__)
    try:
        vuelta = base64.b64decode(crudo)
    except Exception as exc:
        vuelta = None
        print("      no decodifica:", exc)
    revisar(vuelta == RECORTE, "decodifica byte a byte al JPEG original")
    # The whole architecture was sized around this staying a packet, not a video stream.
    revisar(len(json.dumps(con_pois[-1])) < 64_000,
            "el mensaje entero sigue siendo chico",
            "%d bytes" % len(json.dumps(con_pois[-1])))

# ================================================== 3. el marco de coordenadas
print()
print("=" * 64)
print("3. El dron declara el marco en que estan medidos sus metros")
print("=" * 64)

# The bench provider has no origin -- like every desk run ever done.
revisar(bool(mensajes) and "origen_gps" in mensajes[-1],
        "el campo viaja siempre, tambien cuando no hay origen")
revisar(bool(mensajes) and mensajes[-1].get("origen_gps") is None,
        "sin origen en el runtime, viaja como None y no rompe nada")


class ProveedorConMarco(FakeProvider):
    """The embedded runtime's provider carries the frame; IProvider does not declare it."""
    origin_gps_coordinates = (-22.9793, -43.2325, 0.0)


def correr_con(provider_cls, camara, segundos):
    Protocolo = VisionProtocol.with_config(
        camera=camara, pitch_deg=-55.0, yaw_source=lambda: 0.0,
        identidad=IdentidadIncremental(radio_fusion_m=3.5, fps=4.0),
        reportar_preliminares=True)
    provider = provider_cls()
    protocolo = Protocolo.instantiate(provider)
    protocolo.initialize()
    from gradys_embedded.protocol.messages.telemetry import Telemetry
    for k in range(int(segundos / 0.1)):
        provider.time = k * 0.1
        protocolo.handle_telemetry(Telemetry(current_position=(0.0, 0.0, 35.0)))
        provider.fire_due(protocolo)
    return [json.loads(c.message) for c in provider.sent]


con_marco = correr_con(ProveedorConMarco, CamaraDeMentira(), 8.0)
revisar(bool(con_marco), "el runtime con marco emite mensajes")
if con_marco:
    og = con_marco[-1].get("origen_gps")
    revisar(og == [-22.9793, -43.2325, 0.0],
            "el origen del runtime llega al mensaje, como numeros",
            repr(og))
    revisar(all(isinstance(v, float) for v in (og or [])),
            "y como float, no como tupla ni string")
# The beat carries it too: the ground station learns the frame before the first find,
# which is the only order that works -- a sweep may report nothing for minutes.
revisar(bool(con_marco) and con_marco[0].get("latido") is True
        and con_marco[0].get("origen_gps") is not None,
        "el LATIDO ya lo lleva: la GS sabe el marco antes del primer hallazgo")

print()
if FALLOS:
    print("FALLARON %d:" % len(FALLOS))
    for f in FALLOS:
        print("  -", f)
    sys.exit(1)
print("TODO OK")
