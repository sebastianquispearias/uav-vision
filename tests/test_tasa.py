"""
Gates for the frame rate, after it turned out the configured one was never the real one.

Measured on the Pi on 25ago: 2.31 frames per second against 3.00 configured, on an empty
scene. Two faults stacked. The loop rescheduled itself as `now + period`, so the real interval
was `work + period` and the rate was always below the one asked for. And the identity layer
scaled every maturity threshold by the DECLARED rate, so running slower than declared
stretched what "36 seconds of evidence" meant -- to about 47.

Neither shows up in a test that drives the clock itself, which is what every earlier test does:
they advance time in fixed steps and call handle_timer, so the work is free and the loop can
never fall behind. The harness here is event-driven and the camera BURNS CLOCK, which is the
only way the fault is visible at all.

Run with: python tests/test_tasa.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
_GRADYS = os.path.join(os.path.dirname(_HERE), "gradys-embedded")
if os.path.isdir(_GRADYS):
    sys.path.insert(0, _GRADYS)

from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.identity import IdentidadIncremental
from uav_vision.vision_protocol import VisionProtocol

from test_vision_protocol import FakeProvider  # noqa: E402

FALLOS = []


def revisar(condicion, descripcion, detalle=""):
    print("  [%s] %s%s" % ("ok  " if condicion else "FALLA", descripcion,
                           (" -- " + detalle) if detalle else ""))
    if not condicion:
        FALLOS.append(descripcion)


class CamaraQueTarda:
    """A camera whose work costs wall-clock time, because on real hardware it does."""

    def __init__(self, provider, trabajo_s, detecciones=()):
        self.provider = provider
        self.trabajo_s = trabajo_s
        self.detecciones = list(detecciones)
        self.camara = ARDUCAM_MODULE_3

    def ver_alvo(self, pos, yaw):
        self.provider.time += self.trabajo_s
        return [dict(d) for d in self.detecciones]


DETECCION = {
    "px": ARDUCAM_MODULE_3.image_width / 2.0,
    "py": ARDUCAM_MODULE_3.image_height * 0.62,
    "conf": 0.88,
    "track_id": 1,
}


def correr(trabajo_s, periodo_s, hasta_s, detecciones=(), fps_declarado=None,
           dur_reporte_s=36.0):
    """Event-driven: the clock only moves to the next due timer, or forward through work."""
    provider = FakeProvider()
    camara = CamaraQueTarda(provider, trabajo_s, detecciones)
    fps = fps_declarado if fps_declarado is not None else 1.0 / periodo_s
    Protocolo = VisionProtocol.with_config(
        camera=camara, pitch_deg=-55.0, yaw_source=lambda: 0.0,
        see_period_s=periodo_s, report_period_s=2.0,
        identidad=IdentidadIncremental(radio_fusion_m=3.5, fps=fps,
                                       dur_reporte_s=dur_reporte_s),
        reportar_preliminares=True)
    protocolo = Protocolo.instantiate(provider)
    protocolo.initialize()
    protocolo.handle_telemetry(Telemetry(current_position=(0.0, 0.0, 35.0)))

    t_maduro = None
    while provider.timers:
        cuando, nombre = min(provider.timers)
        if cuando > hasta_s:
            break
        provider.timers.remove((cuando, nombre))
        provider.time = max(provider.time, cuando)
        protocolo.handle_timer(nombre)
        if t_maduro is None and provider.sent:
            m = json.loads(provider.sent[-1].message)
            if any(p.get("maduro") for p in m.get("pois", [])):
                t_maduro = provider.time
    mensajes = [json.loads(c.message) for c in provider.sent]
    return protocolo, provider, mensajes, t_maduro


# ================================================ 1. la cadencia es la pedida
print("=" * 68)
print("1. El lazo entrega la cadencia configurada, no 'periodo + trabajo'")
print("=" * 68)

for trabajo, periodo, esperado in [(0.05, 0.25, 4.0), (0.196, 0.333, 3.0)]:
    p, prov, msgs, _ = correr(trabajo, periodo, hasta_s=30.0)
    real = p._frames_seen / prov.time
    antes = 1.0 / (periodo + trabajo)      # what the old fixed-delay loop would have given
    revisar(abs(real - esperado) < 0.12,
            "trabajo %.0f ms, periodo %.0f ms -> %.2f FPS (pedidos %.1f)"
            % (trabajo * 1000, periodo * 1000, real, esperado),
            "el lazo viejo habria dado %.2f" % antes)
    revisar(p._slots_perdidos == 0, "   y sin perder una sola ranura")

# ============================================ 2. saturado, lo dice en vez de mentir
print()
print("=" * 68)
print("2. Cuando NO puede seguir el ritmo, lo declara")
print("=" * 68)

p, prov, msgs, _ = correr(0.45, 0.333, hasta_s=30.0)
real = p._frames_seen / prov.time
revisar(p._slots_perdidos > 0, "cuenta las ranuras que el trabajo se comio",
        "%d perdidas" % p._slots_perdidos)
ultimo = msgs[-1]
revisar(ultimo.get("fps_real") is not None, "el mensaje lleva la tasa REAL")
revisar(abs(ultimo["fps_real"] - real) < 0.2,
        "y la tasa que declara es la que entrego",
        "dice %.2f, entrego %.2f" % (ultimo["fps_real"], real))
revisar(ultimo.get("slots_perdidos") == p._slots_perdidos,
        "y las perdidas viajan tambien: desde tierra se ve un dron saturado")
revisar(real < 3.0, "   (no puede correr mas rapido que su propio trabajo)",
        "%.2f FPS con 450 ms por frame" % real)


# ================================== 3. la madurez ya no depende de la tasa
print()
print("=" * 68)
print("3. '36 s' son 36 s, corra el lazo a la velocidad que corra")
print("=" * 68)

# The whole point: same declared fps, wildly different delivered rates. Before the fix the
# maturity moment tracked the delivered rate; now it tracks the clock.
for trabajo, etiqueta in [(0.05, "lazo holgado"), (0.45, "lazo saturado")]:
    p, prov, msgs, t_mad = correr(trabajo, 0.333, hasta_s=90.0,
                                  detecciones=[DETECCION], fps_declarado=3.0)
    entregados = p._frames_seen / prov.time
    if t_mad is None:
        revisar(False, "%s: nunca madura" % etiqueta)
        continue
    revisar(abs(t_mad - 36.0) < 2.5,
            "%s (%.2f FPS reales): madura a los %.1f s" % (etiqueta, entregados, t_mad),
            "objetivo 36.0 s")

# And the fallback still works for callers replaying recorded data with no clock at all.
ident = IdentidadIncremental(radio_fusion_m=3.5, fps=4.0, dur_reporte_s=10.0)
for f in range(200):
    ident.observar(frame=f, track_id=1, impacto_xy=(5.0, 5.0), conf=0.9)
c = ident.candidatos()
revisar(bool(c) and c[0]["maduro"],
        "sin reloj, la madurez sigue midiendose por span de frames (replays)")

print()
if FALLOS:
    print("FALLARON %d:" % len(FALLOS))
    for f in FALLOS:
        print("  -", f)
    sys.exit(1)
print("TODO OK")
