"""The vision protocol runs on gradys-core, the extracted GrADyS interface, without a change.

gradys-core is the protocol interface the simulator and gradys-embedded are meant to share. Its
legacy adapter promises that a protocol written against the old IProtocol runs on the new runtime
unmodified. This checks the promise on this protocol, the hard way: the 02-ago flight, replayed
through gradys-core's FakeHost, must produce the same reports, byte for byte, as the replay's own
provider. The clock is driven the way the replay drives it -- jump to the frame's time, then fire
the timers that were due at it -- so what is compared is the adapter, not two fake clocks.

One contrast is kept on purpose. FakeHost wraps a legacy protocol automatically only when it
subclasses gradys_core.legacy.IProtocol; this protocol subclasses gradys_embedded's copy, a
different class with the same contract, so installed bare it is refused and has to be wrapped
with legacy(). If gradys-core ever learns to recognise the embedded interface, that section fails
and says so, and the wrapping can go.

Needs the gradys-core clone next to this repository; without it the test is skipped.

Run: python tests/test_gradys_core.py

There is no sensor on a desk, and the see loop still has to run.
"""
import contextlib
import heapq
import io
import json
import os
import runpy
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
LAC = os.path.dirname(RAIZ)
CORE = os.path.join(LAC, "gradys-core", "src")
if not os.path.isdir(CORE):
    print("gradys-core no esta clonado junto a este repo: test saltado")
    sys.exit(0)
sys.path[:0] = [RAIZ, os.path.join(LAC, "gradys-embedded"), CORE]
os.environ.setdefault("UAV_VISION_DATOS", os.path.join(RAIZ, "demo", "data"))
os.environ.pop("UAV_VISION_GS", None)

from gradys_core.commands import Broadcast
from gradys_core.events import TimerEvent
from gradys_core.legacy import is_legacy, legacy
from gradys_core.testing import FakeHost

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol


class Modelo:
    names = dict(enumerate(["pedestrian", "people", "bicycle", "car", "van", "truck",
                            "tricycle", "awning-tricycle", "bus", "motor"]))


def camara_de_escritorio():
    c = OnboardCamera(model="stand-in")
    c._yolo = Modelo()
    c.detect = lambda pos, yaw: []
    return c


def ultimo(host):
    return json.loads(list(host.commands_of(Broadcast))[-1].payload)


print("=" * 70)
print("1. CONTRASTE: instalado tal cual, sin legacy()")
print("=" * 70)
P = VisionProtocol.with_config(camera=camara_de_escritorio(), pitch_deg=-55.0,
                               yaw_source=lambda: 0.0)
assert not is_legacy(P), "gradys-core ya reconoce el IProtocol de gradys_embedded: sobra legacy()"
try:
    FakeHost(node_id=1).install(P)
    raise AssertionError("FakeHost instalo el protocolo sin envolver: sobra legacy()")
except TypeError as e:
    print("  rechazado sin envolver:", str(e)[:80])

print("=" * 70)
print("2. CON legacy(): reporta, toma telemetria y obedece una orden")
print("=" * 70)
cam = camara_de_escritorio()
P = VisionProtocol.with_config(camera=cam, pitch_deg=-55.0, yaw_source=lambda: 0.0)
host = FakeHost(node_id=1)
host.install(legacy(P))
host.advance(2.05)
m = ultimo(host)
assert m["type"] == "vision_poi" and m["latido"] is True and m["pos"] is None, m
host.deliver_telemetry((1.0, 2.0, 30.0))
host.advance(2.0)
assert ultimo(host)["pos"] == [1.0, 2.0, 30.0], ultimo(host)
host.deliver_packet(json.dumps({"type": "vision_buscar", "clases": ["car"], "v": 1,
                                "epoca": "e1"}), source=0)
host.advance(2.0)
assert cam.classes == {"car"} and ultimo(host)["buscando"]["clases"] == ["car"], ultimo(host)
print("  latido, posicion [1.0, 2.0, 30.0] y orden car: los tres llegan por gradys-core")

print("=" * 70)
print("3. EL VUELO ENTERO: los mismos reportes, byte a byte")
print("=" * 70)
REPLAY = os.path.join(RAIZ, "scripts", "replay_vuelo3.py")
sys.argv = [REPLAY]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(REPLAY, run_name="__main__")
referencia = g["reportes"]

estado = {"yaw": 0.0}
camara = g["CamaraReplay"](g["por_frame"], g["camara_cfg"])
P = VisionProtocol.with_config(
    camera=camara, pitch_deg=g["PITCH"], yaw_source=lambda: estado["yaw"],
    see_period_s=0.1, report_period_s=2.0,
    identity=IncrementalIdentity(fusion_radius_m=3.5, fps=g["fps_replay"]))


class HostReplay(FakeHost):
    """FakeHost with the replay's clock: jump to a frame's time, then fire what was due at it."""

    def saltar(self, t):
        self._now = t
        vencidos = []
        while self._timers and self._timers[0][0] <= t:
            vencidos.append(heapq.heappop(self._timers))
        for _, seq, tag in sorted(vencidos):
            pendientes = self._by_tag.get(tag)
            if pendientes is not None and seq in pendientes:
                pendientes.remove(seq)
                if not pendientes:
                    self._by_tag.pop(tag, None)
            if seq in self._cancelled:
                self._cancelled.discard(seq)
                continue
            self.binding.deliver(TimerEvent(tag))


host = HostReplay(node_id=3, start_time=0.0)
host.install(legacy(P))
for f in g["frames_aire"]:
    pose = g["poses"][f]
    t = float(pose["t_mono"]) - g["t0"]
    x, y = g["enu"](float(pose["lat"]), float(pose["lng"]))
    estado["yaw"] = float(pose["yaw"])
    host._now = t
    host.deliver_telemetry((x, y, float(pose["alt_agl"])))
    camara.set_frame(f)
    host.saltar(t)
host.binding.stop()
sobre_core = [json.loads(c.payload) for c in host.commands_of(Broadcast)]

iguales = sum(1 for a, b in zip(referencia, sobre_core) if a == b)
print("  replay: %d reportes | gradys-core: %d reportes | iguales: %d"
      % (len(referencia), len(sobre_core), iguales))
assert len(referencia) == len(sobre_core) == iguales, "el adaptador cambio lo que reporta el protocolo"
p = sobre_core[-1]["pois"][0]
print("  ultimo POI sobre gradys-core: %d obs en (%.2f, %.2f)" % (p["n_obs"], p["x"], p["y"]))
print("test_gradys_core OK")
