"""End-to-end test of VisionProtocol without the full simulator.

A fake provider plays the role of gradys-sim: it keeps the clock, the
timer queue and the outbox. The drone orbits a target for 60 simulated
seconds; the simulated camera adds pixel noise (sigma = C/conf), so the
protocol has to earn its estimate through RANSAC, not get it for free.

PASS criterion: the last reported POI lands within 1.5 m of the true
target. That is the full loop -- detect, back-project, intersect
ground, fuse, report -- working end to end.

Run with:   python tests/test_vision_protocol.py     (from lac/uav_vision)
Needs gradys-embedded importable; the sys.path fallback below picks up
the sibling clone at ../gradys-embedded.
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

from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.camera import CamaraSimulada
from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.vision_protocol import VisionProtocol


class FakeProvider:
    """Minimal stand-in for the runner: clock, timers, outbox."""

    def __init__(self):
        self.time = 0.0
        self.timers = []          # (fire_at, name)
        self.sent = []            # CommunicationCommand list

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
        """Fire every timer whose timestamp has passed, oldest first."""
        due = sorted(t for t in self.timers if t[0] <= self.time)
        self.timers = [t for t in self.timers if t[0] > self.time]
        for _, name in due:
            protocol.handle_timer(name)


ALVO = (3.0, -2.0, 0.0)           # true target on the ground
RADIO = 20.0                      # orbit radius, m
ALTURA = 35.0                     # flight altitude, m (matches real flights)
PITCH = -55.0                     # camera mount pitch since 02ago2026

state = {"yaw": 0.0}


def yaw_actual():
    return state["yaw"]


def drone_pose(t):
    """Orbit around the target, camera always facing it."""
    a = 2 * math.pi * t / 60.0    # one lap per minute
    x = ALVO[0] + RADIO * math.sin(a)
    y = ALVO[1] + RADIO * math.cos(a)
    dx, dy = ALVO[0] - x, ALVO[1] - y
    yaw = math.degrees(math.atan2(dx, dy)) % 360.0   # 0=North, clockwise
    return (x, y, ALTURA), yaw


print("=" * 64)
print("VisionProtocol: vuelo sintetico de 60 s, camara con ruido")
print("=" * 64)

camara = CamaraSimulada(
    alvo=ALVO,
    pitch_deg=PITCH,
    camara=ARDUCAM_MODULE_3,
    rng=np.random.default_rng(1),
)
Protocolo = VisionProtocol.with_config(
    camera=camara,
    pitch_deg=PITCH,
    yaw_source=yaw_actual,
)

provider = FakeProvider()
protocol = Protocolo.instantiate(provider)
protocol.initialize()

DT = 0.2
for step in range(int(60.0 / DT)):
    provider.time = step * DT
    pos, yaw = drone_pose(provider.time)
    state["yaw"] = yaw
    protocol.handle_telemetry(Telemetry(current_position=pos))
    provider.fire_due(protocol)
protocol.finish()

assert provider.sent, "el protocolo no reporto ningun POI"

reportes = [json.loads(c.message) for c in provider.sent]
ultimo = reportes[-1]["pois"][0]
error = math.hypot(ultimo["x"] - ALVO[0], ultimo["y"] - ALVO[1])

print(f"  reportes emitidos : {len(reportes)}")
print(f"  frames procesados : {reportes[-1]['frames_seen']}")
print(f"  observaciones     : {ultimo['n_obs']} (inliers: {ultimo['n_inliers']})")
print(f"  objetivo real     : ({ALVO[0]:.2f}, {ALVO[1]:.2f})")
print(f"  POI reportado     : ({ultimo['x']:.2f}, {ultimo['y']:.2f})")
print(f"  error             : {error:.2f} m")

assert error < 1.5, f"error {error:.2f} m demasiado grande"
assert ultimo["n_obs"] >= 100, "muy pocas observaciones acumuladas"

print()
print("TODO OK")
