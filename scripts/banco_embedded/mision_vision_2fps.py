"""Bench mission at 2 FPS against the REAL uav_api telemetry (via safety proxy).

2 FPS because the Pi is powered from the drone battery through the 5 A UBEC,
whose voltage collapses above ~5 FPS — the mission runs well inside that
margin. Yaw comes from the REAL uav_api on :8000 (real Pixhawk compass).
"""

import os

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import UavApiBattery, UavApiYaw, VisionProtocol

# The pack is read from BANCO_BATERIA_URL when it is set, and from the same service as the yaw
# when it is not. One variable, because on the bench the two do NOT come from the same place.
#
# The protocol never had to change for this: it takes a battery_source and does not care what is
# behind it, which is the whole reason the same file flies and runs on a desk. What was hard
# wired was the URL, here in the mission, which is the composition root and the right place for
# it to be a decision.
#
# On the aircraft both are the real uav_api on 8000 and the default is correct. On the bench the
# fake pilot owns 8000, because it is the one that has to answer the movement commands, so a
# bench that wants the REAL pack voltage on the station runs uav_api beside it on another port
# and points this at it. That is read-only -- UavApiBattery issues one GET -- so no command ever
# reaches the real autopilot.
BATERIA_URL = os.environ.get("BANCO_BATERIA_URL", "http://localhost:8000")

ProtocoloVisionReal = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/yolov8n_ncnn_model",
        threshold=0.3,
        tracker=True,
        fps=2.0,
    ),
    pitch_deg=-20.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    battery_source=UavApiBattery(BATERIA_URL),
    see_period_s=0.5,
    identity=IncrementalIdentity(
        fusion_radius_m=0.6,
        fps=2.0,
        track_dur_s=4.0,
        mobile_dur_s=15.0,
        report_dur_s=20.0,
    ),
)
