"""Bench mission at 2 FPS against the REAL uav_api telemetry (via safety proxy).

2 FPS because the Pi is powered from the drone battery through the 5 A UBEC,
whose voltage collapses above ~5 FPS — the mission runs well inside that
margin. Yaw comes from the REAL uav_api on :8000 (real Pixhawk compass).
"""

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol, UavApiYaw

ProtocoloVisionReal = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/yolov8n_ncnn_model",
        threshold=0.3,
        tracker=True,
        fps=2.0,
    ),
    pitch_deg=-20.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    see_period_s=0.5,
    identity=IncrementalIdentity(
        fusion_radius_m=0.6,
        fps=2.0,
        track_dur_s=4.0,
        mobile_dur_s=15.0,
        report_dur_s=20.0,
    ),
)
