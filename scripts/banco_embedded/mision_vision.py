"""Mission module for the embedded runner: the vision protocol, desk-bench config.

Uploaded to ~/gradys_protocols/ and loaded with
POST /mission/load {"protocol": "mision_vision:ProtocoloVisionLAC", ...}.

Bench configuration: COCO detector (indoor scene; the VisDrone model is
aerial-domain), desk camera pose, short identity maturation so a ~2 minute
mission produces reported candidates. Flight missions swap the model, the pitch
and the identity thresholds — the protocol itself is unchanged.
"""

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol, UavApiYaw

ProtocoloVisionLAC = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/yolov8n_ncnn_model",
        threshold=0.3,
        tracker=True,
        fps=4.0,
    ),
    pitch_deg=-20.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    identity=IncrementalIdentity(
        fusion_radius_m=0.6,
        fps=4.0,
        track_dur_s=4.0,
        mobile_dur_s=15.0,
        report_dur_s=20.0,
    ),
)
