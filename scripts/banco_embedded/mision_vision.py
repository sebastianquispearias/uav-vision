"""Mission module for the embedded runner: the vision protocol, desk-bench config.

Uploaded to ~/gradys_protocols/ and loaded with
POST /mission/load {"protocol": "mision_vision:ProtocoloVisionLAC", ...}.

Bench configuration: COCO detector (indoor scene; the VisDrone model is
aerial-domain), desk camera pose, short identity maturation so a ~2 minute
mission produces reported candidates. Flight missions swap the model, the pitch
and the identity thresholds — the protocol itself is unchanged.
"""

from uav_vision.camera import CamaraArduCam
from uav_vision.identity import IdentidadIncremental
from uav_vision.vision_protocol import VisionProtocol, UavApiYaw

ProtocoloVisionLAC = VisionProtocol.with_config(
    camera=CamaraArduCam(
        modelo="/home/pi/yolov8n_ncnn_model",
        umbral=0.3,
        rastreador=True,
        fps=4.0,
    ),
    pitch_deg=-20.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    identidad=IdentidadIncremental(
        radio_fusion_m=0.6,
        fps=4.0,
        dur_pista_s=4.0,
        dur_movil_s=15.0,
        dur_reporte_s=20.0,
    ),
)
