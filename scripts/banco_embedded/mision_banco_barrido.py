"""Sweep configuration, on the desk: the flight settings with a detector that fires indoors.

mision_barrido.py is what flies. This is that same configuration with one substitution, and it
exists because of a measurement: on a frame of a person sitting three metres from the camera,
both VisDrone models found nothing and COCO yolov8n found `person` at 0.887 (25ago, same frame,
same 0.25 threshold, channel order tested both ways and worth 0.02). The VisDrone weights are
not broken -- an indoor close-up is the far end of the domain gap already measured at altitude,
where they lose 4.4x below ~20 m. They simply cannot be rehearsed on a desk.

So the rehearsal swaps the detector and keeps everything else: preliminary reporting, crops,
the identity layer and its thresholds, the fusion radius, the frame rate. What gets exercised
is the chain -- camera to identity to message to map, and both kinds of pin -- which is what a
desk can actually test. The VisDrone weights are validated at altitude by the flight-3 replay
instead, which is the only place that question can honestly be asked.

The one addition, not a substitution: reid_modelo. mision_barrido.py leaves it unset, so `emb`
comes back None and the identity layer's appearance veto skips itself silently -- and that veto
is what separates a person from the equipment box that captured RANSAC on flight 3. A rehearsal
without it would exercise position matching alone and report a pass the flight config does not
earn. See ESTADO_SESION.md.
"""

from uav_vision.camera import CamaraArduCam
from uav_vision.identity import IdentidadIncremental
from uav_vision.vision_protocol import VisionProtocol, UavApiYaw

ProtocoloBancoBarrido = VisionProtocol.with_config(
    camera=CamaraArduCam(
        modelo="/home/pi/yolov8n_ncnn_model",
        umbral=0.25,
        rastreador=True,
        reid_modelo="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=4.0,
        recortes=True,
    ),
    pitch_deg=-55.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    identidad=IdentidadIncremental(
        radio_fusion_m=3.5,
        fps=4.0,
        dur_reporte_s=36.0,
    ),
    reportar_preliminares=True,
)
