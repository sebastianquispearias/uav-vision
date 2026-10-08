"""Sweep configuration, on the desk: the flight settings with a detector that fires indoors.

mision_barrido.py is what flies. This is that same configuration with one substitution, and it
exists because of a measurement: on a frame of a person sitting three metres from the camera,
both VisDrone models found nothing and COCO yolov8n found `person` at 0.887 (2026-08-25, same frame,
same 0.25 threshold, channel order tested both ways and worth 0.02). The VisDrone weights are
not broken -- an indoor close-up is the far end of the domain gap already measured at altitude,
where they lose 4.4x below ~20 m. They simply cannot be rehearsed on a desk.

So the rehearsal swaps the detector and keeps everything else: preliminary reporting, crops,
the identity layer and its thresholds, the fusion radius, the frame rate. What gets exercised
is the chain -- camera to identity to message to map, and both kinds of pin -- which is what a
desk can actually test. The VisDrone weights are validated at altitude by the flight-3 replay
instead, which is the only place that question can honestly be asked.

The one addition, not a substitution: reid_model. mision_barrido.py leaves it unset, so `emb`
comes back None and the identity layer's appearance veto skips itself silently -- and that veto
is what separates a person from the equipment box that captured RANSAC on flight 3. A rehearsal
without it would exercise position matching alone and report a pass the flight config does not
earn.

The vision timer matches the fps declared for the identity layer: the timer is what sets the
rate, and the fps is only the duty-cycle floor now. COCO is cheap enough here that 4 Hz fits
with room to spare.
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

ProtocoloBancoBarrido = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/yolov8n_ncnn_model",
        threshold=0.25,
        tracker=True,
        reid_model="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=4.0,
        crops=True,
    ),
    pitch_deg=-55.0,
    see_period_s=0.25,
    yaw_source=UavApiYaw("http://localhost:8000"),
    battery_source=UavApiBattery(BATERIA_URL),
    identity=IncrementalIdentity(
        fusion_radius_m=3.5,
        fps=4.0,
        report_dur_s=36.0,
    ),
    report_preliminary=True,
)
