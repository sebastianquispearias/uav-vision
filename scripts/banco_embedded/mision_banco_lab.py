"""The lab demo: real cameras, and a picture of whoever walks in front of them.

mision_vision.py is the flight mission and it is deliberately lean: no crops and no preliminary
candidates, because over a 4G dongle a crop is 3 kB per candidate per report and a preliminary is a
candidate the drone is not yet willing to stand behind. On a desk, on a cable, both of those costs
are gone and both are exactly what a demonstration needs: the picture is what an operator decides
with, and the preliminaries are what makes the evidence bar visible while it fills instead of only
after.

So this is that mission with three things turned on and one lowered:

    crops=True              the card shows what the camera saw, which is the whole point
    report_preliminary      a contact appears while its bar fills, not twenty seconds later
    report_min_looks=8      eight one-second looks instead of twenty: a demonstration where
                            nothing happens for twenty seconds reads as broken, and on a desk the
                            cost of being wrong is that somebody looks at a picture
    station_url             so the station's search orders reach the board

No appearance model. OSNet costs about 33 ms a box on this hardware and what it buys is fusing two
drones' reports by what the target looks like and the operator's appearance template. Neither is
needed to show a person being found, and the model is not cached on the boards.

    POST /mission/load {"protocol": "mision_banco_lab:ProtocoloLab", ...}
"""
import os

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import UavApiYaw, VisionProtocol

FPS = 4.0

ProtocoloLab = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/yolov8n_ncnn_model",
        threshold=0.3,
        tracker=True,
        fps=FPS,
        # The picture of each detection, which is what the card shows and what the operator judges.
        crops=True,
    ),
    # The camera is on a desk looking across the room, not hanging off an aircraft looking down.
    # Twenty degrees is a guess at how the board is propped up and it only affects where the point
    # lands on the map, which on a desk means nothing anyway: there is no autopilot, so the pose
    # comes from the fake one. The detection is real; the metres are not.
    pitch_deg=-20.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    identity=IncrementalIdentity(
        fusion_radius_m=0.6,
        fps=FPS,
        track_dur_s=4.0,
        mobile_dur_s=15.0,
        report_dur_s=20.0,
        report_min_looks=8,
    ),
    report_preliminary=True,
    station_url=os.environ.get("BANCO_ESTACION"),
)
