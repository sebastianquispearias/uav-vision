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

    reid_model              OSNet, on: without it three of the operator's four actions are
                            wired to nothing

That last one was off until 2026-10-03, on the grounds that appearance was not needed to show a
person being found. That was true and beside the point. What it leaves disconnected, measured
rather than argued:

    the station pairs the wrong targets across aircraft. Two people 3 m apart, seen by two
    drones whose bias differs by 2.6 m, come out as two pins 20 cm apart instead of 2.8 m:
    each drone's first person is fused with the other's second. mismo_objetivo says so in
    prose ("two people three metres apart are two people and only their appearance says so");
    the numbers are in the session notes.

    the operator's "no es" never leaves the station. gs_mapa.plantilla_para only considers
    candidates with an 'emb', so with none the refusal is written to disk and the drone goes
    on reporting the same wrong point.

    the operator's click cannot say WHO, only where.

What it does NOT buy, also measured: nothing for track fragmentation and nothing for two people
crossing paths. Position alone already resolves the first and the tracker's ids resolve the
second. Inside one drone the embedding is a veto on merging two distinct objects that stand
closer than fusion_radius_m, and that is all it is.

What it costs is NOT one number, and the one that was written here was wrong for half the
fleet. Measured on 2026-10-03 with scripts/banco_embedded/medir_osnet.py, on a synthetic frame
with pedestrian-sized boxes, milliseconds for the whole batch:

                1 box     3 boxes    6 boxes
    Pi 5         41.9        91.0      162.1
    Pi 4        327.7       689.4     1044.6

So about 30 ms a box on the Pi 5, which is where the "~33 ms" in camera.py came from, and six to
eight times that on the Pi 4. Three people cost that board 689 ms on top of a frame that already
takes around half a second, which takes it under one frame per second.

That is why this is an environment variable and not a constant. It matters beyond throughput:
the identity layer scales every maturity threshold by the DECLARED rate, so a board running
slower than its mission claims silently stretches what "eight looks of evidence" means. Set
BANCO_FPS to what the board actually does, or turn the model off there with BANCO_REID= (empty).

THE DETECTOR HERE IS COCO, NOT THE ONE THAT FLIES, and that is the right way round indoors.
The aircraft carries YOLO26n fine-tuned on VisDrone (modelos_visdrone/y960_ncnn_model, which
mision_barrido.py loads): aerial imagery, people a few dozen pixels tall seen from above. A desk
camera looking across a room is the opposite domain. Measured on the Pi 5 with
comparar_detectores.py, same camera, same scene, threshold 0.3, one after the other:

    COCO yolov8n       111 ms/frame   1.0 boxes/frame   mean conf 0.55   {person: 7}
    VisDrone y960      196 ms/frame   0.0 boxes/frame
    VisDrone y1280     342 ms/frame   0.0 boxes/frame

The aerial model sees NOTHING indoors, and costs two to three times more to see it. So the bench
numbers are not the flight numbers and should not be quoted as if they were: the 46 % recall
measured on real footage belongs to YOLO26 at 960 px, not to what runs here.

    POST /mission/load {"protocol": "mision_banco_lab:ProtocoloLab", ...}

WHAT IS CHOSEN PER BOARD, AND WHY
    BANCO_FPS, BANCO_MODELO and BANCO_REID set what the board is asked to run at, which
    detector it loads, and whether it computes appearance. All three are per board, because the
    two in this bench are not the same machine and they are not always pointed at the same kind
    of scene.

    The detector is COCO by default BECAUSE THE SCENE IS INDOORS, not because it is the better
    model: the aircraft flies the VisDrone one, and indoors that one returns nothing at all. Two
    short names are accepted so a demonstration does not hinge on typing a path correctly twice.

    BANCO_SEE_S is how often it looks, in seconds, and it is kept SEPARATE from fps because
    they are two different things. When they disagree is exactly how "eight looks of evidence"
    stops meaning eight seconds: fps is what the identity layer BELIEVES the loop runs at, and
    this is what really fires it.

    Crops are on because the picture of each detection is what the card shows and what the
    operator judges. The appearance vector is on because without it the operator's "no es" never
    leaves the station, the click cannot say WHO, and the fusion between aircraft falls back to
    the weak path.

    THE METRES ON A DESK MEAN NOTHING. The camera is on a desk looking across the room, not
    hanging off an aircraft looking down. The mount pitch is a guess at how the board is propped
    up, and it only affects where the point lands on the map; with no autopilot the pose comes
    from the fake one. The detection is real; the metres are not.
"""
import os

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import UavApiBattery, UavApiYaw, VisionProtocol

FPS = float(os.environ.get("BANCO_FPS", "4.0"))
REID = os.environ.get("BANCO_REID", "/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt") or None

DETECTORES = {"coco": "/home/pi/yolov8n_ncnn_model",
              "visdrone": "/home/pi/modelos_visdrone/y960_ncnn_model"}
MODELO = DETECTORES.get(os.environ.get("BANCO_MODELO", "coco").strip().lower(),
                        os.environ.get("BANCO_MODELO", "coco"))

SEE_S = float(os.environ.get("BANCO_SEE_S", "0.25"))

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

ProtocoloLab = VisionProtocol.with_config(
    camera=OnboardCamera(
        model=MODELO,
        threshold=0.3,
        tracker=True,
        fps=FPS,
        crops=True,
        reid_model=REID,
    ),
    see_period_s=SEE_S,
    pitch_deg=-20.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    battery_source=UavApiBattery(BATERIA_URL),
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
