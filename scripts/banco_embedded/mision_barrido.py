"""Mission module for a SEARCH SWEEP: the configuration to fly, not the desk bench.

mision_vision.py is the desk bench -- COCO detector, indoor pose, a camera pointing at a
table. This is the one for the evaluation flight, and every value in it comes from something
measured rather than guessed. The protocol itself is identical; only the configuration moves.

Why a separate file at all: a sweep is a different problem from a loiter. A loitering drone
can wait for certainty and should not cry wolf. A sweep crosses each point once and moves on,
and measured over flight 3, a 30 s pass over a person NEVER matures a candidate -- the system
would fly over a victim it tracked perfectly well and say nothing. So this configuration
reports preliminary candidates and sends a crop with them, and the ground station shows them
as POR VERIFICAR rather than as finds.

Where each number comes from:

    modelo 1280      +9 pp of recall over the 640 at mission altitude (71.6% vs 62.5%), and
                     1 alert instead of 4, with 0 false instead of 3. But it LOSES 4.4x at low
                     altitude -- 5.9% vs 26.0% over the 26jul flight at 4.8 m -- because
                     VisDrone never saw a person filling the frame. Correct above ~20 m; below
                     that, swap to best_ncnn_model (640). Its FPS on the Pi is still UNMEASURED:
                     run bench_modelos.py before flying.
    umbral 0.25      swept in punto_operacion.py. It dominates: raising it to 0.40-0.50 loses
                     recall AND adds a false alert, because dropping mid-confidence detections
                     fragments tracks and the fragments stop merging.
    fps 4.0          the vision timer, below the ~5 FPS voltage-collapse point measured with
                     the 5 A UBEC. Since the 25ago fix this is the FRAME rate, which the caller
                     sets and therefore knows exactly -- not a guessed detection rate.
    radio_fusion 3.5 the scene's ground-projection noise at mission altitude. Roughly the GPS
                     bias plus slant range times yaw error; the measured spread on flight 3 was
                     1.6 m around a 2.3 m offset.
    dur_reporte 36   NOT lowered, even though a short pass never reaches it. Lowering it does
                     not help: at 6 s a 30 s pass still only finds the target 17% of the time,
                     because in 30 s the track barely forms. Preliminaries are the answer to
                     short passes; loosening maturity just makes the confirmed pins untrustworthy.

The one thing this file cannot decide is the flight pattern, and it is the one that matters
most: a pass of 20 s finds nothing, 45 s finds 17%, 60 s finds 47%. See
drone-geolocation/docs/PLAN_VUELO_EVALUACION.md.
"""

from uav_vision.camera import CamaraArduCam
from uav_vision.identity import IdentidadIncremental
from uav_vision.vision_protocol import VisionProtocol, UavApiYaw

ProtocoloBarridoLAC = VisionProtocol.with_config(
    camera=CamaraArduCam(
        modelo="/home/pi/modelos_visdrone/best_ncnn_1280",
        umbral=0.25,
        rastreador=True,
        fps=4.0,
        # The crop is what makes a preliminary useful: the drone says "something here, look at
        # this", and RF-DETR on the ground rules. Measured on real flight boxes at 128 px and
        # quality 70: median 2.7 KB, max 3.3 KB -- one small packet, not a video stream, which
        # is the constraint the whole architecture was built around.
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
