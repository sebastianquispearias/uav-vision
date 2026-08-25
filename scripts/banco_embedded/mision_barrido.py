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

    modelo 960       the sweet spot, decided with numbers from both ends. Measured on the Pi
                     with the real camera (bench_modelos.py, 25ago, on the charger):

                         modelo  detector  + OSNet   FPS de mision  CPU
                         640     70.6 ms   ~111 ms      ~9.0        ~65%
                         960    160.7 ms   ~201 ms      ~5.0        ~80%
                         1280   311.7 ms   ~352 ms      ~2.8        ~90%

                     And on VisDrone, pedestrian mAP50: 640 0.305, 960 0.409, 1280 0.464. So
                     960 keeps 78% of the 1280's recall gain at nearly twice the speed, and
                     leaves CPU to spare. The 1280 buys five more points of recall by
                     saturating the board: at 90% any surprise -- a thermal spike, a background
                     process -- drops the rate below usable, with nowhere to fall.

                     The benchmark measured the DETECTOR ALONE. OSNet costs a measured +40 ms
                     per frame and BoT-SORT is free; the mission columns above include that.

                     Altitude caveat, and it is not small: the 1280 weights LOSE 4.4x below
                     ~20 m (5.9% vs 26.0% recall over the 26jul flight at 4.8 m), because
                     VisDrone never saw a person filling the frame. For low-altitude work swap
                     to best_ncnn_model (640).
    umbral 0.25      swept in punto_operacion.py. It dominates: raising it to 0.40-0.50 loses
                     recall AND adds a false alert, because dropping mid-confidence detections
                     fragments tracks and the fragments stop merging.
    fps 4.0          the vision timer, below the ~5 FPS voltage-collapse point measured with
                     the 5 A UBEC. It also fits: one frame every 250 ms against ~201 ms of work
                     leaves 20% of headroom. Since the 25ago fix this is the FRAME rate, which
                     the caller sets and therefore knows exactly -- not a guessed detection
                     rate. Switching to the 1280 would mean dropping this to 3.0, in the camera
                     AND in the identity: declaring 4 Hz while delivering 2.8 makes the loop
                     fall behind and mis-scales every maturity threshold.
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
        modelo="/home/pi/modelos_visdrone/y960_ncnn_model",
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
