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
                     with the real camera (bench_modelos.py, 2026-08-25, on the charger):

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
    reid_modelo      OSNet, on. Without it `emb` comes back None and the identity layer's
                     appearance veto skips itself -- silently, since the check is written as
                     "if both embeddings exist". That veto is what separates a person from the
                     equipment box that captured RANSAC consensus over 930 observations on
                     flight 3, which is the single worst failure this system has produced. The
                     bench sat unpaid for it: this docstring budgeted OSNet's cost while the
                     config never passed the argument.
    fps 3.0          NOT 4.0, and the difference is the whole reason the line above is safe.
                     Measured on the board (2026-08-25, charger, throttled=0x0):

                         960 detector alone, empty frame        196.1 ms   78% of a 250 ms window
                         OSNet, per loaded frame, one box       +33.9 ms
                         960 + OSNet, one person                 230.0 ms   92%
                         960 + OSNet, two people                ~264.0 ms   does not fit

                     At 4 Hz a sweep that finds one person barely fits and a sweep that finds
                     two falls behind -- and falling behind is not a dropped frame, it is a
                     miscalibration: every maturity threshold is scaled by the fps declared
                     here, so a loop delivering 3.8 Hz while claiming 4 makes "36 s" mean
                     something else. At 3 Hz the window is 333 ms and about four people fit.

                     Wall-clock maturity is unchanged: since the 2026-08-25 fix the identity layer
                     measures a span of FRAMES, and fps is the rate the caller sets and
                     therefore knows exactly, so 36 s stays 36 s at either rate. What does
                     change is observations per pass -- roughly a quarter fewer. The
                     pass-length figures quoted below (a 60 s pass finds 47%) were derived at
                     a higher rate and have NOT been re-derived at 3 Hz.

                     Still below the ~5 FPS voltage-collapse point measured with the 5 A UBEC,
                     with more room than before.

                     This number used to be a promise nothing kept. Measured on the board:
                     2.31 FPS delivered against 3.00 declared, because the loop rescheduled as
                     `now + period` (real interval = work + period) and because see_period_s,
                     the parameter that actually drives the timer, was never set here at all.
                     Every maturity threshold was scaled by the declared rate, so "36 s" took
                     about 47. Fixed 2026-08-25 in both places -- fixed-cadence scheduling, and
                     maturity measured off the clock -- and see_period_s is now set below, so
                     this value and the timer cannot drift apart. tests/test_tasa.py holds
                     both ends: at half the frame rate, "36 s" still lands within 2 s.

                     It had gone wrong three ways before that, which
                     is why the identity layer no longer depends on it for timing at all.
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
        reid_modelo="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=3.0,
        # pausa_arranque_s stays OFF, and the measurement is why. Splitting the start-up into
        # three steps 4 s apart was tried on 2026-08-25 against the battery failure: it moved the
        # death from opening the camera (3 s) to loading the detector (9 s) and did not stop
        # it. Each step alone is enough. Buying 8 s of mission time for nothing would only
        # leave a knob that looks like a fix.
        # The crop is what makes a preliminary useful: the drone says "something here, look at
        # this", and RF-DETR on the ground rules. Measured on real flight boxes at 128 px and
        # quality 70: median 2.7 KB, max 3.3 KB -- one small packet, not a video stream, which
        # is the constraint the whole architecture was built around.
        recortes=True,
    ),
    pitch_deg=-55.0,
    # The timer that actually drives the loop. Left unset it defaults to 0.25 s (4 Hz), which
    # is how the fps above quietly meant nothing: a config could ask for 3 and get 4, or --
    # before the scheduler was fixed -- ask for 3 and get 2.31. Set it, and set it to match.
    see_period_s=1.0 / 3.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    identidad=IdentidadIncremental(
        radio_fusion_m=3.5,
        # Must match the camera's fps exactly: this is the number every maturity threshold is
        # scaled by, and the two drifting apart is how "36 s" quietly stops meaning 36 s.
        fps=3.0,
        dur_reporte_s=36.0,
    ),
    reportar_preliminares=True,
)
