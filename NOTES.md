# Notes on decisions and measurements

The project log: where every number in the code comes from and why each
decision was taken. The code does not repeat this history -- the docstrings
say WHAT each thing does; this file says WHY and WHEN.

## Camera calibration

- `focal_px = 1407.0` and `principal_point = (945.7, 547.1)` come from the
  `run_info.json` files recorded by `onboard.py` on the three real flights
  (26-jul, 01-ago and 02-ago 2026). A formal chessboard calibration is
  still pending.
- **02-ago-2026: the mounting changed.** The ArduCam was rotated 180° about
  the optical axis and the pitch went from -45° to -55°. The ISP
  straightens the image at capture (hflip+vflip), which mirrors the
  principal point: `(945.7, 547.1) → (973.3, 531.9)` with the formula
  `size - 1 - c`. That is the reason for `CameraConfig.rotated_180()` and
  for `SimulatedCamera.pitch_deg` having no default: a hidden default with
  the old pitch produced 6.287 m of error at the time.

| Flight | focal_px | principal_point | pitch |
|---|---|---|---|
| 26-jul | 1407.0 | 945.7, 547.1 | -45° |
| 01-ago | 1407.0 | 945.7, 547.1 | -45° |
| 02-ago | 1407.0 | 973.3, 531.9 (mirrored) | -55° |

## Pixel noise models

- `heuristic`: sigma = 5.0 / conf (the paper's).
- `visdrone_1d`: sigma = 1.13 / conf, an empirical fit over
  VisDrone2019-DET-val + YOLOv8s. The paper's ablation showed that the
  ranking of methods does not depend on which one is used.

## Appearance embeddings (OSNet)

- Model: `osnet_x0_25_msmt17` via boxmot, CPU, one batch per image,
  normalised vectors.
- Separation measured on the 02-ago flight (cosine): same identity
  0.63-0.87; different identities 0.33-0.46. They do not overlap.
  - Fusion threshold `emb_dist_max = 0.95` = the midpoint of the gap
    (cosine 0.545) converted to L2 distance.
  - Twin threshold `EMB_DIST_GEMELO = 0.70` = deep inside the
    same-identity zone (same <= 0.86, different >= 1.04 in distance).
- 24-ago-2026: verified that `OnboardCamera._fingerprints` reproduces the
  offline analysis embeddings exactly (cosine 1.000000).
- Decision (23-ago-2026): the embedding is computed ON the camera; the
  image does not leave the module. Over the radio travel 512 float32
  (2048 B) or 128 B with PCA int8 (validated offline in the handoff
  thesis). Future alternative: a crop on demand (2-5 KB once) when the
  ground station wants to verify a POI with a heavy detector.

## Identity thresholds

- Rules and base values validated offline on the 02-ago flight
  (`drone-geolocation/entrenamiento/correr_botsort.py`): fusion radius
  3.5 m, minimums of 6/20/25 observations at that analysis's 0.7 FPS
  cadence. In the module they are expressed as durations
  (8.6 / 29 / 36 s) x declared rate, which reproduce those counts.
- **Lesson C-5 (24-ago-2026, flights 1-2):** the rate that matters is the
  one of real OBSERVATIONS, not of the camera. Flight 1 captured at
  8.7 FPS but the person was detected in ~30% of the frames (2.45 obs/s);
  with camera fps the thresholds demanded impossible evidence and 0
  candidates came out.
- Current position of a MOBILE: the median of the recent window lags the
  walker by half a window (measured: 7.40 m of lag at 1 m/s); the linear
  fit evaluated at the last sample reduces it to 0.48 m. Hence
  `_current_position`.
- MOBILE threshold 4.0 m ~= 1.15 x the fusion radius: a stationary track
  only "trembles" from projection noise; more than that, it walked.
- `COOCURRENCIA_MIN = 3`: two tracks seen together in that many frames are
  two distinct things, whatever position and appearance say. Nothing
  appears twice in the same photograph.
- `DUTY_MIN = 0.10`: the minimum fraction of the frames a track spans in
  which it actually has to have been detected. Below that, the track is a
  handful of sightings stretched out and its span in frames exaggerates
  the evidence behind it. **Careful:** this number was tried as a maturity
  rule and discarded, see `docs/DESCARTADO.md`.
- `POS_FRAC_GEMELO = 0.4`, together with `EMB_DIST_GEMELO`: the exception
  to the co-occurrence veto. Detectors sometimes emit duplicate boxes for
  one person and the tracker turns them into two co-occurring tracks. The
  veto is lifted only when the evidence says "same person": nearly
  identical position AND an appearance distance well inside the measured
  same-identity zone.
- `RADIO_95_2D = 2.4477`: the radius of the circle containing 95% of an
  isotropic gaussian in two dimensions, in sigmas, = `sqrt(-2 ln 0.05)`.
  Derived, not fitted. It turns a per-axis uncertainty into something an
  operator can draw on a map and walk.
- `RANGO_REFERENCIA_M = 20.3`: the slant range at which the chain's
  2.4 m median error was measured, that is the median camera-to-target
  distance of the operator's observations on flight 3 (p10 10.1 m, p90
  53.0 m, at a median altitude of 17.1 m). A position margin quoted
  without the range it holds at is incomplete: a heading error moves the
  impact by range x angle.
- `GPS_SIGMA_M = 1.5`: the assumed horizontal standard deviation for a
  consumer receiver, per axis. **Not measured here:** with one calibration
  point and two sources of error, one has to be assumed, and this is the
  better known of the two. The heading error follows from the measured
  floor.

### Maturity: `span` measures the wrong thing

`span` is the time between the first and last sighting, and it is the rule
the chain used until it was measured against a flight. It is kept so those
numbers stay reproducible, but it measures badly. On flight 3:

- a static object the detector mistakes for a person is sighted
  **86 times in 758 s and matures**;
- a person seen **continuously for 30 s** on a sweep **never** matures;
- the operator, placed within 3 m of the truth **at 2 s**, is only
  reported **at 76 s**.

`looks` counts independent sightings: the distinct `look_s` intervals in
which the thing was detected. Consecutive frames within one second are ONE
sighting, because they share the same pose error and the same background; a
second sighting a minute later is another.

**Neither mode separates a real target from a persistent false positive.**
Measured: the chain's own signals do not do it (detection rate while in
view, confidence, apparent size, and an appearance classifier that does not
transfer between flights). That decision belongs to whoever looks at the
crop.

- `track_min_looks = 3`, `report_min_looks = 20`: **decisions, not
  measurements.** They encode how much a false report costs against a late
  one. The 20 is the value that, on flight 3 with the flight's tracker,
  confirmed the same targets as the `span` rule in **22 s instead of 52**,
  and met the provisional mission requirements
  (`docs/requisitos_mision.json`). With 5 it confirms in 6 s but **also
  confirmed a non-person**.
- `fps` is today a FALLBACK and the history is worth keeping, because this
  parameter was wrong in **three ways.** First it meant the detection rate,
  which nobody can know in advance because it depends on how intermittent
  the scene is. On 25-ago-2026 it was redefined as the FRAME rate, on the
  argument that whoever sets the timer knows it exactly. Measured that same
  night, that was false too: the loop rescheduled itself as `now + period`
  and delivered **2.31 frames per second against 3.00 configured**, and
  every threshold derived from the declared rate stretched by a third. The
  loop and the callers are fixed, but a number that was wrong three times
  is a number to stop depending on. A clock cannot be misconfigured.

### The margin more sightings cannot average away (`bias_sigma_m`)

The per-axis standard deviation of the error that more sightings do **not**
average away: GPS and heading, shared by every sighting of a flight. The
default comes from the chain's measured median error on real flights,
**2.4 m**: for a two-dimensional gaussian the median radial error is 1.1774
sigma. It holds at the range it was measured at, `RANGO_REFERENCIA_M`. When
the observations carry `range_m` the shared error is modelled as
`sqrt(gps_sigma_m² + (range × yaw_sigma)²)`, with the heading error solved
so the model returns `bias_sigma_m` at that range.

| range | 95% radius |
|---|---|
| 12 m | ~4.2 m |
| 20 m | 5.0 m |
| 90 m | ~15 m |

### Motion: the window, the speed and the extrapolation

- `motion_window_s = 5.0`: how far back "where is it now and how fast is it
  going" looks. A moving target has to be described by its recent past.
  Over its whole life, a boat patrolling back and forth has its median in
  the middle of the patrol, and a straight line through its last quarter of
  sightings (**20 to 29 s, 81 to 235 m** on flight 3 with a synthetic target
  at 4 and 8 m/s) crosses the turns and points there too: it was reported
  **10-14 m** from where it was.
- `mobile_speed_mps = 0.5`: **a decision, not a measurement** (a walking
  person goes at ~1.4 m/s). Speed alone is not enough on a short track: on
  flight 3 the **stationary** operator left tracks of 6-21 sightings in
  1-5 s with fitted speeds of **0.75 to 1.95 m/s**, and each became a
  separate mobile candidate: **eight points for one person.** That is why
  the fitted motion has to carry the target further than the mobile
  displacement over the sightings it was fitted on.
- `extrapolation_max_s = 3.0`: how far forward a mobile candidate is
  carried from its last sighting. It is separate from the window its speed
  is estimated over: a target that turns keeps going on paper for as long as
  this allows. At 5 s, a synthetic patrol turning every 6.25 s at 8 m/s was
  reported past its turn. Measured over that patrol (median / p90 error):

| | 1.5 m/s | 4 m/s | 8 m/s |
|---|---|---|---|
| 5 s | 0.35 / 11.39 | 2.80 / 21.38 | 7.91 / 17.09 |
| **3 s** | 2.30 / 14.40 | 2.82 / 13.44 | 6.69 / 17.09 |
| 2 s | 2.56 / 15.90 | 1.89 / 11.29 | 6.69 / 17.91 |

  No value wins at every speed. 3 s has the smallest worst case, and that is
  the choice: **a decision about which failure to tolerate, not an
  optimum.**

### Which crop a candidate carries (`crop_choice`)

`confidence` (default) keeps the most confident sighting. `appearance`
keeps, per track, the sighting whose vector is closest to the track's mean
appearance, so the photograph shows what the evidence mostly is.

The difference is the case of a box that caught two people, or a person
beside clutter: the detector is more confident there, and the appearance
filter on the ground agrees with the detector. On flight 3, with the
stand-in tracker, the only confirmed candidate that mostly is **not** a
person carried such a box under `confidence` and passed the filter; under
`appearance` it carries one of its non-person boxes and gets filtered out.

It is opt-in because the same measurement shows the cost: the chosen crops
are less confident (**median 0.41 against 0.77**), and one more preliminary
non-person passes the filter.

### The density that separates phantoms from people

Of the frames in which this layer received a detection while the candidate
was alive, the fraction in which the candidate itself was seen. A static
false positive is a flicker stretched over a long time; a tracked person is
dense while in view. Measured on flight 3 against the letters a human put on
every box: the **6 phantoms cap at 0.397** and the **7 real-person
candidates floor at 0.714**, with nothing in between.

**Sightings per second does the same on this flight and MUST NOT be used:**
it is not scale-free. A figure of 5.7 sightings per second exists only
because this recording is bursty at 1.64 FPS; on a stable board at 3 FPS the
same count cannot exceed 3. **Frames delivered are the unit of opportunity;
seconds are not.**

A frame where the detector found nothing never reaches `observe`, so the
denominator is the frames in which detection was producing something, not
the frames the camera captured. It is the stricter of the two readings and
the only one available without a second channel from the camera.

**It was tried as a maturity rule and fell apart**, see
`docs/DESCARTADO.md`.

### How the operator's error came down, in three steps

The operator's point on the 02-ago flight, as three distinct rules were
fixed. Each step is a contrast that `tests/test_sin_mezcla.py` pins today:

| | error | what changed |
|---|---|---|
| **2.18 m** | up to describing mobiles by their RECENT past | some of the flight's walkers are classified as mobile and stop fusing into the operator's candidate (**670 → 519 impacts**) |
| **2.25 m** | up to requiring the fitted motion to carry the target further than the projection noise | the **stationary** operator's short tracks stop being mobile and fuse back into one candidate |
| **2.27 m** | up to a mobile track having to ask whether the person already had a candidate before opening one | the mobile branch was the only path that never consulted the matcher, so a stationary operator whose duplicate boxes fake speed became several points |

Measured by identity against that flight's hand labels, the operator went
from **four candidates to three**, with the same 5 people found and the same
4 phantoms, and the point ended up closer.

### Without camera-motion compensation, the map looks BETTER

It is the contrast that justifies `scripts/personas_encontradas.py`
existing. The same detector and the same boxes, with `use_cmc=False` the only
difference: of **2637 boxes only 852 get an id**, against **1608 with CMC**,
and evidence that never reaches the identity layer forms no candidate.

What has to be looked at is that the map **without** CMC looks **cleaner**
and has **two people fewer**. Judged by phantoms, the old version wins. That
is why the scoreboard counts people and not boxes.

### `EMB_DIST_REUNE = 0.63`: the number that is NOT safe

Reassociating a moving candidate is the only case where position argues for
separating and appearance for joining, so appearance decides, and more is
demanded of it than the 0.95 that decides a static fusion.

**That is why rejoining is off by default.** Measured on the 02-ago flight,
track by track, against the letters a human put on every box:

| | distances |
|---|---|
| pieces of the SAME person | 0.26 (the operator) and 0.41 0.41 0.60 0.61 0.64 0.81 (the walker) |
| DIFFERENT people | 0.64 (B vs C) · 0.67 (H vs C) · 0.68 (H vs B) · 0.73 … |

The two ranges **overlap by 0.18**: seven pairs of different people are
closer than the furthest pair of the same person. No threshold separates
them. What joins the walker without fusing anybody is a window between 0.61
and 0.67, and 0.63 falls there, chosen by looking at the flight it is judged
on, which is the mistake this repository has already paid for once. At 0.70
the boy on the balcony is absorbed by another and **disappears from the
map**, which in a search is the worst failure there is: the operator is not
told there is a person there.

It stays off until there is a second flight to choose the number on.

### Minimum separation per class (`SEPARACION_MINIMA_M`)

The smallest centre-to-centre distance at which two members of a class are
still two things. It is **parking geometry, not noise:** a standard bay is
2.4-2.6 m wide, so two cars side by side are 2.5 m apart and a fusion radius
of 3.5 m (the value fitted for people) reports them as a single car. Nothing
downstream can undo it, because by then there is already a candidate.

**People are deliberately absent,** and the reason is worth stating: two
people can also be 0.6 m apart, but nothing puts them there the way bays put
cars. Applying a 0.6 m floor to people would shrink the radius every flight
to date was measured with, to buy a separation the scene does not impose.
Vehicles are the case where the geometry is regular enough to encode.

The trade it makes goes one way on purpose. Below the noise radius, one car
under projection noise can fragment into two candidates a couple of metres
apart; above it, two cars fuse into one. Fragmentation reports the same
thing twice in almost the same place, which an operator resolves at a
glance. Conflation makes a vehicle **disappear**, and nothing on the screen
says so.

## CLIP's second opinion on the station (`--clip-descarta`)

By default, what CLIP doubts goes to the **end** of the queue and nothing
else moves: the order is stable and a POI without a score is never demoted.
With `--clip-descarta` it is **removed** instead of demoted.

Measured on the 02-ago flight, removing instead of demoting takes out the bag
and the cone and leaves every person standing: **phantoms 2 → 1, with 5 of 5
people kept.**

It is off by default for two reasons, and the second is the one that
matters: hiding a point the operator never saw is the operator's decision and
not the system's, and **CLIP still misses the hardest clutter**. The red
object on that flight scored **1.81, above the threshold**, and it also
resembles a person by size.

## Fusing the reports of two drones (`flota.fundir`)

- **The margin of a fused pin is the SMALLER of the two.** Measured over the
  02-ago candidates, the 95% radius of a static target is **82 to 99.9%
  bias** and almost no dispersion, and that bias is `gps_sigma` plus slant
  range times `yaw_sigma`. A target seen from 14 m and from 37 m does not
  have one uncertainty: it has the nearest drone's. Keeping whichever
  arrived first was reporting the worse of two answers for no reason.
- **3-oct-2026, the appearance of the fused pin.** A pin kept the fields of
  the first report that created it, and if THAT drone did not compute
  appearance the pin was left without a vector even though the other one
  brought it. Measured with one board computing appearance and the other
  not: the operator's verdict **was recorded on disk and reached no drone**,
  because `gs_mapa.plantilla_para` filters by `emb`. A fused pin without a
  vector turns the operator's button into a button that does nothing.

## Camera thresholds

### `EMB_DIST_OBJETIVO = 0.85`: the other number that is NOT safe

The appearance distance below which a box the detector doubted is kept
because it **resembles** the target the operator pointed at. Unlike the
window, it does not ask where the box is, and that is the point: the window
is the projection of a ground position and it fails exactly when the
aircraft's attitude is least certain.

Measured over 773 frames of the 02-ago mission (2746-3700), 1673 person
boxes and 484 of the target, keeping the boxes scored between 0.10 and 0.25
that are thrown away today:

| | target recall | total recall | precision |
|---|---|---|---|
| conf >= 0.25 only, as it flies | 91.1% | 48.3% | 57.9% |
| **+ the doubtful ones that RESEMBLE it (0.85)** | **94.4%** | **50.1%** | **55.6%** |
| the same at 1.00 | 97.7% | 52.5% | 52.3% |
| the same at 1.20 | 97.7% | 61.1% | 40.3% |

No weight is retrained and nothing is adapted: the template is the embedding
of the crop the operator clicked, so switching this off returns the system to
exactly what it was.

**THE NUMBER IS NOT SAFE YET, for the same reason `EMB_DIST_REUNE` is not:**
it was chosen by looking at the stretch it is judged on. It has to be
re-chosen on the 01-ago flights before it can be defended, and until then the
gate stays shut unless the caller asks for an explicit distance.

### `BANDA_BAJA = 0.2`: the floor of the BYTE band

The lowest score asked of the detector when a tracker is running. It is 0.2
because **it is BoT-SORT's own `track_low_thresh`**, and a box below the
tracker's floor is dropped before association, so asking for it is paying
OSNet (~33 ms per box) for something nobody will look at.

Measured on the 02-ago flight: of **2443 boxes between 0.10 and 0.20, exactly
zero** received a track id; between 0.20 and 0.25, **30 of 628** did.

Asking below the reporting threshold costs no extra inference, because the
detector already scored those boxes and was throwing them away. The 0.2-0.3
band closes **77 of the 226 gaps** where a person is in one frame and missing
from the next.

### Tiled pass (`slice_every`)

The frame is shrunk to the model's input before inference, so a person 55 px
tall arrives as 28 and the ones already at the limit disappear. The tiled
pass at native resolution skips that shrinking. It costs **six times the
inference**, which is why it does not run on every frame, and why it does not
need to: the identity layer asks for eleven sightings in thirty-six seconds,
a tenth of the frames, so one pass in five keeps the average near the budget
while the cheap pass keeps running always.

Measured on the 02-ago flight with the tile threshold chosen on the 01-ago
flights: recall **55.0 → 57.3%** total and **39.5 → 42.4%** where the drone is
high, at the same precision.

### Camera-motion compensation (CMC)

The drone moves, so every box shifts in the image between frames and the
tracker's prediction misses the next box unless the background motion is
removed first.

Measured with the flight's tracker parameters and hand labels:

| | ids on the stationary operator |
|---|---|
| without CMC, 02-ago labelled windows | 32 |
| with sparse optical flow (`sof`) | **5** |

On flights 2a / 2b from another day the ids on people **halved**
(10 → 6, 22 → 11) without absorbing a single non-person box.

Cost on the Raspberry Pi 5 at boxmot's default image scale of 0.15:
**+8.1 ms median per frame** (1.0 → 9.1 ms), about 4% of the chain's 206 ms,
with no throttling.

boxmot's default method is `ecc` and it is worse on both axes: over the same
flight and the same calibrated thresholds it recovered far fewer ids
(**IDF1 0.376 against 0.656** with `sof`) and cost more on the Pi (**p90
20.5 ms against 10.4 ms**).

### `pausa_s`: the pause between the heavy startups

Default 0, no change for anything plugged into the wall. On battery it is the
only lever the software has against the failure measured on **25-ago-2026**:
the board died **3 s after** opening the camera and loading both models, with
the pack **FULL**, drawing 3.47 W. That is nowhere near saturating a 5 A
UBEC, so what kills it is **the step, not the level.** Opening the camera and
loading the models back to back stacks those steps; this separates them.

### 4-oct-2026: the camera retry could lie, and a linter found it

While configuring ruff (`[tool.ruff]` in `pyproject.toml`) a `B023` came up,
*function definition does not bind loop variable*, in
`OnboardCamera._first_frame`. It was a real bug, in the code that flies.

The loop was like this:

```python
for intento in range(1, self.INTENTOS_ENCENDIDO + 1):
    listo = threading.Event()
    def capture():
        try: picam.capture_array()
        finally: listo.set()        # closes over the CELL, not over the value
    threading.Thread(target=capture, daemon=True).start()
    if listo.wait(self.ESPERA_PRIMER_FRAME_S):
        return
```

`listo` is local to `_first_frame`, so **both definitions of `capture` share
one cell**. The failing sequence:

1. attempt 1 hangs inside `capture_array` (the failure mode this watchdog
   exists to cover: `Camera frontend has timed out`);
2. the deadline expires, the thread stays alive because it is a daemon, on
   purpose;
3. attempt 2 creates a **new** `Event` and rebinds `listo`;
4. attempt 1's thread unsticks and calls `set()` **on attempt 2's `Event`**;
5. attempt 2's `listo.wait()` returns `True` and `_first_frame` **returns as
   if the camera had delivered a frame**, with attempt 2 still hung.

Reproduced with a fake camera that sticks on the first capture and unsticks
during the second:

```
as it was  -> RETURNS saying attempt 2 delivered a frame
fixed      -> raises RuntimeError: the camera did not deliver a frame
```

The fix is to bind the `Event` **per attempt** as a default argument
(`def capture(ev=listo)`), so a late thread can only set its own.

**The lesson is worth more than the bug:** none of the 50 gates caught it,
because none simulates a capture that hangs and then unsticks. A tool found
it in thirty seconds. That is why ruff is configured and runs in CI.

### The camera that enumerates but does not stream

The sensor is detected over I2C and enumerates long before it is going to
stream for real, and sometimes it does not stream at all: libcamera reports
`Camera frontend has timed out` and the capture call **never returns**.

Observed on **25-ago-2026**: five failures in a row within the minute after
boot and right after killing a process mid-capture, and then five successes
out of five once the board had been up a few minutes. Nothing changed in the
configuration.

Without the retry, that failure mode is a lost mission with no diagnosis: the
Pi alive, the protocol running its timer, and zero detections forever because
the first capture never returned.

### The channels arrive RGB, not BGR

picamera2 labels this configuration `BGR888`, but that name is libcamera's
and it lists the components in the **opposite** order to the one the array
arrives in. What comes back is R,G,B. Everything downstream is OpenCV-shaped
and expects B,G,R: ultralytics assumes it for a raw array, the ReID model
assumes it, and `cv2.imencode` assumes it when writing the crop. Left alone,
red and blue are swapped for all three at once.

Measured on the real board (**25-ago-2026**): a person the detector found at
0.887 with the channels swapped scores **0.909** once corrected. It is small,
and it is **not** the reason the VisDrone weights find nothing indoors, which
is a domain gap. The reason to fix it is the **crop**: it is the photograph an
operator looks at to decide whether to send somebody to that point, and it was
arriving with blue skin.

Converting the whole frame once is what keeps the three consumers in
agreement. It costs **1.70 ms against 184 ms** of inference on the Pi 5, 0.9%
of the frame.

## Vision protocol thresholds

All of these lived as comments inside `vision_protocol.py` until 4-oct-2026.
They are decisions, not derivations: their value could be different on
another mission without the system working differently, so the docstring says
WHAT they are and this section says where they come from.

- `STATION_TIMEOUT_S = 0.3` and `STATION_RETRY_S = 10.0`: **not measured.**
  The timeout is a fraction of the vision loop's period (0.25 s at 4 Hz),
  because a query to the station cannot cost the loop more than a sliver of
  its period. The retry avoids asking a downed station on every report.
- `MIN_MEASUREMENTS = 8`: **not derivable from the data.** The minimum number
  of impacts before the RANSAC consensus speaks. It encodes how much evidence
  is needed not to invent a point.
- `RODEO_TOLERANCIA_M = 5.0`: it is the order of the GPS bias the system
  already accounts for (`GPS_SIGMA_M` 1.5 m, and the 95% radius of a static
  candidate on flight 3 is 2.4 m). It is a radius and not a coordinate match
  because an aircraft holding position drifts.
- `RODEO_PLAZO_S = 60.0`: **not derived, it is a guard.** One leg of a 30 m
  orbit is tens of seconds at this aircraft's speeds; it protects against an
  aircraft sent to a point it cannot reach.
- `report_preliminary` off by default: a drone in orbit can wait until it is
  sure. It is switched on for a SWEEP, and the reason is measured on flight 3:
  a 30 s pass over a person **never matures** a candidate, so a search that
  crosses each point once and moves on reports nothing.
- `attitude_source`: the camera is fixed to the body. At 30 m, 10° of
  unaccounted pitch put the impact **7-10 m** away (`tests/test_actitud.py`).
  `None` leaves only the mounting angle, which is right for a drone in orbit
  and wrong for one escorting.

### The target window: the gain does not survive the chain

`FOCO_RADIO_PX = 320` is half the 640 px square the gain was measured over, in
a 1920x1080 frame. `FOCO_UMBRAL = 0.10` is the floor the detector was already
scoring at.

Against real boxes the window raises target recall from **43.9% to 60.7%** and
lowers precision from **71.2% to 53.3%**. But run through the whole chain over
the 02-ago flight and measured in people and phantoms it **changes nothing**:
the same 5 people and the same 6 phantoms, across the four variants tried
(target on the operator and on the most fragile candidate, with the tracker as
it flies and with its floor lowered to accept the band).

The mechanism is in the numbers: 1673 boxes fell inside the window and only
**29** ended up with a track id. BoT-SORT drops everything below its low
threshold of 0.20 **before** associating, and a detection without a track does
not reach the identity layer. Of the 841 boxes between 0.10 and 0.15 and the
494 between 0.15 and 0.20, **zero** were tracked. And above 0.20 it adds
nothing either, because `camera.py` already opens that band over the whole
frame (`low_band=0.2`).

What is left is the plumbing to act on the operator's click.
`docs/RESULTADOS.md` used to publish the 43.9 -> 60.7 as "not implemented": that row was
out of date and was corrected on 5-oct-2026, because it is `fix_target()` today, and it
omitted that in people and phantoms it moves nothing.

### Ground extent per class (`GROUND_EXTENT_M`)

Nominal dimensions of the thing, **not fitted corrections.** The extent along
the line of sight is between the object's width and its length depending on
how it happened to be parked; each value is the midpoint of that pair, so the
residual is bounded by half their difference: **±1.3 m for a car, ±4.7 m for a
bus.** The alternative is a bias of the whole half-length, always towards the
drone.

**People are deliberately absent:** a standing person covers about 0.4 m of
ground, so the correction would be 0.2 m against a system error of 2.4 m, a
twentieth of the noise. And every flight result to date was measured with the
bottom edge.

## Electrical health of the board

- `THROTTLED` is read as a file rather than by calling `vcgencmd`: **3 ms
  against spawning a process** on every report.
- The mask has two halves that mean different things. The low bits are NOW:
  acting on them means the aircraft is in trouble this second. The high ones
  are SINCE BOOT: they stay set after the dip has passed, which is what makes
  them useful on the ground, because dips last an instant and nobody is
  watching at that moment.
- **3-oct-2026, why this exists:** a board ran undervolted for four hours and
  nothing showed it. The kernel had been logging `Undervoltage detected!`
  since 21:56; the station, the operator and the logs being watched all said
  the aircraft was healthy, and at 01:58 it cut off mid-line and did not come
  back. The cause was found the next day by moving its SD to another board and
  reading its journal. In the air that is a drone that disappears with no
  explanation.

## 5-oct-2026: boxmot's YAML moved two appearance gates in silence

While putting the tracker adapter (`uav_vision/trackers.py`) into
`camera._build_tracker`, the equivalence gate -- building the tracker both ways
and comparing attribute by attribute -- caught this **before** the change went
in:

| | as it was | via create_tracker |
|---|---|---|
| `appearance_thresh` | **0.25** | 0.6188818853936099 |
| `proximity_thresh` | **0.5** | 0.6084297894561342 |

boxmot's `create_tracker` merges the caller's settings **over a per-tracker
YAML**, and those files carry the result of a hyperparameter search on MOT
(hence the sixteen decimals). They are two **appearance** gates this flight
never calibrated, for parameters nobody here sets. `appearance_thresh` is the
cosine at which ReID decides two boxes are the same person: from 0.25 to 0.62
changes the tracking for real.

**And the 2.39 m gate would not have seen it**, because the replay uses its own
tracks from an `.npz` and does not build the camera's tracker. It would have
been discovered in flight.

That is why `construir()` instantiates the class **directly** and the YAML is
never loaded: the only values that are not class defaults are the ones passed
in.

**And it is the same thing that makes comparing trackers mean something.** If
the YAML got in, every tracker in the zoo would run with somebody else's tuning
for ground-level pedestrians, and the table would measure that and not the
trackers.

## 5-oct-2026: boxmot's ten trackers over the 02-ago flight

Generated by `scripts/comparar_trackers.py --cmc=on`, which runs the three
steps of each row. **It is not written by hand**, and that is deliberate: the
first version of this table was, and it was false for three reasons that turned
out to be all ours. The story is at the end of the section, because it is what
is worth most.

| tracker | CMC | people | phantoms | not judgeable | with id / 2637 | tracks | median/max boxes |
|---|---|---|---|---|---|---|---|
| botsort | on | **5 of 7** | 6 | 2 | 1608 | 36 | 13 / 336 |
| sam2mot | none | **5 of 7** | 8 | 1 | 1891 | 298 | 2 / 136 |
| occluboost | on | 4 of 7 | **1** | 0 | 1392 | 6 | 192 / 586 |
| deepocsort | on | 4 of 7 | 2 | 2 | 643 | 31 | 7 / 120 |
| hybridsort | on | 4 of 7 | 3 | 1 | 1185 | 17 | 12 / 287 |
| boosttrack | on | 4 of 7 | 3 | 1 | 951 | 51 | 7 / 274 |
| strongsort | on | 4 of 7 | 5 | 3 | 1576 | 106 | 5 / 190 |
| bytetrack | none | 3 of 7 | 0 | 0 | 728 | 109 | 4 / 84 |
| ocsort | none | 2 of 7 | 0 | 0 | 437 | 77 | 3 / 71 |
| sfsort | none | 1 of 7 | 0 | 0 | 1549 | 704 | 1 / 109 |

**The CMC column says what RAN, not what was asked for**, and the difference is
not cosmetic: boxmot exposes motion compensation through four different
mechanisms and ships it ON by default, so a tracker that ignores the request is
not left off, it is left however boxmot wants. Read off the built instance with
`uav_vision.trackers.cmc_real`, the split is three ways:

- **5 switchable**: `botsort`, `boosttrack`, `occluboost` through `use_cmc`;
  `deepocsort` through `cmc_off`, which is the same boolean **inverted**;
  `hybridsort` only through `cmc_method`, because `create_cmc` returns `None`
  for a method of `None` and then the method IS the switch.
- **4 with no mechanism**: `bytetrack`, `ocsort`, `sfsort`, `sam2mot`. And
  `none` is not `off`: lumping them together counts a tracker that has no
  compensation as evidence that turning it off did no harm.
- **1 forced**: `strongsort` does `create_cmc("ecc")` in `__init__` with no
  parameter and applies it with no guard (`strongsort.py:68` and `:83`). **It
  compensates always and there is no way to turn it off.**

### What compensation is worth, measured with both tables

`scripts/comparar_trackers.py --cmc=off` gives the other half of the contrast.
It only makes sense for the five switchable ones:

| tracker | CMC off | CMC on | people | phantoms |
|---|---|---|---|---|
| botsort | 3 of 7 / 1 phantom | **5 of 7 / 6** | +2 | +5 |
| boosttrack | 2 of 7 / 0 | 4 of 7 / 3 | +2 | +3 |
| deepocsort | 3 of 7 / 0 | 4 of 7 / 2 | +1 | +2 |
| occluboost | 3 of 7 / 2 | 4 of 7 / 1 | +1 | −1 |
| hybridsort | 4 of 7 / 3 | 4 of 7 / 3 | 0 | 0 |

Compensation is worth **two people** on the tracker that flies, and it **costs
five phantoms**. Both halves are published together on purpose: the product
scoreboard is both numbers, and keeping the convenient one is how an
improvement stops being an improvement. `hybridsort` is the case that stops
this being read as a rule: its run changes (1033 → 1185 detections with an id)
and its scoreboard does not.

### What this table does NOT authorise concluding

**1. It is ONE flight, one day, one place, seven people.** Ten trackers over a
single scenario multiply the ways of overfitting to it. The deliverable is the
machine for swapping pieces; this comparison is of one flight.

**2. "More detections with an id" is not better, and both ends show it.**
`sfsort` assigns 1549 and finds ONE person, because it splits them into 704
tracks of median 1 box. `occluboost` assigns 1392 across **6 tracks** of median
192: it is merging identities, which is the dangerous failure, because a merged
person DISAPPEARS from the map. Its single phantom is not precision, it is that
almost everything fell into the same contact.

**3. Seven of the ten do not receive the full calibration.** A row that did not
receive it is not compared on the same terms as one that did, so the lists go
beside the numbers and not below them:

| tracker | our 5 settings, untranslated | translated | no equivalent |
|---|---|---|---|
| botsort | 5 of 5 | 5 of 5 | — |
| occluboost | 2 of 5 | 5 of 5 | — |
| sfsort | 0 of 5 | 5 of 5 | — |
| bytetrack | 2 of 5 | 4 of 5 | `new_track_thresh` |
| ocsort | 0 of 5 | 4 of 5 | `new_track_thresh` |
| strongsort | 0 of 5 | 4 of 5 | `new_track_thresh` |
| sam2mot | 1 of 5 | 4 of 5 | `track_low_thresh` |
| boosttrack | 0 of 5 | 3 of 5 | `new_track_thresh`, `track_low_thresh` |
| deepocsort | 0 of 5 | 3 of 5 | `new_track_thresh`, `track_low_thresh` |
| hybridsort | 0 of 5 | 3 of 5 | `new_track_thresh`, `track_low_thresh` |

Renames that **change the value**: `match_thresh=0.85 → iou_threshold=0.15`.
Renames by role only: `track_high_thresh → det_thresh`.

### Why the first version of this table was false, and all three reasons were ours

It matters more than the numbers, because the numbers are of one flight and
this is not.

**1. `match_thresh → iou_threshold` ran INVERTED, not "approximate".** BoT-SORT
caps a COST of `1-IoU` (`matching.py:79` builds it, `matching.py:35` passes it
to `lap.lapjv` as `cost_limit`), so `match_thresh=0.85` means "associate from
IoU >= 0.15" and is the **most permissive** setting this calibration carries.
Everywhere else in boxmot the same number is a FLOOR on the IoU
(`stages.py:152`, `boost.py:196`, `hybrid.py:343`, `hybrid.py:363`,
`occluboost.py:956`), so five trackers were told "the boxes have to overlap by
0.85 to be the same person". At 25 m that never happens. The conversion is
`1 - v` and it is **derived**: it is the value at which both expressions admit
the same pairs. It lives in `uav_vision.trackers.CONVERSIONES` and section 9 of
`tests/test_elegir_tracker.py` fails if it is lost.

**2. "No row ran with CMC" was false for three of the nine.** `strongsort`,
`hybridsort` and `deepocsort` were compensating, and `strongsort` topped the
table. The premise that supposedly invalidated the comparisons was itself
incorrect.

**3. `botsort_pistas.py`'s docstring asserted the opposite of the flight.** It
said "configured exactly as `_build_tracker` builds it for the flight mission
(camera-motion compensation off)", and `_build_tracker` has it ON with method
`sof` (`camera.py:479-480`, `compensate_motion=True`). That line is the origin
of the table giving `botsort` 3 of 7 while the repository's front page says
5 of 7.

**The combined effect: the ordering flipped entirely.** The tracker that flies
went from fourth place to first, `ocsort` stopped being "does not work for this
case" (6 detections with an id out of 2637) and went to tracking 16.6 % of the
flight, and `sam2mot`, which the table omitted without saying why, ties on
people with the one that flies. **The preliminary table suggested replacing the
flight's tracker with `strongsort`, and that was an artefact of two bugs of
ours.** All three failures had the same shape: a signal that did not measure
what one believed. A setting that was "honoured" and arrived inverted, a
setting that was "ignored" whose effect stayed on, and a comment that
contradicted the code it described. None of them broke anything; all three
produced a publishable table. What found them was looking at the built
INSTANCE and at the library's source, not at the signatures.

A row of the table is regenerated with two commands:

    V=../drone-geolocation/entrenamiento/venv/Scripts/python.exe
    "$V" scripts/comparar_trackers.py --trackers=botsort --cmc=on
    "$V" scripts/medir/umbral_invertido.py      # the ocsort diagnosis


## Tracker (BoT-SORT)

- Parameters validated offline: high 0.35 / low 0.2 / new 0.4 /
  buffer 40 frames / match 0.85, with external embeddings and CMC.
- The buffer is declared in seconds because 40 frames are 24 s at the
  02-ago flight's cadence but only 8 s at the drone's 5 Hz.
- CMC (camera-motion compensation) **ON by default on the drone**, method
  `sof` and not boxmot's `ecc` (`camera.py:267` and `:479`). Nothing turns it
  off: `compensate_motion` is not set to `False` in any mission or any
  script. Its cost is ALREADY inside the 199 ms / 5.04 FPS that
  `scripts/medir/medir_vuelo.py` measures on the Pi 5, because that probe
  builds the camera without touching the switch; what is not measured is its
  isolated share. It is worth +2 people and +5 phantoms on the 02-ago flight
  (ten-tracker table, above).
- BotSort with `with_reid=True` demands embeddings; with no ReID configured
  the module creates it in motion-only mode.

## Raspberry Pi budget (bench 22-ago-2026)

- torch 3.74 FPS / 258 ms · NCNN 12.47 FPS / 77 ms (3.3x).
- Thermal: 3 fps → 47 °C / 20% CPU · 5 fps → 52 °C / 33% ·
  10 fps → 58 °C / 76%.
- With a 5 A UBEC the voltage sinks above ~5 FPS and the system collapses;
  that is why the default vision rate is 4 Hz (lowered from 5 to 4 on
  24-ago-2026 to operate with margin below the collapse point). **The test
  with the 7 A one is done**, and it is the one-variable experiment of
  `docs/deploying-yolo-reid-tracking-on-a-raspberry-pi.md:45-52`: the 7 A one
  survived the full 87 s with 0 rows in undervoltage and the 5 A one died at
  55 s with 153, **drawing less** (p95 1.19 A against 1.50 A). It was not the
  current, it was the step.
- Full-chain live rehearsal (24-ago-2026, wall power): detector alone
  7.23 FPS / CPU 54% · +BoT-SORT 8.84 FPS / 65% · +OSNet 6.59 FPS / 75.5%.
  OSNet ~= +40 ms/frame with 1 person. The capacity (6.59) exceeds the demand
  (4 Hz) even with embeddings.
- 24-ago-2026: `OnboardCamera` ran on the real Pi at 5.58 FPS (contract tested
  on hardware). Config: `camera_auto_detect=0` + `dtoverlay=imx708` in
  config.txt (auto-detect does not recognise Arducam's IMX708); `numpy<2`
  required by picamera2/simplejpeg.

## Accumulated validation (02-ago flight unless stated)

- Geometry: protocol rays vs real flight rays = 0.000° of difference
  (268 samples).
- Replay with identity: the operator as a separate POI at 2.32 m (offline
  analysis: 2.49 m); the equipment case no longer steals the consensus.
- Stack with RF-DETR as the detector: 2.39 m, 1106 obs, conf 0.80, case
  absent -- a tie on accuracy (the ~2.3-2.5 m floor is set by the GPS/yaw
  bias), a clear improvement in robustness. Role assigned: verifier on the
  ground station.
- Healthy flights 1-2 (C-5): a clean dominant candidate in all 3 runs --
  0.41 / 3.13 / 1.43 m (simple fusion: 1.03 / 2.08 / 0.59).
- Synthetic identity gates (`tests/test_identity.py`): static 0.09 m; mobile
  with no lag; the co-occurrence veto keeps 2; twins fuse to 1.
