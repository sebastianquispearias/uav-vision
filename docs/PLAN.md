# What comes next, in order, with its acceptance criterion

Every item says **when it is done**, so the work has an end and is not a treadmill.
What is already measured and discarded lives in `DESCARTADO.md`; the current numbers in
`RESULTADOS.md`.

## Active verification: the new block, and it goes before the rest

The three pieces below are one mechanism, not three loose features. The operator says what is being
looked for, the system measures how much evidence it has and how much it is missing, and the drone
goes and gets what is missing. In the literature that is called **active sequential hypothesis
testing**, and the stopping rule is Wald's sequential test (1945): evidence is accumulated and the
process stops on crossing one of two thresholds, one to confirm and one to discard.

They go before items 1 to 6 because none of them needs to fly, and because the 30sep measurement
(below, under what not to do) took the supposed justification for them off the table.

### A. The certainty bar, visible

Today a candidate is either mature or not, and the operator sees nothing while it is being decided.
Measured on flight 3: a 30 s sweep over a person **never** matures a candidate, and a 60 s one does
so 47 % of the time. During those 30 s the system is halfway there and the operator does not know
it.

The count already exists: `identity.py` computes `c["n"]` against `n_reporte` and the temporal
coverage on every frame. What is missing is showing it, and two deeper things. First, **weighting
each sighting**: today the tenth from the same angle counts as much as the first from a new angle,
and in a sequential test each observation contributes according to how informative it is. Second,
**the lower threshold**: today the bar only rises, and a candidate accumulating evidence against
should empty out and discard itself instead of hanging around as a preliminary.

**Done when:** the station shows, per candidate, the fraction of evidence gathered; a candidate with
evidence against discards itself; and the product scoreboard does not get worse.

### B. The click that defines the target

The operator's verdict exists and today does little: "this is what I am looking for" lowers the
threshold in that window, and "not it" erases a point from the map. The first is measured and is no
use (see item 5). The second throws away the most valuable signal there is.

The measured route is the appearance template: the crop of the click is the embedding, and a
doubtful detection that resembles it is accepted. The target rises from 91.1 to 94.4 % without
touching a weight and with immediate rollback. **Weights are not trained in flight**, which is
already on the list below.

And "not it" is kept as a hard negative. In active learning the most valuable example is not the
positive but the negative the model got wrong with confidence, which is exactly what that button
produces.

**Done when:** the operator's click changes which candidates are reported in the 02ago replay, the
negative verdicts are stored and usable, and the product scoreboard judges it.

### C. The manoeuvre: sending a drone to look from another side

It is phase 2 of the original mission ("the drones go, circle and hold position") and it was never
built. Today the station proposes a second look and the pilot decides.

**What justifies it is the operator's decision, and that is NOT a consolation prize.** Of the 11
candidates the system reports, 6 are nobody. What kills them is the operator, and today the
operator decides with a 128 px crop taken from almost directly above, which is the worst viewpoint
there is: measured on boats, the camera at 36 degrees gives 0.866 recall and pointing straight down
**0.067**. Thirteen times, from the angle alone. If the viewpoint is worth that to a detector, to a
human eye judging whether a blob is a person it is worth the same: from above a person has no
posture.

**The contrast that makes it clear:** RF-DETR's second opinion re-reads **the same photograph**,
better. The manoeuvre brings **a new photograph**. Only one of the two adds information that was not
there before, and it is the only one a bigger model cannot replace.

**It is NOT justified by detector recall**, and that still stands: the detector loses people larger
than the ones it finds in the same image, so going to fetch more pixels fixes nothing. Nor does it
shrink the margin: measured over the 02ago candidates, the 95 % radius of a stationary target is 82
to 99.9 % the GPS and compass bias of that aircraft, and taking the dispersion to zero shrinks it
0.1 to 0.5 %. That is why the order goes to **another** aircraft, whose bias is a different one.

What is missing is not the criterion but its objective. `view_selection.py` scores views by
**geometric diversity**, which is right for triangulating and wrong for recognising. To verify what
something is, the criterion has to be the expected reduction of **classification** uncertainty.
First material obstacle: `correr.py` is the only thing importing `view_selection.py` and `fusion.py`,
and it does not start (it imports `CamaraSimulada`, which does not exist: `camera.py` defines
`SimulatedCamera`).

**Done when:** the operator points at a candidate, a non-leader drone reaches a computed position and
returns an image from another angle, and how much it moved the bar of item A is measured.

## Product

### 1. Fly again, at 20-25 m, another day and another site

**Why it is first and cannot be skipped:** it does not only give the first clean evaluation. Today we
**do not have a single validation frame in the regime that matters**, because all the high-altitude
material that exists is either the test or its immediate neighbours. That is why validation chose
wrong three times running and **we cannot even choose between two models** before touching the test.

**Done when:** there is a 10-minute flight at 20-25 m, with at least two people besides the operator,
and at least one surveyed position that is not the operator's.

**Help that exists:** the labelling tool's `/plan` page draws altitude against frames and says, for
whatever stretch is chosen, how much of it is new and how large a person would look there.

### 2. Measure the on-demand frame on the Pi

None of the new work has ever run in the air. The Pi takes 206 ms per frame.

**Done when:** `<= 330 ms/frame` measured on the Pi with the model frozen and the on-demand frame
answering.

**The tiles left this item on 17sep.** Measuring them on the Pi was expensive and no longer needs
deciding there: by the product scoreboard they buy not one person, remove no phantom and push the
point 57 cm away, in exchange for six times the compute. The number is in `RESULTADOS.md`. If they
are ever switched on, it gets measured then.

### 3. Judge the new improvements with the product scoreboard — DONE on 17sep

The fear was that the 12 points of precision adaptation pays would be new phantoms. **They were:**
from 2 to 8. And in exchange the point comes 42 % closer, from 1.92 to 1.12 m, so it is a trade to
be chosen, not an improvement to be applied. The tiles buy no person at all.

The table is in `RESULTADOS.md`. The scoreboard stopped living in a temporary folder: it is
`scripts/personas_encontradas.py`, with `tests/test_personas_encontradas.py` pinning it, and the
candidates are regenerated with `scripts/replay_vuelo3.py --candidatos=`, so it scores today's code
and not a frozen run.

**What is still open:** deciding whether adaptation is switched on, and with what safeguard. As the
end of this document says, the failure mode is silent and it has to be possible to go back to the
original model.

### 4. RF-DETR on the ground, end to end — DONE on 17sep (`39f8f86`)

The button exists and is pinned by `tests/test_gs_segunda.js`, which walks the whole flow:

```
  click on drone 2's       : {"ruta":"/mirar","cuerpo":{"dron":"2"}}
  while waiting            : both phases are visible and neither asserts a result
  it answers               : 4 people on the ground, in 1.44 s
  and only the drone asked : drone 2's card still shows no answer
```

1.44 s against the 5 s criterion. The button sits next to the drone and not next to the map point,
because the second opinion is asked of **a drone**: it is its frame that gets looked at.

**What is left, and it belongs to the portfolio and not the product:** the comparison video (item 9)
is cited in `docs/README.md` and embedded nowhere.

### 5. "Fixed target" mode — DONE on 17sep, NEGATIVE

Lowering the threshold only inside the target window takes recall over the target from 43.9 to
60.7 %, without a millisecond of extra compute and reversibly. **Run through the whole chain it
changes nothing**: the same five people and the same six phantoms, across the four variants tried,
with the target set on the operator and on the most fragile candidate (commit `e186c50`).

Seventeen points of recall that buy no person. It is the measured ceiling of the cheap version of
"let the click teach", and the reason the route is the appearance template of item B.

### 6. Rejoin a person after a gap — BUILT AND SWITCHED OFF, because of the threshold

The criterion **is met**, and `tests/test_reunir_movil.py` measures it:

```
  as it ships (off)                  A(1) B(1) C(1) G(3) H(1)   6 phantoms
  switched on (0.63, 30 s)           A(1) B(1) C(1) G(1) H(1)   6 phantoms
  a little looser (0.70)             A(1) B(1) G(1) H(1)        6 phantoms
```

G goes from three points to one without losing anybody and without one more phantom. H was already
one.

**Why it is still off**, and it is the reason not to skip: between joining the walking woman and
erasing the boy on the balcony there are **0.07** of OSNet distance, on a scale where two pieces of
the same person get as far apart as 0.81. The `EMB_DIST_REUNE = 0.63` was chosen by looking at the
flight it is judged against, and its own comment declares it: *"THIS NUMBER IS NOT SAFE AND THAT IS
WHY REJOINING IS OFF BY DEFAULT"*.

**Done when:** the 0.63 is re-chosen on a flight that is not the test one. That is item 1 again.

## Portfolio

The declared objective is getting work as a vision or perception engineer. For that the system
**does not need to be finished**, it needs to be defensible and quick to understand.

### 7. A page understood in 40 seconds — STANDING, the middle is missing

`README.md` already opens with the claim and the number, and already has the **What it does not do**
section, which is the part a reader who hires values: the 5 people of 7 with their 6 phantoms, the
44 % the detector loses, the measurement that getting closer does not fix it, the 46.2 % against
90.5 % of the model that cannot fly, that the system never touches the flight, and that it was never
tried over water.

**What is missing:** the middle. Between the headline and the limitations there are usage and
camera-contract sections, which are for whoever is going to work in the repo, not for whoever is
judging it in 40 seconds. And the comparison video (item 9) is cited but not embedded.

### 8. The training runs that lost, told as a result — DONE

There are four, not three, and `DESCARTADO.md` already tells them as an argument and not as a record:
the table with each one's number, and below it the lesson that was not obvious, that run 3 won when
read through recall and precision and lost when read through people, **and that this is why the
people-and-phantoms scoreboard exists**. That is the argument: the failure explains why the system is
measured the way it is.

And all four were handicapped by one line of Ultralytics that reinitialises the classification head,
so none of them was fine-tuning anything. It is stated there with the literal message.

`README.md` sends the reader to that document saying what it is for: *"the document to read first if
you are judging the method rather than the result"*.

### 9. The comparison video as a centrepiece — ONE STEP MISSING, AND IT IS NOT CODE

`docs/yolo26_vs_rfdetr.mp4` shows 46.2 % against 90.5 % with the running total on screen, and it is
now cited in the **What it does not do** section of `README.md`, which is where whoever judges the
repo will see it.

**What is missing is uploading it.** Binaries do not enter the repository, by the same convention as
`demo.gif`: it is uploaded to a GitHub issue and the URL pasted in. The gap with the instruction is
already in the README, in the exact place. It is a manual step and nobody but the author can take it.

**Done when:** the video plays inside the README without the file being versioned.

## What NOT to do

It is measured and exhausted, the detail in `DESCARTADO.md`:

- **More training runs with another recipe over the same data.** Three lost.
- **More labelling of the flights already held.** The material is squeezed dry: what is left
  unlabelled above 12 m consists of immediate neighbours of what is already labelled.
- **RF-DETR as a teacher while there is no new footage.** Pseudo-labelling the same flights would
  triple the overfitting, not the variety.
- **Flying closer so the detector sees better.** Measured on 30sep over the 764 labelled misses of
  02ago: they are not small (median 49.3 px, only 8.2 % below the 28 px the detector was trained
  for). And comparing **within the same frame**, where the drone's altitude and the scene are
  identical, in 59.1 % of the 176 frames the detector missed a person **larger** than one it found in
  the same image (f2586: missed 186 px and found 141). The failure is not one of resolution, so more
  pixels do not fix it. Digital zoom had already measured worse (43.9 → 11.2 → 2.3 %). The signal
  that remains is **shape**: height/width ratio 1.27 in the missed against 1.79 in the found, and
  wider than tall 20.0 % against 0.9 %. For recall the route is the detector (item 4, second opinion
  on the ground), not the manoeuvre.
- **In-flight weight adaptation without being able to revert.** Adaptation works, but the failure
  mode is silent: both models have to be held and going back to the original has to be possible.

## Future work: visual servoing (closing the loop)

Today the system is **open-loop perception**: it looks, computes the position and reports it, but
never touches the flight. The only commands a GrADyS protocol can issue are `GotoCoords`,
`GotoGeoCoords` and `SetSpeed`: coordinates and speed, nothing about the image.

Closing the loop is called **visual servoing**. Two variants: **IBVS** (image-based), where the error
is measured in pixels, and **PBVS** (position-based), which first turns the image into a position.
The system already does the hard half of PBVS: `pinhole_local.py` + `identity.py` turn image into
metres. What is missing is using that result to send a command.

**The primitives already exist in `uav_api`**, they are simply not exposed as a GrADyS command:

    /drive_body, /drive_body_wait    velocity in the drone's own frame
    /travel_at_ned                   velocity
    /set_heading, /set_yaw_rate      heading and turn rate

**The case that justifies it:** at 40 m a person is 42 px and the detector fails past 24 m. A loop
holding the target at a fixed size in pixels (say 80 px tall) chooses the altitude on its own,
instead of fixing it by hand before taking off.

**What makes it hard, and why it is not done:**
- A new command has to be exposed in GrADyS, or uav_api has to be called directly from the protocol
  bypassing the abstraction, which is exactly what the interface exists to prevent.
- A closed loop at 3 FPS with a detector that loses 45 % of the frames runs out of signal every time
  the detector fails. What the controller does meanwhile has to be decided.
- It changes what the system IS: from "I observe and report" to "I observe and fly", which is much
  harder to defend as safe to an operator.
