# What was tried and did not work, with its numbers

This document exists because an idea discarded without a number gets tried again. Every entry says
what was tried, what it gave, and **why it failed**, which is the only part that stops a repeat.

## Training runs

### Four consecutive training runs lost

| Run | Data | Result | Why |
|---|---|---|---|
| 1 | low flights only | 30.9 / 20.9 | boxes at the wrong scale: at IoU 0.5 it collapsed to 11.0. Trained at 165 px to be evaluated at 66 px |
| 2 | + VisDrone mixed in | 55.4 / 46.9 | fixed the scale but did not beat the baseline (54.3 / 60.9) |
| 3 | + 184 frames at 25 m | 57.5 / 76.6 | **won on totals and lost two people**: it overfitted to the single person in those 184 frames |
| 4 | + 14jun and everything mixed | 51.9 / 94.0 | spectacular precision, **4 people out of 7**. Loses G |

**The lesson that was not obvious:** run 3 won when read through recall and precision, and lost when
read through people. That is why the people-and-phantoms scoreboard exists.

### All four were handicapped by one detail of Ultralytics

```
Overriding model.yaml nc=10 with nc=1
```

The base model carries VisDrone's 10 classes. Fine-tuning it on a **one**-class dataset makes
Ultralytics **throw the classification head away and reinitialise it**. So none of the four was
fine-tuning: each was retraining a brand-new head from scratch on our handful of images.

**Not verified:** whether all four improve when the 10 classes are kept. That is the thing that
could overturn all four verdicts, and it was never tested.

## Inference

| Idea | Result | Why it failed |
|---|---|---|
| Run at 1280 instead of 960 | 52.9 / 56.2 (was 55.0 / 62.7) | the model is specialised in people of about 28 px and gets worse with larger ones |
| Run at 1920, shrinking nothing | 44.4 / 47.4 | the same, worse |
| Upscale the image over the target | recall 43.9 to **11.2** | the same reason, in its most brutal form |
| SAHI with tiles only, replacing the frame | 55.0 / 47.0 | a tile decides over a piece of scene and calls a shadow a person more readily |
| Raise the threshold on top of SAHI | 37.2 / 46.6 on the balcony | worse than the control **on both axes** |
| TTA (multi-scale and mirror) | impossible | this version of Ultralytics does not implement it for yolo26 |

**What did work** was not replacing but **adding**: whole frame plus tiles, which is in
`RESULTADOS.md`.

## Filters against the phantoms

| Idea | Result |
|---|---|
| CLIP as a second judge | **works**: phantoms 2 to 1, without losing people |
| Filter by physical size in metres | good at box level (30 % of the junk for 2 % of the people) but **does not beat CLIP** at candidate level |
| Lower the detector's NMS (iou 0.45) | **no effect at all**: NMS is per class and VisDrone's classes 0 and 1 are both people |
| NMS across classes | removes 44 boxes out of 2655 and **does not change a single candidate** |

**The phantom nobody kills:** a red object measuring 1.56 m (like a person) and scoring 1.81 on
CLIP (above the threshold). No technique tried touches it.

**A useful fact about why physical size is noisy:** a cone measured 2.11 m because the box includes
its shadow. The detector does not enclose the object, it encloses the object plus its shadow.

## Identity layer

| Idea | Result |
|---|---|
| Turn the speed rule off | fixes the fragmentation but **loses every mobile**: G stops being marked as moving |
| Demand temporal coverage from the speed rule | keeps the **false** mobile (the stationary operator) and loses the **real** one (G) |
| Have mobiles ask before opening a candidate | **works**: the operator from 2.37 to 1.92 m. Committed |
| ReID adjustments | they make it worse: they merge distinct objects |
| Demand density to confirm (duty-cycle floor in looks mode) | **reverted**: see below |

### The density floor: it separates on one flight and falls apart with another tracker

In `looks` mode, which is the one that flies, confirming a candidate is just counting twenty
one-second looks. That rewards permanence, and the most permanent thing in a scene is a static
phantom. It was crossed, candidate by candidate, against the letters a human put on every box of
the 02-ago flight:

```
of the 4 confirmed candidates, 3 are phantoms and 1 is the operator
of the real people B, C, G and H, none confirms
```

Density (the fraction of the detection frames within its own lifetime in which the candidate was
seen) separated them with a clean gap: phantoms up to **0.397**, people from **0.714**, nothing in
between. And a floor at 0.55 removed the three confirmed phantoms without touching the operator.

**Why it was reverted.** One test in the suite runs with the fallback tracker instead of BoT-SORT,
and there the operator scores density 0.126 and lands in the middle of the pack. The cause is
fragmentation: a target split into pieces keeps a long span and spreads its sightings, so its
density sinks. Tracks per candidate: with motion compensation the operator has 5 and the phantoms 1
to 3; with the fallback tracker the operator has 14.

**And the flight that flew had compensation off**, so the regime where the rule fails is the real
one. The per-track variant (the maximum over a candidate's tracks) separates in neither case: a
phantom with three tracks has one that is perfectly dense within its own window and scores 1.000.

**What it would take:** a second flight with compensation on, or a quantity that does not depend on
how the tracker splits the tracks. The density measurement itself stayed, as a measurement and with
no threshold, and the station's bar displays it.

## Labelling

| Idea | Result |
|---|---|
| Resolve overlapping pairs with "the largest wins" (the X key) | fails **25 %** at IoU 0.3 and 7.6 % at IoU 0.5, and would touch only about 100 pairs in a whole flight |
| Label more of the flights already held | exhausted: what is left above 12 m consists of immediate neighbours of what is already labelled |
| Label the 8 to 12 m stretches | a person is 168 px there, practically the same as at 3-8 m (183 px): more of what is already in surplus |

## Mistakes of method that cost time

- **Reading the letters file as if it were the truth.** The letters cover only the boxes the
  detectors proposed; two thirds of the balcony's truth carries no letter because it was drawn by
  hand. That produced two false alarms: "the balcony is unlabelled" and "recall is inflated". Both
  were false.
- **Scoring candidates outside the reviewed windows.** It marked a real person as a phantom.
- **Choosing a threshold by looking at the test set.** Corrected by re-choosing it on validation,
  which gave the same value.
- **Measuring with 200 sampled frames.** The gain of adaptation over the target looked like +7.6
  points and over the whole mission it is +2.9.
- **Cutting a training run short with `| head`.** The closed pipe killed the process at the first
  epoch.
- **Measuring the low band against the product scoreboard, over the replay.** It does not give
  zero: it cannot give anything. The track file holds 2,637 rows and the detections are 5,708,
  exactly those at confidence >= 0.25, because the tracker was run over that cut. The band's 3,071
  boxes reach the protocol **without a `track_id`**, and the identity layer only sees detections
  that carry a track. So neither the target window nor the appearance template can move a candidate
  in that replay, whatever they do: the three runs give a total `n_obs` of **1299**, to the unit.
  Any conclusion about the band drawn that way measures the plumbing, not the idea.
- **And re-running the tracker with the floor lowered is not enough either, for a worse reason.**
  It was run (`scripts/botsort_pistas.py --piso 0.10`, two configurations) and the result leaves
  the idea with nothing to measure on this flight:

  | confidence band | boxes | with `track_id` | |
  |---|---|---|---|
  | 0.10 - 0.20 | 2,443 | **0** (0.0 %) | below BoT-SORT's `track_low_thresh`, which drops them before association |
  | 0.20 - 0.25 | 628 | **30** (4.8 %) | the only part of the band the tracker can claim |
  | >= 0.25 | 2,637 | 1,284 (48.7 %) | above the reporting threshold |

  Of the band's 3,071 boxes, **30 can reach the identity layer**: 1 %. Against the base run's 1,299
  observations, that cannot move a candidate and never could. And tracking everything else gets
  worse when the band is fed in: with the flight's thresholds it assigns **860** tracks against the
  **1,608** of the floor at 0.25, so there is no clean A/B either, because the baseline moves with
  the change.

  **The design detail that comes out of it:** the band's floor and the tracker's floor have to match
  and they do not. The replay loads the band from 0.10 and `OnboardCamera` asks for it from 0.20,
  which is exactly `track_low_thresh`. Asking the detector for boxes below the tracker's floor is
  paying OSNet (~33 ms per box) for boxes that are dropped without being looked at.
