# Putting YOLO, ReID and tracking on a Raspberry Pi 5 — and the failure I almost blamed on software

This is the deployment half of a UAV geolocation project: a drone that finds people
on the ground from a single camera and reports their coordinates. The research half
is a paper. This is the part nobody writes down — what it actually costs to run a
detector, an appearance model and a tracker on a board that fits on an airframe.

Three things are worth your time here: a latency table, a failure that looked like
software and was not, and an ablation that says which of these numbers matter.

## The stack

- **Raspberry Pi 5**, CPU only. No GPU, no accelerator.
- **YOLOv8n fine-tuned on VisDrone** (3.2 M parameters), exported to **NCNN**.
- **BoT-SORT** for tracking, with **OSNet x0.25** appearance embeddings.
- Camera at 3 FPS, one report every 2 s.

## What it costs

Measured on the board, with the real camera, detector plus re-identification:

| input size | detector | + OSNet | FPS | CPU |
|---|---|---|---|---|
| 640  |  70.6 ms | 111 ms | 9.0 | 65 % |
| 960  | 160.7 ms | 201 ms | 5.0 | 80 % |
| 1280 | 311.7 ms | 352 ms | 2.8 | 90 % |

At 1280 the board sits at 90 % CPU. That is the edge of the hardware, and we are
already touching it.

**Moving from torch to NCNN bought 3.3x** — 12.47 FPS against 3.74 on the same
model and the same board. That trick works once. There is no second NCNN waiting.

## The failure that looked like software

The board kept dying mid-flight, always during inference. The obvious suspects were
all software: too many threads, thermal throttling, the model, the camera pipeline.

It was the power path. Specifically, the **BEC**: a 5 A unit could not follow the
current steps that inference produces.

The experiment was one variable. Same battery, same board, same SD card, same code.
Only the BEC changed:

| run | outcome | V min | A p95 | rows in undervoltage | kernel events |
|---|---|---|---|---|---|
| 7 A BEC | **survived the full 87 s** | 4.792 | 1.50 | **0** | **0** |
| 5 A BEC | died at 55 s | 4.458 | 1.19 | 153 | 1 |

Read the current column again. **The run that survived drew more** — 1.50 A at p95
against 1.19 A in the run that died — and never raised a flag. So it was never the
current. It was the delivery path: the 5 A unit could not follow the step.

Ten minutes of sustained load afterwards, ~14000 samples: 2.40 A peak, 11.71 W peak,
60.9 °C maximum, zero undervoltage events, `throttled=0x0`.

The lesson that generalises: when a board dies under load, measure the rail before
you touch the code. A plausible software explanation will always be available, and
it will be wrong in a way that costs weeks.

## The ablation that decides where to spend effort

The instinct on an embedded detector is to make the detector better. We measured
what that buys. Same 2983 frames from a real flight, same everything, only the
detector changed:

| | YOLO-640 | YOLO-1280 | RF-DETR |
|---|---|---|---|
| mAP50 on the benchmark | 0.305 | 0.464 | — |
| error to the target | 2.49 m | 2.45 m | 2.39 m |

**Raising mAP50 by 52 % moved the end-to-end error by 4 centimetres.** Everything
lands between 2.39 and 2.49 m because the floor is GPS and heading bias, not
perception.

That result is worth more than the speedup. It says the detector is not where the
error lives, and it says it with numbers rather than intuition — which means the
next optimisation went somewhere else.

## What I would tell someone starting this

1. **Measure the rail before you debug the code.** The most convincing wrong
   explanation is a software one.
2. **Export to NCNN early.** 3.3x is not a tuning gain, it is a different regime,
   and it changes what resolution you can afford.
3. **Measure end to end, not per stage.** A detector metric is not a system metric,
   and the gap between them is where the effort gets wasted.
4. **Declare rates in seconds, not frames.** A 40-frame tracker buffer is 24 s in
   one recording and 8 s on the aircraft. The same number means two different things.
5. **The observation rate is not the camera rate.** Ours captured at 8.7 FPS while
   the target was detected in ~30 % of frames — 2.45 observations per second. Every
   threshold expressed in frames was quietly asking for impossible evidence.

---

Code: https://github.com/sebastianquispearias/uav-vision — it ships a demo that
replays a real flight on a laptop, no drone required.
