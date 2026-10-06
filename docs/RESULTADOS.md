# What the system does, measured

Everything here is measured on our own data and can be reproduced with the commands given. What is
not measured says "not measured" instead of an estimate.

**The caveat that applies to the whole document:** except where stated otherwise, the numbers come
from **one flight, one day, one place and seven people** (the 2 August flight). There is no second
evaluation confirming them.

## The metric that matters: people and phantoms

Per-box recall and precision do not say whether the system is useful. What an operator suffers is
how many real people reach the map and how many false points have to be walked to and dismissed.
That metric was built on 16sep from the letters the user assigned box by box.

| | the detector that flies |
|---|---|
| people reported | **5 of 7** |
| phantoms | **6** |
| error of the best POI | **2.29 m** |
| candidates formed | 15 |

Those are the numbers of the flight as it was recorded, which is what the repository ships and what
anyone who clones it runs. Until 17sep this document published 2 phantoms and 1.92 m: that row came
not from the flight but from an **offline re-detection** of the same frames, with 2655 boxes instead
of the 2637 that were recorded. It reproduces, and it is further down beside adaptation and tiles,
but it is not what flew nor what comes out of running this repository.

The two missing people (D and E) have 2 and 5 boxes in the whole flight: they are not found **even
with a perfect detector**, because the evidence threshold discards them on purpose.

Reproduced with `scripts/personas_encontradas.py`, and `tests/test_personas_encontradas.py` pins it.

## Whose problem each one is

The perfect-detector experiment: the chain was fed only the boxes the user marked as real people,
and what still failed was examined.

| | real detector | perfect detector |
|---|---|---|
| phantoms | 4 | **0** |
| the operator appears as | 3 points | **1 point** |
| G and H appear as | 2 points each | **2 points each** |

- **The phantoms are 100 % the detector's.** With perfect boxes they disappear.
- **The operator's fragmentation too.** Duplicate boxes cause it.
- **G and H splitting in two survives the perfect detector**: that one is the algorithm's, and the
  route would be rejoining by appearance after a gap. It was never attempted.
- **The error in metres is moved by neither.** Geometry sets it (GPS and yaw).

## The detector

Baseline of the model that flies (yolo26n VisDrone, imgsz 960, conf 0.25, IoU 0.3), against the
corrected truth of 1851 boxes:

```
2551-2641   recall 98.9 %   precision 85.4 %
2746-2952   recall 76.1 %   precision 61.1 %
3000-3700   recall 41.8 %   precision 55.5 %   <- the balcony, at 24 m: where it fails
TOTAL       recall 54.3 %   precision 60.9 %
```

**The size of the person explains almost all of it**, and it does not follow the optics because the
camera looks forward and down:

```
at  3- 8 m: 183 px    at  8-12 m: 168 px    at 12-18 m: 109 px
at 18-30 m:  62 px    at 30-99 m:  42 px
```

## A detector that cannot fly finds twice as many

RF-DETR with tiles, running on the ground, over 201 frames of the balcony:

```
YOLO26 onboard      46.2 % recall, 59.0 % precision,   35 ms/frame
RF-DETR on ground   90.5 % recall, 63.4 % precision, 1410 ms/frame
```

It is the biggest number we have and it requires training nothing. RF-DETR **will never fly** (40x
slower), but it can run on the ground over a frame the drone sends on demand, which is already
wired.

Video: `docs/yolo26_vs_rfdetr.mp4`.

## Measured improvements, without retraining

| Improvement | What it gives | Status |
|---|---|---|
| CLIP as a second judge | phantoms 2 → 1, without losing people | on the station, optional with `--clip-descarta` |
| Whole frame + tiles every 5 frames | recall 55.0 → 57.3 total and 39.5 → 42.4 on the balcony, same precision, but **not one person more** on the product scoreboard | in `camera.py`, off by default |
| Low threshold only inside the target window | recall 43.9 → 60.7 % over the target **against boxes, but zero change in people and phantoms**: the same 5 of 7 and the same 6, across 4 variants | in `vision_protocol.py` (`fix_target()`, `FOCO_RADIO_PX=320`), triggered by the operator's click |

The tile threshold (0.55) was chosen on the 01ago flights, not on the test set, and validation chose
the same value independently.

## In-flight adaptation

Adapting the detector to the scene during the flight, 91 frames, 3 laps, frozen backbone, 89 s:

```
                                   recall TARGET   recall ALL   precision
the one that flies                     91.1 %        49.6 %       58.3 %
adapted to the scene                   94.0 %        56.1 %       46.3 %
adapted with simulated clicks          94.0 %        56.1 %       46.3 %
```

**Adapting from operator clicks gives exactly the same as adapting from hand-drawn boxes**, because
of the 1851 boxes that come out of the click, 1217 are the detector's own boxes: the operator points
at who, the detector already knew where the edge was.

The requirement without which it does not work: **keep VisDrone's 10 classes** in the dataset. With a
single class Ultralytics reinitialises the classification head and everything collapses.

Video: `docs/base_vs_adaptado.mp4`.

## The two improvements, judged by people and phantoms

Measured on 17sep. The three detection caches were regenerated over **the same 1659 frames**, because
changing the set would change the cadence the tracker sees and the comparison would stop being about
the detector. All three pass through the tracker that flies (BoT-SORT with motion compensation) and
through the same identity chain.

```
                        base (re-detection)   + tiles every 5   adapted to the SCENE
  boxes conf>=0.25            2655                 2678              3629  (+37 %)
  tracks / with id          44 / 1889            43 / 1899         75 / 2222
  candidates                    13                   12                22
  people reported           5 of 7               5 of 7            5 of 7
  which ones                A B C G H            A B C G H         A B C G H
  PHANTOMS                       2                    2                 8
  not judgeable                  2                    2                 5
  error of the POI          1.92 m               2.49 m            1.12 m
  fragmentation            A(3) G(2)            A(2) G(2)         A(3) G(1)
```

**The tiles do not pay.** They add 23 boxes over 2655, find not one person more, add no phantoms, and
the point comes out 57 cm worse. The only thing they improve is that the operator appears as two
points instead of three. That costs six times the compute on the frame that carries tiles.

**Adaptation is a trade, not an improvement.** It finds not one person more, the same five and the
same letters; it **quadruples the phantoms**, from 2 to 8; and it **brings the point 42 % closer**,
from 1.92 to 1.12 m. The 12 points of per-box precision that adapting costs did turn into false
points somebody has to walk to and dismiss. Who wins that trade depends on the mission, not on the
metric.

Reproduced with `scripts/personas_encontradas.py` over the three caches, which were left in
`../drone-geolocation/entrenamiento/cadena_02ago/`.

## The complete chain

```
error of the best POI against the surveyed operator:  1.92 m
POIs reported at the end of the stretch:              6 (was 15)
boxes with a track id:                                80 % (was 63 %)
```

The repository's gate (`python demo/demo.py --sin-mapa`) prints **2.39 m** and is what pins the chain
against changing by accident.

## Operator load

```
false confirmations: 0 to 4 per hour    (requirement: <= 6)   MEETS IT
queue to verify: 27.7 -> 4.0 per hour with CLIP
```

## What is NOT measured

- What everything new costs on the Pi (tiles, frame on demand, adaptation).
- Any of the new work running on a real flight.
- Anything at all on a second flight, another day, another place.
