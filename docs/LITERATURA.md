# Papers consulted, what they gave and what is still untested

Only what was read and checked against our own data. Every entry says **what it gave us**, which is
usually not what the paper promises.

## Small object detection from the air

**[Slicing Aided Hyper Inference (SAHI), arXiv 2202.06934](https://arxiv.org/pdf/2202.06934)**
Reports +6.8 / +5.1 / +5.3 AP on VisDrone and xView, without retraining, by cutting the image into
tiles at native resolution.
**What it gave us:** the technique works here but yields less, and we know why. Replacing the frame
with the tiles does not pay; **adding them does**: +2.3 total recall and +2.9 on the balcony, at the
same precision. The difference from the paper is explained by our model being specialised in people
of about 28 px and getting worse when larger ones arrive.

**[Evaluation of YOLO Models with Sliced Inference, arXiv 2203.04799](https://arxiv.org/pdf/2203.04799)**
**What it gave us:** confirmation of the order of magnitude to expect.

**[SAHI-Improved-YOLOv8 for UAV imagery](https://www.sciencedirect.com/science/article/pii/S2772375525004125)**
**What it gave us:** the same pattern applied to a drone case.

**[Maritime Small Object Detection with Altitude-Aware Dynamic Tiling, arXiv 2511.19728](https://arxiv.org/pdf/2511.19728)**
Tiles adapted to the flight altitude.
**Untested, and it fits:** use tiles **only when the drone is high**, which is when they pay.
We have `persona_px()` measuring the expected size by altitude, so the rule would come out of our
own data.

## Interaction with the operator

**[UAVDB: Point-Guided Masks for UAV Detection and Segmentation, arXiv 2409.06490](https://arxiv.org/pdf/2409.06490)**
A point as a prompt to generate masks on drone imagery.
**What it gave us:** the name of what the user proposed (point-guided detection) and the
confirmation that it is a live line of work. Our cheap version is measured: lowering the threshold
only inside the target window gives +17 points of recall at no compute cost.

**[A reliable UAV tracking system with online re-detection network](https://www.sciencedirect.com/science/article/abs/pii/S0019057825004744)**
Uncertainty estimation and online re-detection.
**Untested, and it is what we are missing:** having the system know **when it stopped being sure**
and go looking again, instead of carrying on asserting. It is exactly the problem of G and H split
in two.

## Handoff between drones

**[Continuous Marine Tracking via Autonomous UAV Handoff, arXiv 2507.12763](https://arxiv.org/pdf/2507.12763)**
82.9 % target coverage at 4-5 Hz on a Jetson Nano. The handoff is resolved with position and **an
ArUco marker stuck to the other drone**.
**What it gave us:** that handoff works, and that they solve it by geometry rather than by
recognising the target. We already send the OSNet embedding between drones, so we have a different
route to hand.

**[Multi-Drone based Single Object Tracking with Agent Sharing Network, arXiv 2003.06994](https://arxiv.org/pdf/2003.06994)**
**Unexplored.**

## Compute split between drone and ground

**[Supporting UAVs with Edge Computing, arXiv 2310.11957](https://arxiv.org/html/2310.11957v1)**,
**[Offloading Deep Learning Vision Tasks from UAV, arXiv 2302.01991](https://arxiv.org/pdf/2302.01991)**,
**[Real-time UAV object detection and task offloading](https://link.springer.com/article/10.1007/s11370-026-00736-z)**

The warning that served us most: **offloading simple tasks degrades the drone**; the optimum is
usually to compute onboard.
**What it gave us:** the split we implemented. One frame is 303 KB, that is 7.3 Mbps at 3 FPS:
sending the whole video is not an option. Sending **one frame on demand** is, and that is what was
wired.

## Ideas without a source yet

- **Distil RF-DETR into yolo26.** RF-DETR finds 90.5 % where ours finds 46.2 %. As a teacher over
  **new** footage it would be the most promising route. Over the flights we already have it is no
  use: it would triple the overfitting, not the variety.
- **Adapt only the appearance template instead of the weights.** Measured: the target rises from
  91.1 to 94.4 % by accepting doubtful detections that resemble it, without touching a single
  weight and with immediate rollback.
- **Have the operator's verdict teach.** Today "not it" only erases the point from the map. It is
  the only route against the phantom that neither CLIP nor physical size detects.
- **Tiles only when the drone is high**, following the adaptive tiling paper, using our own
  measurement of size by altitude.
