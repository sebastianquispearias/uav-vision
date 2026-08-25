# uav-vision

Detection and camera geometry for vision-based target geolocation from UAVs.

The same interface runs in **simulation** (GrADyS-SIM) and on the **real drone**
(Raspberry Pi + camera). The package imports nothing from GrADyS, the simulator
or MAVLink, except for the single protocol module noted below — it can be
installed on its own.

## The camera contract

The whole module reduces to one method, exposed identically by both cameras:

```python
camara.ver_alvo(pos, yaw) -> list[dict]
```

| Input | Type | Meaning |
|---|---|---|
| `pos` | `(x, y, z)` float, meters | camera position, local ENU frame (x=East, y=North, z=Up) |
| `yaw` | float, degrees | heading. 0 = North, 90 = East, clockwise |

The output is a list of detections; an empty list means nothing was detected.

```python
[{'px': 960.0, 'py': 805.0, 'conf': 0.846}, ...]
```

| Field | Required | Meaning |
|---|---|---|
| `px` | yes | horizontal center of the detection, pixels |
| `py` | yes | **bottom edge** — the point where the object touches the ground |
| `conf` | yes | detector confidence in (0, 1] |
| `emb` | no | appearance embedding: 512 normalized `float32` (OSNet) |
| `track_id` | no | stable identity assigned by the tracker |

`py` is the bottom edge because a standing person touches the ground with
their feet; using the box center would aim the ray at the waist and place the
ground intersection too far. `conf` feeds the view selector and is not
informational. Optional fields appear only when the camera can compute them —
read them with `det.get(...)`, never `det[...]`.

## Usage

```python
from uav_vision.camera import CamaraSimulada, CamaraArduCam

# simulation
camara = CamaraSimulada(alvo=(0.0, 0.0, 0.0), pitch_deg=-55.0)

# real drone, full pipeline: detector + embeddings + tracker
camara = CamaraArduCam(modelo="best_ncnn_model", rastreador=True, fps=4.0,
                       reid_modelo="osnet_x0_25_msmt17.pt")

# from here on the consumer code is identical
for det in camara.ver_alvo(pos, yaw):
    ...
```

The constructors differ on purpose — the simulated camera needs to know where
the target is, the real one needs a detector model. Only `ver_alvo` must
match. `CamaraArduCam` imports `picamera2`, `ultralytics` and `boxmot` on the
first capture, so the package imports cleanly on machines without them.

`correr.py` runs the full geolocation pipeline of the paper in either world:

```bash
python correr.py sim     # geometric camera, no images
python correr.py dron    # real captures + YOLO
```

## The GrADyS protocol

`vision_protocol.py` — the only module that imports `gradys_embedded` — wraps
the camera as an observe-only GrADyS protocol: it never sends mobility
commands, so it composes with any mission. On a timer it captures, detects,
back-projects each detection to a ground impact and feeds the identity layer;
every report period it broadcasts the consolidated POI list as JSON.

- **Position** arrives through `handle_telemetry` (the shared local frame).
- **Yaw** is not part of GrADyS telemetry; on the drone it comes from
  `UavApiYaw`, which polls the `uav_api` HTTP service on localhost (that
  service owns the MAVLink serial link). In simulation a function is injected.
- **Configuration** goes through
  `VisionProtocol.with_config(camera=..., pitch_deg=..., yaw_source=...)`,
  because GrADyS instantiates protocols with no constructor arguments.

`identity.py` turns tracked detections into named POIs: image-plane tracks are
summarized on the ground, classified static vs mobile, and merged by position
and appearance under a co-occurrence veto. Thresholds are parameters whose
defaults come from measurements — see `NOTES.md` for the provenance of every
number.

`manifest.yaml` describes the module as a discoverable skill (capabilities,
requirements, activation, message schema) for the ground-station LLM agent.

## Installation

```bash
pip install -e .              # laptop / simulation
pip install -e ".[dron]"      # + ultralytics, boxmot (picamera2 ships with Raspberry Pi OS)
```

## Tests

```bash
python tests/test_contrato.py          # both cameras honor the same contract
python tests/test_identidad.py         # identity rules against known ground truth
python tests/test_vision_protocol.py   # end-to-end synthetic flight (needs ../gradys-embedded)
python tests/test_rastreador.py        # tracker wiring (needs boxmot)
python scripts/replay_vuelo3.py        # replay of a recorded real flight through the protocol
```

## Contents

| File | Purpose |
|---|---|
| `camera.py` | the two cameras; the contract |
| `pinhole_local.py` | pixel ↔ ray projection, pure math |
| `camera_config.py` | camera intrinsics |
| `confidence.py` | confidence-to-noise models |
| `identity.py` | tracked detections → named POIs (static/mobile) |
| `vision_protocol.py` | the GrADyS protocol; only module importing `gradys_embedded` |
| `fusion.py` | robust multi-view triangulation (RANSAC and variants) |
| `view_selection.py` | joint geometry+confidence view selector |
| `noise.py` | sensor noise models for simulation |
| `manifest.yaml` | skill manifest for the LLM agent |
| `NOTES.md` | decision log: where every calibrated number comes from |

## Why this package exists

The pixel-to-ray math used to live in four places (the onboard script, the
offline pipeline, an earlier field script and the simulator showcase) with
different signatures and different constants. One copy with a stale mount
angle produced a position error larger than the total error the method
reports. With a single module imported by everyone, that divergence cannot
happen.

## Known limitations

- POI coordinates are in the local mission frame, not lat/lng; conversion
  requires the mission origin from the ground station.
- Flat-terrain assumption (ground plane at a fixed z).
- The full onboard chain (camera + tracker + identity on the Pi) has been
  validated piecewise and in replay, not yet end-to-end in flight.
- Focal length and principal point come from flight logs; checkerboard
  calibration is pending.
