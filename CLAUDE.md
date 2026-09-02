# `uav_vision` — bearing-only geolocation from a single UAV camera

**Read `ESTADO_SESION.md` first.** It is the resume point: what is in flight, what was last
done and why, and what comes next. This file is the standing context; that one is the news.

## What this is

A drone camera finds people (or any enabled class) on the ground and reports where they are,
in metres, to a ground station. One camera, no lidar, no stereo. It runs at 3 FPS on a
Raspberry Pi 5 and its median error on real flights is 2.4 m.

The chain: `camera.py` (capture → YOLO → BoT-SORT → OSNet ReID → crop) → `pinhole_local.py`
(pixel → bearing ray) → ground intersection → `identity.py` (accumulate per track, classify
static/mobile, merge tracks into candidates) → `vision_protocol.py` (the GrADyS protocol that
broadcasts POIs) → `scripts/banco_embedded/gs_mapa.py` (the ground station, with a map).

**The goal right now is a PORTFOLIO to get hired** as an ML / computer vision / perception
engineer — not the thesis, not BeEyes. Titles to aim at: *Edge AI / Embedded CV* and
*Perception Engineer*. That decides priorities: what a reader sees in 40 seconds beats what is
intellectually interesting. The reasoning, and what was ruled out, is in `ESTADO_SESION.md`.

## How to check the state in two minutes

```bash
for t in tests/test_*.py; do echo "== $t"; python "$t" >/dev/null && echo OK || echo FAIL; done
node tests/test_gs_filtro.js     # the ground station page, without a browser
python demo/demo.py --sin-mapa   # the flight-3 replay: must print 2.39 m
```

Tests are plain scripts, not pytest: each one prints its numbers and ends in `TODO OK`. They
are **empirical gates**, not unit tests — every section is a contrast that would fail if the
behaviour it claims were not true. Keep them that way: a test that cannot fail proves nothing.

## Conventions that are not negotiable

- **Docstrings and comments in English**, GrADyS style: descriptive prose in `"""..."""` that
  explains *what the thing is*. No session notes with dates inside the source — those go to
  `ESTADO_SESION.md`. The repo is read by the LAC group, in English.
- **Never `git add -A`.** A second Claude Code window may be working on this same tree. Stage
  explicit paths. `uav_vision/vision_protocol.py` and `tests/test_vision_protocol.py` are the
  other window's territory; coordinate before touching them.
- **No `Co-Authored-By: Claude`** and no mention of Claude in commit messages. This repo is
  shown to the group as the author's own work.
- **Measure, don't argue.** Before fixing a parameter or accepting a change, find data already
  on disk, measure, and paste the numbers. Before changing validated code, build an exact
  equivalence gate first — the flight-3 replay printing 2.39 m is that gate.

## Parameters that are decisions, not constants

`fusion_radius_m` has its formula in its own docstring (`gps_sigma + slant_range * yaw_sigma`)
and every term is in the telemetry. `MIN_MEASUREMENTS`, `DUTY_MIN`, `COOCURRENCIA_MIN` and the
detector threshold are **not** derivable from data: they encode the cost of being wrong, which
changes per mission. Do not present either kind as physically derived.

## Hardware notes that have bitten before

- The Pi's clock comes up wrong after a cold boot. `sudo date -s` before anything, or the logs
  cannot be crossed with the laptop's.
- Two `gs_mapa.py` can hold the same port without complaining. Check
  `netstat -ano | findstr :8300` before starting the ground station.
- `localhost` costs ~1 s per POST on Windows (IPv6 first). Use `127.0.0.1`.
- An `ssh` that launches background processes does not return the prompt even when they are
  `setsid`-ed. Launch and query in separate calls, with `ssh -n`.
