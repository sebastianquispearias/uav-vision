# The documents of this system

`ESTADO_SESION.md`, at the root, **is read first and always**: it is the resume point, with what is
in flight and the open threads. These documents do not replace it, they complement it: they are the
conclusions pulled out of the chronological order so they can be consulted on their own and shown to
somebody.

| Document | Answers |
|---|---|
| [RESULTADOS.md](RESULTADOS.md) | **What the system does**, with the numbers that hold up and the caveats that go with each one |
| [PLAN.md](PLAN.md) | **What comes next**, in order, and when each item counts as finished |
| [DESCARTADO.md](DESCARTADO.md) | **What has already been tried and did not work**, with the number and the reason, so it is not repeated |
| [LITERATURA.md](LITERATURA.md) | **What the literature says** and what it gave us, which is almost never what the paper promises |
| [ARQUITECTURA.md](ARQUITECTURA.md) | **How the pieces fit**: the ground station, the replay and the labelling tool, with the reason behind every knob |
| [../NOTES.md](../NOTES.md) | **Where every number in the code comes from** and why each decision was taken. At the root because it is the log, not a conclusion |

The split between the last three and the source is deliberate: **the docstrings say what each thing
is**, `ARQUITECTURA.md` says how it fits together, and `NOTES.md` says where the number came from. A
threshold measured on one flight does not belong in a docstring: it could change on another flight
and it does not describe how the thing works.

## Where to start depending on who you are

- **If you came to see whether the system is any use:** `RESULTADOS.md`, the people-and-phantoms
  table.
- **If you came to read the code:** `ARQUITECTURA.md`, and `scripts/sistema_esencial.py`, which is
  the whole chain in one runnable file.
- **If you came to work on it:** `PLAN.md`, and before proposing anything, `DESCARTADO.md`.
- **If you came to judge the work:** `DESCARTADO.md`. Measured failures say more about the method
  than successes do.

## Videos

| File | What it shows |
|---|---|
| `yolo26_vs_rfdetr.mp4` | the detector that flies (46 %) against one that cannot fly (90 %), with the running total |
| `base_vs_adaptado.mp4` | the same detector before and after adapting to the scene in 89 seconds |
| `antes_despues_detector_20260916.mp4` | two detectors over the same stretch, in the station's standard format |
| `sistema_arreglado_20260916.mp4` | the complete system: real camera and station, side by side |

The videos are not committed (`.gitignore`), they live on disk only.
