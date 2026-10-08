# Probes

Each one answers a question that came up on the bench and could not be settled by reading code.
They live here because the answers are decisions: the numbers below changed what the system does.

| Probe | Where it runs | What it answered |
|---|---|---|
| `clon_pelado.py` | any machine | **The first command of the README did not work.** Runs anything as if this repository had just been cloned alone, by BLOCKING the sibling runtime's import rather than moving directories -- `mv` failed with "Device or resource busy" and the loop around it reported 50 of 50 passing while the siblings were still there. |
| `cierre_de_bucle.py` | any machine | **A wedged camera retry could report a frame that never arrived.** Reproduces the closure bug ruff found in `OnboardCamera._first_frame`, and the fix, side by side. |
| `sin_comentarios.py` | any machine | Strips loose comments BY TOKENIZER. A regex over `gs_mapa.py` destroys all 33 of its CSS colours and the file still compiles. |
| `umbral_invertido.py` | the laptop | **Why OC-SORT tracked 6 of 2637 detections, and it was our bug.** BoT-SORT caps a cost of `1 - IoU` and boxmot floors the IoU itself, so the calibration's most permissive value arrived as the strictest possible one. Converting it takes ocsort from 6 boxes with a track id to 265, and to 487 of 764 once its consecutive-frame output gate is also accounted for, against BoT-SORT's 454 |
| `probar_pi4.py` | a board | Does this board detect, or only capture? Six frames through the real camera |
| `medir_vuelo.py` | a board | What one frame of the FLIGHT configuration costs. Pi 5: 199 ms, 5.04 FPS |
| `medir_reid.py` | a board | What the appearance model costs per frame. Useless with an empty room: zero boxes means it never runs |
| `hueco_por_tamano.py` | the laptop | Does the appearance gap survive when people are small? Under 35 px there are 5 detections in the whole flight: not measured |
| `hueco_promediado.py` | the laptop | The same gap on averaged vectors, which is what the system compares |

The two on the laptop read `demo/data/examen_v3_datos.npz` and the ground truth in
`scripts/personas_encontradas.py`, so they need no boards:

    PYTHONPATH="$PWD:$PWD/../gradys-embedded" python scripts/medir/hueco_promediado.py

The three on a board are copied over with `scp` and run with `PYTHONPATH=$HOME/banco`. The runner
holds the camera, so it has to be stopped first or picam2 fails with "Camera __init__ sequence did
not complete", which does not name the cause:

    ssh pi@<placa> 'pkill -f "[g]radys_embedded.runner.cli"'

The brackets are what keep that pattern from matching its own command line and killing the ssh.

## The MAVLink probes

Four read the bus directly instead of going through `uav_api`, which matters because `uav_api`
binds its UDP port exclusively and the bench's fake pilot wants port 8000. They connect as an
extra client to the router's **server** endpoint, so nothing has to be stopped:

| Script | Answers |
|---|---|
| `quien_es_el_fc.py` | Which component is the autopilot. Run this FIRST |
| `diag_mav_sr.py` | What is actually arriving, and every `SRx_*` stream rate |
| `bateria_real.py` | The real pack voltage, per cell, while the bench shows a simulated one |
| `aplicar_en_caliente.py` | Turns EXTENDED_STATUS on for a link that is already up |

They are copied over and run on the board, which is where the router is:

    scp scripts/medir/bateria_real.py pi@<placa>:~/ && ssh -n pi@<placa> 'python3 ~/bateria_real.py'

**`wait_heartbeat()` latches the FIRST heartbeat, and that is usually not the autopilot.** On
this airframe the ground station announces itself as sysid 250 and the probe's own heartbeat is
echoed back by the router, both of them `MAV_AUTOPILOT_INVALID`. A `PARAM_SET` addressed to
either is accepted by the socket and lands nowhere. `quien_es_el_fc.py` exists to name the right
one, and on this aircraft it is **sysid 3, component 1**.

**A stream rate written to `SRx_*` does nothing to a running autopilot.** ArduPilot turns those
parameters into message intervals once, when the channel initialises, and never re-reads them.
The parameter is for the next boot; a live link also needs `aplicar_en_caliente.py`. Measured on
2026-10-08: with `SR2_EXT_STAT` already at 2, `SYS_STATUS` was absent from 557 packets in 15 s,
and one `REQUEST_DATA_STREAM` brought it to 2 Hz.
