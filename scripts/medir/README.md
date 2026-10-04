# Probes

Each one answers a question that came up on the bench and could not be settled by reading code.
They live here because the answers are decisions: the numbers below changed what the system does.

| Probe | Where it runs | What it answered |
|---|---|---|
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
