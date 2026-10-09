# Bench measurements, 9 October 2026

What a Raspberry Pi 4 and a Pi 5 really do with the vision chain, measured rather than estimated.
The report built from these files: `informe.html`.

## The headline

| | Pi 5 | Pi 4 |
|---|---|---|
| median per frame | 130 ms | 720 ms |
| ceiling by compute | 7.7 fps | **1.4 fps** |
| at 1 fps, after 8 min | — | **79.9 °C**, still climbing |
| throttle word at shutdown | `0x0` | `0xe0000` |

`0xe0000` is the board's own firmware saying it hit the soft temperature limit, throttled, and
capped the ARM frequency. It is not inferred from the temperature curve.

That one number explains the rest. Three frames per second need a cycle under 333 ms and the
Pi 4 takes 720, so it is not slow, it is arithmetically unable. Even asked for half a frame per
second it works 36 % of the time against the Pi 5's 7 %, which is the whole of the thermal gap.
And the detector is essentially the entire cycle on both boards: ray projection, tracker and the
identity layer together do not add a measurable millisecond.

## The files

| file | what it holds |
|---|---|
| `barrido_fps.csv` | 20 samples at 0.5 fps, both boards: rate, temperature, latency, memory, battery |
| `temperatura_1fps.csv` | the 8-minute thermal curve at 1 fps that reaches the throttle limit |
| `precision_por_tasa.csv` | people found and false positives at four simulated rates |
| `estado_pi5.txt`, `estado_pi4.txt` | temperature, throttle word, uptime and memory at shutdown |
| `logs_pi5.tgz`, `logs_pi4.tgz` | each board's runner log and its bench config |
| `estacion.log`, `barrido_consola.log` | the ground station and the sweep's own console |

## How to reproduce

    python scripts/medir/barrido_fps.py --estacion <ip:puerto> \
        --placas pi@<pi5> pi@<pi4> --tasas 0.5 1 2 3 4 --minutos 10

Ten minutes per point is not caution. It is how long the temperature takes to show its slope,
and at two minutes it lies: the Pi 4 reads 66 °C and delivers 100 % of what it declares.

For the accuracy column, which needs no hardware:

    python scripts/replay_vuelo3.py --pistas=demo/data/pistas_bot_cmc_sof.npz \
        --preliminares --submuestreo=N --candidatos=<salida>.json
    python scripts/personas_encontradas.py --candidatos <salida>.json

The external tracks and `--preliminares` are not optional: with the replay's own tracker the
baseline scores 1 of 7 instead of 5 of 7, and nothing after that compares.

## What these numbers do not say

The accuracy sweep is one flight and seven people. The 4 of 7 at 0.55 fps is noise, since with
seven people four and five differ by one person.

And the 95th percentile of latency is not yet meaningful: at half a frame per second one look
fits in a report interval, so p95 and the median are the same sample. It needs 3 to 4 fps, where
six or eight looks fit.
