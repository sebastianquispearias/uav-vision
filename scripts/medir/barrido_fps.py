"""Sweeps the declared frame rate on the bench and records what each board actually does with it.

The question this answers is not "how fast is the board", which a thirty-second run will happily
overstate. It is "what rate can this board hold for the length of a flight", and those are
different numbers because the limit is thermal. Measured on 2026-10-09: at one frame per second
the Pi 4 delivered 100 % of what it declared and looked perfect after two minutes, and reached
80 C -- the soft throttle limit -- after eight. A flight is fifteen to twenty.

So each point of the sweep is a full relaunch of the bench at that rate, held for long enough
for the temperature to show its slope, sampled every half minute. What comes out is a table with
one row per sample, which is what lets the temperature be plotted against time per rate rather
than reduced to a single number that hides the climb.

BOTH BOARDS RUN THE SAME RATE AT EACH POINT, on purpose. The interesting result is not each
board's own ceiling in isolation but where the two diverge: the Pi 5 flat at 52 C while the Pi 4
climbs past 80 is the whole argument, and it only exists if they were asked for the same thing.

    python scripts/medir/barrido_fps.py --estacion 10.213.110.80:8300 \
        --placas pi@10.213.110.141 pi@10.213.110.186

Costs roughly twelve minutes per rate: two to bring the bench up, ten to hold it. The CSV is
written row by row as the sweep runs, so a sweep that is interrupted still leaves everything it
measured.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))      # scripts/medir
RAIZ = os.path.dirname(os.path.dirname(AQUI))          # la raiz del repositorio
LEVANTAR = os.path.join(RAIZ, "scripts", "banco_embedded", "levantar_banco.sh")

MISION = "mision_banco_lab:ProtocoloLab"
CAMPOS = ["t_s", "fps_declarado", "dron", "fps_real", "fps_pedido", "temp_c",
          "slots_perdidos", "slots_total", "throttle_ahora", "bateria_v",
          "mem_mb", "lat_ciclo_p50_ms", "lat_ciclo_p95_ms", "lat_detector_p50_ms",
          "lat_muestras"]


def estado(base, timeout=5):
    with urllib.request.urlopen(base + "/estado", timeout=timeout) as r:
        return json.loads(r.read())


def esperar_drones(base, cuantos, limite_s):
    """Waits until `cuantos` drones have reported a real rate, or gives up."""
    t0 = time.time()
    while time.time() - t0 < limite_s:
        try:
            d = estado(base).get("drones") or {}
            if sum(1 for f in d.values() if f.get("fps_real")) >= cuantos:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def levantar(estacion, placas, fps):
    """Brings the bench up with every board declaring the same rate."""
    entorno = dict(os.environ,
                   BANCO_MISION=MISION,
                   BANCO_FPS=" ".join(str(fps) for _ in placas))
    return subprocess.run(["bash", LEVANTAR, estacion] + list(placas),
                          env=entorno, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estacion", required=True, help="ip:puerto de la estacion, p.ej. 10.213.110.80:8300")
    ap.add_argument("--placas", nargs="+", required=True, help="usuario@ip de cada placa")
    ap.add_argument("--tasas", nargs="+", type=float, default=[0.5, 1, 2, 3, 4])
    ap.add_argument("--minutos", type=float, default=10.0, help="cuanto sostener cada tasa")
    ap.add_argument("--intervalo", type=float, default=30.0, help="segundos entre muestras")
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "barrido_fps.csv"))
    args = ap.parse_args()

    # The address passed is the one the DRONES are told to reach, which is this same laptop by
    # another route. This process asks on the loopback instead, and on the literal 127.0.0.1
    # rather than "localhost", which on Windows costs about a second per request resolving IPv6
    # first -- an invisible second, multiplied by every sample of every rate.
    consulta = "http://127.0.0.1:" + args.estacion.rsplit(":", 1)[1]

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    nuevo = not os.path.isfile(args.salida)
    fh = open(args.salida, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=CAMPOS)
    if nuevo:
        w.writeheader()
        fh.flush()

    print("barrido de %s, %.0f min por tasa, muestra cada %.0f s"
          % (", ".join(str(t) for t in args.tasas), args.minutos, args.intervalo))
    print("salida: %s" % args.salida)

    for tasa in args.tasas:
        print("\n=== %s FPS declarados en las %d placas ===" % (tasa, len(args.placas)))
        r = levantar(args.estacion, args.placas, tasa)
        listo = [ln for ln in (r.stdout or "").splitlines() if "setup:" in ln or "entorno" in ln]
        for ln in listo[-4:]:
            print("   " + ln.strip()[:110])
        if not esperar_drones(consulta, len(args.placas), 240):
            print("   NO reportaron las %d placas: salto esta tasa" % len(args.placas))
            continue

        t0 = time.time()
        while time.time() - t0 < args.minutos * 60:
            try:
                d = estado(consulta)
            except Exception as e:
                print("   la estacion no contesta: %s" % e)
                time.sleep(args.intervalo)
                continue
            t = round(time.time() - t0, 1)
            for ident, f in sorted((d.get("drones") or {}).items()):
                salud = f.get("salud") or {}
                w.writerow({
                    "t_s": t, "fps_declarado": tasa, "dron": ident,
                    "fps_real": f.get("fps_real"), "fps_pedido": f.get("fps_pedido"),
                    "temp_c": f.get("temp_c"),
                    "slots_perdidos": f.get("slots_perdidos"),
                    "slots_total": f.get("slots_perdidos_total"),
                    "throttle_ahora": "|".join(salud.get("ahora") or []),
                    "bateria_v": f.get("bateria_v"),
                    "mem_mb": f.get("mem_mb"),
                    "lat_ciclo_p50_ms": f.get("lat_ciclo_p50_ms"),
                    "lat_ciclo_p95_ms": f.get("lat_ciclo_p95_ms"),
                    "lat_detector_p50_ms": f.get("lat_detector_p50_ms"),
                    "lat_muestras": f.get("lat_muestras"),
                })
            fh.flush()
            linea = "  t+%5.1f s " % t
            for ident, f in sorted((d.get("drones") or {}).items()):
                linea += " | dron %s: %-5s/%-4s FPS  %-5s C" % (
                    ident, f.get("fps_real"), f.get("fps_pedido"), f.get("temp_c"))
            print(linea)
            time.sleep(args.intervalo)

    fh.close()
    print("\nbarrido terminado. %s" % args.salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
