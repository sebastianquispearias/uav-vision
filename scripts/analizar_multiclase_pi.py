"""Reads what scripts/medir_multiclase_pi.py left on the Pi and says what the second class cost.

Per condition and repetition, both pooled, and a least-squares fit of latency against the number of
embeddings in the frame. The fit is the mechanism behind the headline: the class list does not
change inference, the boxes it lets through each cost an OSNet embedding and tracker work.

    python scripts/analizar_multiclase_pi.py [folder with resultados_*]
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "drone-geolocation", "entrenamiento", "pi_multiclase_13sep")
dirs = sorted(glob.glob(os.path.join(BASE, "resultados_*")))
if not dirs:
    sys.exit("no hay resultados traidos todavia")
D = dirs[-1]
filas = list(csv.DictReader(open(os.path.join(D, "frames.csv"))))
print("resultados:", os.path.basename(D), "|", len(filas), "frames medidos")

por = defaultdict(list)
for f in filas:
    por[(f["condicion"], int(f["repeticion"]))].append(f)

print("\n%-16s %4s %6s %9s %9s %7s %7s %7s" % ("condicion", "rep", "frames", "mediana", "p90", "FPS", "cajas", "embs"))
for (c, r), fs in sorted(por.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    lat = np.array([float(x["lat_ms"]) for x in fs])
    print("%-16s %4d %6d %6.1f ms %6.1f ms %7.2f %7.2f %7.2f" % (
        c, r, len(fs), np.median(lat), np.percentile(lat, 90), 1000 / np.median(lat),
        np.mean([int(x["cajas"]) for x in fs]), np.mean([int(x["embeddings"]) for x in fs])))

print("\nJUNTANDO REPETICIONES")
med = {}
for c in ("personas", "personas+coches"):
    fs = [f for f in filas if f["condicion"] == c]
    lat = np.array([float(x["lat_ms"]) for x in fs])
    med[c] = np.median(lat)
    print("  %-16s %4d frames | mediana %.1f ms | p90 %.1f ms | %.2f FPS | cajas %.2f | coches %.2f" % (
        c, len(fs), np.median(lat), np.percentile(lat, 90), 1000 / np.median(lat),
        np.mean([int(x["cajas"]) for x in fs]), np.mean([int(x["coches"]) for x in fs])))
d = med["personas+coches"] - med["personas"]
print("  diferencia de medianas: +%.1f ms (+%.0f %%)" % (d, 100 * d / med["personas"]))

emb = np.array([int(f["embeddings"]) for f in filas], dtype=float)
lat = np.array([float(f["lat_ms"]) for f in filas])
A = np.vstack([np.ones_like(emb), emb]).T
(a, b), *_ = np.linalg.lstsq(A, lat, rcond=None)
r2 = 1 - np.sum((lat - A @ [a, b]) ** 2) / np.sum((lat - lat.mean()) ** 2)
print("\nCOSTE POR HUELLA (regresion lat = a + b * embeddings, todos los frames)")
print("  a = %.1f ms (frame sin cajas) | b = %.1f ms por embedding | R2 = %.2f" % (a, b, r2))
print("  mediana de latencia por numero de embeddings en el frame:")
for k in range(7):
    m = emb == k if k < 6 else emb >= 6
    if m.sum():
        print("    %s%d: %4d frames, %.1f ms" % (">=" if k == 6 else "  ", k, m.sum(), np.median(lat[m])))

tel = list(csv.DictReader(open(os.path.join(D, "telemetria.csv"))))
print("\nTELEMETRIA (una muestra por segundo)")
grupos = defaultdict(list)
for t in tel:
    grupos[t["condicion"]].append(t)
for c, ts in grupos.items():
    temps = [float(x["temp_c"]) for x in ts if x["temp_c"]]
    cpus = [float(x["cpu_total"]) for x in ts if x["cpu_total"]]
    print("  %-18s %4d s | temp %.1f-%.1f C | CPU media %.0f %% | throttled %s" % (
        c, len(ts), min(temps), max(temps), np.mean(cpus), sorted({x["throttled"] for x in ts})))
