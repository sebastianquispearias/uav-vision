"""One video: where the chain reports a moving target, against where it is, before and after.

Flight 3 with the synthetic patrolling target of replay --sintetico (class 'boat', known ground truth,
detected only when in frame and as often as a real target was). The same seconds are drawn twice:
left, with the span rule every earlier number was measured with (--span); right, the chain as it is
now. Each map shows the target (a star), the reported point with its 95 % radius, and the error line
between them. Underneath, both errors over time.

    python scripts/render_blanco_movil.py --velocidad 4 --desde 80 --hasta 200

Writes an H.264 mp4 when ffmpeg is available.
"""
import argparse
import contextlib
import io
import math
import os
import runpy
import shutil
import subprocess
import sys

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
REPLAY = os.path.join(AQUI, "replay_vuelo3.py")
LAT0, LNG0, R = -22.978029946, -43.23214256266666, 6378137.0
E0, E1, N0, N1 = -32.0, 32.0, -14.0, 40.0
MW, MH = 960, 810
W, H = 1920, 1080
F = cv2.FONT_HERSHEY_SIMPLEX
VERDE, AMBAR, DRON, BLANCO, GRIS = (120, 230, 120), (60, 190, 250), (255, 190, 120), (245, 245, 245), (90, 90, 100)
COLORES = {"antes": (120, 120, 255), "ahora": (120, 230, 120)}


def enu(la, ln):
    return (math.radians(ln - LNG0) * R * math.cos(math.radians(LAT0)), math.radians(la - LAT0) * R)


def a_pix(e, n):
    return (int((e - E0) / (E1 - E0) * MW), int((1 - (n - N0) / (N1 - N0)) * MH))


def correr(v, extra):
    os.environ["UAV_VISION_DATOS"] = os.path.join(RAIZ, "demo", "data")
    os.environ.pop("UAV_VISION_GS", None)
    sys.argv = [REPLAY, "--preliminares", "--sintetico=%g" % v, *extra]
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(REPLAY, run_name="__main__")
    return [(float(r["time"]), r.get("pois") or []) for r in g["reportes"]], g


def barco_en(reportes, t, verdad):
    actual = []
    for tr, pois in reportes:
        if tr > t:
            break
        actual = pois
    b = [p for p in actual if "x" in p and p.get("cls") == "boat"]
    if not b:
        return None
    return min(b, key=lambda p: math.hypot(p["x"] - verdad[0], p["y"] - verdad[1]))


def mapa(titulo, poi, verdad, rastro, dron, color_modo):
    m = np.full((MH, MW, 3), 20, np.uint8)
    px_m = MW / (E1 - E0)
    for e in range(int(E0) // 10 * 10, int(E1) + 1, 10):
        cv2.line(m, a_pix(e, N0), a_pix(e, N1), (34, 34, 40), 1)
    for n in range(int(N0) // 10 * 10, int(N1) + 1, 10):
        cv2.line(m, a_pix(E0, n), a_pix(E1, n), (34, 34, 40), 1)
    cv2.line(m, a_pix(-25, 25), a_pix(25, 25), (60, 60, 70), 2)
    if len(rastro) > 1:
        cv2.polylines(m, [np.array([a_pix(*q) for q in rastro[-400:]], np.int32)], False, DRON, 1, cv2.LINE_AA)
    cv2.circle(m, a_pix(*dron), 7, DRON, -1, cv2.LINE_AA)
    g = a_pix(*verdad)
    cv2.drawMarker(m, g, BLANCO, cv2.MARKER_STAR, 26, 2, cv2.LINE_AA)
    cv2.putText(m, "blanco (verdad)", (g[0] + 14, g[1] - 12), F, 0.55, BLANCO, 1, cv2.LINE_AA)
    error = None
    if poi is not None:
        q = a_pix(poi["x"], poi["y"])
        c = VERDE if poi.get("mature") else AMBAR
        if poi.get("radius_m") is not None:
            cv2.circle(m, q, int(poi["radius_m"] * px_m), c, 1, cv2.LINE_AA)
        cv2.line(m, q, g, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(m, q, 9, c, -1, cv2.LINE_AA)
        error = math.hypot(poi["x"] - verdad[0], poi["y"] - verdad[1])
        estado = ("CONFIRMADO" if poi.get("mature") else "POR VERIFICAR") + (" MOVIL" if poi.get("mobile") else " (quieto)")
        cv2.putText(m, estado, (q[0] + 13, q[1] + 20), F, 0.55, c, 1, cv2.LINE_AA)
        cv2.putText(m, "error %.1f m" % error, (q[0] + 13, q[1] + 42), F, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    else:
        cv2.putText(m, "todavia sin reporte del blanco", (20, MH - 24), F, 0.65, GRIS, 1, cv2.LINE_AA)
    cv2.rectangle(m, (0, 0), (MW, 40), (14, 14, 18), -1)
    cv2.putText(m, titulo, (14, 28), F, 0.8, color_modo, 2, cv2.LINE_AA)
    cv2.rectangle(m, (0, 0), (MW - 1, MH - 1), (70, 70, 80), 1)
    return m, error


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--velocidad", type=float, default=4.0)
    ap.add_argument("--desde", type=float, default=80.0)
    ap.add_argument("--hasta", type=float, default=200.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "blanco_movil.mp4"))
    args = ap.parse_args()

    print("replay con la regla vieja (--span)...")
    rep_antes, g = correr(args.velocidad, ["--span"])
    print("replay con la cadena de ahora...")
    rep_ahora, _ = correr(args.velocidad, [])
    verdad_de, poses = g["posicion_sintetica"], g["poses"]
    t_base = float(poses[g["frames_aire"][0]]["t_mono"])
    frames = sorted(f for f, p in poses.items() if float(p["alt_agl"]) > 3.0
                    and args.desde <= float(p["t_mono"]) - t_base <= args.hasta)

    serie = {"antes": [], "ahora": []}
    crudo = args.salida + ".crudo.mp4"
    vw = cv2.VideoWriter(crudo, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    rastro, lienzo = [], None
    for f in frames:
        p = poses[f]
        t = float(p["t_mono"]) - t_base
        verdad = verdad_de(t_base + t)
        x, y = enu(float(p["lat"]), float(p["lng"]))
        if not rastro or abs(rastro[-1][0] - x) + abs(rastro[-1][1] - y) > 0.3:
            rastro.append((x, y))
        m1, e1 = mapa("ANTES: regla vieja (--span)", barco_en(rep_antes, t, verdad), verdad, rastro, (x, y), COLORES["antes"])
        m2, e2 = mapa("AHORA: estado reciente + extrapolacion", barco_en(rep_ahora, t, verdad), verdad, rastro, (x, y), COLORES["ahora"])
        serie["antes"].append((t, e1))
        serie["ahora"].append((t, e2))

        lienzo = np.full((H, W, 3), 12, np.uint8)
        lienzo[40:40 + MH, :MW] = m1
        lienzo[40:40 + MH, MW:] = m2
        cv2.putText(lienzo, "Blanco sintetico patrullando a %.0f m/s dentro del vuelo 3: donde lo reporta el sistema vs donde esta   t = %5.1f s"
                    % (args.velocidad, t), (14, 28), F, 0.72, (235, 235, 235), 1, cv2.LINE_AA)
        x0, x1, ytop, ybot = 70, W - 30, 40 + MH + 30, H - 30
        cv2.rectangle(lienzo, (x0, ytop), (x1, ybot), (40, 40, 48), 1)
        emax = 30.0
        for nivel in (10, 20):
            yy = int(ybot - nivel / emax * (ybot - ytop))
            cv2.line(lienzo, (x0, yy), (x1, yy), (32, 32, 38), 1)
            cv2.putText(lienzo, "%d m" % nivel, (18, yy + 5), F, 0.45, GRIS, 1, cv2.LINE_AA)
        for modo, pts in serie.items():
            xy = [(int(x0 + (tt - args.desde) / (args.hasta - args.desde) * (x1 - x0)),
                   int(ybot - min(e, emax) / emax * (ybot - ytop))) for tt, e in pts if e is not None]
            if len(xy) > 1:
                cv2.polylines(lienzo, [np.array(xy, np.int32)], False, COLORES[modo], 2, cv2.LINE_AA)
        med = {m: [e for _, e in s_ if e is not None] for m, s_ in serie.items()}
        cv2.putText(lienzo, "error: antes mediana %s | ahora mediana %s" % tuple(
            ("%.1f m" % np.median(med[m])) if med[m] else "-" for m in ("antes", "ahora")),
            (x0 + 10, ytop + 22), F, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        vw.write(lienzo)
    for _ in range(args.fps * 3):
        if lienzo is not None:
            vw.write(lienzo)
    vw.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", crudo, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", args.salida], check=True)
        os.remove(crudo)
    else:
        os.replace(crudo, args.salida)
    med = {m: [e for _, e in s_ if e is not None] for m, s_ in serie.items()}
    print("%s | %d frames | error mediana antes %.2f m, ahora %.2f m" % (
        args.salida, len(frames), np.median(med["antes"]), np.median(med["ahora"])))


if __name__ == "__main__":
    main()
