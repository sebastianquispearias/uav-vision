"""One video: how long the chain takes to commit to a target, under the old rule and the new one.

Flight 3, the flying chain (BoT-SORT track ids from the flight), replayed twice through the same
protocol. Left, the camera. Right, two maps of the same seconds: above, maturity by the span
between first and last sighting, the rule the chain has always used; below, maturity by
independent looks, with the drone's own 95 % radius drawn around each point. Underneath, a
timeline marks when each rule commits to the operator.

What it does not show, on purpose, is a fix for false alarms: neither rule separates a person
from an object the detector keeps confusing with one. The last line of the video says so.

    python scripts/render_madurez.py --segundos 90 --miradas-min 20

Needs the recording in the sibling repo. Writes an H.264 mp4 when ffmpeg is available.
"""
import argparse
import contextlib
import csv
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
HNO = os.path.join(RAIZ, "..", "drone-geolocation")
VUELO = os.path.join(HNO, "data", "flight_02ago", "20260802_133309")
FRAMES = os.path.join(VUELO, "frames")
PISTAS = os.path.join(HNO, "entrenamiento", "botsort_pistas_02ago.npz")
REPLAY = os.path.join(AQUI, "replay_vuelo3.py")

LAT0, LNG0, R = -22.978029946, -43.23214256266666, 6378137.0
PIES = (-1.3, 8.8)                       # the operator, surveyed
E0, E1, N0, N1 = -22.0, 18.0, -9.0, 24.75
MW, MH = 640, 540                        # each map panel; 16 px per metre
CAM_W, CAM_H = 1280, 720
W, H = CAM_W + MW, 1080
F = cv2.FONT_HERSHEY_SIMPLEX
VERDE, AMBAR, DRON, GT = (120, 230, 120), (60, 190, 250), (255, 190, 120), (250, 250, 250)


def enu(la, ln):
    return (math.radians(ln - LNG0) * R * math.cos(math.radians(LAT0)),
            math.radians(la - LAT0) * R)


def a_pix(e, n):
    return (int((e - E0) / (E1 - E0) * MW), int((1 - (n - N0) / (N1 - N0)) * MH))


def correr(flags):
    """The replay in-process, returning its reports as (flight second, pois) and its globals."""
    os.environ["UAV_VISION_DATOS"] = os.path.join(RAIZ, "demo", "data")
    os.environ.pop("UAV_VISION_GS", None)
    sys.argv = [REPLAY, "--preliminares", "--pistas=" + PISTAS, *flags]
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(REPLAY, run_name="__main__")
    return [(float(r["time"]), r.get("pois") or []) for r in g["reportes"]], g


def confirmacion(reportes):
    """First report in which the point nearest the operator, within 5 m, is mature."""
    for t, pois in reportes:
        for p in pois:
            if "x" in p and p.get("mature") and math.hypot(p["x"] - PIES[0], p["y"] - PIES[1]) < 5.0:
                return t, p
    return None, None


def vigentes(reportes, t):
    actual = []
    for tr, pois in reportes:
        if tr > t:
            break
        actual = pois
    return actual


def fondo_mapa():
    sat = os.path.join(HNO, "entrenamiento", "satelite_zona.png")
    geo = os.path.join(HNO, "entrenamiento", "satelite_georef.txt")
    img = cv2.imread(sat)
    if img is None or not os.path.exists(geo):
        return np.full((MH, MW, 3), 24, np.uint8)
    lat0, lon0, lat1, lon1, _z = [float(x) for x in open(geo).read().split(",")]
    h, w = img.shape[:2]

    def pix(e, n):
        lat = LAT0 + n / R * 180 / math.pi
        lng = LNG0 + e / (R * math.cos(math.radians(LAT0))) * 180 / math.pi
        return ((lng - lon0) / (lon1 - lon0) * w, (lat - lat0) / (lat1 - lat0) * h)

    xa, yb = pix(E0, N0)
    xb, ya = pix(E1, N1)
    x0, x1 = int(min(xa, xb)), int(max(xa, xb))
    y0, y1 = int(min(ya, yb)), int(max(ya, yb))
    return (cv2.resize(img[y0:y1, x0:x1], (MW, MH), interpolation=cv2.INTER_CUBIC) * 0.7).astype(np.uint8)


def pintar_mapa(base, titulo, pois, rastro, dron, con_radio):
    m = base.copy()
    if len(rastro) > 1:
        cv2.polylines(m, [np.array([a_pix(*q) for q in rastro], np.int32)], False, DRON, 2, cv2.LINE_AA)
    g = a_pix(*PIES)
    cv2.drawMarker(m, g, GT, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
    cv2.putText(m, "operador (medido)", (g[0] + 12, g[1] + 22), F, 0.42, GT, 1, cv2.LINE_AA)
    for p in pois:
        if "x" not in p:
            continue
        maduro = bool(p.get("mature"))
        c = VERDE if maduro else AMBAR
        q = a_pix(p["x"], p["y"])
        if con_radio and p.get("radius_m") is not None:
            rr = int(p["radius_m"] / (E1 - E0) * MW)
            cv2.circle(m, q, rr, c, 1, cv2.LINE_AA)
        cv2.circle(m, q, 7, c, -1, cv2.LINE_AA)
        txt = "CONFIRMADO" if maduro else "POR VERIFICAR"
        cv2.putText(m, txt, (q[0] + 11, q[1] - 4), F, 0.46, c, 1, cv2.LINE_AA)
        det = ("%d miradas, +-%.1f m" % (p.get("looks", 0), p["radius_m"])
               if con_radio and p.get("radius_m") is not None else "%s obs" % p.get("n_obs"))
        cv2.putText(m, det, (q[0] + 11, q[1] + 13), F, 0.4, c, 1, cv2.LINE_AA)
    cv2.circle(m, a_pix(*dron), 6, DRON, -1, cv2.LINE_AA)
    cv2.rectangle(m, (0, 0), (MW, 30), (16, 16, 20), -1)
    cv2.putText(m, titulo, (10, 21), F, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(m, (0, 0), (MW - 1, MH - 1), (70, 70, 80), 1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segundos", type=float, default=90.0)
    ap.add_argument("--miradas-min", type=int, default=20)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "madurez_vuelo3.mp4"))
    args = ap.parse_args()
    if not os.path.isdir(FRAMES):
        sys.exit("no estan los frames del vuelo: %s" % FRAMES)

    print("replay con la regla de hoy (separacion temporal)...")
    rep_span, g = correr([])
    print("replay con miradas (>= %d)..." % args.miradas_min)
    rep_look, _ = correr(["--miradas", "--miradas-min=%d" % args.miradas_min])
    t_span, _ = confirmacion(rep_span)
    t_look, p_look = confirmacion(rep_look)
    print("operador confirmado: hoy %s s | miradas %s s" % (t_span, t_look))

    poses = g["poses"]
    t0 = float(poses[g["frames_aire"][0]]["t_mono"])
    cajas = {}
    for d in g["dets"]:
        cajas.setdefault(int(d[0]), []).append(d)
    frames = sorted(f for f, p in poses.items()
                    if 0.0 <= float(p["t_mono"]) - t0 <= args.segundos and float(p["alt_agl"]) > 3.0)
    base = fondo_mapa()

    crudo = args.salida + ".crudo.mp4"
    vw = cv2.VideoWriter(crudo, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    rastro, lienzo = [], None
    for f in frames:
        img = cv2.imread(os.path.join(FRAMES, "frame_%04d.jpg" % f))
        if img is None:
            continue
        p = poses[f]
        t = float(p["t_mono"]) - t0
        esc = CAM_W / float(img.shape[1])
        cam = cv2.resize(img, (CAM_W, CAM_H))
        for d in cajas.get(f, []):
            a = (int(d[2] * esc), int(d[3] * CAM_H / img.shape[0]))
            b = (int(d[4] * esc), int(d[5] * CAM_H / img.shape[0]))
            cv2.rectangle(cam, a, b, (235, 235, 235), 2)
            cv2.putText(cam, "%.2f" % d[1], (a[0], max(14, a[1] - 6)), F, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.rectangle(cam, (0, 0), (CAM_W, 34), (16, 16, 20), -1)
        cv2.putText(cam, "CAMARA DEL DRON   t = %5.1f s   altura %.1f m   %d detecciones"
                    % (t, float(p["alt_agl"]), len(cajas.get(f, []))), (12, 24), F, 0.65,
                    (235, 235, 235), 1, cv2.LINE_AA)

        x, y = enu(float(p["lat"]), float(p["lng"]))
        if not rastro or abs(rastro[-1][0] - x) + abs(rastro[-1][1] - y) > 0.3:
            rastro.append((x, y))
        m1 = pintar_mapa(base, "HOY: madurez por separacion temporal",
                         vigentes(rep_span, t), rastro, (x, y), False)
        m2 = pintar_mapa(base, "NUEVO: madurez por miradas (>= %d) + radio 95%%" % args.miradas_min,
                         vigentes(rep_look, t), rastro, (x, y), True)

        lienzo = np.full((H, W, 3), 14, np.uint8)
        lienzo[:CAM_H, :CAM_W] = cam
        lienzo[:MH, CAM_W:] = m1
        lienzo[MH:, CAM_W:] = m2

        y0 = CAM_H + 48
        for i, (nombre, tc, extra) in enumerate((
                ("HOY", t_span, ""),
                ("NUEVO", t_look, "   radio +-%.1f m" % p_look["radius_m"] if p_look else ""))):
            if tc is not None and t >= tc:
                txt, c = "%-6s operador CONFIRMADO a los %.0f s%s" % (nombre, tc, extra), VERDE
            else:
                txt, c = "%-6s esperando evidencia...   (%.0f s)" % (nombre, t), AMBAR
            cv2.putText(lienzo, txt, (24, y0 + i * 46), F, 0.95, c, 2, cv2.LINE_AA)

        bx0, bx1, by = 24, CAM_W - 40, CAM_H + 190
        cv2.line(lienzo, (bx0, by), (bx1, by), (90, 90, 100), 3)
        px = lambda s_: int(bx0 + min(1.0, s_ / args.segundos) * (bx1 - bx0))
        for nombre, tc, c in (("hoy", t_span, (170, 170, 170)), ("nuevo", t_look, VERDE)):
            if tc is not None and tc <= args.segundos:
                cv2.line(lienzo, (px(tc), by - 16), (px(tc), by + 16), c, 3)
                cv2.putText(lienzo, "%s %.0f s" % (nombre, tc), (px(tc) - 30, by + 40), F, 0.6, c, 1, cv2.LINE_AA)
        cv2.circle(lienzo, (px(t), by), 9, DRON, -1, cv2.LINE_AA)

        cv2.putText(lienzo, "Lo que ninguna de las dos arregla: un objeto fijo que el detector confunde con una persona.",
                    (24, H - 62), F, 0.62, (175, 175, 185), 1, cv2.LINE_AA)
        cv2.putText(lienzo, "Eso lo decide quien mira el recorte en la estacion de tierra.",
                    (24, H - 32), F, 0.62, (175, 175, 185), 1, cv2.LINE_AA)
        vw.write(lienzo)
    for _ in range(args.fps * 3):
        if lienzo is not None:
            vw.write(lienzo)
    vw.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", crudo, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", args.salida], check=True)
        os.remove(crudo)
    else:
        os.replace(crudo, args.salida)
    print("%s  (%d frames de vuelo, %.0f s de video)" % (args.salida, len(frames), len(frames) / args.fps + 3))


if __name__ == "__main__":
    main()
