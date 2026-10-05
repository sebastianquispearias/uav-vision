"""Real BoT-SORT track ids for every cached detection of the 02-ago flight.

The replay stands in for the tracker with nearest-centre continuity, which splits this flight
into 614 short tracks. Scoring identity against that would score the stand-in, not the system.
This runs the tracker the camera flies -- boxmot's BotSort, configured as
``OnboardCamera._build_tracker`` builds it for the flight mission (appearance on, camera-motion
compensation ON through the 'sof' method, an 8 s buffer) -- over the frames the chain processed,
and stores one track id per detection, aligned row for row with ``examen_v3_datos.npz`` filtered
at conf >= 0.25.

THE CMC DEFAULT IS ON AND THAT IS NOT A PREFERENCE. An earlier version of this file hardwired
``use_cmc=False`` while claiming in this same paragraph to configure the tracker exactly as the
flight does, and the flight has it on (``camera.py``: ``use_cmc=self.compensate_motion`` with
``compensate_motion: bool = True``, ``cmc_method="sof"``). That one line is why the nine-tracker
table scored BoT-SORT at 3 of 7 people while the repository's own front page says 5 of 7: without
compensation the same 2637 detections carry 114 track ids instead of 36, 852 of them get an id
instead of 1608, and two people fall off the scoreboard. --cmc=off still reproduces that run.

Only the processed frames, never every recorded one. The camera recorded 8.7 frames per second
and the chain detected on roughly one in five; feeding the frames in between hands the tracker
empty frames the real loop never saw, and BoT-SORT deletes a new track that is not matched on the
very next frame. The first run did exactly that and tracked 849 of 2637 detections. The replay
uses the same frame set, which is also what makes the two trackers comparable.

The buffer is converted to frames with the cadence of that frame set, measured from the
timestamps, since a buffer counted in frames means a different time at a different rate.

Needs boxmot, which the training venv has:
    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/botsort_pistas.py

THE THRESHOLDS ARE THE FLIGHT CAMERA'S, and two switches change them.

--calibrado lowers the two that gate a detection's way into a track down to the chain's own
reporting threshold. A tracker that demands more confidence than the chain reports with
silently drops detections the chain decided to keep: measured on five flight recordings,
23-50 % of the detector's boxes fall below 0.35 and 32-67 % below 0.40.

--piso lowers the confidence floor of the detections FED to the tracker. It exists because
anything measured about the BYTE band on the replay is otherwise unmeasurable. The band is the
boxes between the camera's floor and the reporting threshold, and a band box only earns its way
into the chain by being claimed by an existing track. With the floor at the reporting threshold
the tracker never sees those boxes, so they arrive at the protocol without a track id, so the
identity layer ignores them, so the window and the appearance template cannot change a single
candidate no matter what they do. Measured: the three runs give n_obs 1299 each, to the unit.

The cached embs_osnet.npy only covers the detections above the reporting cut; the flight's npz
carries ONE EMBEDDING PER DETECTION, band included, which is what makes a lower floor possible
at all. Above the cut the two are the same vectors.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

LAC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATOS = os.path.join(LAC, "uav_vision", "demo", "data")
FR = os.path.join(LAC, "drone-geolocation", "data", "flight_02ago", "20260802_133309")
SALIDA = os.path.join(LAC, "drone-geolocation", "entrenamiento", "botsort_pistas_02ago.npz")
UMBRALES_VUELO = {"track_high_thresh": 0.35, "new_track_thresh": 0.4}
CMC_METODO = "sof"
CONF_MIN = 0.25
TRACK_BUFFER_S = 8.0
UMBRALES_CALIBRADOS = {"track_high_thresh": CONF_MIN, "new_track_thresh": CONF_MIN}


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from uav_vision.trackers import catalogo, cmc_real, construir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracker", default="botsort",
                    help="cual de los de boxmot: %s" % ", ".join(catalogo()))
    ap.add_argument("--hasta", type=int, default=0,
                    help="parar tras N cuadros, para un smoke test antes de la corrida larga")
    ap.add_argument("--cmc", choices=("on", "off"), default="on",
                    help="camera-motion compensation, as the flight has it (default on, '%s'). "
                         "'off' is the state the preliminary nine-tracker table ran in."
                         % CMC_METODO)
    ap.add_argument("--calibrado", action="store_true")
    ap.add_argument("--piso", type=float, default=CONF_MIN,
                    help="confidence floor of the detections fed in (default %.2f)" % CONF_MIN)
    args = ap.parse_args()
    umbrales = UMBRALES_CALIBRADOS if args.calibrado else UMBRALES_VUELO
    salida = SALIDA.replace(".npz", "_calibrado.npz") if args.calibrado else SALIDA
    if args.piso != CONF_MIN:
        salida = salida.replace(".npz", "_piso%03d.npz" % round(args.piso * 100))
    if args.tracker != "botsort":
        salida = salida.replace("botsort_pistas", args.tracker + "_pistas")
    salida = salida.replace(".npz", "_cmc%s.npz" % args.cmc)
    if args.hasta:
        salida = salida.replace(".npz", "_hasta%d.npz" % args.hasta)

    D = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))
    dets_all = D["dets"]
    mascara = dets_all[:, 1] >= args.piso
    dets = dets_all[mascara]
    if args.piso >= CONF_MIN:
        embs = np.load(os.path.join(DATOS, "embs_osnet.npy")).astype(np.float32)
    else:
        embs = D["embs"][mascara].astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    assert len(embs) == len(dets), "embeddings not aligned with detections"
    n = len(dets)

    por_frame = {}
    for i, d in enumerate(dets):
        por_frame.setdefault(int(d[0]), []).append(i)

    poses = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(FR, "frames.csv")))}
    aire = sorted(f for f, p in poses.items()
                  if float(p["alt_agl"]) > 3.0 and f in por_frame)
    t = np.array([float(poses[f]["t_mono"]) for f in aire])
    fps = float(1.0 / np.median(np.diff(t)))
    buffer_frames = max(2, round(TRACK_BUFFER_S * fps))
    print(f"{n} detections | {len(aire)} airborne frames | fed at {fps:.2f} fps "
          f"-> track_buffer {buffer_frames} frames ({TRACK_BUFFER_S} s)", flush=True)

    usa_apariencia = catalogo()[args.tracker]
    tracker, honra, ignora, aprox = construir(
        args.tracker,
        embs_propias=usa_apariencia,
        use_cmc=(args.cmc == "on"),
        cmc_method=CMC_METODO,
        track_high_thresh=umbrales["track_high_thresh"],
        track_low_thresh=0.2,
        new_track_thresh=umbrales["new_track_thresh"],
        track_buffer=buffer_frames,
        match_thresh=0.85,
    )
    # Lo PEDIDO y lo que QUEDO CORRIENDO son dos preguntas, y la tabla preliminar las
    # confundio: strongsort ignora todo ajuste de CMC y compensa en cada cuadro igual.
    real = cmc_real(tracker)
    print("%s: apariencia %s | CMC pedida %s -> corriendo %s | honra %d ajustes%s%s"
          % (args.tracker, "si" if usa_apariencia else "no", args.cmc, real.upper(), len(honra),
             " | NO RECIBE %s" % sorted(ignora) if ignora else "",
             " | renombrados cambiando el valor o solo el rol: %s" % aprox if aprox else ""),
          flush=True)
    if real != args.cmc and real != "ninguno":
        print("  OJO: %s corre con CMC %s aunque se pidio %s, y no hay parametro para cambiarlo"
              % (args.tracker, real.upper(), args.cmc.upper()), flush=True)

    track = -np.ones(n, dtype=np.int64)
    t0 = time.time()
    if args.hasta:
        aire = aire[:args.hasta]
        print("  SMOKE TEST: solo los primeros %d cuadros" % len(aire), flush=True)
    for k, f in enumerate(aire):
        img = cv2.imread(os.path.join(FR, "frames", f"frame_{f:04d}.jpg"))
        if img is None:
            continue
        idx = por_frame.get(f, [])
        if idx:
            dts = np.array([[*dets[i][2:6], dets[i][1], 0] for i in idx], dtype=np.float32)
            ee = embs[idx]
        else:
            dts, ee = np.empty((0, 6), dtype=np.float32), None
        for fila in np.asarray(tracker.update(dts, img,
                                              embs=ee if usa_apariencia else None)):
            di = int(fila[7])
            if 0 <= di < len(idx):
                track[idx[di]] = int(fila[4])
        if k % 1000 == 0:
            print(f"  {k}/{len(aire)} frames, {len(set(track[track >= 0]))} tracks, "
                  f"{time.time() - t0:.0f} s", flush=True)

    ids, cuenta = np.unique(track[track >= 0], return_counts=True)
    print(f"{args.tracker}: {len(ids)} tracks | {int((track >= 0).sum())}/{n} detections assigned | "
          f"boxes per track: median {int(np.median(cuenta))}, max {int(cuenta.max())} | "
          f"tracks with >=10 boxes: {int((cuenta >= 10).sum())}", flush=True)
    np.savez(salida, track=track, fps=fps, track_buffer=buffer_frames, conf_min=args.piso,
             tracker=args.tracker, honra=sorted(honra), ignora=sorted(ignora),
             aproximados=aprox, cmc_pedida=args.cmc, cmc_real=real, cmc_method=CMC_METODO,
             **umbrales)
    print("saved", salida, umbrales)


if __name__ == "__main__":
    sys.exit(main())
