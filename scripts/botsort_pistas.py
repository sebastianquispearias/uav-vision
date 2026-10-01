"""Real BoT-SORT track ids for every cached detection of the 02-ago flight.

The replay stands in for the tracker with nearest-centre continuity, which splits this flight
into 614 short tracks. Scoring identity against that would score the stand-in, not the system.
This runs the tracker the camera flies -- boxmot's BotSort, configured exactly as
``OnboardCamera._build_tracker`` builds it for the flight mission (appearance on, camera-motion
compensation off, an 8 s buffer) -- over the frames the chain processed, and stores one track id
per detection, aligned row for row with ``examen_v3_datos.npz`` filtered at conf >= 0.25.

Only the processed frames, never every recorded one. The camera recorded 8.7 frames per second
and the chain detected on roughly one in five; feeding the frames in between hands the tracker
empty frames the real loop never saw, and BoT-SORT deletes a new track that is not matched on the
very next frame. The first run did exactly that and tracked 849 of 2637 detections. The replay
uses the same frame set, which is also what makes the two trackers comparable.

The buffer is converted to frames with the cadence of that frame set, measured from the
timestamps, since a buffer counted in frames means a different time at a different rate.

Needs boxmot, which the training venv has:
    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/botsort_pistas.py
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
# The flight camera's thresholds. --calibrado lowers the two that gate a detection's way into a
# track to the chain's own reporting threshold: a tracker that demands more confidence than the
# chain reports with silently drops detections the chain decided to keep. Measured on five flight
# recordings, 23-50 % of the detector's boxes fall below 0.35 and 32-67 % below 0.40.
UMBRALES_VUELO = {"track_high_thresh": 0.35, "new_track_thresh": 0.4}
CONF_MIN = 0.25
TRACK_BUFFER_S = 8.0  # OnboardCamera default, unchanged by the flight mission
UMBRALES_CALIBRADOS = {"track_high_thresh": CONF_MIN, "new_track_thresh": CONF_MIN}


def main():
    from boxmot.trackers.bbox.botsort import BotSort

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrado", action="store_true")
    # --piso lowers the confidence floor of the detections FED to the tracker. It exists because
    # anything measured about the BYTE band on the replay is otherwise unmeasurable: the band is
    # the boxes between 0.10 and the reporting threshold, and a band box only earns its way into
    # the chain by being claimed by an existing track. With the floor at 0.25 the tracker never
    # sees those boxes, so they arrive at the protocol without a track id, so the identity layer
    # ignores them, so the window and the appearance template cannot change a single candidate no
    # matter what they do. Measured: the three runs give n_obs 1299 each, to the unit.
    ap.add_argument("--piso", type=float, default=CONF_MIN,
                    help="confidence floor of the detections fed in (default %.2f)" % CONF_MIN)
    args = ap.parse_args()
    umbrales = UMBRALES_CALIBRADOS if args.calibrado else UMBRALES_VUELO
    salida = SALIDA.replace(".npz", "_calibrado.npz") if args.calibrado else SALIDA
    if args.piso != CONF_MIN:
        salida = salida.replace(".npz", "_piso%03d.npz" % round(args.piso * 100))

    D = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))
    dets_all = D["dets"]
    mascara = dets_all[:, 1] >= args.piso
    dets = dets_all[mascara]
    # The cached embs_osnet.npy only covers the detections above 0.25; the flight's npz carries
    # one embedding per detection, band included, which is what makes a lower floor possible at
    # all. Above 0.25 the two are the same vectors.
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

    tracker = BotSort(
        reid_model=None,
        with_reid=True,
        use_cmc=False,
        track_high_thresh=umbrales["track_high_thresh"],
        track_low_thresh=0.2,
        new_track_thresh=umbrales["new_track_thresh"],
        track_buffer=buffer_frames,
        match_thresh=0.85,
    )

    track = -np.ones(n, dtype=np.int64)
    t0 = time.time()
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
        for fila in np.asarray(tracker.update(dts, img, embs=ee)):
            di = int(fila[7])
            if 0 <= di < len(idx):
                track[idx[di]] = int(fila[4])
        if k % 1000 == 0:
            print(f"  {k}/{len(aire)} frames, {len(set(track[track >= 0]))} tracks, "
                  f"{time.time() - t0:.0f} s", flush=True)

    ids, cuenta = np.unique(track[track >= 0], return_counts=True)
    print(f"BoT-SORT: {len(ids)} tracks | {int((track >= 0).sum())}/{n} detections assigned | "
          f"boxes per track: median {int(np.median(cuenta))}, max {int(cuenta.max())} | "
          f"tracks with >=10 boxes: {int((cuenta >= 10).sum())}", flush=True)
    np.savez(salida, track=track, fps=fps, track_buffer=buffer_frames, conf_min=args.piso,
             **umbrales)
    print("saved", salida, umbrales)


if __name__ == "__main__":
    sys.exit(main())
