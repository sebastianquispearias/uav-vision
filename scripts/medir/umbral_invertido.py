"""Why OC-SORT tracked 6 of the flight's 2637 detections, measured one setting at a time.

The first nine-tracker table scored ocsort and deepocsort at 0 of 7 people, on runs that gave a
track id to 6 and 31 detections out of 2637. That is not a tracker performing badly, and
``docs/DESCARTADO.md`` exists because this repository has already published a conclusion that a
measurement later took back. So the configuration was taken apart before anything was scored.

TWO CAUSES, BOTH OURS, AND THEY MULTIPLY.

THE THRESHOLD RAN BACKWARDS. BoT-SORT's ``match_thresh`` caps a cost of ``1 - IoU``
(``matching.py:79`` builds it, ``matching.py:35`` passes it to ``lap.lapjv`` as ``cost_limit``),
so 0.85 means "associate anything overlapping by 0.15 or more" -- the most permissive value this
calibration carries. Everywhere else in boxmot the same quantity is a FLOOR on the IoU itself
(``stages.py:152``, ``boost.py:196``, ``hybrid.py:343``, ``occluboost.py:956``), so the adapter's
rename handed five trackers "boxes must overlap by 0.85 to be the same person". For someone a few
dozen pixels tall at 25 m that essentially never happens.

THE OUTPUT GATE NEEDS CONSECUTIVE FRAMES. ``ocsort.py:257`` and ``deepocsort.py:273`` emit a
track only when ``hit_streak >= min_hits``, and ``predict()`` resets the streak to zero on any
frame the track was not updated, so with the default ``min_hits=3`` a person has to be detected
on three consecutive PROCESSED frames before anything is reported. BoT-SORT does not use that
rule: its gate is ``is_activated`` (``botsort.py:285``), which admits a track on its first
high-confidence frame and recovers it from the lost pool afterwards.

The second half is measured here too, because the mechanism alone does not say whether it bites:
this flight detects a visible target in bursts, so the probe counts how many of its boxes ever
reach a third consecutive frame.

WHAT IT FOUND, on the first 400 processed frames (764 detections):

    iou_threshold 0.85, min_hits 3     6 with an id ( 0.8 %)   <- what the table scored
    iou_threshold 0.15, min_hits 3   265 with an id (34.7 %)
    iou_threshold 0.85, min_hits 1    61 with an id ( 8.0 %)
    iou_threshold 0.30, min_hits 1   487 with an id (63.7 %)
    botsort, same calibration        454 with an id (59.4 %)

So OC-SORT is not the worst of the zoo on this flight; it was handed a setting that runs in the
opposite sense. The conversion now lives in ``uav_vision.trackers.CONVERSIONES`` and
``tests/test_elegir_tracker.py`` section 9 fails if it is dropped. ``min_hits`` is NOT changed:
it is the tracker's own decision about how much evidence a report needs, and overriding it to
make a number look better is tuning on one flight.

Needs boxmot, and reads only what is already on disk:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/medir/umbral_invertido.py
    ... --hasta 1381    for the whole flight instead of the first 400 frames
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAC = os.path.dirname(REPO)
DATOS = os.path.join(REPO, "demo", "data")
FR = os.path.join(LAC, "drone-geolocation", "data", "flight_02ago", "20260802_133309")
CAL = {"track_high_thresh": 0.35, "track_low_thresh": 0.2, "new_track_thresh": 0.4,
       "track_buffer": 75}
NUESTRO_MATCH = 0.85


def cargar(hasta):
    """The flight's cached detections, their embeddings, and the frames the chain processed."""
    D = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))
    dets = D["dets"][D["dets"][:, 1] >= 0.25]
    embs = np.load(os.path.join(DATOS, "embs_osnet.npy")).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    poses = {int(r["frame"]): r
             for r in csv.DictReader(open(os.path.join(FR, "frames.csv")))}
    por_frame = {}
    for i, d in enumerate(dets):
        por_frame.setdefault(int(d[0]), []).append(i)
    aire = [f for f in sorted(poses)
            if float(poses[f]["alt_agl"]) > 3.0 and f in por_frame][:hasta]
    return dets, embs, por_frame, aire


def correr(nombre, reid, dets, embs, por_frame, aire, **extra):
    """One pass of a tracker over those frames, returning how much of it carries a track id."""
    from uav_vision.trackers import construir
    tracker, _, _, _ = construir(nombre, embs_propias=reid, use_cmc=False, **CAL, **extra)
    track = -np.ones(len(dets), dtype=np.int64)
    for f in aire:
        img = cv2.imread(os.path.join(FR, "frames", f"frame_{f:04d}.jpg"))
        if img is None:
            continue
        idx = por_frame.get(f, [])
        dts = (np.array([[*dets[i][2:6], dets[i][1], 0] for i in idx], dtype=np.float32)
               if idx else np.empty((0, 6), dtype=np.float32))
        for fila in np.asarray(tracker.update(dts, img, embs=embs[idx] if (reid and idx)
                                              else None)):
            j = int(fila[7])
            if 0 <= j < len(idx):
                track[idx[j]] = int(fila[4])
    return len(np.unique(track[track >= 0])), int((track >= 0).sum()), tracker.iou_threshold


def rachas_consecutivas(dets, por_frame, aire):
    """How often a single identity is detected on consecutive processed frames.

    The identities are the published run's track ids, which is the closest thing to per-person
    truth that does not require re-reading the labels, and the question asked of them is only
    about frame continuity, which no tracker setting changes.
    """
    pub = np.load(os.path.join(DATOS, "pistas_bot_cmc_sof.npz"))["track"]
    donde = {f: k for k, f in enumerate(aire)}
    todas = []
    for t in np.unique(pub[pub >= 0]):
        ks = sorted({donde[int(dets[i][0])] for i in np.where(pub == t)[0]
                     if int(dets[i][0]) in donde})
        if not ks:
            continue
        largo = 1
        for a, b in zip(ks, ks[1:]):
            if b == a + 1:
                largo += 1
            else:
                todas.append(largo)
                largo = 1
        todas.append(largo)
    return np.array(todas)


def main():
    sys.path.insert(0, REPO)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hasta", type=int, default=400,
                    help="processed frames to feed (default 400; the flight has 1381)")
    args = ap.parse_args()

    dets, embs, por_frame, aire = cargar(args.hasta)
    n = sum(len(por_frame[f]) for f in aire)
    print("%d cuadros procesados, %d detecciones de las %d del vuelo"
          % (len(aire), n, len(dets)))

    print()
    print("=" * 78)
    print("1. EL UMBRAL INVERTIDO. match_thresh=%.2f es nuestro ajuste MAS permisivo"
          % NUESTRO_MATCH)
    print("=" * 78)
    print("   BoT-SORT acepta 1-IoU <= %.2f, o sea IoU >= %.2f (matching.py:79 y :35)"
          % (NUESTRO_MATCH, 1 - NUESTRO_MATCH))
    print("   boxmot acepta IoU > iou_threshold (stages.py:152, boost.py:196, hybrid.py:343)")
    print()
    print("   %-12s %-30s %6s %16s" % ("tracker", "iou_threshold", "ids", "con id"))
    for nombre, reid in (("ocsort", False), ("deepocsort", True)):
        for v, glosa in ((NUESTRO_MATCH, "0.85 CRUDO, como lo paso la tabla"),
                         (0.5, "0.50"),
                         (0.3, "0.30"),
                         (1 - NUESTRO_MATCH, "0.15 = 1 - match_thresh, CONVERTIDO")):
            ids, asg, _ = correr(nombre, reid, dets, embs, por_frame, aire, iou_threshold=v)
            print("   %-12s %-30s %6d %6d (%4.1f %%)"
                  % (nombre, glosa, ids, asg, 100.0 * asg / n))

    print()
    print("=" * 78)
    print("2. LA PUERTA DE SALIDA. hit_streak>=min_hits pide cuadros CONSECUTIVOS")
    print("=" * 78)
    r = rachas_consecutivas(dets, por_frame, aire)
    if len(r):
        alcanzan = int(r[r >= 3].sum() - 2 * (r >= 3).sum())
        print("   %d tramos de cuadros consecutivos en las pistas publicadas" % len(r))
        for L in (1, 2):
            print("     largo == %d : %4d tramos (%4.1f %%), %4d cajas"
                  % (L, (r == L).sum(), 100.0 * (r == L).sum() / len(r), r[r == L].sum()))
        print("     largo >= 3 : %4d tramos (%4.1f %%), %4d cajas"
              % ((r >= 3).sum(), 100.0 * (r >= 3).sum() / len(r), r[r >= 3].sum()))
        print("   cajas que llegan al 3er cuadro consecutivo: %d de %d (%.1f %%)"
              % (alcanzan, r.sum(), 100.0 * alcanzan / r.sum()))
        print("   o sea que min_hits=3 ya descarta el %.1f %% ANTES de cualquier umbral"
              % (100.0 - 100.0 * alcanzan / r.sum()))
    print()
    print("   %-12s %-30s %6s %16s" % ("tracker", "configuracion", "ids", "con id"))
    for nombre, reid in (("ocsort", False), ("deepocsort", True)):
        for kw, glosa in (({"iou_threshold": NUESTRO_MATCH, "min_hits": 1},
                           "min_hits=1, umbral crudo 0.85"),
                          ({"iou_threshold": 0.3, "min_hits": 1},
                           "min_hits=1 Y umbral 0.30")):
            ids, asg, _ = correr(nombre, reid, dets, embs, por_frame, aire, **kw)
            print("   %-12s %-30s %6d %6d (%4.1f %%)"
                  % (nombre, glosa, ids, asg, 100.0 * asg / n))

    print()
    print("=" * 78)
    print("3. LA REFERENCIA. botsort con la MISMA calibracion, sobre los mismos cuadros")
    print("=" * 78)
    ids, asg, _ = correr("botsort", True, dets, embs, por_frame, aire,
                         match_thresh=NUESTRO_MATCH)
    print("   %-12s %-30s %6d %6d (%4.1f %%)"
          % ("botsort", "match_thresh=0.85, sin traducir", ids, asg, 100.0 * asg / n))
    print()
    print("   min_hits NO se cambia en la tabla: es la decision del tracker sobre cuanta")
    print("   evidencia pide un reporte, y moverla para que un numero quede mejor es")
    print("   ajustar sobre un vuelo. El umbral SI se convierte, porque 0.85 crudo no era")
    print("   nuestra calibracion: era su opuesto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
