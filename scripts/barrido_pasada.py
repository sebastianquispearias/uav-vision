# -*- coding: utf-8 -*-
"""Would this system report anything at all during a real search sweep?

Every number this project has produced comes from flight 3, where the drone LOITERED over one
cooperative man for minutes. A search sweep does not do that: it crosses each point once, for
seconds, and moves on. IncrementalIdentity needs 36 s of accumulated observation before it
reports a candidate -- so there is a real possibility that the whole pipeline, validated and
deployed and flying, would stay silent over a victim it saw perfectly well.

This measures that without a new flight. The flight-3 observation stream is replayed through
windows of a fixed length, each window a fresh identity instance that sees only that slice:
a stand-in for a single pass. Sliding the window across the flight gives many passes over the
same scene, some with the person centred, some catching only the edge.

Reported per pass length: what fraction of passes produced ANY candidate, what fraction found
the operator, and what it cost in false alerts. Then the same sweep over dur_reporte_s, to
find the setting where a pass of realistic length actually speaks -- and what that setting
costs in noise.

Three tables come out, and they are NOT interchangeable -- citing one for another is how two
incompatible dwell-time tables ended up in this project's notes:

    1. Every window, including the ones where the drone was pointing elsewhere. This measures
       the pipeline AND the flight plan together.
    2. The same, lowering the maturity threshold, to see what that setting buys and what it
       costs in false alerts.
    3. Only the windows where the target was genuinely in frame for at least a third of the
       pass. This one measures the PIPELINE alone, and it is the one to quote.

Stage 1 caches the observation stream, so the parameter sweeps are cheap.
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uav_vision.camera_config import ARDUCAM_MODULE_3
from uav_vision.pinhole_local import pixel_to_ray
from uav_vision.identity import IncrementalIdentity

FR = r'C:\Users\User\Desktop\lac\drone-geolocation\data\flight_02ago\20260802_133309'
# The cached observation stream and the embeddings live in the flight archive,
# next to this repo: they are 5 MB of intermediate data, not source.
AQUI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'drone-geolocation', 'entrenamiento')
REF = (-22.978029946, -43.23214256266666)
PITCH = -55.0
CAM = ARDUCAM_MODULE_3.rotated_180()
PIES = np.array([-1.3, 8.8])
OBJ = np.array([2.5, 4.4])
RADIO = 3.5

CUAL = sys.argv[1] if len(sys.argv) > 1 else '1280'
OBS_NPZ = os.path.join(AQUI, 'obs_stream_%s.npz' % CUAL)

DURACIONES = [10, 15, 20, 30, 45, 60, 90, 120]     # seconds per simulated pass
PASO_S = 10                                        # window step
REPORTES = [6, 9, 12, 18, 24, 36]                  # dur_reporte_s to try


def enu(lat, lng):
    R = 6378137.0
    return (math.radians(lng - REF[1]) * R * math.cos(math.radians(REF[0])),
            math.radians(lat - REF[0]) * R)


def construir_stream():
    """Runs the tracker once and stores every ground observation it produced."""
    import cv2
    from boxmot.trackers.bbox.botsort import BotSort

    filas = [r for r in csv.DictReader(open(os.path.join(AQUI, 'y%s_cajas.csv' % CUAL)))]
    embs = np.load(os.path.join(AQUI, 'y%s_embs.npy' % CUAL))
    por_frame = {}
    for i, r in enumerate(filas):
        por_frame.setdefault(int(r['frame']), []).append(i)

    poses = {int(r['frame']): r for r in
             csv.DictReader(open(os.path.join(FR, 'frames.csv')))}
    frames_aire = sorted(f for f, p in poses.items() if float(p['alt_agl']) > 3.0)
    t0 = float(poses[frames_aire[0]]['t_mono'])

    tracker = BotSort(reid_model=None, with_reid=True, use_cmc=True,
                      track_high_thresh=0.35, track_low_thresh=0.2,
                      new_track_thresh=0.4, track_buffer=40, match_thresh=0.85)
    obs = []
    for k, f in enumerate(frames_aire):
        img = cv2.imread(os.path.join(FR, 'frames', 'frame_%04d.jpg' % f))
        if img is None:
            continue
        ix = por_frame.get(f, [])
        if ix:
            dts = np.array([[float(filas[i]['x1']), float(filas[i]['y1']),
                             float(filas[i]['x2']), float(filas[i]['y2']),
                             float(filas[i]['conf']), 0] for i in ix], dtype='float32')
            ee = embs[ix]
        else:
            dts = np.empty((0, 6), dtype='float32')
            ee = None
        res = np.asarray(tracker.update(dts, img, embs=ee))

        p = poses[f]
        x, y = enu(float(p['lat']), float(p['lng']))
        pos = (x, y, float(p['alt_agl']))
        for fila in res:
            di = int(fila[7])
            if not (0 <= di < len(ix)):
                continue
            i = ix[di]
            px = (float(filas[i]['x1']) + float(filas[i]['x2'])) / 2
            py = float(filas[i]['y2'])
            o, d = pixel_to_ray(pos, float(p['yaw']), (px, py), PITCH,
                                CAM.focal_length_px, CAM.image_width,
                                CAM.image_height, CAM.principal_point)
            if d[2] >= -1e-9:
                continue
            t = o[2] / -d[2]
            obs.append((float(poses[f]['t_mono']) - t0, k, int(fila[4]),
                        o[0] + t * d[0], o[1] + t * d[1], float(fila[5]), i))
        if k % 500 == 0:
            print('  stream %d/%d' % (k, len(frames_aire)), flush=True)

    a = np.asarray(obs, dtype='float64')
    dur = len(frames_aire) / (float(poses[frames_aire[-1]]['t_mono']) - t0)
    np.savez(OBS_NPZ, obs=a, fps=dur)
    print('%d observaciones guardadas' % len(a), flush=True)


def evaluar(obs, embs, fps, dur_reporte_s):
    """Feeds one slice of the stream to a fresh identity and judges what came out.

    The timestamp goes in with every observation. Without it, maturity falls back to a frame
    span scaled by the declared rate -- the mechanism that has been wrong three times over
    (ESTADO_SESION.md, 25ago). The stream already carries the clock; there was never a reason
    to infer what could be measured.
    """
    ident = IncrementalIdentity(fusion_radius_m=RADIO, fps=fps,
                                report_dur_s=dur_reporte_s)
    for t, k, tid, x, y, cf, i in obs:
        ident.observe(frame=int(k), track_id=int(tid), ground_xy=(x, y),
                      conf=cf, emb=embs[int(i)], t=float(t))
    cands = ident.candidates()
    hallado = any(np.linalg.norm(np.array([c['x'], c['y']]) - PIES) < RADIO
                  for c in cands)
    falsas = sum(1 for c in cands
                 if np.linalg.norm(np.array([c['x'], c['y']]) - PIES) >= RADIO)
    return hallado, falsas, len(cands)


if __name__ == '__main__':
    if not os.path.exists(OBS_NPZ):
        construir_stream()
    D = np.load(OBS_NPZ)
    obs = D['obs']
    # The cadence frames actually arrive at. The value cached in the npz was frames over the
    # wall-clock span, which counts 573 s of the drone sitting on the ground as if it were
    # flying and understates the rate ~3x -- shrinking every maturity threshold with it.
    _p = [r for r in csv.DictReader(open(os.path.join(FR, 'frames.csv')))
          if float(r['alt_agl']) > 3.0]
    _t = np.array([float(r['t_mono']) for r in _p])
    _dt = np.diff(_t)
    fps = float(1.0 / np.median(_dt[_dt > 0]))
    embs = np.load(os.path.join(AQUI, 'y%s_embs.npy' % CUAL))
    T = obs[:, 0]
    print('\nvariante %s | %d observaciones | %.2f obs/s | vuelo de %.0f s'
          % (CUAL, len(obs), fps, T.max()))

    print('\n=== 1. PASADA UNICA: fraccion de pasadas que reportan (dur_reporte_s=36, '
          'el default que vuela) ===')
    print('%-12s %8s %12s %14s %12s' % ('pasada', 'n pas.', 'reporta algo',
                                        'halla al operad.', 'falsas/pas.'))
    for dur in DURACIONES:
        inicios = np.arange(0, max(1.0, T.max() - dur), PASO_S)
        if len(inicios) == 0:
            continue
        algo = op = 0
        fal = 0
        for s in inicios:
            sl = obs[(T >= s) & (T < s + dur)]
            if len(sl) == 0:
                continue
            h, f, n = evaluar(sl, embs, fps, 36.0)
            algo += (n > 0)
            op += h
            fal += f
        print('%-12s %8d %11.0f%% %13.0f%% %12.2f'
              % ('%d s' % dur, len(inicios), 100.0 * algo / len(inicios),
                 100.0 * op / len(inicios), fal / len(inicios)))

    print('\n=== 2. BAJAR EL UMBRAL: pasada de 30 s, barriendo dur_reporte_s ===')
    print('%-16s %14s %14s %12s' % ('dur_reporte_s', 'halla al oper.',
                                    'reporta algo', 'falsas/pas.'))
    dur = 30
    inicios = np.arange(0, max(1.0, T.max() - dur), PASO_S)
    for rep in REPORTES:
        algo = op = 0
        fal = 0
        for s in inicios:
            sl = obs[(T >= s) & (T < s + dur)]
            if len(sl) == 0:
                continue
            h, f, n = evaluar(sl, embs, fps, float(rep))
            algo += (n > 0)
            op += h
            fal += f
        print('%-16s %13.0f%% %13.0f%% %12.2f'
              % ('%.0f s' % rep, 100.0 * op / len(inicios),
                 100.0 * algo / len(inicios), fal / len(inicios)))

    # -- the control the first table needs ---------------------------------
    # A pass that reports nothing because the drone was pointing elsewhere is not a failure
    # of the pipeline, it is a failure of the flight plan. Separating the two means asking
    # only about passes where the target was genuinely in the field of view, which the
    # reprojection gives for free.
    from uav_vision.pinhole_local import project_to_pixel
    poses = [r for r in csv.DictReader(open(os.path.join(FR, 'frames.csv')))]
    aire = [r for r in poses if float(r['alt_agl']) > 3.0]
    t_ini = float(aire[0]['t_mono'])
    visibles = []
    for r in aire:
        alt = float(r['alt_agl'])
        x, y = enu(float(r['lat']), float(r['lng']))
        pix = project_to_pixel((x, y, alt), (PIES[0], PIES[1], 0.0), float(r['yaw']),
                               PITCH, CAM.focal_length_px, CAM.image_width,
                               CAM.image_height, CAM.principal_point)
        dentro = (pix is not None
                  and 40 <= pix[0] <= CAM.image_width - 40
                  and 40 <= pix[1] <= CAM.image_height - 40)
        visibles.append((float(r['t_mono']) - t_ini, 1.0 if dentro else 0.0))
    V = np.asarray(visibles)

    print('\n=== 3. SOLO PASADAS QUE DE VERDAD LO SOBREVUELAN ===')
    print('(descarta las ventanas donde el objetivo nunca estuvo en cuadro: esas no miden'
          ' al detector, miden al plan de vuelo)')
    print('%-12s %10s %14s %14s %12s' % ('pasada', 'n utiles', 'seg. en cuadro',
                                         'halla al oper.', 'falsas/pas.'))
    for dur in DURACIONES:
        inicios = np.arange(0, max(1.0, T.max() - dur), PASO_S)
        utiles, op, fal, envista = 0, 0, 0.0, []
        for s0 in inicios:
            vv = V[(V[:, 0] >= s0) & (V[:, 0] < s0 + dur)]
            if len(vv) == 0:
                continue
            seg_vista = float(vv[:, 1].sum()) / fps
            if seg_vista < 0.3 * dur:      # target in view under a third of the pass
                continue
            sl = obs[(T >= s0) & (T < s0 + dur)]
            if len(sl) == 0:
                continue
            utiles += 1
            envista.append(seg_vista)
            h, f, n = evaluar(sl, embs, fps, 36.0)
            op += h
            fal += f
        if utiles == 0:
            print('%-12s %10d %14s %14s %12s' % ('%d s' % dur, 0, '-', '-', '-'))
            continue
        print('%-12s %10d %13.1f %13.0f%% %12.2f'
              % ('%d s' % dur, utiles, float(np.mean(envista)),
                 100.0 * op / utiles, fal / utiles))
