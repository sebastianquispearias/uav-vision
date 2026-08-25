# -*- coding: utf-8 -*-
"""Which detector can the Raspberry actually carry? Measured, not extrapolated.

The 1280 model wins on everything the mission cares about -- recall, alert load, false alerts,
latency -- but it costs 23.3 GFLOPs against 5.8 for the 640 that flies today. The only FPS
figure anyone has measured on this board is 12.47, and that was the 640 one; every number
quoted for 960 and 1280 so far is that figure divided by a FLOPs ratio, which is a guess.

This measures them, on the real camera, one after another in the same session so the thermal
state carries over the way it will in flight.

Watts matter more than frames here. Three missions died from millisecond transients on the 5 V
rail, not from sustained load, so this logs the throttling flags and the core clock alongside
the frame rate: a model that hits the target FPS while raising the throttling flag has not
passed.

Run on the Pi:
    python bench_modelos.py --modelos ~/modelos_visdrone/best_ncnn_model \\
                                      ~/modelos_visdrone/best_ncnn_960 \\
                                      ~/modelos_visdrone/best_ncnn_1280 \\
                            --frames 60
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np


def leer(cmd, defecto=''):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return defecto


def temperatura():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as fh:
            return int(fh.read().strip()) / 1000.0
    except Exception:
        return float('nan')


def reloj_mhz():
    s = leer('vcgencmd measure_clock arm')
    try:
        return int(s.split('=')[1]) / 1e6
    except Exception:
        return float('nan')


def banderas():
    """throttled=0x0 means the board never complained. Any other value is a failed run."""
    return leer('vcgencmd get_throttled', 'throttled=?')


def cpu_pct(t_muestra=0.4):
    def leer_jiffies():
        with open('/proc/stat') as fh:
            v = [float(x) for x in fh.readline().split()[1:]]
        return sum(v), v[3]
    t0, i0 = leer_jiffies()
    time.sleep(t_muestra)
    t1, i1 = leer_jiffies()
    dt, di = t1 - t0, i1 - i0
    return 100.0 * (1.0 - di / dt) if dt > 0 else float('nan')


def medir(modelo, picam, frames, imgsz):
    from ultralytics import YOLO
    import cv2

    yolo = YOLO(modelo, task='detect')
    lat, n_det = [], 0

    # The first inference includes graph setup and is not representative; it is timed
    # separately so it does not contaminate the median.
    img = picam.capture_array()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    t0 = time.time()
    yolo.predict(img, imgsz=imgsz, verbose=False)
    arranque = time.time() - t0

    temp0 = temperatura()
    for _ in range(frames):
        img = picam.capture_array()
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        t0 = time.time()
        r = yolo.predict(img, imgsz=imgsz, verbose=False)[0]
        lat.append((time.time() - t0) * 1000.0)
        n_det += len(r.boxes)
    lat = np.asarray(lat)

    return {
        'modelo': os.path.basename(modelo),
        'imgsz': imgsz,
        'arranque_s': round(arranque, 2),
        'lat_mediana_ms': round(float(np.median(lat)), 1),
        'lat_p90_ms': round(float(np.percentile(lat, 90)), 1),
        'fps': round(1000.0 / float(np.median(lat)), 2),
        'detecciones': n_det,
        'cpu_%': round(cpu_pct(), 1),
        'temp_inicio_C': round(temp0, 1),
        'temp_fin_C': round(temperatura(), 1),
        'reloj_MHz': round(reloj_mhz(), 0),
        'throttled': banderas(),
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modelos', nargs='+', required=True)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--imgsz', nargs='+', type=int, default=None,
                    help='uno por modelo; por defecto se deduce del nombre (…_960 -> 960)')
    ap.add_argument('--objetivo-fps', type=float, default=4.0,
                    help='la tasa de visión configurada; por debajo el modelo no entra')
    ap.add_argument('--salida', default='bench_modelos.json')
    ap.add_argument('--rot180', action='store_true')
    args = ap.parse_args()

    def deducir(m):
        base = os.path.basename(m)
        for s in (1280, 960, 640):
            if str(s) in base:
                return s
        return 640

    tams = args.imgsz if args.imgsz else [deducir(m) for m in args.modelos]
    if len(tams) != len(args.modelos):
        raise SystemExit('--imgsz debe tener un valor por modelo')

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from uav_vision.camera_config import ARDUCAM_MODULE_3
    from libcamera import Transform
    from picamera2 import Picamera2

    picam = Picamera2()
    picam.configure(picam.create_still_configuration(
        main={"size": (ARDUCAM_MODULE_3.image_width, ARDUCAM_MODULE_3.image_height)},
        transform=Transform(hflip=1, vflip=1) if args.rot180 else Transform()))
    picam.start()
    time.sleep(2)

    filas = []
    try:
        print('reposo: %.1f C, %s, CPU %.1f%%'
              % (temperatura(), banderas(), cpu_pct()), flush=True)
        for m, sz in zip(args.modelos, tams):
            if not os.path.exists(m):
                print('FALTA: %s' % m, flush=True)
                continue
            print('\n--- %s @ %d ---' % (os.path.basename(m), sz), flush=True)
            f = medir(m, picam, args.frames, sz)
            filas.append(f)
            print('  %.2f FPS | %.1f ms | CPU %.1f%% | %.1f->%.1f C | %s'
                  % (f['fps'], f['lat_mediana_ms'], f['cpu_%'], f['temp_inicio_C'],
                     f['temp_fin_C'], f['throttled']), flush=True)
    finally:
        picam.stop()
        picam.close()

    print('\n%-24s %6s %8s %9s %7s %10s' % ('modelo', 'imgsz', 'FPS', 'lat ms',
                                            'temp', 'throttled'))
    for f in filas:
        marca = 'OK ' if f['fps'] >= args.objetivo_fps else 'NO '
        print('%s%-21s %6d %8.2f %9.1f %6.1fC %10s'
              % (marca, f['modelo'], f['imgsz'], f['fps'], f['lat_mediana_ms'],
                 f['temp_fin_C'], f['throttled']))
    print('\n"NO" = por debajo de los %.1f FPS de la tasa de vision configurada.'
          % args.objetivo_fps)
    print('Cualquier throttled distinto de 0x0 invalida la corrida: el problema es '
          'electrico, no de modelo.')

    with open(args.salida, 'w') as fh:
        json.dump(filas, fh, indent=2)
    print('-> %s' % args.salida)
