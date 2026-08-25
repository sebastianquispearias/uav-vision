# -*- coding: utf-8 -*-
"""Checkerboard calibration for the ArduCam, ready to run the day the board is printed.

The intrinsics in ARDUCAM_MODULE_3 were never calibrated: the focal length comes from the
flights' run_info.json and the principal point was recovered from a mount reflection. That is
an angular error, and an angular error misses the ground by more the higher you fly -- about
0.4 m at 35 m for every 14 px the principal point is off.

To be clear about what this does NOT fix: the ~2.3 m the system is currently off by is a
POSITION bias (measured in diagnostico_sesgo.py -- the error stays the same size from 6 m to
41 m of altitude, which no lens error can do). Calibration is worth doing, but it will not
move that number.

Two modes, because they happen at different times:

    capturar   grabs frames from the real camera, keeps only the ones where the board is fully
               visible, and reports how much of the frame the accepted views have covered so
               far. Corner coverage is what makes a calibration trustworthy: a pile of views
               all in the middle gives a confident and wrong distortion model.
    calibrar   runs the solve over the saved views and prints the CameraConfig values.

Usage on the Raspberry:
    python calibrar_tablero.py capturar --salida ~/calib --vistas 25
    python calibrar_tablero.py calibrar --salida ~/calib --casillas 9x6 --lado-mm 25
"""
import argparse
import glob
import os
import sys

import numpy as np
import cv2


def parse_casillas(s):
    a, b = s.lower().split('x')
    return (int(a), int(b))


def capturar(args):
    """Interactive capture: only board-visible frames are kept, with coverage feedback.

    Picamera2 is driven directly rather than through CamaraArduCam: calibration has no use for
    a detector, and CamaraArduCam loads YOLO on start. The capture configuration mirrors the
    one in camera.py so the calibration describes the frames the mission will actually see --
    same resolution, same ISP transform.
    """
    import time

    from libcamera import Transform
    from picamera2 import Picamera2

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from uav_vision.camera_config import ARDUCAM_MODULE_3

    os.makedirs(args.salida, exist_ok=True)
    patron = parse_casillas(args.casillas)

    picam = Picamera2()
    picam.configure(picam.create_still_configuration(
        main={"size": (ARDUCAM_MODULE_3.image_width, ARDUCAM_MODULE_3.image_height)},
        transform=Transform(hflip=1, vflip=1) if args.rot180 else Transform()))
    picam.start()
    time.sleep(2)  # the sensor needs time to stabilize exposure

    # Coverage is tracked as a coarse grid over the frame: each accepted view marks the cell
    # its board centre falls in. Corners are the cells that matter and the ones people skip.
    REJ = 4
    visto = np.zeros((REJ, REJ), dtype=int)
    n = 0
    print('Movimiento sugerido: tablero cerca y lejos, inclinado, y sobre todo en las '
          'CUATRO ESQUINAS del cuadro. Ctrl-C para terminar.')
    try:
        while n < args.vistas:
            img = picam.capture_array()
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, esq = cv2.findChessboardCorners(
                gris, patron,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK)
            if not ok:
                continue
            c = esq.reshape(-1, 2).mean(axis=0)
            celda = (min(REJ - 1, int(c[1] / img.shape[0] * REJ)),
                     min(REJ - 1, int(c[0] / img.shape[1] * REJ)))
            # Enough views of the same cell add cost and no information.
            if visto[celda] >= max(1, args.vistas // (REJ * REJ)):
                continue
            visto[celda] += 1
            n += 1
            cv2.imwrite(os.path.join(args.salida, 'vista_%03d.jpg' % n), img)
            print('  vista %d/%d  celda %s  cobertura %d/%d celdas'
                  % (n, args.vistas, celda, int((visto > 0).sum()), REJ * REJ), flush=True)
    except KeyboardInterrupt:
        print('\ninterrumpido')
    finally:
        picam.stop()
        # stop() alone keeps the device acquired; without close() no later
        # Picamera2 instance can open it.
        picam.close()

    faltan = [(i, j) for i in range(REJ) for j in range(REJ) if visto[i, j] == 0]
    if faltan:
        print('AVISO: %d celdas sin ninguna vista %s. La distorsion en esas zonas queda '
              'extrapolada.' % (len(faltan), faltan))
    print('%d vistas en %s' % (n, args.salida))


def calibrar(args):
    patron = parse_casillas(args.casillas)
    lado = args.lado_mm / 1000.0

    objp = np.zeros((patron[0] * patron[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2) * lado

    puntos_obj, puntos_img = [], []
    forma = None
    archivos = sorted(glob.glob(os.path.join(args.salida, '*.jpg')))
    if not archivos:
        print('no hay vistas en %s' % args.salida)
        return
    for f in archivos:
        img = cv2.imread(f)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        forma = gris.shape[::-1]
        ok, esq = cv2.findChessboardCorners(gris, patron, None)
        if not ok:
            print('  descartada (sin tablero): %s' % os.path.basename(f))
            continue
        esq = cv2.cornerSubPix(
            gris, esq, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        puntos_obj.append(objp)
        puntos_img.append(esq)
    print('%d vistas usables de %d' % (len(puntos_obj), len(archivos)))
    if len(puntos_obj) < 8:
        print('DEMASIADO POCAS. Con menos de ~8 vistas bien repartidas el resultado no es '
              'confiable; volver a capturar.')
        return

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        puntos_obj, puntos_img, forma, None, None)

    # Reprojection error per view: the honest quality number. Under ~0.5 px is good; a single
    # view far above the rest is usually a mis-detected board and should be removed.
    errs = []
    for i in range(len(puntos_obj)):
        proj, _ = cv2.projectPoints(puntos_obj[i], rvecs[i], tvecs[i], K, dist)
        errs.append(float(cv2.norm(puntos_img[i], proj, cv2.NORM_L2) / len(proj)))
    errs = np.asarray(errs)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = forma
    fov = 2.0 * np.degrees(np.arctan(w / (2.0 * fx)))

    print('\n=== resultado ===')
    print('  RMS global        : %.3f px' % rms)
    print('  error por vista   : mediana %.3f  max %.3f px' % (np.median(errs), errs.max()))
    print('  fx, fy            : %.1f , %.1f  (diferencia %.2f %%)'
          % (fx, fy, 100.0 * abs(fx - fy) / fx))
    print('  principal point   : %.1f , %.1f   (centro geometrico %.1f , %.1f)'
          % (cx, cy, w / 2.0, h / 2.0))
    print('  desviacion del centro: %.1f px -> %.2f grados de inclinacion del rayo'
          % (np.hypot(cx - w / 2.0, cy - h / 2.0),
             np.degrees(np.arctan(np.hypot(cx - w / 2.0, cy - h / 2.0) / fx))))
    print('  distorsion k1,k2,p1,p2,k3: %s'
          % np.array2string(dist.ravel(), precision=4, suppress_small=True))

    print('\n=== para camera_config.py ===')
    print("""ARDUCAM_MODULE_3 = CameraConfig(
    name="ArduCam IMX708 (calibrado %s)",
    focal_length_px=%.1f,
    image_width=%d,
    image_height=%d,
    fov_deg=%.1f,
    calibrated_principal_point=(%.1f, %.1f),
)""" % (args.fecha or 'sin fecha', (fx + fy) / 2.0, w, h, fov, cx, cy))
    print('\nOJO: si el montaje va girado 180 grados, usar .rotated_180(), no editar '
          'el punto principal a mano.')

    np.savez(os.path.join(args.salida, 'calibracion.npz'), K=K, dist=dist,
             rms=rms, errs=errs, forma=np.asarray(forma))
    print('-> %s' % os.path.join(args.salida, 'calibracion.npz'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('modo', choices=['capturar', 'calibrar'])
    ap.add_argument('--salida', default='calib')
    ap.add_argument('--casillas', default='9x6',
                    help='esquinas INTERNAS, no casillas: un tablero de 10x7 casillas es 9x6')
    ap.add_argument('--lado-mm', type=float, default=25.0)
    ap.add_argument('--vistas', type=int, default=25)
    ap.add_argument('--fecha', default=None)
    ap.add_argument('--rot180', action='store_true',
                    help='montaje girado 180 grados, como en el vuelo 3')
    args = ap.parse_args()
    (capturar if args.modo == 'capturar' else calibrar)(args)
