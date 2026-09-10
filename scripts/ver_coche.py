# -*- coding: utf-8 -*-
"""Reproyecta el POI del coche sobre los frames reales del vuelo.

El POI sale del replay como una coordenada en el suelo. Para comprobar que
corresponde a un coche de verdad se hace el camino inverso: se lleva esa
coordenada a la imagen con el mismo modelo de camara y se mira si cae sobre
la caja que detecto el detector.
"""
import csv, math, os, sys
import numpy as np, cv2

RAIZ = r'C:/Users/User/Desktop/lac'
sys.path.insert(0, os.path.join(RAIZ, 'uav_vision'))
from uav_vision.pinhole_local import project_to_pixel
from uav_vision.camera_config import DEFAULT_CAMERA as CAM

BASE = os.path.join(RAIZ, 'drone-geolocation/data/flight_02ago/20260802_133309')
LAT0, LNG0, R = -22.978029946, -43.23214256266666, 6378137.0
PITCH = -55.0
POI = (-12.24, 13.06, 0.0)          # el coche que reporto el replay

enu = lambda la, ln: (math.radians(ln-LNG0)*R*math.cos(math.radians(LAT0)),
                      math.radians(la-LAT0)*R)

poses = {int(r['frame']): r for r in csv.DictReader(open(os.path.join(BASE,'frames.csv')))}
d = np.load(os.path.join(RAIZ,'uav_vision/demo/data/vehiculos.npz'), allow_pickle=True)
dets, clases = d['dets'], d['clases']

pp = CAM.image_center
filas = []
for det, cl in zip(dets, clases):
    if cl != 'car': continue
    f = int(det[0])
    p = poses.get(f)
    if p is None: continue
    x, y = enu(float(p['lat']), float(p['lng']))
    alt = float(p['alt_agl'])
    if alt < 10: continue
    px = project_to_pixel((x, y, alt), POI, float(p['yaw']), PITCH,
                          CAM.focal_length_px, CAM.image_width, CAM.image_height, pp)
    if px is None: continue
    x1, y1, x2, y2 = det[2:6]
    base = ((x1+x2)/2.0, y2)                       # borde inferior = contacto con el suelo
    res = math.hypot(px[0]-base[0], px[1]-base[1])
    exc = math.hypot(px[0]-pp[0], px[1]-pp[1])     # cuan centrado esta en la imagen
    filas.append((exc, res, f, alt, det, px, base))

filas.sort()
print('detecciones de coche utilizables (alt >= 10 m): %d' % len(filas))
r = np.array([x[1] for x in filas])
print('residuo de reproyeccion: mediana %.1f px, p90 %.1f px' % (np.median(r), np.percentile(r,90)))
print()
print('%-8s %-7s %-8s %-9s %s' % ('frame','alt_m','residuo','excentr','conf'))
for exc, res, f, alt, det, px, base in filas[:6]:
    print('%-8d %-7.1f %-8.1f %-9.0f %.2f' % (f, alt, res, exc, det[1]))

exc, res, f, alt, det, px, base = filas[0]
img = cv2.imread(os.path.join(BASE, 'frames', 'frame_%04d.jpg' % f))
x1, y1, x2, y2 = [int(v) for v in det[2:6]]
cv2.rectangle(img, (x1,y1), (x2,y2), (62,155,190), 3)
cv2.putText(img, 'car %.2f  (lo que vio el detector)' % det[1], (x1, max(24,y1-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (62,155,190), 2)
u, v = int(px[0]), int(px[1])
cv2.line(img, (u-26,v), (u+26,v), (128,95,201), 3)
cv2.line(img, (u,v-26), (u,v+26), (128,95,201), 3)
cv2.putText(img, 'POI reportado, traido de vuelta a la imagen', (u+34, v+6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (128,95,201), 2)
cv2.putText(img, 'frame %d   altura %.1f m   residuo %.0f px (~%.1f m)' %
            (f, alt, res, res*alt/CAM.focal_length_px), (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
sal = r'C:/Users/User/AppData/Local/Temp/claude/coche_verificado.png'
cv2.imwrite(sal, img)
print()
print('imagen:', sal, '| frame elegido:', f, '| residuo %.1f px = %.2f m a %.1f m de altura'
      % (res, res*alt/CAM.focal_length_px, alt))
