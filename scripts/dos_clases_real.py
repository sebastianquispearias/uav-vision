# -*- coding: utf-8 -*-
"""Las dos clases sobre un frame real del vuelo.

El mapa de la estacion de tierra muestra pines sobre coordenadas. Esto muestra
lo mismo sobre la imagen que vio el dron: las cajas que detecto y los dos POI
reportados, traidos de vuelta a la imagen con el mismo modelo de camara.
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
POIS = {'person': (-0.19, 6.69, 0.0), 'car': (-12.24, 13.06, 0.0)}
COLOR = {'person': (201, 95, 128), 'car': (62, 155, 190)}   # BGR

enu = lambda la, ln: (math.radians(ln-LNG0)*R*math.cos(math.radians(LAT0)),
                      math.radians(la-LAT0)*R)
poses = {int(r['frame']): r for r in csv.DictReader(open(os.path.join(BASE,'frames.csv')))}

per = np.load(os.path.join(RAIZ,'uav_vision/demo/data/examen_v3_datos.npz'), allow_pickle=True)['dets']
veh = np.load(os.path.join(RAIZ,'uav_vision/demo/data/vehiculos.npz'), allow_pickle=True)
vd, vc = veh['dets'], veh['clases']

cajas = {}
for d in per:
    cajas.setdefault(int(d[0]), {}).setdefault('person', []).append(d[2:6])
for d, c in zip(vd, vc):
    if c == 'car':
        cajas.setdefault(int(d[0]), {}).setdefault('car', []).append(d[2:6])

pp, W, H = CAM.image_center, CAM.image_width, CAM.image_height
cand = []
for f, cj in cajas.items():
    if 'person' not in cj or 'car' not in cj:
        continue
    p = poses.get(f)
    if p is None or float(p['alt_agl']) < 10:
        continue
    x, y = enu(float(p['lat']), float(p['lng']))
    alt = float(p['alt_agl'])
    px = {k: project_to_pixel((x, y, alt), v, float(p['yaw']), PITCH,
                              CAM.focal_length_px, W, H, pp) for k, v in POIS.items()}
    if any(v is None for v in px.values()):
        continue
    res = {}
    for k in POIS:
        bs = cj[k]
        b = min(bs, key=lambda b: math.hypot(px[k][0]-(b[0]+b[2])/2, px[k][1]-b[3]))
        res[k] = (math.hypot(px[k][0]-(b[0]+b[2])/2, px[k][1]-b[3]), b)
    peor = max(r[0] for r in res.values())
    cand.append((peor, f, alt, px, res))

cand.sort()
print('frames con persona Y coche a la vez, ambos POI dentro de la imagen: %d' % len(cand))
print()
print('%-8s %-7s %-11s %s' % ('frame','alt_m','res_person','res_car'))
for peor, f, alt, px, res in cand[:6]:
    print('%-8d %-7.1f %-11.1f %.1f' % (f, alt, res['person'][0], res['car'][0]))

peor, f, alt, px, res = cand[0]
img = cv2.imread(os.path.join(BASE, 'frames', 'frame_%04d.jpg' % f))
for k in ('car', 'person'):
    c = COLOR[k]
    x1, y1, x2, y2 = [int(v) for v in res[k][1]]
    cv2.rectangle(img, (x1,y1), (x2,y2), c, 3)
    cv2.putText(img, k, (x1, max(26, y1-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    u, v = int(px[k][0]), int(px[k][1])
    cv2.line(img, (u-30,v), (u+30,v), c, 3)
    cv2.line(img, (u,v-30), (u,v+30), c, 3)
    cv2.putText(img, 'POI %s  (%.0f px = %.1f m)' % (k, res[k][0], res[k][0]*alt/CAM.focal_length_px),
                (u+38, v+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
cv2.putText(img, 'frame %d   altura %.1f m   las dos clases, misma cadena, sin reentrenar' % (f, alt),
            (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
sal = r'C:/Users/User/AppData/Local/Temp/claude/dos_clases_real.png'
cv2.imwrite(sal, img)
print()
print('imagen:', sal, '| frame', f, '| altura %.1f m' % alt)
