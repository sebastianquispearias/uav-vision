"""Sigue UNA deteccion real del vuelo del 02ago por toda la cadena, imprimiendo cada valor.

Es lo que veria alguien poniendo un punto de interrupcion en cada etapa: la caja como sale del
detector, el rayo que produce, donde aterriza, y como esa vista se suma a las otras hasta volverse
un punto en el mapa del operador.
"""
import csv
import os
import sys

import numpy as np

UAV = r"C:\Users\User\Desktop\lac\uav_vision"
os.chdir(UAV)
sys.path.insert(0, UAV)
from uav_vision.camera_config import ARDUCAM_MODULE_3  # noqa: E402
from uav_vision.pinhole_local import _world_to_camera_rotation, pixel_to_ray  # noqa: E402

DATOS = os.path.join(UAV, "demo", "data")
PITCH = -55.0
LAT0, LNG0, R = -22.978029946, -43.23214256266666, 6378137.0
cam = ARDUCAM_MODULE_3.rotated_180()

D = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))
dets_all, embs_all = D["dets"], D["embs"].astype(np.float32)
sel = dets_all[:, 1] >= 0.25
dets, embs = dets_all[sel], embs_all[sel]
track = np.load(os.path.join(DATOS, "pistas_bot_cmc_sof.npz"))["track"]
poses = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(DATOS, "frames.csv")))}


def enu(lat, lng):
    return (np.radians(lng - LNG0) * R * np.cos(np.radians(LAT0)), np.radians(lat - LAT0) * R)


# La pista mas larga: la del operador. Tomamos su primera deteccion como la que vamos a seguir.
ids, cuentas = np.unique(track[track >= 0], return_counts=True)
pista = int(ids[np.argmax(cuentas)])
idx = np.where(track == pista)[0]
i = int(idx[0])
d = dets[i]
f = int(d[0])
p = poses[f]
x, y = enu(float(p["lat"]), float(p["lng"]))
pos = (x, y, float(p["alt_agl"]))
yaw = float(p["yaw"])

print("=" * 78)
print("ETAPA 1 - lo que devolvio camera.detect(), la fila cruda del cache")
print("=" * 78)
print("  frame        %d" % f)
print("  caja xyxy    (%.1f, %.1f, %.1f, %.1f)" % (d[2], d[3], d[4], d[5]))
px, py = (d[2] + d[4]) / 2.0, d[5]
print("  px, py       (%.1f, %.1f)   <- centro horizontal, y el borde INFERIOR" % (px, py))
print("  conf         %.3f" % d[1])
print("  track_id     %d" % pista)
print("  emb          vector de %d numeros, norma %.4f" % (len(embs[i]), np.linalg.norm(embs[i])))
print()
print("  telemetria de ese cuadro:")
print("    pos dron   (%.2f, %.2f, %.2f) m   <- x este, y norte, z altura sobre el suelo" % pos)
print("    yaw        %.2f grados" % yaw)
print("    pitch/roll %.2f / %.2f grados  (la actitud, hoy NO se usa)" % (float(p["pitch"]), float(p["roll"])))

print()
print("=" * 78)
print("ETAPA 2 - pixel_to_ray: el pixel se vuelve una direccion")
print("=" * 78)
cx, cy = cam.principal_point if cam.principal_point else (cam.image_width / 2, cam.image_height / 2)
d_cam = np.array([(px - cx) / cam.focal_length_px, (py - cy) / cam.focal_length_px, 1.0])
print("  focal        %.1f px      imagen %dx%d      punto principal (%.1f, %.1f)"
      % (cam.focal_length_px, cam.image_width, cam.image_height, cx, cy))
print("  d_cam        [%.5f, %.5f, %.5f]   <- (px-cx)/focal, (py-cy)/focal, 1" % tuple(d_cam))
Rm = _world_to_camera_rotation(yaw, PITCH)
print("  R (mundo->camara), sus filas son los ejes de la camara en el mundo:")
for nombre, fila in zip(("derecha", "abajo  ", "optico "), Rm):
    print("    %s  [%7.4f, %7.4f, %7.4f]" % (nombre, fila[0], fila[1], fila[2]))
origin, direction = pixel_to_ray(pos, yaw, (px, py), PITCH, cam.focal_length_px,
                                 cam.image_width, cam.image_height, cam.principal_point)
print("  direction    [%.5f, %.5f, %.5f]   norma %.6f" % (*direction, np.linalg.norm(direction)))
print("  >> NO es una posicion. Es una recta que sale del dron.")

print()
print("=" * 78)
print("ETAPA 3 - _ground_impact: la recta se corta con el suelo")
print("=" * 78)
t = (0.0 - origin[2]) / direction[2]
suelo = (origin[0] + t * direction[0], origin[1] + t * direction[1])
print("  dz           %.5f   (negativo: el rayo baja; si fuera >= 0 se devuelve None)" % direction[2])
print("  t            %.3f    <- t = (0 - z_dron) / dz : cuanto avanzar por el rayo" % t)
print("  punto suelo  (%.2f, %.2f) m" % suelo)
print("  distancia en el suelo desde el dron: %.2f m" % np.hypot(suelo[0] - pos[0], suelo[1] - pos[1]))

print()
print("=" * 78)
print("ETAPA 4 - identity.observe: esta vista se suma a las otras de la pista %d" % pista)
print("=" * 78)
puntos = []
for j in idx:
    dj = dets[j]
    fj = int(dj[0])
    pj = poses.get(fj)
    if pj is None:
        continue
    xj, yj = enu(float(pj["lat"]), float(pj["lng"]))
    o, dirj = pixel_to_ray((xj, yj, float(pj["alt_agl"])), float(pj["yaw"]),
                           ((dj[2] + dj[4]) / 2.0, dj[5]), PITCH, cam.focal_length_px,
                           cam.image_width, cam.image_height, cam.principal_point)
    if dirj[2] >= -1e-9:
        continue
    tj = (0.0 - o[2]) / dirj[2]
    puntos.append((o[0] + tj * dirj[0], o[1] + tj * dirj[1]))
P = np.array(puntos)
print("  vistas de esta pista        %d" % len(P))
print("  primera / ultima            frame %d .. %d" % (int(dets[idx[0], 0]), int(dets[idx[-1], 0])))
print("  los puntos van de           x %.1f .. %.1f     y %.1f .. %.1f"
      % (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()))
print("  MEDIA     (%.2f, %.2f)" % (P[:, 0].mean(), P[:, 1].mean()))
print("  MEDIANA   (%.2f, %.2f)   <- la que usa el sistema" % (np.median(P[:, 0]), np.median(P[:, 1])))
print("  desviacion tipica          %.2f m en x, %.2f m en y" % (P[:, 0].std(), P[:, 1].std()))
print("  >> un rayo malo mueve la media y NO mueve la mediana: por eso se usa la mediana")

print()
print("=" * 78)
print("ETAPA 5 - el candidato que resulta, contra la verdad topografiada")
print("=" * 78)
PIES = np.array([-1.3, 8.8])
med = np.array([np.median(P[:, 0]), np.median(P[:, 1])])
print("  candidato                  (%.2f, %.2f)" % (med[0], med[1]))
print("  operador topografiado      (%.2f, %.2f)" % (PIES[0], PIES[1]))
print("  error                      %.2f m" % np.linalg.norm(med - PIES))
print("  n_obs                      %d   (hacen falta 8 para reportar)" % len(P))
