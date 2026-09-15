"""The station's CLIP scorer gives the numbers the measurement gave.

scripts/medir_filtro_clip.py fixed the threshold on image features cached on disk. The station
scores with its own code, so the question is whether it is the same score, not a similar one.
Two contrasts:

    equivalence  the station's scorer, on the exact array medir_filtro_clip.recortar(...,
                 degradar=128) builds, against the score recomputed from that script's cached
                 features for the same box: a non-person and a person of flight 02ago.
    travelling   the path the station really runs: the crop cut the way the camera cuts it,
                 128 px JPEG q70 bytes, decoded and scored. Printed for the operator's frame 2571
                 (a person) and for the non-person above.

Needs open_clip, which only the training venv has; any other python prints SALTADO:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe tests/test_filtro_clip_real.py
"""
import importlib.util, json, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
HNO = os.path.join(RAIZ, '..', 'drone-geolocation')
FRAMES = os.path.join(HNO, 'data', 'flight_02ago', '20260802_133309', 'frames')
CACHE = os.path.join(HNO, 'entrenamiento', 'clip_ViT-B-32_openai_02ago_d128.npz')

print('======================================================================')
print('PUNTAJE CLIP DE LA ESTACION CONTRA LA MEDICION')
print('======================================================================')
faltan = [n for n, ok in (('open_clip', importlib.util.find_spec('open_clip') is not None),
                          ('frames 02ago', os.path.isdir(FRAMES)),
                          ('cache de features', os.path.exists(CACHE))) if not ok]
if faltan:
    print('  SALTADO: falta %s' % ', '.join(faltan))
    print()
    print('TODO OK')
    raise SystemExit(0)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(RAIZ, 'scripts'))
sys.path.insert(0, os.path.join(RAIZ, 'scripts', 'banco_embedded'))
import medir_filtro_clip as m  # noqa: E402
import filtro_clip  # noqa: E402

d = np.load(os.path.join(RAIZ, 'demo', 'data', 'examen_v3_datos.npz'))['dets']
d = d[d[:, 1] >= 0.25]
et = json.load(open(os.path.join(HNO, 'entrenamiento', 'identidad_gt_02ago.json'),
                    encoding='utf-8'))['etiquetas']
ix = [i for i in range(len(d)) if 3000 <= d[i, 0] <= 3700 and et.get(str(i)) not in (None, '?')]
feats = np.load(CACHE)['feats']
assert len(feats) == len(ix), 'la cache no corresponde a estas cajas'
feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

puntuador = filtro_clip.PuntuadorClip()
t = puntuador.texto.cpu().numpy()


def de_cache(k):
    sim = 100.0 * feats[k] @ t.T
    n = len(m.POSITIVOS)
    return float(np.log(np.exp(sim[:n]).sum()) - np.log(np.exp(sim[n:]).sum()))


def img(frame):
    return cv2.imread(os.path.join(FRAMES, 'frame_%04d.jpg' % frame))


def como_la_camara(frame, caja, lado_px=128, calidad=70, margen=0.25):
    """The crop a drone ships: uav_vision/camera.py _crop, reproduced without the camera."""
    f = img(frame)
    h, w = f.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in caja]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lado = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margen)
    a = max(0, int(cx - lado / 2)), max(0, int(cy - lado / 2))
    b = min(w, int(cx + lado / 2)), min(h, int(cy + lado / 2))
    p = f[a[1]:b[1], a[0]:b[0]]
    if max(p.shape[:2]) > lado_px:
        e = lado_px / max(p.shape[:2])
        p = cv2.resize(p, (max(1, int(p.shape[1] * e)), max(1, int(p.shape[0] * e))),
                       interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', p, [int(cv2.IMWRITE_JPEG_QUALITY), calidad])[1].tobytes()


print('  modelo ViT-B-32/openai en %s | umbral %.3f' % (puntuador.dev, filtro_clip.UMBRAL))
print('  %-28s %9s %9s %8s' % ('caja', 'medicion', 'estacion', 'dif'))
k_no = next(k for k, i in enumerate(ix) if et[str(i)] in ('X', 'x'))
k_si = next(k for k, i in enumerate(ix) if et[str(i)] not in ('X', 'x'))
for k, nombre in ((k_no, 'no persona'), (k_si, 'persona')):
    i = ix[k]
    rgb = m.recortar(img(int(d[i, 0])), *d[i, 2:6], degradar=128)
    a, b = de_cache(k), puntuador.puntuar_rgb(rgb)
    print('  fila %4d frame %4d %-9s %9.3f %9.3f %8.4f' % (i, d[i, 0], nombre, a, b, abs(a - b)))
    assert abs(a - b) < 0.05, 'la estacion no puntua como la medicion'

print('  crop que viaja (JPEG 128 px q70):')
for i, nombre in ((189, 'persona'), (ix[k_no], 'no persona')):
    jpeg = como_la_camara(int(d[i, 0]), d[i, 2:6])
    s = puntuador.puntuar(jpeg)
    print('  fila %4d frame %4d %-10s %5d bytes  CLIP %6.3f -> %s' % (
        i, d[i, 0], nombre, len(jpeg), s,
        'probable no persona' if s < filtro_clip.UMBRAL else 'persona'))
print()
print('TODO OK')
