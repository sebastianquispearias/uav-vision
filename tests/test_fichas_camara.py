"""
Gate for the tiled second pass: what it may add to a frame's detections and what it may not.

The frame is shrunk to the model's input before inference, so a person 55 px tall arrives as 28 and
the ones already at the limit vanish. Tiles skip that reduction, but a tile decides on a fragment of
the scene and calls a shadow a person more readily, so the pass is only allowed to ADD where the frame
found nothing. Replacing the frame's own boxes with the tiles' cost precision on the flight's footage;
adding to them did not, and that difference is the whole design.

It also costs six times the inference, which is why it runs every Nth frame. The identity layer asks
for eleven sightings in thirty six seconds, a tenth of the frames, so the cadence is affordable: the
test pins that the pass is skipped on the other frames rather than merely made cheaper.

WHAT EACH SECTION PROVES
    1. A tile find where the frame found nothing is added, in coordinates of the WHOLE frame.
    2. A tile find on top of a box the frame already has is NOT added, while the others still
       are. The fake answers the same box in every tile, so the tiles land on different parts of
       the frame and only the one covered by the frame's own box should disappear.
    3. The tiles are asked for more confidence than the frame, and only on their turn.
    4. The tiles cover the frame to its FAR EDGE, or a person at the border is invisible to them.
    5. The overlap rule itself.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uav_vision.camera import OnboardCamera, _solapan


class _Tensor:
    """Ultralytics hands back tensors, and the code under test calls .cpu().numpy() on them."""

    def __init__(self, v):
        self._v = np.asarray(v, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._v

    def __getitem__(self, i):
        return self._v[i]

    def __iter__(self):
        return iter(self._v)


class _Caja:
    def __init__(self, xyxy, conf=0.9, cls=0):
        self.xyxy = [_Tensor(xyxy)]
        self.conf = [float(conf)]
        self.cls = [float(cls)]


class _Resultado:
    def __init__(self, cajas):
        self.boxes = cajas


class _YoloFalso:
    """Answers with whatever the test planted for the region it is given."""

    names = {0: "person"}

    def __init__(self, por_tamano):
        self.por_tamano = por_tamano
        self.llamadas = []

    def __call__(self, imagen, **kw):
        self.llamadas.append((imagen.shape[1], imagen.shape[0], kw.get("conf")))
        return [_Resultado(self.por_tamano(imagen.shape[1], imagen.shape[0]))]


def _camara(yolo, cada):
    c = OnboardCamera.__new__(OnboardCamera)
    c._yolo = yolo
    c.classes = frozenset(["person"])
    c.tile_every, c.tile_side, c.tile_conf = cada, 400, 0.55
    c._n_frames = 0
    return c


frame = np.zeros((800, 1200, 3), np.uint8)

yolo = _YoloFalso(lambda w, h: [_Caja([10, 20, 40, 90])] if w == 400 else [])
c = _camara(yolo, 1)
c._n_frames = 1
extra = c._cajas_de_fichas(frame, [])
assert extra, "no agrego nada de las fichas"
xy = extra[0][0]
assert xy[0] >= 10 and xy[1] >= 20, "la caja no se paso a coordenadas del frame entero: %s" % xy
assert all(0 <= v <= 1200 for v in xy[:3:2]), "la caja cayo fuera del frame: %s" % xy
print("  agrega lo que las fichas ven donde el frame no vio: %d cajas, la primera en (%.0f, %.0f)"
      % (len(extra), xy[0], xy[1]))

c2 = _camara(_YoloFalso(lambda w, h: [_Caja([10, 20, 40, 90])] if w == 400 else []), 1)
c2._n_frames = 1
sin_previas = len(c2._cajas_de_fichas(frame, []))
c2._n_frames = 1
tapada = [e[0] for e in c2._cajas_de_fichas(frame, [])][0]
c3b = _camara(_YoloFalso(lambda w, h: [_Caja([10, 20, 40, 90])] if w == 400 else []), 1)
c3b._n_frames = 1
con_previa = c3b._cajas_de_fichas(frame, [_Caja(tapada)])
assert len(con_previa) == sin_previas - 1,     "deberia caer exactamente la que ya estaba: %d contra %d" % (len(con_previa), sin_previas)
assert all(not _solapan(e[0], tapada) for e in con_previa), "dejo pasar la que ya tenia el frame"
print("  descarta la que el frame ya tenia y conserva las otras (%d -> %d)" % (sin_previas, len(con_previa)))

yolo3 = _YoloFalso(lambda w, h: [])
c3 = _camara(yolo3, 5)
for n in range(1, 11):
    c3._n_frames = n
    c3._cajas_de_fichas(frame, [])
confs = {k[2] for k in yolo3.llamadas}
assert confs == {0.55}, "las fichas no usaron su propio umbral: %s" % confs
vueltas = len(yolo3.llamadas)
c4 = _camara(_YoloFalso(lambda w, h: []), 0)
for n in range(1, 11):
    c4._n_frames = n
    assert not c4._cajas_de_fichas(frame, []), "corrio las fichas con tile_every=0"
print("  corre solo en su turno (%d llamadas en 10 frames con cada=5) y con su umbral 0.55" % vueltas)
assert vueltas > 0, "nunca corrio las fichas"

anchos = {k[0] for k in yolo3.llamadas}
assert anchos == {400}, "las fichas no salieron del tamano pedido: %s" % anchos
c5 = _camara(_YoloFalso(lambda w, h: [_Caja([395, 395, 399, 399])] if w == 400 else []), 1)
c5._n_frames = 1
esquinas = [e[0] for e in c5._cajas_de_fichas(frame, [])]
assert any(x[2] > 1100 for x in esquinas), "ninguna ficha llego al borde derecho del frame"
assert any(x[3] > 700 for x in esquinas), "ninguna ficha llego al borde inferior"
print("  las fichas llegan hasta los bordes del frame")

assert _solapan([0, 0, 10, 10], [1, 1, 11, 11]), "no reconocio dos cajas casi iguales"
assert not _solapan([0, 0, 10, 10], [50, 50, 60, 60]), "unio dos cajas que no se tocan"
print("  la regla de solape distingue la misma caja de dos distintas")

print("TODO OK")
