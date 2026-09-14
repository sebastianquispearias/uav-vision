"""La inclinacion del dron entra en el rayo de la camara.

La camara va fija al cuerpo. Cuando el dron baja la nariz para avanzar, o se ladea para girar, la
camara mira a otro sitio, y un rayo calculado con el angulo de montaje solo cae en otro punto del
suelo. Verdad conocida: un punto de suelo proyectado con la actitud real del dron y devuelto a rayo.

  - con la actitud compensada, el rayo vuelve al punto exacto;
  - sin compensar, cae a metros (lo que se hacia hasta ahora);
  - con actitud cero, la matriz es identica a la de antes, bit a bit.

Run: python tests/test_actitud.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from uav_vision.pinhole_local import _world_to_camera_rotation, pixel_to_ray, project_to_pixel

F, W, H = 1407.0, 1920, 1080
DRON = (3.0, -4.0, 30.0)
MONTAJE = -55.0
YAW = 37.0


def impacto(o, d):
    t = -o[2] / d[2]
    return np.array([o[0] + t * d[0], o[1] + t * d[1]])


R0 = _world_to_camera_rotation(YAW, MONTAJE)
assert np.array_equal(R0, _world_to_camera_rotation(YAW, MONTAJE, 0.0, 0.0)), "actitud cero cambio la matriz"
print("  actitud cero: matriz identica a la de antes")

for nombre, bp, br in (("nariz abajo 10 grados", -10.0, 0.0), ("ladeado 12 grados", 0.0, 12.0),
                       ("nariz abajo 8 y ladeado -6", -8.0, -6.0)):
    peor_sin, peor_con = 0.0, 0.0
    for objetivo in ((18.0, 25.0, 0.0), (-6.0, 30.0, 0.0), (10.0, 12.0, 0.0), (0.0, 20.0, 0.0)):
        px = project_to_pixel(DRON, objetivo, YAW, MONTAJE, F, W, H, None, bp, br)
        if px is None:
            continue
        con = impacto(*pixel_to_ray(DRON, YAW, px, MONTAJE, F, W, H, None, bp, br))
        sin = impacto(*pixel_to_ray(DRON, YAW, px, MONTAJE, F, W, H, None))
        peor_con = max(peor_con, float(np.hypot(*(con - objetivo[:2]))))
        peor_sin = max(peor_sin, float(np.hypot(*(sin - objetivo[:2]))))
    print("  %-28s peor error sin compensar %6.2f m | compensado %.1e m" % (nombre, peor_sin, peor_con))
    assert peor_con < 1e-6, "compensando, el rayo tiene que volver al punto exacto"
    assert peor_sin > 2.0, "sin compensar tiene que caer lejos, o el caso no prueba nada"

print()
print("TODO OK")
