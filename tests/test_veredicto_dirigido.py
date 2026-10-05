"""The operator's "no es" has to reach each drone in that drone's own coordinates.

Until 2026-10-03 one refusal was built from the fused pin and pushed identically to every
aircraft, and nothing tested the path at all. Two things made it not work on the bench, and the
second is the one that is easy to argue yourself out of:

  A board with no appearance model cannot honour a verdict, ever. VisionProtocol._fue_descartado
  opens with `if not self._descartados or poi.get("emb") is None: return False`, so the refusal
  is received, stored, and never matches anything.

  A fused pin sits at the weighted mean of what each drone reported, so its position is not any
  drone's own. The drone matches a refusal with mismo_objetivo, which bounds by distance, so a
  refusal carrying the fused position can land outside the radius of the very candidate it was
  meant to silence. Measured: the refusal was delivered to both drones and neither one's POI
  count moved.

Run: python tests/test_veredicto_dirigido.py

The templates travel as base64 and THE STATION NEVER DECODES THEM.

THE PIN THE OPERATOR CLICKS BELONGS TO NEITHER DRONE. One drone sees the target in one place and
the other, with its own bias, a couple of metres away. The station fuses them into a pin at the
MEAN, and that is where the operator clicks, because it is the only thing drawn. Neither drone
has a candidate there, which is why the refusal has to be rebuilt per drone in that drone's own
coordinates.

THE APPEARANCE IS NOT SHARED BETWEEN AIRCRAFT. Two of them photograph the same person with
different lenses and different exposures, so handing drone 2 the vector drone 1 computed is
asking it to compare its own crops against another camera's. Here the vectors are deliberately
different so the split is visible.

SILENCING A POINT A DRONE NEVER REPORTED is silencing whatever it finds there later, so it is
refused.

A BOARD WITH NO APPEARANCE MODEL PUBLISHES NO VECTOR. There is no template to send, and with no
template the refusal cannot exist at all: position alone would suppress whoever walks past there
next. The operator has to learn that by two routes: the station's log on pressing, and the
drone's own line on screen before pressing.

A drone that has been quiet too long may have fallen out of the sky, so its last POIs are not
current state. Same criterion as pois_vigentes.
"""
import math
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))
sys.path.insert(0, os.path.join(AQUI, '..', 'scripts', 'banco_embedded'))

import gs_mapa

from uav_vision.flota import fundir


def poi(x, y, emb=None, cls='person'):
    d = {'x': x, 'y': y, 'cls': cls, 'n_obs': 50, 'mature': True,
         'age_s': 1.0, 't': 100.0, 'radius_m': 4.0}
    if emb is not None:
        d['emb'] = emb
    return d


ana = 'AAAA'
beto = 'BBBB'
AHORA = 1000.0


def montar(por_dron, vivos=('1', '2')):
    """Leaves the station holding exactly these reports, as if both drones had just spoken."""
    gs_mapa.ESTADO['pois_por_dron'] = dict(por_dron)
    gs_mapa.ESTADO['drones'] = {d: {'t': AHORA} for d in vivos}


print('=' * 70)
print('1. CADA DRON RECIBE SU PROPIA POSICION, NO LA DEL PIN FUNDIDO')
print('=' * 70)
montar({'1': [poi(10.0, 10.0, ana)], '2': [poi(12.0, 10.0, ana)]})
fundido = fundir({'1': [poi(10.0, 10.0, ana)], '2': [poi(12.0, 10.0, ana)]})
assert len(fundido) == 1, 'el montaje tenia que producir UN pin fundido: %r' % fundido
cx, cy = fundido[0]['x'], fundido[0]['y']
print('  el operador hace clic sobre el pin fundido, en (%.1f, %.1f)' % (cx, cy))

r = gs_mapa.descartes_por_dron(cx, cy, AHORA)
for dron in sorted(r):
    print('    dron %s recibe (%.1f, %.1f)' % (dron, r[dron]['x'], r[dron]['y']))
assert set(r) == {'1', '2'}, 'los dos drones vieron ese punto y los dos tienen que recibirlo: %r' % sorted(r)
assert abs(r['1']['x'] - 10.0) < 1e-6, 'el dron 1 recibio %.2f y su candidato esta en 10.0' % r['1']['x']
assert abs(r['2']['x'] - 12.0) < 1e-6, 'el dron 2 recibio %.2f y su candidato esta en 12.0' % r['2']['x']
assert r['1']['x'] != r['2']['x'], ('los dos recibieron la MISMA posicion, que es el fallo que '
                                    'esto arregla: el pin fundido no es de ninguno')
print('  -> cada uno recibe la suya; ninguno recibe la media')

print()
print('=' * 70)
print('2. Y SU PROPIA APARIENCIA, NO LA DE LA OTRA CAMARA')
print('=' * 70)
montar({'1': [poi(10.0, 10.0, ana)], '2': [poi(11.0, 10.0, beto)]})
r = gs_mapa.descartes_por_dron(10.5, 10.0, AHORA)
print('  dron 1 -> plantilla %r     dron 2 -> plantilla %r' % (r['1']['plantilla'], r['2']['plantilla']))
assert r['1']['plantilla'] == ana and r['2']['plantilla'] == beto, \
    'las plantillas se cruzaron o se compartieron: %r' % r

print()
print('=' * 70)
print('3. UN DRON QUE NO VIO ESE PUNTO NO RECIBE NADA')
print('=' * 70)
montar({'1': [poi(10.0, 10.0, ana)], '2': [poi(60.0, 10.0, beto)]})
r = gs_mapa.descartes_por_dron(10.0, 10.0, AHORA)
print('  el dron 2 tiene su candidato a 50 m del clic -> recibe: %s' % (sorted(r) or 'nada'))
assert set(r) == {'1'}, 'alguien recibio un rechazo de algo que no vio: %r' % sorted(r)
print('  y el radio que lo decide es OBJETIVO_RADIO_M = %.1f m' % gs_mapa.OBJETIVO_RADIO_M)

print()
print('=' * 70)
print('4. SIN APARIENCIA NO SE MANDA NADA, Y ESO NO ES UN SILENCIO')
print('=' * 70)
montar({'1': [poi(10.0, 10.0, None)], '2': [poi(10.2, 10.0, None)]})
r = gs_mapa.descartes_por_dron(10.0, 10.0, AHORA)
print('  dos drones sin vector de apariencia -> rechazos construidos: %d' % len(r))
assert r == {}, 'se construyo un rechazo sin apariencia: %r' % r
fuente = open(os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py'),
              encoding='utf-8').read()
assert 'NINGUN DRON PUEDE OBEDECERLO' in fuente, 'la estacion se queda callada cuando no puede'
assert 'cannot act on a verdict' in fuente, 'la pantalla no avisa que ese dron no puede obedecer'
proto = open(os.path.join(AQUI, '..', 'uav_vision', 'vision_protocol.py'), encoding='utf-8').read()
assert '"apariencia"' in proto, 'el dron no declara si puede obedecer un veredicto'
print('  -> la estacion lo dice al apretar, y la pantalla lo dice ANTES de apretar')

print()
print('=' * 70)
print('5. UN DRON CALLADO NO RECIBE ORDENES')
print('=' * 70)
montar({'1': [poi(10.0, 10.0, ana)], '2': [poi(10.2, 10.0, beto)]}, vivos=('1',))
gs_mapa.ESTADO['drones']['2'] = {'t': AHORA - gs_mapa.DRON_CALLADO_S - 5}
r = gs_mapa.descartes_por_dron(10.0, 10.0, AHORA)
print('  el dron 2 lleva %.0f s callado -> recibe: %s'
      % (gs_mapa.DRON_CALLADO_S + 5, sorted(r) or 'nada'))
assert set(r) == {'1'}, 'se le mando una orden a un dron que puede no estar: %r' % sorted(r)

print()
print('test_veredicto_dirigido OK')
