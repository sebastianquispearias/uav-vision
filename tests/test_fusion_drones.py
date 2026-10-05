"""Un objetivo, un pin, aunque lo vean dos drones.

Tres condiciones y ninguna sola alcanza: la clase veta, la distancia acota, y la apariencia
decide lo que queda. Y una regla que no es un umbral: dos POI del MISMO dron no se funden
nunca, porque su capa de identidad ya decidio que eran distintos y decidio con la pista
entera delante.

Run: python tests/test_fusion_drones.py

WHAT EACH SECTION PROVES
    1. The same person, two drones, almost the same place and the same appearance: ONE pin. The
       position is weighted by observations, so the drone that looked more weighs more.
    2. Two people close together but different: appearance separates them.
    3. The class vetoes, however close they are and however alike the vector.
    4. Far apart does not fuse even when everything else matches.
    5. Two POIs from the SAME drone are not touched, even when they look like the same thing.
    6. The age of the last sighting reaches the page, and a fused pin is as fresh as the drone
       that saw it LAST: one drone lost the person a while ago, the other is looking at it now,
       and keeping the first drone's age would fade out a target that is in view.
    7. The fused margin is the SMALLER of the two and not the one that arrived first. The same
       target seen from close and from far does not have one uncertainty, it has the closer
       drone's; NOTES.md has what share of that radius is bias rather than scatter.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py')
PUERTO = 8385
BASE = 'http://127.0.0.1:%d' % PUERTO

rng = np.random.default_rng(7)


def vector(semilla, ruido=0.0):
    r = np.random.default_rng(semilla)
    v = r.normal(size=512).astype(np.float32)
    if ruido:
        v = v + ruido * rng.normal(size=512).astype(np.float32)
    v /= np.linalg.norm(v)
    return base64.b64encode(v.astype(np.float16).tobytes()).decode()


def pedir(ruta, cuerpo=None):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    q = urllib.request.Request(BASE + ruta, data=d, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=2) as r:
        return json.loads(r.read())


def reportar(dron, pois):
    pedir('/', {'message': json.dumps({'type': 'vision_poi', 'pois': pois}), 'source': dron})


def poi(x, y, cls, emb=None, n=100):
    return {'x': x, 'y': y, 'cls': cls, 'mature': True, 'n_obs': n, 'emb': emb}


est = subprocess.Popen([sys.executable, GS, '--puerto', str(PUERTO), '--origen=-22.978,-43.232'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            pedir('/estado'); break
        except Exception:
            time.sleep(0.25)

    reportar(1, [poi(0.0, 6.0, 'person', vector(1), n=200)])
    reportar(2, [poi(1.0, 6.4, 'person', vector(1, ruido=0.05), n=50)])
    e = pedir('/estado')
    assert len(e['pois']) == 1, 'no se fundieron: %r' % [(p['x'], p['dron']) for p in e['pois']]
    p = e['pois'][0]
    assert p['dron'] == '1+2', p['dron']
    assert p['n_obs'] == 250, p['n_obs']
    assert abs(p['x'] - 0.2) < 0.01, p['x']
    print('  misma persona vista por dos drones -> 1 pin, dron "1+2", x ponderado %.2f' % p['x'])

    reportar(1, [poi(0.0, 6.0, 'person', vector(1), n=200)])
    reportar(2, [poi(1.0, 6.4, 'person', vector(99), n=50)])
    e = pedir('/estado')
    assert len(e['pois']) == 2, 'la apariencia no las separo: %r' % e['pois']
    print('  dos personas a 1 m con apariencias distintas -> 2 pines')

    reportar(1, [poi(0.0, 6.0, 'person', vector(1))])
    reportar(2, [poi(0.1, 6.0, 'car', vector(1))])
    e = pedir('/estado')
    assert len(e['pois']) == 2, 'se fundio una persona con un coche: %r' % e['pois']
    print('  persona y coche en el mismo punto con el mismo vector -> 2 pines')

    reportar(1, [poi(0.0, 6.0, 'person', vector(1))])
    reportar(2, [poi(0.0, 30.0, 'person', vector(1))])
    e = pedir('/estado')
    assert len(e['pois']) == 2, e['pois']
    print('  la misma persona a 24 m de distancia -> 2 pines: la distancia acota')

    reportar(1, [poi(0.0, 6.0, 'person', vector(1)), poi(0.3, 6.1, 'person', vector(1))])
    reportar(2, [])
    e = pedir('/estado')
    assert len(e['pois']) == 2, 'la estacion deshizo lo que decidio el dron: %r' % e['pois']
    print('  dos POI del mismo dron, identicos -> se respetan los 2')

    viejo = dict(poi(0.0, 6.0, 'person', vector(1), n=200), age_s=40.0)
    fresco = dict(poi(1.0, 6.4, 'person', vector(1, ruido=0.05), n=50), age_s=0.5)
    reportar(1, [viejo])
    reportar(2, [fresco])
    e = pedir('/estado')
    assert len(e['pois']) == 1, e['pois']
    p = e['pois'][0]
    print('  dron 1 lo vio hace 40 s, dron 2 hace 0.5 s -> age_s del pin fundido: %s' % p.get('age_s'))
    assert p.get('age_s') is not None, 'la estacion se come age_s'
    assert p['age_s'] < 2.0, 'el pin fundido tomo la edad del dron que lo perdio: %s' % p['age_s']
    reportar(2, [dict(fresco, age_s=0.5)])
    reportar(1, [viejo])
    e = pedir('/estado')
    print('  mismo caso, reportes en el otro orden -> age_s: %s' % e['pois'][0].get('age_s'))
    assert e['pois'][0]['age_s'] < 2.0, e['pois'][0]['age_s']
    v = vector(77)
    for primero, segundo, nombre in ((7.2, 4.4, 'el lejano primero'), (4.4, 7.2, 'el cercano primero')):
        a = poi(60.0, 60.0, 'person', v); a['radius_m'] = primero
        b = poi(60.4, 60.2, 'person', v); b['radius_m'] = segundo
        reportar('8', [a]); reportar('9', [b])
        e = pedir('/estado')
        fundidos = [q for q in e['pois'] if abs(q['x'] - 60.0) < 2.0 and '+' in str(q.get('dron', ''))]
        assert fundidos, 'los dos reportes tenian que fundirse en un pin'
        r = fundidos[0].get('radius_m')
        print('  %-22s radios %.1f y %.1f -> el pin fundido dice %.1f' % (nombre, primero, segundo, r))
        assert abs(r - min(primero, segundo)) < 1e-6,             'el pin fundido tiene que quedarse con el margen MENOR, no con el que llego primero'

    print('test_fusion_drones OK')
finally:
    est.terminate()
