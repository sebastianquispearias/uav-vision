"""Un objetivo, un pin, aunque lo vean dos drones.

Tres condiciones y ninguna sola alcanza: la clase veta, la distancia acota, y la apariencia
decide lo que queda. Y una regla que no es un umbral: dos POI del MISMO dron no se funden
nunca, porque su capa de identidad ya decidio que eran distintos y decidio con la pista
entera delante.

Run: python tests/test_fusion_drones.py
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

    # 1. Misma persona, dos drones, casi el mismo sitio y la misma apariencia -> UN pin.
    reportar(1, [poi(0.0, 6.0, 'person', vector(1), n=200)])
    reportar(2, [poi(1.0, 6.4, 'person', vector(1, ruido=0.05), n=50)])
    e = pedir('/estado')
    assert len(e['pois']) == 1, 'no se fundieron: %r' % [(p['x'], p['dron']) for p in e['pois']]
    p = e['pois'][0]
    assert p['dron'] == '1+2', p['dron']
    assert p['n_obs'] == 250, p['n_obs']
    # 200 obs en x=0.0 y 50 en x=1.0 -> 0.2, no 0.5: el que miro mas pesa mas
    assert abs(p['x'] - 0.2) < 0.01, p['x']
    print('  misma persona vista por dos drones -> 1 pin, dron "1+2", x ponderado %.2f' % p['x'])

    # 2. Dos personas cerca pero distintas: la apariencia las separa.
    reportar(1, [poi(0.0, 6.0, 'person', vector(1), n=200)])
    reportar(2, [poi(1.0, 6.4, 'person', vector(99), n=50)])
    e = pedir('/estado')
    assert len(e['pois']) == 2, 'la apariencia no las separo: %r' % e['pois']
    print('  dos personas a 1 m con apariencias distintas -> 2 pines')

    # 3. La clase veta, por cerca que esten y por parecido que sea el vector.
    reportar(1, [poi(0.0, 6.0, 'person', vector(1))])
    reportar(2, [poi(0.1, 6.0, 'car', vector(1))])
    e = pedir('/estado')
    assert len(e['pois']) == 2, 'se fundio una persona con un coche: %r' % e['pois']
    print('  persona y coche en el mismo punto con el mismo vector -> 2 pines')

    # 4. Lejos no se funde aunque todo lo demas coincida.
    reportar(1, [poi(0.0, 6.0, 'person', vector(1))])
    reportar(2, [poi(0.0, 30.0, 'person', vector(1))])
    e = pedir('/estado')
    assert len(e['pois']) == 2, e['pois']
    print('  la misma persona a 24 m de distancia -> 2 pines: la distancia acota')

    # 5. Dos POI del MISMO dron no se tocan, aunque parezcan el mismo.
    reportar(1, [poi(0.0, 6.0, 'person', vector(1)), poi(0.3, 6.1, 'person', vector(1))])
    reportar(2, [])
    e = pedir('/estado')
    assert len(e['pois']) == 2, 'la estacion deshizo lo que decidio el dron: %r' % e['pois']
    print('  dos POI del mismo dron, identicos -> se respetan los 2')
    print('test_fusion_drones OK')
finally:
    est.terminate()
