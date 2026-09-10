"""Dos drones reportando a la misma estacion no se borran entre si.

La estacion guardaba UNA lista de POI y la reemplazaba con cada reporte. Con un dron eso
era correcto: el ultimo reporte es el estado actual. Con dos, el reporte de B borraba los
objetivos de A y el mapa parpadeaba entre las dos vistas.

Run: python tests/test_dos_drones.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py')
PUERTO = 8383
BASE = 'http://127.0.0.1:%d' % PUERTO


def pedir(ruta, cuerpo=None):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(BASE + ruta, data=datos,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def reportar(dron, pois):
    pedir('/', {'message': json.dumps({'type': 'vision_poi', 'pois': pois}), 'source': dron})


estacion = subprocess.Popen(
    [sys.executable, GS, '--puerto', str(PUERTO), '--origen=-22.978,-43.232'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            pedir('/estado'); break
        except Exception:
            time.sleep(0.25)

    reportar(1, [{'x': -12.2, 'y': 13.1, 'cls': 'car', 'mature': True, 'n_obs': 132}])
    e = pedir('/estado')
    assert len(e['pois']) == 1, e['pois']
    print('  el dron 1 reporta un coche: 1 POI')

    reportar(2, [{'x': -0.2, 'y': 6.7, 'cls': 'person', 'mature': True, 'n_obs': 203}])
    e = pedir('/estado')
    clases = sorted(p['cls'] for p in e['pois'])
    assert len(e['pois']) == 2, 'el dron 2 borro al dron 1: %r' % (e['pois'],)
    assert clases == ['car', 'person'], clases
    drones = sorted(str(p['dron']) for p in e['pois'])
    assert drones == ['1', '2'], drones
    print('  el dron 2 reporta una persona: siguen los 2 POI, uno de cada dron')

    # Un dron que deja de ver algo si tiene que poder retirarlo: su propia lista se
    # reemplaza, la del otro no se toca.
    reportar(1, [])
    e = pedir('/estado')
    assert len(e['pois']) == 1 and e['pois'][0]['cls'] == 'person', e['pois']
    print('  el dron 1 deja de ver el coche: se va el suyo y queda el del dron 2')
    print('test_dos_drones OK')
finally:
    estacion.terminate()
