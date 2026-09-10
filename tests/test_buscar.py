"""The control plane: the station stores what the operator asks for.

Two properties are worth pinning. The order survives a poll, which is what the
drone depends on; and asking for nothing is not the same as never having asked,
which is what tells a freshly booted drone to keep its own defaults.

Run: python tests/test_buscar.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py')
PUERTO = 8377
BASE = 'http://127.0.0.1:%d' % PUERTO


def pedir(ruta, cuerpo=None):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(BASE + ruta, data=datos,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


estacion = subprocess.Popen(
    [sys.executable, GS, '--puerto', str(PUERTO), '--origen=-22.978,-43.232'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            pedir('/buscar'); break
        except Exception:
            time.sleep(0.25)

    d = pedir('/buscar')
    assert d['clases'] is None, 'recien arrancada no manda nada: %r' % (d,)
    v0 = d['v']

    pedir('/buscar', {'clases': ['car']})
    d = pedir('/buscar')
    assert d['clases'] == ['car'], d
    assert d['v'] == v0 + 1, 'la version tiene que subir: %r' % (d,)
    print('  la orden sobrevive al sondeo:', d)

    pedir('/buscar', {'clases': ['person', 'car']})
    d = pedir('/buscar')
    assert set(d['clases']) == {'person', 'car'}, d
    v_dos = d['v']

    # Asking for nothing is a real order ("everything again"), and it has to be
    # distinguishable from a station that was never told anything.
    pedir('/buscar', {'clases': []})
    d = pedir('/buscar')
    assert d['clases'] is None and d['v'] == v_dos + 1, \
        'pedir nada tiene que contar como orden: %r' % (d,)
    print('  pedir nada cuenta como orden y sube la version:', d)

    # The data plane must not have moved: a POST to / is still a report.
    pedir('/', {'message': json.dumps({'type': 'vision_poi', 'pois': []}), 'source': 1})
    e = pedir('/estado')
    assert 'pois' in e and 'reportes' in e, e
    print('  el plano de datos sigue intacto: %d reportes' % e['reportes'])
    print('test_buscar OK')
finally:
    estacion.terminate()
