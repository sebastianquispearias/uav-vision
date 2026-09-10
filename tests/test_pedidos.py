"""La estacion propone una verificacion; no da ordenes.

Un objetivo se gana un pedido cuando esta sin confirmar, lo vio un solo dron, y hay otro dron
vivo que podria ir. Sin confirmar y visto por dos NO es un pedido: la segunda mirada ya
ocurrio y la respuesta siguio siendo "no estoy seguro", que es otro problema.

Run: python tests/test_pedidos.py
"""
import json, os, subprocess, sys, time, urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py')
P = 8387
BASE = 'http://127.0.0.1:%d' % P


def pedir(ruta, cuerpo=None):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    q = urllib.request.Request(BASE + ruta, data=d, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=2) as r:
        return json.loads(r.read())


def reportar(dron, pois):
    pedir('/', {'message': json.dumps({'type': 'vision_poi', 'pois': pois}), 'source': dron})


def poi(x, y, cls, mature):
    return {'x': x, 'y': y, 'cls': cls, 'mature': mature, 'n_obs': 40}


est = subprocess.Popen([sys.executable, GS, '--puerto', str(P), '--origen=-22.978,-43.232'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            pedir('/estado'); break
        except Exception:
            time.sleep(0.25)

    # Un solo dron en el aire: no hay a quien pedirselo.
    reportar(1, [poi(0.0, 6.0, 'person', False)])
    assert pedir('/pedidos')['pedidos'] == [], 'con un dron no hay pedido posible'
    print('  un solo dron volando -> ningun pedido')

    # Llega el segundo: ahora el POR VERIFICAR del primero si se puede delegar.
    reportar(2, [])
    d = pedir('/pedidos')['pedidos']
    assert len(d) == 1, d
    assert d[0]['visto_por'] == '1' and d[0]['puede_ir'] == ['2'], d[0]
    assert d[0]['lat'] is not None, 'el pedido tiene que traer el punto al que ir'
    print('  entra el dron 2 -> 1 pedido: lo vio el 1, puede ir el 2')

    # Confirmado no se pide.
    reportar(1, [poi(0.0, 6.0, 'person', True)])
    assert pedir('/pedidos')['pedidos'] == [], 'un CONFIRMADO no necesita segunda mirada'
    print('  el objetivo pasa a CONFIRMADO -> el pedido desaparece')

    # Visto por los dos y aun sin confirmar: la segunda mirada ya ocurrio.
    reportar(1, [poi(0.0, 6.0, 'person', False)])
    reportar(2, [poi(0.2, 6.1, 'person', False)])
    e = pedir('/estado')
    assert len(e['pois']) == 1 and e['pois'][0]['dron'] == '1+2', e['pois']
    assert pedir('/pedidos')['pedidos'] == [], 'ya lo miraron los dos'
    print('  lo ven los dos y sigue sin confirmar -> no se vuelve a pedir')
    print('test_pedidos OK')
finally:
    est.terminate()
