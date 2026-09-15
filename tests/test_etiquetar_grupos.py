"""Etiquetar por grupos: un clic etiqueta todo el grupo, y "mezcla" lo parte limpio.

Datos sinteticos con verdad conocida: 20 cajas con una apariencia y 20 con otra, en un unico grupo
inicial (mezclado a proposito). Contrastes que fallan si la herramienta miente: partir tiene que
separar exactamente las dos apariencias; marcar tiene que cubrir a todos los miembros; una etiqueta
invalida se rechaza sin tocar el archivo; el recorte es un JPEG de verdad; al reabrir no se pierde
nada.

Run: python tests/test_etiquetar_grupos.py
"""
import csv, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
HERRAMIENTA = os.path.join(AQUI, '..', 'scripts', 'etiquetar_grupos.py')
P = 8394
BASE = 'http://127.0.0.1:%d' % P


def pedir(ruta, cuerpo=None, crudo=False):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    q = urllib.request.Request(BASE + ruta, data=d, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=5) as r:
        b = r.read()
        return b if crudo else json.loads(b)


def arrancar(dirt, grupos, salida='etiquetas.json', extra=()):
    p = subprocess.Popen([sys.executable, HERRAMIENTA, '--cajas', os.path.join(dirt, 'cajas.csv'),
                          '--embs', os.path.join(dirt, 'embs.npy'), '--frames', dirt,
                          '--salida', os.path.join(dirt, salida), '--grupos', str(grupos),
                          '--puerto', str(P), *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            pedir('/estado'); return p
        except Exception:
            time.sleep(0.25)
    p.terminate(); raise RuntimeError('la herramienta no arranco')


dirt = tempfile.mkdtemp(prefix='grupos_')
rng = np.random.default_rng(3)
for f in range(1, 5):
    cv2.imwrite(os.path.join(dirt, 'frame_%04d.jpg' % f), rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
a, b = rng.normal(size=512), rng.normal(size=512)
emb = np.array([a + 0.05 * rng.normal(size=512) for _ in range(20)] +
               [b + 0.05 * rng.normal(size=512) for _ in range(20)])
np.save(os.path.join(dirt, 'embs.npy'), emb)
with open(os.path.join(dirt, 'cajas.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['frame', 'conf', 'x1', 'y1', 'x2', 'y2'])
    for i in range(40):
        w.writerow([1 + i % 4, 0.5, 50 + i, 60, 90 + i, 140])

proc = arrancar(dirt, grupos=1)
try:
    e = pedir('/estado')
    assert len(e['grupos']) == 1 and e['grupos'][0]['n'] == 40, e
    print('  arranque             : 1 grupo de 40 (mezclado a proposito)')

    nuevos = pedir('/partir', {'g': e['grupos'][0]['g']})['grupos']
    e = pedir('/estado')
    assert sorted(g['n'] for g in e['grupos']) == [20, 20], e['grupos']
    print('  "mezcla: partir"     : grupos de', [g['n'] for g in e['grupos']])

    pedir('/marcar', {'g': nuevos[0], 'v': 'persona'})
    pedir('/marcar', {'g': nuevos[1], 'v': 'no'})
    et = json.load(open(os.path.join(dirt, 'etiquetas.json'), encoding='utf-8'))['etiquetas']
    assert len(et) == 40, 'dos clics tenian que etiquetar las 40 cajas: %d' % len(et)
    primera, segunda = {et[str(i)] for i in range(20)}, {et[str(i)] for i in range(20, 40)}
    assert len(primera) == 1 and len(segunda) == 1 and primera != segunda, \
        'partir no separo las dos apariencias: %s / %s' % (primera, segunda)
    print('  dos clics            : 40 cajas, cada apariencia con una sola etiqueta')

    try:
        pedir('/marcar', {'g': nuevos[0], 'v': 'quizas'})
        raise AssertionError('una etiqueta invalida tenia que dar 400')
    except urllib.error.HTTPError as err:
        assert err.code == 400
    assert json.load(open(os.path.join(dirt, 'etiquetas.json'), encoding='utf-8'))['etiquetas'] == et
    print('  etiqueta invalida    : 400, archivo intacto')

    jpg = pedir('/recorte/5', crudo=True)
    assert jpg[:2] == b'\xff\xd8' and cv2.imdecode(np.frombuffer(jpg, np.uint8), 1).shape == (96, 96, 3)
    print('  recorte              : JPEG 96x96 real')
finally:
    proc.terminate(); proc.wait(timeout=5)

proc = arrancar(dirt, grupos=2)
try:
    assert pedir('/estado')['etiquetadas'] == 40, 'al reabrir se perdieron etiquetas'
    print('  reabrir              : 40 etiquetadas, nada perdido')
    # The person / not-person mode must keep refusing a letter: identities are a separate mode.
    try:
        pedir('/marcar', {'g': pedir('/estado')['grupos'][0]['g'], 'v': 'F'})
        raise AssertionError('sin --identidad una letra tenia que dar 400')
    except urllib.error.HTTPError as err:
        assert err.code == 400
    print('  modo persona/no      : una letra da 400')
finally:
    proc.terminate(); proc.wait(timeout=5)

# Identity mode: one letter per real person, X for not a person, ? for cannot tell. Same grouping,
# its own output file, so the person / not-person labels above are never touched.
proc = arrancar(dirt, grupos=2, salida='identidad.json', extra=('--identidad',))
try:
    g1, g2 = [g['g'] for g in pedir('/estado')['grupos']]
    pedir('/marcar', {'g': g1, 'v': 'F'})
    pedir('/marcar', {'g': g2, 'v': 'x'})
    et = json.load(open(os.path.join(dirt, 'identidad.json'), encoding='utf-8'))['etiquetas']
    assert len(et) == 40 and set(et.values()) == {'F', 'X'}, 'letras: %s' % set(et.values())
    print('  modo identidad       : F y x -> 40 cajas con {F, X} (x normalizada a X)')
    for mala in ('persona', 'AB', '', '7'):
        try:
            pedir('/marcar', {'g': g1, 'v': mala})
            raise AssertionError('%r tenia que dar 400 en modo identidad' % mala)
        except urllib.error.HTTPError as err:
            assert err.code == 400
    assert json.load(open(os.path.join(dirt, 'identidad.json'), encoding='utf-8'))['etiquetas'] == et
    assert json.load(open(os.path.join(dirt, 'etiquetas.json'), encoding='utf-8'))['etiquetas'] != et
    print("  identidad invalida   : 'persona', 'AB', '', '7' dan 400; archivos intactos y separados")
    jpg = pedir('/frame/5', crudo=True)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1)
    assert jpg[:2] == b'\xff\xd8' and img.shape[1] > 96, 'la vista de frame tiene que ser el frame, no el recorte'
    print('  vista de frame       : JPEG %dx%d con la caja marcada' % (img.shape[1], img.shape[0]))
finally:
    proc.terminate(); proc.wait(timeout=5)
    shutil.rmtree(dirt, ignore_errors=True)

print()
print('TODO OK')
