"""El veredicto del operador queda en disco, con el recorte sobre el que se dio.

Cada "no es" es un negativo duro etiquetado por una persona: justo lo que le falta a un
detector entrenado con datos aereos publicos. Tres contrastes: el "no es" con recorte guarda
los bytes exactos, el "es" sin recorte guarda solo la fila, y un veredicto invalido se rechaza
sin escribir nada.

Run: python tests/test_veredictos_disco.py
"""
import base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py')
P = 8391
BASE = 'http://127.0.0.1:%d' % P


def pedir(ruta, cuerpo=None):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    q = urllib.request.Request(BASE + ruta, data=d, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=2) as r:
        return json.loads(r.read())


carpeta = tempfile.mkdtemp(prefix='veredictos_')
est = subprocess.Popen([sys.executable, GS, '--puerto', str(P), '--veredictos', carpeta],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            pedir('/estado'); break
        except Exception:
            time.sleep(0.25)

    jpeg = b'\xff\xd8\xff\xe0 un recorte de prueba \xff\xd9'
    r = pedir('/veredicto', {'v': 'no', 'x': 10.5, 'y': 4.2, 'cls': 'pedestrian', 'dron': 1,
                             'n_obs': 31, 'looks': 12, 'radius_m': 5.0,
                             'crop': base64.b64encode(jpeg).decode()})
    assert r['crop'], 'con recorte tiene que devolver el archivo guardado'
    with open(os.path.join(carpeta, r['crop']), 'rb') as f:
        assert f.read() == jpeg, 'el recorte guardado no son los bytes enviados'
    print('  "no es" con recorte -> %s, %d bytes identicos' % (r['crop'], len(jpeg)))

    r2 = pedir('/veredicto', {'v': 'si', 'x': -0.7, 'y': 6.6, 'cls': 'pedestrian', 'dron': 1})
    assert r2['crop'] is None, 'sin recorte no se inventa un archivo'
    print('  "es" sin recorte    -> solo la fila')

    try:
        pedir('/veredicto', {'v': 'quizas', 'x': 1, 'y': 1})
        raise AssertionError('un veredicto invalido tenia que dar 400')
    except urllib.error.HTTPError as e:
        assert e.code == 400, e.code
    print('  veredicto invalido  -> 400')

    filas = [json.loads(l) for l in open(os.path.join(carpeta, 'veredictos.jsonl'), encoding='utf-8')]
    assert [f['v'] for f in filas] == ['no', 'si'], 'el invalido no puede quedar escrito: %s' % filas
    assert filas[0]['looks'] == 12 and filas[0]['radius_m'] == 5.0 and filas[0]['x'] == 10.5
    jpgs = [n for n in os.listdir(carpeta) if n.endswith('.jpg')]
    assert len(jpgs) == 1, jpgs
    print('  veredictos.jsonl    -> %d filas, %d recorte' % (len(filas), len(jpgs)))
finally:
    est.terminate()
    est.wait(timeout=5)
    shutil.rmtree(carpeta, ignore_errors=True)

print()
print('TODO OK')
