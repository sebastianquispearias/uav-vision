"""The station's CLIP score: it orders and marks, it never hides.

On flight 3 the operator's verification queue held 8 non-person candidates and 20 people. CLIP on
the crop each candidate already carries, with a threshold fixed on another day, pushed 7 of the 8
below the threshold and none of the 20. That is worth using and not worth trusting blindly, so the
station only scores: a doubtful candidate goes to the end of the list with a visible mark, and the
operator's buttons still decide.

The model is not loaded here. A fake scorer is injected through the same Anotador the real one
uses, so what is checked is the station's handling of a score, not CLIP itself (that is
test_filtro_clip_real.py). Contrasts:

    A  no scorer        the same report carries no score field and keeps the drone's order
    B  fake scorer      the doubtful candidate moves after the confident one and is flagged;
                        a candidate with no crop, and a car, are left unscored and not demoted
    C  cache            the same crop is scored once; a new crop is scored again
    D  --clip without open_clip   the station warns and still serves

Run: python tests/test_gs_clip.py
"""
import base64, importlib.util, json, os, subprocess, sys, threading, time, urllib.request
from http import server

AQUI = os.path.dirname(os.path.abspath(__file__))
BANCO = os.path.join(AQUI, '..', 'scripts', 'banco_embedded')
sys.path.insert(0, BANCO)
import gs_mapa as gs  # noqa: E402
import filtro_clip  # noqa: E402

P = 8398
BASE = 'http://127.0.0.1:%d' % P


def estado():
    with urllib.request.urlopen(BASE + '/estado', timeout=2) as r:
        return json.loads(r.read())


def b64(bytes_):
    return base64.b64encode(bytes_).decode('ascii')


srv = server.ThreadingHTTPServer(('127.0.0.1', P), gs.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

# The drone's order: the doubtful one first, so a station that does not reorder leaves it first.
REPORTE = {'type': 'vision_poi', 'frames_seen': 100, 'pois': [
    {'x': 0.0, 'y': 0.0, 'cls': 'person', 'mature': False, 'n_obs': 9,
     'crop': b64(b'\xff\xd8 maniqui \xff\xd9')},
    {'x': 20.0, 'y': 0.0, 'cls': 'person', 'mature': False, 'n_obs': 9,
     'crop': b64(b'\xff\xd8 persona \xff\xd9')},
    {'x': 40.0, 'y': 0.0, 'cls': 'person', 'mature': False, 'n_obs': 9},
    {'x': 60.0, 'y': 0.0, 'cls': 'car', 'mature': False, 'n_obs': 9,
     'crop': b64(b'\xff\xd8 maniqui en un coche \xff\xd9')},
]}

print('======================================================================')
print('PUNTAJE CLIP EN LA ESTACION: ordena y marca, no oculta')
print('======================================================================')

# -- A: without a scorer ------------------------------------------------------
gs.CLIP = None
gs.registrar(REPORTE, 1)
pois = estado()['pois']
campos = sorted({k for p in pois for k in p if k.startswith('clip')})
print('  A sin --clip     : orden x =', [p['x'] for p in pois], '| campos clip:', campos)
assert campos == [], 'sin --clip no puede aparecer ningun campo de CLIP: %s' % campos
assert [p['x'] for p in pois] == [0.0, 20.0, 40.0, 60.0], 'sin puntaje el orden es el del dron'

# -- B: a fake scorer ---------------------------------------------------------
llamadas = []


def falso(jpeg):
    llamadas.append(jpeg)
    return 0.12 if b'maniqui' in jpeg else 2.5


gs.CLIP = filtro_clip.Anotador(falso, umbral=filtro_clip.UMBRAL)
gs.registrar(REPORTE, 1)
pois = estado()['pois']
por_x = {p['x']: p for p in pois}
print('  B con puntaje    : orden x =', [p['x'] for p in pois])
for p in pois:
    print('     x=%-5s %-6s crop=%-3s clip=%-5s no_persona=%s' % (
        p['x'], p['cls'], 'si' if p.get('crop') else 'no', p.get('clip'), p.get('clip_no_persona')))
assert por_x[0.0]['clip'] == 0.12 and por_x[0.0]['clip_no_persona'] is True
assert por_x[20.0]['clip'] == 2.5 and por_x[20.0]['clip_no_persona'] is False
assert [p['x'] for p in pois].index(0.0) > [p['x'] for p in pois].index(20.0), \
    'el dudoso tiene que ir despues del confiable'
assert 'clip' not in por_x[40.0] and 'clip_no_persona' not in por_x[40.0], 'sin crop no hay puntaje'
assert [p['x'] for p in pois].index(40.0) < [p['x'] for p in pois].index(0.0), \
    'un POI sin crop no puede quedar degradado'
assert 'clip' not in por_x[60.0], 'un coche no se puntua contra "persona"'
assert [p['x'] for p in pois] == [20.0, 40.0, 60.0, 0.0], 'orden estable salvo el dudoso al final'
assert len(pois) == 4, 'nada se oculta'

# -- C: cache -----------------------------------------------------------------
antes = len(llamadas)
gs.registrar(REPORTE, 1)
repetido = len(llamadas) - antes
otro = json.loads(json.dumps(REPORTE))
otro['pois'][1]['crop'] = b64(b'\xff\xd8 persona, otra mirada \xff\xd9')
gs.registrar(otro, 1)
nuevo = len(llamadas) - antes - repetido
print('  C cache          : %d puntajes en el primer reporte, %d al repetirlo, %d con un crop nuevo'
      % (antes, repetido, nuevo))
assert antes == 2 and repetido == 0 and nuevo == 1

# -- D: --clip in a python without open_clip ------------------------------------
if importlib.util.find_spec('open_clip') is not None:
    print('  D --clip sin open_clip: SALTADO (este python tiene open_clip)')
else:
    P2 = P + 1
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    est = subprocess.Popen([sys.executable, os.path.join(BANCO, 'gs_mapa.py'), '--puerto', str(P2),
                            '--clip'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    responde = False
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen('http://127.0.0.1:%d/estado' % P2, timeout=1) as r:
                    responde = 'pois' in json.loads(r.read())
                break
            except Exception:
                time.sleep(0.25)
    finally:
        est.terminate()
        salida = est.communicate(timeout=10)[0].decode('utf-8', 'replace')
    aviso = [l for l in salida.splitlines() if l.startswith('AVISO') and 'CLIP' in l]
    print('  D --clip sin open_clip: responde /estado=%s | %s' % (responde, aviso[0] if aviso else '(sin aviso)'))
    assert responde, 'sin open_clip la estacion tiene que seguir sirviendo'
    assert aviso and 'AVISO' in aviso[0], 'sin open_clip tiene que avisar: %r' % salida

srv.shutdown()
print()
print('TODO OK')
