"""Etiquetar por grupos: un clic etiqueta todo el grupo, y "mezcla" lo parte limpio.

Datos sinteticos con verdad conocida: 20 cajas con una apariencia y 20 con otra, en un unico grupo
inicial (mezclado a proposito). Contrastes que fallan si la herramienta miente: partir tiene que
separar exactamente las dos apariencias; marcar tiene que cubrir a todos los miembros; una etiqueta
invalida se rechaza sin tocar el archivo; el recorte es un JPEG de verdad; al reabrir no se pierde
nada. Los frames son grises uniformes para que los colores de las cajas dibujadas se puedan contar
despues del JPEG: el borde grueso de la caja propia tiene que aparecer en el recorte, y una caja de
--contexto solo si cae dentro del recorte.

Run: python tests/test_etiquetar_grupos.py
"""
import csv, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
HERRAMIENTA = os.path.join(AQUI, '..', 'scripts', 'etiquetar_grupos.py')
P = 8394
sys.path.insert(0, os.path.join(AQUI, '..', 'scripts'))
from etiquetar_grupos import LADO, REVISION
BASE = 'http://127.0.0.1:%d' % P


def pedir(ruta, cuerpo=None, crudo=False):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    q = urllib.request.Request(BASE + ruta, data=d, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=5) as r:
        b = r.read()
        return b if crudo else json.loads(b)


def contar(jpg, color):
    """Pixels of a JPEG that are clearly one drawn color over the uniform gray frame."""
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1).astype(int)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    if color == 'gruesa':      # yellow: this box
        return int(((r > 170) & (g > 150) & (b < 90)).sum())
    if color == 'contexto':    # magenta: the flight's detections
        return int(((r - g > 50) & (b - g > 50)).sum())
    raise ValueError(color)


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
    cv2.imwrite(os.path.join(dirt, 'frame_%04d.jpg' % f), np.full((300, 400, 3), 128, np.uint8))
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
    assert jpg[:2] == b'\xff\xd8' and cv2.imdecode(np.frombuffer(jpg, np.uint8), 1).shape == (LADO, LADO, 3)
    print('  recorte              : JPEG %dx%d real' % (LADO, LADO))
    n_gruesa, n_ctx = contar(jpg, 'gruesa'), contar(jpg, 'contexto')
    assert n_gruesa > 40, 'el recorte tiene que traer la caja propia con borde grueso: %d px' % n_gruesa
    assert n_ctx == 0, 'sin --contexto no puede haber cajas de contexto: %d px' % n_ctx
    print('  borde grueso         : %d px amarillos en el recorte, 0 de contexto sin --contexto' % n_gruesa)
finally:
    proc.terminate(); proc.wait(timeout=5)

def rechaza(ruta, cuerpo, que):
    try:
        pedir(ruta, cuerpo)
    except urllib.error.HTTPError as err:
        assert err.code == 400, '%s: codigo %d' % (que, err.code)
        return
    raise AssertionError('%s tenia que dar 400' % que)


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
    rechaza('/letras/renombrar', {'de': 'A', 'a': 'B'}, 'renombrar letras sin --identidad')
    assert pedir('/estado')['catalogo'] == [], 'sin --identidad no hay catalogo de letras'
    print('  sin identidad        : renombrar da 400 y el catalogo viene vacio')
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
    # The letter catalogue: each letter with the first crop labelled with it, and renaming, which is how
    # two letters given to the same person are merged.
    cat = {c['letra']: c for c in pedir('/estado')['catalogo']}
    assert set(cat) == {'F', 'X'} and cat['F']['n'] == 20 and isinstance(cat['F']['i'], int), cat
    print('  catalogo de letras   : F y X, con %d y %d cajas y un recorte de referencia' % (cat['F']['n'], cat['X']['n']))
    assert pedir('/letras/renombrar', {'de': 'F', 'a': 'G'})['cajas'] == 20
    et = json.load(open(os.path.join(dirt, 'identidad.json'), encoding='utf-8'))['etiquetas']
    assert set(et.values()) == {'G', 'X'}, set(et.values())
    assert {c['letra'] for c in pedir('/estado')['catalogo']} == {'G', 'X'}
    rechaza('/letras/renombrar', {'de': 'F', 'a': 'G'}, 'renombrar una letra que ya no existe')
    print('  renombrar o fundir   : F -> G cambia 20 cajas; repetirlo da 400')
    jpg = pedir('/frame/5', crudo=True)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1)
    assert jpg[:2] == b'\xff\xd8' and img.shape[1] > 96, 'la vista de frame tiene que ser el frame, no el recorte'
    print('  vista de frame       : JPEG %dx%d con la caja marcada' % (img.shape[1], img.shape[0]))
finally:
    proc.terminate(); proc.wait(timeout=5)

# Context: the flight's detections drawn thin over the thumbnails. Row 5 is in frame 2 and its crop
# spans x 45-105, y 40-160; the context box in frame 2 lies inside it. The context box in frame 3
# lies far from row 6's crop (x 46-106), so it must not show there.
with open(os.path.join(dirt, 'contexto.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['frame', 'conf', 'x1', 'y1', 'x2', 'y2'])
    w.writerow([2, 0.9, 60, 70, 85, 130])
    w.writerow([3, 0.9, 250, 180, 300, 280])
json.dump({'etiquetas': {'0': 'a'}}, open(os.path.join(dirt, 'contexto_letras.json'), 'w'))
proc = arrancar(dirt, grupos=2, extra=('--contexto', os.path.join(dirt, 'contexto.csv'),
                                       '--contexto-etiquetas', os.path.join(dirt, 'contexto_letras.json')))
try:
    dentro, fuera = pedir('/recorte/5', crudo=True), pedir('/recorte/6', crudo=True)
    n_dentro, n_fuera = contar(dentro, 'contexto'), contar(fuera, 'contexto')
    assert n_dentro > 20, 'una caja de contexto dentro del recorte tiene que dibujarse: %d px' % n_dentro
    assert n_fuera == 0, 'una caja de contexto fuera del recorte no puede aparecer: %d px' % n_fuera
    assert contar(dentro, 'gruesa') > 40, 'con --contexto la caja propia sigue en grueso'
    print('  contexto dentro      : %d px magenta en el recorte de la fila 5' % n_dentro)
    print('  contexto fuera       : %d px magenta en el recorte de la fila 6' % n_fuera)
    fr = pedir('/frame/5', crudo=True)
    n_fr_g, n_fr_c = contar(fr, 'gruesa'), contar(fr, 'contexto')
    assert n_fr_g > 100 and n_fr_c > 20, 'el frame tiene que traer caja propia y contexto: %d / %d' % (n_fr_g, n_fr_c)
    print('  frame con contexto   : %d px de caja propia, %d px de contexto' % (n_fr_g, n_fr_c))
    assert 'grueso' in pedir('/', crudo=True).decode('utf-8'), 'la ayuda tiene que explicar grueso / fino'
    print('  ayuda                : explica grueso = esta caja, fino = las del vuelo')
finally:
    proc.terminate(); proc.wait(timeout=5)


# Frame review, the unit a detector trains on. Every CSV box starts at x 50+i, 40 px wide, so the boxes
# of one frame overlap: once two of them are persona they must be flagged as a possible double. Frame 9
# has no candidate: it must be listed only because --lista-frames names it, since that is where a
# person every detector missed would have to be drawn.
cv2.imwrite(os.path.join(dirt, 'frame_0009.jpg'), np.full((300, 400, 3), 128, np.uint8))
open(os.path.join(dirt, 'lista.txt'), 'w').write('1\n2\n9\n')
grupos_antes = json.load(open(os.path.join(dirt, 'etiquetas.json'), encoding='utf-8'))['etiquetas']
extra = ('--lista-frames', os.path.join(dirt, 'lista.txt'), '--nombre', 'vuelo_prueba')
proc = arrancar(dirt, grupos=2, extra=extra)
try:
    e = pedir('/frames/estado')
    assert [x['f'] for x in e['frames']] == [1, 2, 3, 4, 9] and e['nombre'] == 'vuelo_prueba', e
    assert [x['doble'] for x in e['frames']] == [True, True, True, True, False], e['frames']
    print('  estado por frame     : doble en los 4 frames con 5 personas encimadas, no en el 9 vacio')
    print('  lista de frames      : %s (el 9 sin candidatas, por --lista-frames), nombre %s'
          % ([x['f'] for x in e['frames']], e['nombre']))

    d = pedir('/frames/1')
    per = [b for b in d['cajas'] if b['etiqueta'] == 'persona']
    assert len(d['cajas']) == 10 and len(per) == 5, d['cajas']
    dobles = sum(b['doble'] for b in d['cajas'])
    assert dobles == 5 and not any(b['doble'] for b in d['cajas'] if b['etiqueta'] != 'persona'), d['cajas']
    for b in per[1:]:
        pedir('/frames/caja', {'i': b['i'], 'v': 'duplicado'})
    d = pedir('/frames/1')
    assert sum(b['doble'] for b in d['cajas']) == 0 and sum(b['corregida'] for b in d['cajas']) == 4
    print('  doble                : 5 personas que se pisan -> 5 marcadas; 4 pasadas a duplicado -> 0')

    k0 = pedir('/frames/nueva', {'f': 9, 'caja': [100, 100, 140, 180]})['k']
    assert not pedir('/frames/9')['nuevas'][0]['doble']
    pedir('/frames/nueva', {'f': 9, 'caja': [104, 104, 140, 180]})
    assert all(b['doble'] for b in pedir('/frames/9')['nuevas'])
    pedir('/frames/borrar', {'f': 9, 'k': 1})
    d9 = pedir('/frames/9')
    assert len(d9['nuevas']) == 1 and not d9['nuevas'][0]['doble'] and k0 == 0
    print('  dibujar              : 1 caja sola no es doble; 2 encimadas si; borrar la 2a -> vuelve a 1')

    rechaza('/frames/nueva', {'f': 9, 'caja': [10, 10, 12, 30]}, 'una caja de 2 px')
    rechaza('/frames/nueva', {'f': 7, 'caja': [10, 10, 60, 90]}, 'un frame fuera de la lista')
    rechaza('/frames/caja', {'i': 0, 'v': 'quizas'}, 'una etiqueta inventada')
    rechaza('/frames/borrar', {'f': 9, 'k': 5}, 'borrar una caja que no existe')
    print('  invalidos            : caja de 2 px, frame 7, "quizas", borrar k=5 -> 400')

    pedir('/frames/caja', {'i': per[0]['i'], 'v': 'ignorar'})
    ign = [b for b in pedir('/frames/1')['cajas'] if b['i'] == per[0]['i']][0]
    assert ign['etiqueta'] == 'ignorar' and 'ignorar' in REVISION and not ign['doble'], ign
    print('  ignorar              : la caja queda "ignorar" y deja de contar como persona para los dobles')

    # The groups page shows the label of each crop, so one box corrected apart from its group is visible
    # there, and Z undoes it by sending a null label, which drops the correction.
    crops = {m['i']: m for g in pedir('/estado')['grupos'] for m in g['muestra']}
    assert crops[per[0]['i']]['etiqueta'] == 'ignorar' and crops[per[0]['i']]['corregida'], crops[per[0]['i']]
    sin_corregir = [m for m in crops.values() if not m['corregida']][0]
    assert sin_corregir['etiqueta'] in ('persona', 'no'), sin_corregir
    print('  grupos por recorte   : el recorte corregido dice "ignorar"; los demas, la etiqueta de su grupo')
    pedir('/frames/caja', {'i': per[0]['i'], 'v': None})
    vuelto = {m['i']: m for g in pedir('/estado')['grupos'] for m in g['muestra']}[per[0]['i']]
    assert vuelto['etiqueta'] == 'persona' and not vuelto['corregida'], vuelto
    pedir('/frames/caja', {'i': per[0]['i'], 'v': 'ignorar'})
    print('  deshacer (Z)         : etiqueta nula quita la correccion y la caja vuelve a la de su grupo')
finally:
    proc.terminate(); proc.wait(timeout=5)

# A frame with a box nobody labelled cannot be reviewed: rows 40-41 are new boxes in frame 2, beyond the
# 40 the groups labelled, so they have no label until the review gives them one.
with open(os.path.join(dirt, 'cajas.csv'), 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow([2, 0.5, 300, 60, 340, 140]); w.writerow([2, 0.5, 300, 160, 340, 240])
np.save(os.path.join(dirt, 'embs.npy'), np.vstack([emb, rng.normal(size=(2, 512))]))
proc = arrancar(dirt, grupos=2, extra=extra)
try:
    d = pedir('/frames/1')
    assert pedir('/frames/estado')['revisados'] == 0
    assert sum(b['corregida'] for b in d['cajas']) == 5 and len(pedir('/frames/9')['nuevas']) == 1
    print('  reabrir              : 5 correcciones y 1 caja dibujada siguen ahi')
    rechaza('/frames/revisado', {'f': 2, 'v': True}, 'revisar un frame con 2 cajas sin etiquetar')
    sin = [b['i'] for b in pedir('/frames/2')['cajas'] if b['etiqueta'] is None]
    assert len(sin) == 2, sin
    for i in sin:
        pedir('/frames/caja', {'i': i, 'v': 'no'})
    pedir('/frames/revisado', {'f': 2, 'v': True}); pedir('/frames/revisado', {'f': 9, 'v': True})
    assert pedir('/frames/estado')['revisados'] == 2
    print('  revisado             : con 2 sin etiquetar -> 400; etiquetadas -> 2 frames revisados')
    rev = json.load(open(os.path.join(dirt, 'etiquetas_frames.json'), encoding='utf-8'))
    assert rev['revisados'] == [2, 9] and rev['nuevas'] == {'9': [[100.0, 100.0, 140.0, 180.0]]}, rev
    e = {x['f']: x for x in pedir('/frames/estado')['frames']}
    assert not e[1]['doble'] and e[3]['doble'] and e[9]['personas'] == 1, e
    # The mosaic's button: frame 3 and 4 have every box labelled and are accepted; frame 7 is not in the
    # list and comes back with its reason, without stopping the others.
    lote = pedir('/frames/revisados_lote', {'frames': [3, 7, 4]})
    assert lote['hechos'] == [3, 4] and [x['f'] for x in lote['rechazados']] == [7], lote
    assert pedir('/frames/estado')['revisados'] == 4
    print('  lote del mosaico     : [3, 7, 4] -> hechos [3, 4], rechazado 7; 4 frames revisados')

    # Resizing a drawn box by a corner: the box is replaced, not added, and a box of 2 px is still refused.
    pedir('/frames/mover', {'f': 9, 'k': 0, 'caja': [100, 100, 150, 220]})
    assert pedir('/frames/9')['nuevas'] == [{'k': 0, 'caja': [100.0, 100.0, 150.0, 220.0],
                                             'etiqueta': 'persona', 'doble': False}], pedir('/frames/9')
    rechaza('/frames/mover', {'f': 9, 'k': 5, 'caja': [10, 10, 60, 90]}, 'mover una caja que no existe')
    rechaza('/frames/mover', {'f': 9, 'k': 0, 'caja': [10, 10, 12, 12]}, 'achicar a 2 px')
    print('  redimensionar        : la caja 0 del frame 9 pasa a 100,100-150,220; k=5 y 2 px dan 400')

    # Blind re-check: it asks about a fixed sample of the reviewed frames and must not reveal the count
    # before the answer. Frame 9 has one drawn box, so its true count is 1.
    e = pedir('/repaso/estado')
    assert len(e['frames']) == 1 and e['frames'][0]['dicho'] is None and e['frames'][0]['tenia'] is None, e
    f = e['pendiente']
    assert f in (2, 3, 4, 9), e
    tenia = pedir('/repaso', {'f': f, 'n': 99})['tenia']
    e = pedir('/repaso/estado')
    assert e['contestadas'] == 1 and e['acuerdo'] == 0 and e['frames'][0]['tenia'] == tenia, e
    pedir('/repaso', {'f': f, 'n': tenia})
    e = pedir('/repaso/estado')
    assert e['contestadas'] == 1 and e['acuerdo'] == 1 and e['pendiente'] is None, e
    # Two person boxes on the same person: X settles them, the smaller one becomes the duplicate. And C
    # copies a decision to the same box in the neighbouring frames, which is where it repeats.
    d1 = pedir('/frames/3')          # frame 3 is untouched: frame 1 was already cleaned up above
    per = [b for b in d1['cajas'] if b['etiqueta'] == 'persona']
    assert len(per) >= 2, per
    r = pedir('/frames/resolver', {'i': per[0]['i']})
    tras = {b['i']: b['etiqueta'] for b in pedir('/frames/3')['cajas']}
    # The synthetic boxes are all 40x80, so which one is "the smaller" is a tie: what has to hold is that
    # of the two the endpoint touched, one ended as the duplicate and the other stayed a person.
    assert r['duplicado'] != r['persona'], r
    assert tras[r['duplicado']] == 'duplicado' and tras[r['persona']] == 'persona', (r, tras)
    assert sum(v == 'persona' for v in tras.values()) == len(per) - 1, tras
    print('  resolver (X)         : de %d personas encimadas, la fila %d queda duplicado y la %d persona'
          % (len(per), r['duplicado'], r['persona']))
    for j, v in r['antes']:
        pedir('/frames/caja', {'i': j, 'v': v})

    # A resolves every pair of the frame at once. Frame 4 has five person boxes on top of each other, so
    # one call has to leave exactly one person standing, and Z has to put the five back.
    antes4 = {b['i']: b['etiqueta'] for b in pedir('/frames/4')['cajas']}
    r4 = pedir('/frames/resolver_todo', {'f': 4, 'propagar': False})
    tras4 = {b['i']: b['etiqueta'] for b in pedir('/frames/4')['cajas']}
    assert r4['pares'] == 4 and sum(v == 'persona' for v in tras4.values()) == 1, (r4, tras4)
    print('  resolver todo (A)    : %d pares resueltos en un frame -> queda 1 persona de %d'
          % (r4['pares'], sum(v == 'persona' for v in antes4.values())))
    for j, v in r4['antes']:
        pedir('/frames/caja', {'i': j, 'v': v})
    assert {b['i']: b['etiqueta'] for b in pedir('/frames/4')['cajas']} == antes4, 'Z tenia que devolver el frame'
    print('  deshacer el lote     : el frame 4 vuelve a sus %d personas' % sum(v == 'persona' for v in antes4.values()))

    i0 = per[0]['i']
    pedir('/frames/caja', {'i': i0, 'v': 'duplicado'})
    prop = pedir('/frames/propagar', {'i': i0})
    assert prop['etiqueta'] == 'duplicado' and prop['cambiadas'] >= 1, prop
    # propagar reaches every neighbouring frame of the list, so the check has to look at all of them, and
    # at the local rows the endpoint reports, not at the rows of the output file.
    iguales = {b['i']: b['etiqueta'] for f in (1, 2, 3, 4, 9) for b in pedir('/frames/%d' % f)['cajas']}
    assert all(iguales[j] == 'duplicado' for j in prop['locales']), (prop, iguales)
    print('  propagar (C)         : la misma decision se copio a %d cajas iguales de los frames vecinos' % prop['cambiadas'])
    for j, v in prop['antes']:
        pedir('/frames/caja', {'i': j, 'v': v})
    pedir('/frames/caja', {'i': i0, 'v': 'persona'})
    # Row 40 already carries a label by now, so to check the refusal its label is taken away first.
    pedir('/frames/caja', {'i': 40, 'v': None})
    rechaza('/frames/propagar', {'i': 40}, 'propagar una caja sin etiqueta')
    pedir('/frames/caja', {'i': 40, 'v': 'no'})

    # A box of the CSV can be resized too, which is how a detection that covers only the legs is fixed.
    # The label stays where it was; what moves is the box, and the checks see the new one.
    fila = pedir('/frames/1')['cajas'][0]
    pedir('/frames/ajustar', {'i': fila['i'], 'caja': [10, 20, 60, 140]})
    despues = [b for b in pedir('/frames/1')['cajas'] if b['i'] == fila['i']][0]
    assert despues['caja'] == [10.0, 20.0, 60.0, 140.0] and despues['ajustada'] and despues['etiqueta'] == fila['etiqueta'], despues
    guardado = json.load(open(os.path.join(dirt, 'etiquetas_frames.json'), encoding='utf-8'))['ajustes']
    assert list(guardado.values()) == [[10.0, 20.0, 60.0, 140.0]], guardado
    rechaza('/frames/ajustar', {'i': fila['i'], 'caja': [10, 20, 12, 22]}, 'ajustar a 2 px')
    pedir('/frames/ajustar', {'i': fila['i'], 'caja': None})
    vuelta = [b for b in pedir('/frames/1')['cajas'] if b['i'] == fila['i']][0]
    assert vuelta['caja'] == fila['caja'] and not vuelta['ajustada'], vuelta
    print('  ajustar una caja     : %s -> [10, 20, 60, 140] y con caja nula vuelve a la del CSV' % (fila['caja'],))

    # A drawn box is a person unless it is marked to be ignored, and then it stops counting as one.
    pedir('/frames/nueva_etiqueta', {'f': 9, 'k': 0, 'v': 'ignorar'})
    d9 = pedir('/frames/9')
    assert d9['nuevas'][0]['etiqueta'] == 'ignorar' and not d9['nuevas'][0]['doble'], d9
    guardado = json.load(open(os.path.join(dirt, 'etiquetas_frames.json'), encoding='utf-8'))['nuevas']['9'][0]
    assert guardado[4] == 'ignorar' and len(guardado) == 5, guardado
    rechaza('/frames/nueva_etiqueta', {'f': 9, 'k': 0, 'v': 'duplicado'}, 'una dibujada no puede ser duplicado')
    rechaza('/frames/nueva_etiqueta', {'f': 9, 'k': 3, 'v': 'ignorar'}, 'etiquetar una dibujada que no existe')
    print('  dibujada a ignorar   : queda "ignorar" en la pagina y en el archivo; duplicado y k=3 dan 400')

    # The suspicions panel: what a stronger detector found where no label of ours lies. One of the two
    # rows sits on the drawn box, so it must not be listed; the other is in an empty part of the frame.
    with open(os.path.join(dirt, 'olvidadas_vuelo_prueba.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frame', 'conf', 'x1', 'y1', 'x2', 'y2'])
        w.writerow([9, 0.91, 105, 105, 145, 215])
        w.writerow([9, 0.44, 300, 200, 340, 260])
    s = pedir('/sospechas/estado')
    assert s['hay_archivo'] and [x['caja'] for x in s['frames']] == [[300.0, 200.0, 340.0, 260.0]], s
    print('  sospechas del modelo : la que cae sobre una caja nuestra se filtra; queda %d de 2' % len(s['frames']))
    jpg = pedir('/sospecha/9/300,200,340,260', crudo=True)
    assert jpg[:2] == b'\xff\xd8'
    assert 'sospechas' in pedir('/sospechas', crudo=True).decode('utf-8')
    assert 'video' in pedir('/video', crudo=True).decode('utf-8')
    print('  paginas nuevas       : /sospechas con recorte JPEG y /video sirven')
    pedir('/frames/nueva_etiqueta', {'f': 9, 'k': 0, 'v': 'persona'})

    rechaza('/repaso', {'f': 1, 'n': 1}, 'repasar un frame sin revisar')
    rechaza('/repaso', {'f': f, 'n': -2}, 'un numero negativo de personas')
    print('  repaso ciego         : el frame %d no revela su cuenta antes de contestar; 99 -> acuerdo 0, %d -> acuerdo 1' % (f, tenia))
    assert 'saltar' in pedir('/frames', crudo=True).decode('utf-8'), 'la pagina tiene que traer el campo de salto'
    print('  saltar a un frame    : la pagina trae el campo')
    mini = pedir('/miniatura/9', crudo=True)
    im = cv2.imdecode(np.frombuffer(mini, np.uint8), 1)
    assert mini[:2] == b'\xff\xd8' and im.shape[1] == 480 and contar(mini, 'contexto') == 0
    verde = int(((im[..., 1] > 170) & (im[..., 0] < 130) & (im[..., 2] < 130)).sum())
    assert verde > 50, 'la caja dibujada del frame 9 tiene que verse verde en la miniatura: %d px' % verde
    # The checks page: frame 1 was cleaned up in the review, frames 3 and 4 still have five people on
    # top of each other. A gap is a frame with nobody between two frames with somebody: emptying frame 3
    # makes one, because frames 2 and 4 keep their people.
    ch = pedir('/chequeos/estado')
    assert 1 not in ch['dobles'] and 1 not in ch['medias'], ch
    assert 3 in ch['dobles'] and 3 in ch['medias'] and 4 in ch['dobles'], ch
    assert ch['huecos'] == [] and ch['sin_revisar'] == [1], ch
    print('  chequeos             : dobles %s, medias %s, huecos %s, sin revisar %s' % (ch['dobles'], ch['medias'], ch['huecos'], ch['sin_revisar']))
    for b in pedir('/frames/3')['cajas']:
        if b['etiqueta'] == 'persona':
            pedir('/frames/caja', {'i': b['i'], 'v': 'no'})
    ch = pedir('/chequeos/estado')
    assert ch['huecos'] == [3] and 3 not in ch['dobles'], ch
    print('  hueco                : vaciar el frame 3 entre el 2 y el 4 lo deja como hueco %s' % ch['huecos'])
    assert 'mosaico' in pedir('/mosaico', crudo=True).decode('utf-8')
    assert 'chequeos' in pedir('/chequeos', crudo=True).decode('utf-8')
    print('  miniatura            : JPEG 480 px de ancho con %d px verdes de la caja dibujada; /mosaico sirve' % verde)
    grupos_despues = json.load(open(os.path.join(dirt, 'etiquetas.json'), encoding='utf-8'))['etiquetas']
    assert grupos_despues == grupos_antes, 'la revision no puede reescribir las etiquetas de grupos'
    print('  archivos             : revision en etiquetas_frames.json; etiquetas.json de grupos intacto')
    assert pedir('/imagen/9', crudo=True) == open(os.path.join(dirt, 'frame_0009.jpg'), 'rb').read()
    pagina = pedir('/frames', crudo=True).decode('utf-8')
    assert 'duplicado' in pagina and 'ignorar' in pagina and 'lupa' in pagina
    print('  pagina               : /imagen es el archivo tal cual; /frames explica duplicado e ignorar')
finally:
    proc.terminate(); proc.wait(timeout=5)
    shutil.rmtree(dirt, ignore_errors=True)

print()
print('TODO OK')
