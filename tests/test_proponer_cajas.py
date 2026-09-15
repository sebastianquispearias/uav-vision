"""Proponer cajas: la fusion entre detectores y el muestreo de frames, sin GPU.

Contrastes que fallan si la fusion miente: dos cajas de fuentes distintas sobre la misma persona
(IoU 0.8) tienen que salir como UNA, con la confianza mas alta y las dos fuentes; una caja corrida
sobre la misma persona (IoU 0.33) tiene que seguir siendo otra caja, porque decidir si es una persona
nueva o un duplicado es trabajo del etiquetado, no de la fusion; y la misma fuente dos veces no se
lista dos veces. El muestreo tiene que respetar el paso y el rango, y saltar numeros que no existen.

Run: python tests/test_proponer_cajas.py
"""
import os, shutil, sys, tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'scripts'))
from proponer_cajas import fusionar, frames_de, iou
import numpy as np

a = (0.40, 100, 100, 140, 200, 'vuelo')
b = (0.90, 102, 104, 142, 204, 'rfdetr')    # the same person, another detector
c = (0.70, 120, 100, 160, 200, 'coco')      # shifted half a body: a separate candidate
d = (0.30, 101, 101, 141, 201, 'rfdetr')    # rfdetr again on the same person, from a tile
print('  IoU a-b %.2f, a-c %.2f' % (iou(np.array(a[1:5]), np.array([b[1:5]]))[0], iou(np.array(a[1:5]), np.array([c[1:5]]))[0]))

out = fusionar([a, b, c, d])
assert len(out) == 2, 'a, b y d son la misma persona y c otra caja: %s' % (out,)
misma = [o for o in out if 'vuelo' in o[5]]
assert len(misma) == 1 and misma[0][0] == 0.90 and misma[0][5] == 'rfdetr+vuelo', misma
print('  misma persona        : 3 cajas -> 1, conf %.2f, fuentes %s' % (misma[0][0], misma[0][5]))
corrida = [o for o in out if o is not misma[0]][0]
assert corrida[5] == 'coco' and corrida[1] == 120, corrida
print('  caja corrida (IoU .33): sigue aparte, fuente %s' % corrida[5])
assert fusionar([]) == []
print('  sin cajas            : []')

dirt = tempfile.mkdtemp(prefix='prop_')
try:
    for n in list(range(1, 31)) + [45]:
        open(os.path.join(dirt, 'frame_%04d.jpg' % n), 'wb').close()
    # The step counts existing frames, not frame numbers: 1..30 and 45 are 31 files, every 10th is 1, 11, 21, 45.
    assert frames_de(dirt, 10) == [1, 11, 21, 45], frames_de(dirt, 10)
    assert frames_de(dirt, 5, desde=12, hasta=45) == [12, 17, 22, 27], frames_de(dirt, 5, desde=12, hasta=45)
    assert frames_de(dirt, 1, desde=29) == [29, 30, 45]
    print('  muestreo             : paso 10 -> [1, 11, 21, 45]; paso 5 en 12-45 -> [12, 17, 22, 27]; salta huecos')
    # Heights: frames 1-10 on the ground, 11-30 flying, 45 flying. The step must be taken over the flying
    # frames, not over all of them and then filtered, or a ground stretch would thin the sample.
    alt = {n: (0.2 if n <= 10 else 6.0) for n in list(range(1, 31)) + [45]}
    assert frames_de(dirt, 10, alturas=alt, alt_min=3) == [11, 21, 45], frames_de(dirt, 10, alturas=alt, alt_min=3)
    assert frames_de(dirt, 10) != frames_de(dirt, 10, alturas=alt, alt_min=3)
    print('  altura minima 3 m    : paso 10 sobre los que vuelan -> [11, 21, 45] (sin filtro [1, 11, 21, 45])')
finally:
    shutil.rmtree(dirt, ignore_errors=True)

print()
print('TODO OK')
