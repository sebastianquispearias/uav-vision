"""Exportar a YOLO: que lo que sale sea exactamente lo que se etiqueto, y el reparto sea por dia.

Un vuelo de juguete con verdad conocida: 4 frames, de los cuales solo 3 estan revisados. Contrastes que
fallan si el exportador miente: el frame sin revisar NO sale (un frame sin confirmar puede estar
incompleto, y un frame incompleto ensena que la gente que le falta es fondo); "duplicado" y "no" no
generan caja; "ignorar" tapa esos pixeles en vez de dejarlos como fondo; un frame sin personas sale
igual, con su .txt vacio, porque los negativos hacen falta; y una caja dibujada a mano vale como
cualquier otra. Las coordenadas se comprueban de vuelta: del .txt normalizado a pixeles tiene que salir
la caja original.

Run: python tests/test_exportar_yolo.py
"""
import csv, json, os, shutil, subprocess, sys, tempfile

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
HERRAMIENTA = os.path.join(AQUI, '..', 'scripts', 'exportar_yolo.py')
sys.path.insert(0, os.path.join(AQUI, '..', 'scripts'))
import exportar_yolo

ANCHO, ALTO = 400, 300
dirt = tempfile.mkdtemp(prefix='yolo_')
ent = os.path.join(dirt, 'entrenamiento')
frames = os.path.join(dirt, 'data', '20260726_195524', 'frames')
os.makedirs(ent)
os.makedirs(frames)
for f in (1, 2, 3, 4):
    cv2.imwrite(os.path.join(frames, 'frame_%04d.jpg' % f), np.full((ALTO, ANCHO, 3), 40, np.uint8))

# frame 1: una persona y un duplicado encima | frame 2: sin nadie (negativo) | frame 3: un "no" y un
# "ignorar" | frame 4: etiquetado pero SIN revisar, no tiene que salir.
CAJAS = [(1, 100, 50, 140, 150), (1, 104, 54, 136, 146), (3, 10, 10, 60, 90), (3, 200, 100, 260, 200),
         (4, 20, 20, 80, 120)]
with open(os.path.join(ent, 'candidatas_26jul.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['frame', 'conf', 'x1', 'y1', 'x2', 'y2', 'fuentes'])
    for f, x1, y1, x2, y2 in CAJAS:
        w.writerow([f, 0.9, x1, y1, x2, y2, 'vuelo'])
json.dump({'etiquetas': {'0': 'persona', '1': 'duplicado', '2': 'no', '3': 'ignorar', '4': 'persona'}},
          open(os.path.join(ent, 'etiquetas_detector_26jul.json'), 'w'))
json.dump({'revisados': [1, 2, 3], 'correcciones': {}, 'ajustes': {},
           'nuevas': {'1': [[300.0, 100.0, 340.0, 200.0]]}, 'repaso': {}, 'pares_ok': []},
          open(os.path.join(ent, 'etiquetas_detector_26jul_frames.json'), 'w'))

salida = os.path.join(dirt, 'dataset')
r = subprocess.run([sys.executable, HERRAMIENTA, '--entrenamiento', ent, '--datos', dirt,
                    '--salida', salida, '--solo', '26jul'], capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
print(r.stdout.strip().splitlines()[0])

imgs = sorted(os.listdir(os.path.join(salida, 'images', 'train')))
assert imgs == ['26jul_00001.jpg', '26jul_00002.jpg', '26jul_00003.jpg'], imgs
print('  solo los revisados   : salieron %s (el frame 4 estaba etiquetado pero sin revisar)' % [i[-9:-4] for i in imgs])

leer = lambda n: [l.split() for l in open(os.path.join(salida, 'labels', 'train', n)).read().splitlines() if l]
uno = leer('26jul_00001.txt')
assert len(uno) == 2, 'el duplicado no tenia que salir, y la caja dibujada si: %s' % uno
cajas_px = []
for c, cx, cy, bw, bh in [[float(v) for v in l] for l in uno]:
    assert c == 0
    cajas_px.append([round((cx - bw / 2) * ANCHO), round((cy - bh / 2) * ALTO),
                     round((cx + bw / 2) * ANCHO), round((cy + bh / 2) * ALTO)])
assert sorted(cajas_px) == [[100, 50, 140, 150], [300, 100, 340, 200]], cajas_px
print('  persona y dibujada   : %s (el duplicado encima no genero caja)' % sorted(cajas_px))

assert leer('26jul_00002.txt') == [], 'un frame sin personas sale con .txt vacio'
print('  frame sin nadie      : .txt vacio, que es el negativo que el detector necesita')

assert leer('26jul_00003.txt') == [], 'un "no" y un "ignorar" no son personas'
img3 = cv2.imread(os.path.join(salida, 'images', 'train', '26jul_00003.jpg'))
tapado = img3[150, 230]
fondo = img3[20, 20]
assert abs(int(tapado[0]) - 114) < 6 and abs(int(fondo[0]) - 40) < 6, (tapado, fondo)
print('  ignorar              : esos pixeles quedan tapados (%d) y el resto intacto (%d)' % (tapado[0], fondo[0]))

man = json.load(open(os.path.join(salida, 'manifiesto.json'), encoding='utf-8'))
assert man['vuelos']['26jul']['parte'] == 'train' and man['reparto']['train'] == ['26jul'], man
assert exportar_yolo.REPARTO['02ago'] == 'test' and exportar_yolo.REPARTO['01ago_2a'] == 'val', exportar_yolo.REPARTO
dias = {v: exportar_yolo.REPARTO[v] for v in exportar_yolo.REPARTO}
assert len({dias['01ago_2a'], dias['01ago_2b']}) == 1, 'los dos vuelos del mismo dia van al mismo lado: %s' % dias
print('  reparto por dia      : %s; los dos vuelos del 01ago caen juntos en val' % dias)

yaml = open(os.path.join(salida, 'personas.yaml')).read()
assert 'names:\n  0: persona' in yaml and 'train: images/train' in yaml, yaml
print('  personas.yaml        : una clase, 0 = persona')
shutil.rmtree(dirt, ignore_errors=True)

print()
print('TODO OK')
