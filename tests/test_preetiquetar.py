"""Pre-etiquetar: decide solo lo que los proponentes ya contestan, y no toca lo dudoso.

Contrastes que fallan si la regla miente: tres detectores de acuerdo con confianza alta es persona; uno
solo, o dos flojos, NO se deciden y quedan para la persona; lo que solo vio COCO o solo el vuelo se
descarta; y el archivo de salida es el mismo que lee la herramienta, con las filas como clave, para que
todo se pueda corregir despues a mano. La regla se midio contra 985 cajas etiquetadas por el usuario:
decide el 41 % con 1.2 % de error.

Run: python tests/test_preetiquetar.py

The three labels the tool can produce are persona, dudosa and no.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
HERRAMIENTA = os.path.join(AQUI, '..', 'scripts', 'preetiquetar.py')
sys.path.insert(0, os.path.join(AQUI, '..', 'scripts'))
from preetiquetar import decidir

CASOS = [
    (("coco+rfdetr+vuelo", 0.91), "persona", "los tres de acuerdo y confianza alta"),
    (("coco+rfdetr", 0.75), "persona", "coco y rfdetr de acuerdo, confianza alta"),
    (("coco+rfdetr+vuelo", 0.55), None, "los tres pero confianza media: la decide el usuario"),
    (("rfdetr", 0.95), None, "rfdetr solo, por alta que sea la confianza"),
    (("rfdetr+vuelo", 0.80), None, "sin coco no alcanza"),
    (("vuelo", 0.60), "no", "solo el detector del vuelo"),
    (("coco", 0.80), "no", "solo coco, a cualquier confianza"),
    (("coco+vuelo", 0.40), "no", "coco y vuelo sin rfdetr"),
    (("rfdetr", 0.20), None, "rfdetr flojo: dudosa"),
]
for (fuentes, conf), esperado, porque in CASOS:
    got = decidir(fuentes, conf)
    assert got == esperado, "%s conf %.2f -> %s, esperaba %s (%s)" % (fuentes, conf, got, esperado, porque)
print('  la regla                 : %d casos, incluidos los que NO decide' % len(CASOS))
assert sum(1 for c in CASOS if c[1] is None) >= 4, 'tiene que dejar varios sin decidir'
print('  deja lo dudoso           : %d de %d casos quedan para la persona' % (
    sum(1 for c in CASOS if c[1] is None), len(CASOS)))

dirt = tempfile.mkdtemp(prefix='preet_')
try:
    with open(os.path.join(dirt, 'candidatas_prueba.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frame','conf','x1','y1','x2','y2','fuentes'])
        w.writerow([1, 0.90, 10, 10, 50, 90, 'coco+rfdetr+vuelo'])
        w.writerow([1, 0.50, 60, 10, 90, 90, 'rfdetr'])
        w.writerow([2, 0.60, 10, 10, 40, 60, 'vuelo'])
    r = subprocess.run([sys.executable, HERRAMIENTA, '--vuelo', 'prueba', '--entrenamiento', dirt],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    print('  %s' % r.stdout.strip().splitlines()[0])
    et = json.load(open(os.path.join(dirt, 'etiquetas_detector_prueba.json'), encoding='utf-8'))
    assert et['etiquetas'] == {'0': 'persona', '2': 'no'}, et['etiquetas']
    assert '1' not in et['etiquetas'], 'la dudosa no se puede decidir sola'
    assert et['clave'] == 'fila del CSV de cajas'
    print('  archivo para la herramienta: filas 0 y 2 decididas, la 1 sin etiqueta')
    r2 = subprocess.run([sys.executable, HERRAMIENTA, '--vuelo', 'prueba', '--entrenamiento', dirt],
                        capture_output=True, text=True)
    assert r2.returncode != 0 and 'ya existe' in r2.stdout + r2.stderr, r2.stdout + r2.stderr
    print('  no pisa lo ya etiquetado : el segundo intento se niega')
finally:
    shutil.rmtree(dirt, ignore_errors=True)

print()
print('TODO OK')
