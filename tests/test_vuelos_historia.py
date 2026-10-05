"""
Gate for the flight picker and for the trail every save leaves behind.

Two failures motivated both. Labels were judged against the wrong file for a whole afternoon because
comparing two flights meant restarting the tool, so nobody compared; and when a box turned out to be
wrong there was no way to ask when it changed, because the only thing on disk was the current answer.
The picker has to report each flight without loading it and open another without losing what the
current one saved, and every save has to leave a line and a recoverable copy.

The grouping clusters by cosine, which has no answer for a zero vector, so every synthetic box
is given a direction.

WHAT EACH SECTION PROVES
    1. Every save leaves a line whose counts follow what the review holds.
    2. A copy of the state is recoverable, and it is THE STATE and not a summary.
    3. The picker reports a flight WITHOUT LOADING IT, and the numbers are that flight's own.
    4. Opening a flight gives a working session, and refuses one it does not know.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import cv2
import etiquetar_grupos as E

base = tempfile.mkdtemp()
try:
    for k in range(4):
        cv2.imwrite(os.path.join(base, "frame_%04d.jpg" % (100 + k)), np.full((200, 320, 3), 180, np.uint8))
    cajas = os.path.join(base, "cajas.csv")
    with open(cajas, "w", newline="") as fh:
        fh.write("frame,conf,x1,y1,x2,y2,fuente\n100,0.9,10,10,60,120,vuelo\n101,0.8,12,10,62,120,vuelo\n")
    salida = os.path.join(base, "etiquetas.json")
    embs = np.zeros((2, 512), "float32")
    embs[0, 0], embs[1, 1] = 1.0, 1.0
    np.save(os.path.join(base, "embs.npy"), embs)
    s = E.Sesion(cajas, embs, base, salida, 1, lista_frames=[100, 101, 102, 103])

    s.corregir(0, "persona")
    s.nueva(102, [20, 20, 70, 130])
    hist = salida.replace(".json", "_frames_historia.jsonl")
    lineas = [json.loads(l) for l in open(hist, encoding="utf-8") if l.strip()]
    assert len(lineas) >= 2, "no quedo una linea por guardado: %d" % len(lineas)
    assert lineas[-1]["dibujadas"] == 1, "la historia no cuenta la caja dibujada"
    assert lineas[-1]["personas"] >= 2, "la historia no cuenta las personas: %s" % lineas[-1]
    assert lineas[-1]["t"] >= lineas[0]["t"], "las lineas no estan en orden de tiempo"
    print("  historia: %d guardados, el ultimo con %d personas y %d dibujadas"
          % (len(lineas), lineas[-1]["personas"], lineas[-1]["dibujadas"]))

    copias = salida.replace(".json", "_frames_copias")
    archivos = sorted(os.listdir(copias))
    assert archivos, "no se guardo ninguna copia"
    d = json.load(open(os.path.join(copias, archivos[-1]), encoding="utf-8"))
    assert d["nuevas"]["102"], "la copia no trae la caja dibujada"
    print("  copias: %d, la ultima trae la caja dibujada" % len(archivos))

    E.VUELOS["_prueba"] = (cajas, os.path.join(base, "embs.npy"), base, salida,
                           os.path.join(base, "lista.txt"))
    open(os.path.join(base, "lista.txt"), "w").write("100\n101\n102\n103\n")
    r = E.resumen_vuelo("_prueba")
    assert r["existe"] and r["cajas"] == 2, "el resumen no leyo las cajas: %s" % r
    assert r["dibujadas"] == 1 and r["personas"] >= 2, "el resumen no cuenta lo dibujado: %s" % r
    assert r["frames"] == 4, "el resumen no conto los frames: %s" % r
    print("  resumen sin cargar: %d cajas, %d personas, %d frames" % (r["cajas"], r["personas"], r["frames"]))

    otra = E.abrir_vuelo("_prueba")
    assert otra.nombre == "_prueba" and len(otra.lista) == 4, "la sesion abierta no es la del vuelo"
    assert otra.final(0) == "persona", "la sesion abierta perdio lo que ya estaba guardado"
    try:
        E.abrir_vuelo("no_existe")
        raise AssertionError("acepto un vuelo inventado")
    except ValueError:
        print("  abre el vuelo con lo ya guardado y rechaza uno inventado")
finally:
    E.VUELOS.pop("_prueba", None)
    shutil.rmtree(base, ignore_errors=True)

print("TODO OK")
