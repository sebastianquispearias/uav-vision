"""Says whether a capture card is really delivering an image, and not merely opening.

A camera that is plugged in but has nothing behind it opens, reports a resolution and hands
back frames: black ones, or the same frozen one forever. `cap.isOpened()` is true in every one
of those cases, which is why "it opened" is not an answer to "is it working".

So this measures three things per frame instead:

    brillo      the mean pixel value. A capture card with no HDMI signal returns a frame of
                zeros, and a lens cap returns something close to it.
    nitidez     the variance of the Laplacian, which is the standard focus measure. A flat
                grey test pattern and a real scene differ here by orders of magnitude.
    cambio      how much the frame moved since the last one. A frozen buffer repeats the same
                bytes, and the difference is exactly zero -- which a single snapshot cannot
                tell apart from a very still scene.

It walks the capture indices rather than trusting one, because the index of a USB capture card
moves with what else is plugged in: the laptop's own webcam is usually 0 and the card lands on
1, but not reliably.

    python scripts/medir/ver_siyi.py                 looks at every index it finds
    python scripts/medir/ver_siyi.py --indice 1      only that one
    python scripts/medir/ver_siyi.py --guardar out   writes one JPEG per working index

On Windows the backend is DirectShow explicitly: the default backend reports a resolution it
has not actually applied, and a card asked for 1080p quietly delivers 640x480.
"""

import argparse
import os
import sys

import cv2
import numpy as np

ANCHO, ALTO, FPS = 1920, 1080, 30
CUADROS = 12          # enough for the capture chain to settle and for motion to show
NEGRO = 8.0           # mean below this is, for practical purposes, no signal
PLANO = 15.0          # Laplacian variance below this is a flat field, not a scene


def mirar(indice, guardar=None):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cuadros, previo, cambios = [], None, []
    for _ in range(CUADROS):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        gris = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cuadros.append((f, gris))
        if previo is not None:
            cambios.append(float(np.mean(cv2.absdiff(gris, previo))))
        previo = gris
    cap.release()

    if not cuadros:
        return {"indice": indice, "abre": True, "leidos": 0, "w": w, "h": h, "fps": fps}

    marco, gris = cuadros[-1]
    info = {
        "indice": indice, "abre": True, "leidos": len(cuadros),
        "w": marco.shape[1], "h": marco.shape[0], "fps": fps,
        "pedido": "%dx%d" % (w, h),
        "brillo": float(np.mean(gris)),
        "nitidez": float(cv2.Laplacian(gris, cv2.CV_64F).var()),
        "cambio": float(np.mean(cambios)) if cambios else 0.0,
    }
    if guardar:
        os.makedirs(guardar, exist_ok=True)
        info["jpg"] = os.path.join(guardar, "captura_%d.jpg" % indice)
        cv2.imwrite(info["jpg"], marco)
    return info


def veredicto(d):
    if d["leidos"] == 0:
        return "ABRE PERO NO ENTREGA NI UN CUADRO"
    if d["brillo"] < NEGRO:
        return "CUADRO NEGRO: abre, entrega, y no hay senal detras"
    if d["cambio"] == 0.0:
        return "CONGELADO: el mismo cuadro byte por byte, el buffer no avanza"
    if d["nitidez"] < PLANO:
        return "CAMPO PLANO: hay senal pero no una escena (tapa puesta, o barra de color)"
    return "IMAGEN DE VERDAD"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indice", type=int, default=None)
    ap.add_argument("--hasta", type=int, default=4, help="ultimo indice a probar")
    ap.add_argument("--guardar", default=None, help="carpeta donde dejar un JPEG por indice")
    args = ap.parse_args()

    indices = [args.indice] if args.indice is not None else range(args.hasta + 1)
    encontrados = 0
    for i in indices:
        d = mirar(i, args.guardar)
        if d is None:
            print("indice %d: no abre" % i)
            continue
        encontrados += 1
        print("")
        print("indice %d" % i)
        print("  entrega      : %d de %d cuadros, %dx%d a %.0f fps (pedidos %s)"
              % (d["leidos"], CUADROS, d["w"], d["h"], d["fps"], d.get("pedido", "?")))
        if d["leidos"]:
            print("  brillo medio : %.1f    (menos de %.0f es sin senal)" % (d["brillo"], NEGRO))
            print("  nitidez      : %.1f    (menos de %.0f es un campo plano)" % (d["nitidez"], PLANO))
            print("  cambio entre cuadros: %.3f  (0.000 es un buffer congelado)" % d["cambio"])
            if "jpg" in d:
                print("  guardado     : %s" % d["jpg"])
        print("  VEREDICTO    : %s" % veredicto(d))

    print("")
    if not encontrados:
        print("NINGUN dispositivo de captura. Si la placa UGREEN esta enchufada, Windows no la ve.")
        return 1
    print("%d dispositivo(s) encontrado(s). El de la SIYI es el que NO es la webcam de la tapa:" % encontrados)
    print("una captura HDMI entrega 1920x1080; la webcam integrada no suele pasar de 1280x720.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
