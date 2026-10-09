"""Says whether an RTSP camera is really delivering video, and not merely accepting the socket.

An RTSP server that is up but has no encoder running accepts the TCP connection, completes the
handshake, and then sends nothing. OpenCV's `isOpened()` is true for that, and so is a port
scan, which is why neither answers the question "is there a picture".

So this measures the frames instead, the same three ways `ver_siyi.py` does for a capture card:

    brillo      the mean pixel value. A stream with the lens capped, or a black test pattern,
                is close to zero.
    nitidez     the variance of the Laplacian. A flat field and a real scene differ by orders
                of magnitude.
    cambio      how much the frame moved since the last one. A decoder stuck on one keyframe
                repeats the same bytes, which a single snapshot cannot tell from a still scene.

TCP and not UDP for the transport, because a dropped UDP packet in an H.264 stream shows up as
a smeared frame rather than as an error, and that would be measured here as a real picture.

    python scripts/medir/ver_rtsp.py
    python scripts/medir/ver_rtsp.py --url rtsp://192.168.144.25:8554/main.264 --guardar docs/
"""

import argparse
import os
import sys
import time

# Must be set before cv2 is imported: the FFmpeg backend reads it at load time.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

URL = "rtsp://192.168.144.25:8554/main.264"
CUADROS = 12
NEGRO = 8.0
PLANO = 15.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--guardar", default=None, help="carpeta donde dejar un JPEG")
    ap.add_argument("--espera", type=float, default=15.0, help="segundos para abrir el stream")
    args = ap.parse_args()

    print("abriendo %s" % args.url)
    t0 = time.time()
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    print("  abrio en %.1f s: %s" % (time.time() - t0, cap.isOpened()))
    if not cap.isOpened():
        print("  NO ABRE. Revisar que el puerto 8554 este accesible desde esta maquina y que")
        print("  esta maquina tenga una direccion en la red de la camara.")
        return 1

    cuadros, previo, cambios = [], None, []
    t0 = time.time()
    while len(cuadros) < CUADROS and time.time() - t0 < args.espera:
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
        print("  ABRE PERO NO ENTREGA NI UN CUADRO en %.0f s" % args.espera)
        return 1

    marco, gris = cuadros[-1]
    brillo = float(np.mean(gris))
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    cambio = float(np.mean(cambios)) if cambios else 0.0

    print("  entrega      : %d cuadros, %dx%d" % (len(cuadros), marco.shape[1], marco.shape[0]))
    print("  brillo medio : %.1f    (menos de %.0f es sin imagen)" % (brillo, NEGRO))
    print("  nitidez      : %.1f    (menos de %.0f es un campo plano)" % (nitidez, PLANO))
    print("  cambio entre cuadros: %.3f  (0.000 es un decodificador trabado)" % cambio)

    if args.guardar:
        os.makedirs(args.guardar, exist_ok=True)
        ruta = os.path.join(args.guardar, "siyi_rtsp.jpg")
        cv2.imwrite(ruta, marco)
        print("  guardado     : %s" % ruta)

    if brillo < NEGRO:
        print("  VEREDICTO    : CUADRO NEGRO: el stream corre y no hay imagen detras")
    elif cambio == 0.0:
        print("  VEREDICTO    : CONGELADO: el mismo cuadro, el decodificador no avanza")
    elif nitidez < PLANO:
        print("  VEREDICTO    : CAMPO PLANO: hay senal pero no una escena (tapa, o barra de color)")
    else:
        print("  VEREDICTO    : IMAGEN DE VERDAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
