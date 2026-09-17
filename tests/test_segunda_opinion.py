"""
Gate for the second opinion: the aircraft hands over the frame it judged and the ground judges it again.

The point of the exchange is that the ground can run a detector the aircraft cannot afford. On the
02ago flight, where the drone is high, the aircraft's detector found 46.2 % of the people and the
ground's RF-DETR found 90.5 % of them with better precision, at 1.4 s per frame against 35 ms. So the
wiring has to hold two properties or the gain is fiction: the frame that travels must be the one the
detector actually looked at, not a fresh capture that would answer a different question and excuse a
miss; and it must travel only when asked, because a frame is 300 KB and three a second is seven
megabits sitting on top of the telemetry.

RF-DETR itself is not loaded here: it lives in the training venv and this suite runs on the plain
interpreter. What is pinned is the plumbing and the merge, which is where the mistakes would be silent.
"""
import base64
import json
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts", "banco_embedded"))
import cv2
import segunda_opinion as SO
from uav_vision.camera import OnboardCamera

# 1. the camera hands over the frame it JUDGED, not a new one
c = OnboardCamera.__new__(OnboardCamera)
c._ultimo_frame = None
assert c.ultimo_marco_jpeg() is None, "devolvio algo antes de haber mirado un solo frame"
marca = np.zeros((120, 160, 3), np.uint8)
cv2.rectangle(marca, (20, 20), (60, 90), (255, 255, 255), -1)
c._ultimo_frame = marca
datos = c.ultimo_marco_jpeg()
assert datos and datos[:2] == b"\xff\xd8", "lo que salio no es un jpeg"
vuelta = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
assert vuelta.shape == marca.shape, "el marco cambio de tamano al viajar"
assert vuelta[50, 40].mean() > 200, "el marco que viajo no es el que la camara miro"
print("  la camara entrega el frame que miro (%d bytes) y nada antes de mirar" % len(datos))

# 2. the packet carries it whole, and the ground reads it back
mensaje = {"type": "vision_marco", "sender": 1, "t": 12.5,
           "jpeg": base64.b64encode(datos).decode("ascii")}
assert base64.b64decode(mensaje["jpeg"]) == datos, "el base64 no devuelve los mismos bytes"
malo = SO.mirar_mensaje({"type": "vision_poi"})
assert malo["n"] == 0 and "error" in malo, "acepto un paquete que no trae marco"
roto = SO.mirar_jpeg(b"esto no es un jpeg")
assert roto["n"] == 0 and "error" in roto, "no aviso que el jpeg venia roto"
print("  el paquete lleva el marco entero y un paquete equivocado se rechaza con motivo")

# 3. the merge: the same person seen by two tiles comes back once, two people stay two
una = SO.fusionar([[10, 10, 50, 110], [12, 11, 52, 112]], [0.9, 0.8])
assert len(una) == 1, "la misma persona en dos fichas quedo como dos: %s" % una
assert una[0]["conf"] == 0.9, "no se quedo con la caja mas segura"
dos = SO.fusionar([[10, 10, 50, 110], [400, 300, 440, 400]], [0.9, 0.8])
assert len(dos) == 2, "fusiono dos personas que estan lejos"
assert not SO.fusionar([], []), "invento algo sin cajas"
print("  la fusion deja una caja por persona y no junta a dos que estan separadas")

# 4. the tiles cover the frame: a person in any corner falls inside at least one
alto, ancho = 1080, 1920
for x, y in ((5, 5), (1900, 5), (5, 1070), (1900, 1070), (960, 540)):
    dentro = [f for f in SO.FICHAS if f[0] <= x <= f[2] and f[1] <= y <= f[3]]
    assert dentro, "el punto (%d, %d) no cae en ninguna ficha" % (x, y)
assert any(f == (0, 0, ancho, alto) for f in SO.FICHAS), "falta la ficha del frame entero"
print("  las %d fichas cubren las cuatro esquinas y el centro, y una es el frame entero" % len(SO.FICHAS))

print("TODO OK")
