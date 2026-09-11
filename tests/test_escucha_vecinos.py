"""Un dron oye lo que encontraron los demas y lo dice en su propio reporte.

El reporte sale como broadcast, asi que los hallazgos del vecino ya llegaban y se tiraban:
handle_packet era un `pass`. Guardarlos es lo que permite que este dron sepa, sin estacion de
tierra en medio, que un objetivo del que no esta seguro lo esta viendo alguien mas.

No actua sobre ello. Escucha, corrobora, y lo dice.

Run: python tests/test_escucha_vecinos.py
"""
import base64
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
# El protocolo importa gradys-embedded; el clon hermano sirve, igual que en test_vision_protocol.
_GRADYS = os.path.join(os.path.dirname(_HERE), "gradys-embedded")
if os.path.isdir(_GRADYS):
    sys.path.insert(0, _GRADYS)

from uav_vision.flota import mismo_objetivo


def emb(semilla, ruido=0.0):
    r = np.random.default_rng(semilla)
    v = r.normal(size=512).astype(np.float32)
    if ruido:
        v = v + ruido * np.random.default_rng(semilla + 500).normal(size=512).astype(np.float32)
    v /= np.linalg.norm(v)
    return base64.b64encode(v.astype(np.float16).tobytes()).decode()


class Reloj:
    def __init__(self): self.t = 0.0
    def current_time(self): return self.t
    def get_id(self): return 7


class Oyente:
    """Solo las dos piezas nuevas del protocolo, sin camara ni temporizadores."""

    def __init__(self, provider):
        self.provider = provider
        self._ajenos = {}
        self.ventana_ajenos_s = 15.0

    # copiadas tal cual del protocolo: si divergen, este test deja de valer
    handle_packet = None
    _corroboracion = None


from uav_vision.vision_protocol import VisionProtocol
Oyente.handle_packet = VisionProtocol.handle_packet
Oyente._corroboracion = VisionProtocol._corroboracion

reloj = Reloj()
d = Oyente(reloj)
mio = {'x': 0.0, 'y': 6.0, 'cls': 'person', 'emb': emb(1)}

# 1. Sin vecinos, nadie corrobora.
assert d._corroboracion(mio) == []
print('  sin vecinos -> nadie corrobora')

# 2. Un paquete que no es nuestro, o que es nuestro propio eco, no cuenta.
d.handle_packet(json.dumps({'type': 'otra_cosa', 'sender': 9, 'pois': [mio]}))
d.handle_packet(json.dumps({'type': 'vision_poi', 'sender': 7, 'pois': [mio]}))
d.handle_packet('esto no es json')
assert d._corroboracion(mio) == [], 'se colo un paquete ajeno al tipo, o el eco propio'
print('  otro tipo, el eco de uno mismo, y basura: se ignoran los tres')

# 3. El vecino 9 ve lo mismo.
d.handle_packet(json.dumps({'type': 'vision_poi', 'sender': 9,
                            'pois': [{'x': 0.4, 'y': 6.2, 'cls': 'person',
                                      'emb': emb(1, ruido=0.05)}]}))
assert d._corroboracion(mio) == [9], d._corroboracion(mio)
print('  el vecino 9 ve lo mismo -> corroborado por [9]')

# 4. Otro objetivo del mismo vecino no corrobora este.
otro = {'x': 20.0, 'y': 3.0, 'cls': 'person', 'emb': emb(1)}
assert d._corroboracion(otro) == [], 'corroboro algo que esta a 20 m'
print('  un objetivo distinto del mismo vecino -> no corrobora')

# 5. Lo que oyo hace rato dice donde estaba algo, no que siga ahi.
reloj.t = 30.0
assert d._corroboracion(mio) == [], 'un reporte viejo se promociono a confirmacion'
print('  pasados 30 s con la ventana en 15 -> el reporte viejo deja de contar')
print('test_escucha_vecinos OK')
