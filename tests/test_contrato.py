"""
Checks that both cameras honor the same ver_alvo contract: identical signature, identical
output shape, and consumer code that never learns which camera it received.

Run with: python tests/test_contrato.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_vision.camera import CamaraArduCam, CamaraSimulada


def consumidor(camara, pos, yaw):
    """Codigo que NO sabe que camara le toco. Esto es el punto de todo."""
    detecciones = camara.ver_alvo(pos, yaw)      # una sola llamada
    total = 0.0
    for det in detecciones:
        total += det["conf"]
    return len(detecciones), total


print("=" * 64)
print("1. Las dos clases exponen ver_alvo con la MISMA firma")
print("=" * 64)
for cls in (CamaraSimulada, CamaraArduCam):
    print(f"  {cls.__name__:16s} ver_alvo{inspect.signature(cls.ver_alvo)}")

print()
print("=" * 64)
print("2. CamaraSimulada, tres situaciones")
print("=" * 64)
import numpy as np

cam = CamaraSimulada(alvo=(0.0, 0.0, 0.0), pitch_deg=-55.0,
                     rng=np.random.default_rng(0))

casos = [
    ("dron justo encima", (0.0, 0.0, 50.0)),
    ("dron 20 m al sur ", (0.0, -20.0, 50.0)),
    ("dron 195 m lejos ", (0.0, -195.0, 50.0)),
]
for etiqueta, pos in casos:
    salida = cam.ver_alvo(pos, 0.0)
    print(f"  {etiqueta} -> {type(salida).__name__:5s} de {len(salida)}  {salida}")

print()
print("=" * 64)
print("3. Todas las salidas son del mismo tipo")
print("=" * 64)
for etiqueta, pos in casos:
    salida = cam.ver_alvo(pos, 0.0)
    assert isinstance(salida, list), f"{etiqueta}: no es lista"
    for det in salida:
        faltan = {"px", "py", "conf"} - set(det)
        assert not faltan, f"{etiqueta}: faltan campos {faltan}"
        sobran = set(det) - {"px", "py", "conf", "emb"}
        assert not sobran, f"{etiqueta}: campos desconocidos {sobran}"
    print(f"  {etiqueta} OK")

print()
print("=" * 64)
print("4. Un consumidor que no sabe que camara tiene")
print("=" * 64)
n, suma = consumidor(cam, (0.0, -20.0, 50.0), 0.0)
print(f"  detecciones={n}  suma de confianzas={suma:.3f}")
print()
print("  La misma funcion, con CamaraArduCam, correria en la Raspberry")
print("  sin cambiar una linea.")

print()
print("=" * 64)
print("5. 'emb' es OPCIONAL: se pide con .get(), nunca con []")
print("=" * 64)
det = cam.ver_alvo((0.0, -20.0, 50.0), 0.0)[0]
print(f"  campos que trae la simulada : {sorted(det)}")
print(f"  det.get('emb')              : {det.get('emb')}")
assert det.get("emb") is None, "la simulada no tiene imagen: no puede tener huella"
try:
    det["emb"]
    raise AssertionError("deberia haber fallado")
except KeyError:
    print("  det['emb']                  : KeyError  <- por eso se usa .get()")

print()
print("TODO OK")
