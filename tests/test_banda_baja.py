"""The BYTE band: weak boxes continue tracks, they do not create targets.

Opening the detector below the reporting threshold is free -- it already scored those boxes --
but the band is only useful as evidence about something already being followed. A box nobody
was following is a guess, and the fusion would use it: unlike the identity layer, it does not
check 'track_id'. So the camera filters the band back out before returning.

Run: python tests/test_banda_baja.py

THE CASE THAT REALLY MATTERS: without the band that frame contributes nothing, and with it the
frame contributes one more observation of somebody the system was already following. And it
does not leak in the other way round: with no track_id, no quantity of weak boxes adds up.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from uav_vision.camera import solo_confirmadas

U = 0.3

fuerte = {"conf": 0.62}
fuerte_seguida = {"conf": 0.55, "track_id": 4}
floja_seguida = {"conf": 0.24, "track_id": 4}
floja_suelta = {"conf": 0.22}
justo = {"conf": 0.30}

salida = solo_confirmadas([fuerte, fuerte_seguida, floja_seguida, floja_suelta, justo], U)

assert fuerte in salida, "una deteccion por encima del umbral no depende del tracker"
assert fuerte_seguida in salida
assert floja_seguida in salida, "la banda baja existe justo para esto: continuar una pista"
assert floja_suelta not in salida, "una caja floja sin pista NO puede llegar a la fusion"
assert justo in salida, "el umbral es inclusivo"
print('  pasan: fuerte, fuerte+pista, floja+pista, justo en el umbral')
print('  se cae: floja sin pista')

solo_floja = solo_confirmadas([floja_seguida], U)
assert len(solo_floja) == 1
assert solo_confirmadas([floja_suelta], U) == []
print('  un frame con solo una caja floja: aporta si hay pista, nada si no la hay')

assert solo_confirmadas([{"conf": 0.29} for _ in range(50)], U) == []
print('  cincuenta cajas flojas sueltas siguen sin ser un objetivo')
print('test_banda_baja OK')
