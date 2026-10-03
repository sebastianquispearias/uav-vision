"""A mission the operator acts on has to carry appearance, or the actions reach nothing.

test_fusion_drones.py already proves that the station uses appearance correctly, and it proves it
by building POIs with an 'emb' by hand. That is the whole blind spot: every test hands the station
what it wants, so a MISSION that ships without an appearance model is invisible to all of them.
That is not hypothetical. mision_banco_lab.py ran on both boards on 2026-10-02 with the model off,
and the bench looked healthy: contacts appeared, the fusion said '2+1', nothing errored.

Three things were silently dead, and the first section below measures the one that is worst,
because it does not look like a failure on the screen.

Run: python tests/test_misiones_operador.py
"""
import os
import re
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))

from uav_vision.flota import fundir


def poi(x, y, emb):
    d = {'x': x, 'y': y, 'n_obs': 50, 'mature': True, 'cls': 'person',
         'age_s': 1.0, 't': 100.0, 'radius_m': 4.0}
    if emb is not None:
        d['emb'] = emb.tolist()
    return d


print('=' * 70)
print('1. SIN APARIENCIA, LA ESTACION CRUZA LOS EMPAREJAMIENTOS ENTRE AERONAVES')
print('=' * 70)
# Ana y Beto, quietos, a 3 m uno del otro. El dron 1 los ve donde estan. El dron 2 tiene su
# propio sesgo de GPS y brujula, 2.6 m hacia el este, y los reporta en el orden en que los
# encontro, que no tiene por que ser el mismo. Nada de esto es rebuscado: medido sobre los
# candidatos del 02ago, el radio del 95 % de un objetivo quieto es 82 a 99.9 % SESGO, y el
# sesgo es por aeronave. Dos drones no comparten el error, cada uno tiene el suyo.
ana = np.zeros(512, dtype=np.float32); ana[0] = 1.0
beto = np.zeros(512, dtype=np.float32); beto[9] = 1.0

VERDAD = ((10.0, 10.0), (13.0, 10.0))


def escena(con_emb):
    ea, eb = (ana, beto) if con_emb else (None, None)
    return fundir({'1': [poi(10.0, 10.0, ea), poi(13.0, 10.0, eb)],
                   '2': [poi(12.0, 10.0, eb), poi(9.4, 10.0, ea)]})


def separacion(pines):
    xs = sorted(p['x'] for p in pines)
    return xs[-1] - xs[0]


sin_v, con_v = escena(False), escena(True)
d_sin, d_con = separacion(sin_v), separacion(con_v)
print('  verdad de terreno: dos personas a %.1f m' % (VERDAD[1][0] - VERDAD[0][0]))
print('  sin vector: %d pines, separados %.2f m   %s'
      % (len(sin_v), d_sin, [round(p['x'], 2) for p in sin_v]))
print('  con vector: %d pines, separados %.2f m   %s'
      % (len(con_v), d_con, [round(p['x'], 2) for p in con_v]))

# Los dos casos dan DOS pines, asi que contarlos no detecta nada: el fallo no es que falte un
# contacto, es que los dos quedan encima uno del otro. En el mapa del operador son dos personas
# a 20 cm, cuando estan a 3 m. Eso no se lee como un error, se lee como dos personas juntas.
assert len(sin_v) == len(con_v) == 2, 'el numero de pines no es lo que distingue los dos casos'
assert d_sin < 1.0, ('sin apariencia los dos pines tendrian que colapsar; dieron %.2f m. '
                     'Si esto deja de pasar, el emparejamiento ya no depende del orden '
                     'de llegada y esta seccion hay que rehacerla, no borrarla.' % d_sin)
assert d_con > 2.0, 'con apariencia los pines tienen que quedar separados, dieron %.2f m' % d_con
print('  -> sin vector el mapa pone a 20 cm a dos personas que estan a 3 m, y no parece un error')

print()
print('=' * 70)
print('2. LA MISION QUE EL OPERADOR USA TIENE QUE LLEVAR EL MODELO')
print('=' * 70)
# Esta es la seccion que habria atrapado el fallo del 2oct. Una mision con recortes es una mision
# pensada para que una persona mire y decida: el recorte no sirve para nada mas. Y las tres
# acciones que esa persona tiene dependen del vector:
#   gs_mapa.plantilla_para solo mira candidatos con 'emb', asi que sin el ni el "no es" ni el
#   clic de objetivo salen de la estacion, y flota.mismo_objetivo cae al camino debil.
MISIONES = os.path.join(AQUI, '..', 'scripts', 'banco_embedded')
revisadas = 0
for nombre in sorted(os.listdir(MISIONES)):
    if not nombre.startswith('mision_') or not nombre.endswith('.py'):
        continue
    texto = open(os.path.join(MISIONES, nombre), encoding='utf-8').read()
    # Solo el codigo: el docstring de una mision puede nombrar cualquiera de los dos para
    # explicar por que NO lo usa, y eso no es configuracion.
    codigo = re.sub(r'"""[\s\S]*?"""', '', texto)
    recortes = re.search(r'crops\s*=\s*True', codigo) is not None
    reid = re.search(r'reid_model\s*=\s*["\']', codigo) is not None
    estado = ('recortes SI, modelo ' + ('SI' if reid else 'NO')) if recortes else 'sin recortes'
    print('  %-34s %s' % (nombre, estado))
    revisadas += 1
    assert not (recortes and not reid), (
        '%s manda recortes para que un operador decida, pero no calcula apariencia. '
        'Entonces su "no es" no sale de la estacion, su clic no puede decir quien, y la '
        'fusion entre drones cae al camino debil. O enciende reid_model, o quita crops.'
        % nombre)
assert revisadas >= 3, 'solo se revisaron %d misiones, el patron de busqueda esta mal' % revisadas
print('  -> %d misiones revisadas' % revisadas)

print()
print('test_misiones_operador OK')
