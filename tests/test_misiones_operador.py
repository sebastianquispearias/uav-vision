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
# propio sesgo de GPS y brujula, 2 m hacia el este, y los ve a los dos corridos.
#
# Con ese sesgo, la Ana del dron 2 (en 12.0) queda a 1 m del Beto del dron 1 (en 13.0) y a 2 m
# de su propia Ana (en 10.0). El pin EQUIVOCADO esta mas cerca. Elegir por distancia, aunque se
# elija el mejor y no el primero, se equivoca: hace falta mirar la apariencia.
#
# El sesgo de 2 m no es rebuscado. Medido sobre los candidatos del 02ago, entre el 82 % y el
# 99.9 % del radio del 95 % de un objetivo quieto es SESGO del gps y la brujula de esa aeronave,
# no dispersion. El sesgo no se promedia a lo largo del vuelo y no se comparte entre aviones.
ana = np.zeros(512, dtype=np.float32); ana[0] = 1.0
beto = np.zeros(512, dtype=np.float32); beto[9] = 1.0

SEPARACION_REAL = 3.0
SESGO = 2.0


def escena(con_emb):
    ea, eb = (ana, beto) if con_emb else (None, None)
    return fundir({'1': [poi(10.0, 10.0, ea), poi(13.0, 10.0, eb)],
                   '2': [poi(10.0 + SESGO, 10.0, ea), poi(13.0 + SESGO, 10.0, eb)]})


def separacion(pines):
    xs = sorted(p['x'] for p in pines)
    return xs[-1] - xs[0]


sin_v, con_v = escena(False), escena(True)
d_sin, d_con = separacion(sin_v), separacion(con_v)
print('  verdad de terreno: dos personas a %.1f m, y %.1f m de sesgo entre aeronaves'
      % (SEPARACION_REAL, SESGO))
print('  sin vector: %d pines, separados %.2f m   %s'
      % (len(sin_v), d_sin, [round(p['x'], 2) for p in sin_v]))
print('  con vector: %d pines, separados %.2f m   %s'
      % (len(con_v), d_con, [round(p['x'], 2) for p in con_v]))

# Contar los pines no detecta nada: los dos casos dan DOS. El fallo no es que falte un contacto,
# es que los dos quedan en el MISMO sitio. En el mapa del operador eso no se lee como un error.
assert len(sin_v) == len(con_v) == 2, 'el numero de pines no es lo que distingue los dos casos'
assert d_sin < 0.5, ('sin apariencia los dos pines tendrian que colapsar; dieron %.2f m. '
                     'Si esto deja de pasar, la geometria sola ya resuelve este caso y hay que '
                     'buscar el sesgo al que vuelve a fallar, no borrar la seccion.' % d_sin)
assert abs(d_con - SEPARACION_REAL) < 0.5, (
    'con apariencia los pines tienen que reproducir la separacion real de %.1f m, dieron %.2f m'
    % (SEPARACION_REAL, d_con))
print('  -> sin vector los dos contactos caen encima; con vector reproducen los 3 m reales')

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
    # Dos formas valen: la ruta escrita ahi mismo, y un nombre que la mision resuelve (hoy
    # mision_banco_lab lo lee del entorno, porque el modelo cuesta unos 30 ms por caja en la
    # Pi 5 y entre 170 y 330 en la Pi 4, asi que la decision es por placa). Lo que NO vale es
    # reid_model=None. La comprobacion es textual porque importar una mision pide el runtime
    # de GrADyS y una camara; se le escapa una mision que nombre un modelo y lo resuelva a None
    # en ejecucion, y para eso esta el arranque, que falla ruidosamente sin el archivo.
    m_reid = re.search(r'reid_model\s*=\s*([^,\n]+)', codigo)
    asignado = m_reid is not None and m_reid.group(1).strip() not in ('None', '')
    hay_ruta = re.search(r'osnet\w*\.pt', codigo) is not None
    reid = asignado and hay_ruta
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
print('=' * 70)
print('3. LA ESTACION SE FUNDE CON EL PIN MAS CERCANO, NO CON EL PRIMERO')
print('=' * 70)
# Sin vectores, para que decida solo la geometria. El dron 1 reporta dos objetivos y crea dos
# pines, en ese orden. El dron 2 reporta uno que cae a 2.4 m del primer pin y a 0.6 m del
# segundo. La version voraz, que se quedaba con el primero que encajara, lo metia en el pin de
# la izquierda; con dos aeronaves eso casi nunca se nota, porque el unico pin al alcance suele
# ser el correcto. Con tres empieza a notarse, y lo que decide pasa a ser el orden de llegada
# de los reportes, que no significa nada.
pines = fundir({'1': [poi(10.0, 10.0, None), poi(13.0, 10.0, None)],
                '2': [poi(12.4, 10.0, None)]})
fundidos = [p for p in pines if len(p['drones']) > 1]
print('  pines del dron 1 en x=10.0 y x=13.0; el dron 2 reporta x=12.4')
print('  resultado: %s' % [(round(p['x'], 2), p.get('dron', '+'.join(p['drones']))) for p in pines])
assert len(fundidos) == 1, 'el reporte del dron 2 tenia que fundirse con alguno: %r' % pines
x = fundidos[0]['x']
# Primero: (10.0 + 12.4) / 2 = 11.2.   Mas cercano: (13.0 + 12.4) / 2 = 12.7.
assert abs(x - 12.7) < 0.05, (
    'se fundio en x=%.2f. 11.2 significa que eligio el PRIMER pin que encajaba en vez del mas '
    'cercano, que es la asignacion voraz que se quito el 3oct.' % x)
print('  -> se fundio en x=%.2f, que es el pin de 13.0 y no el de 10.0' % x)

print()
print('test_misiones_operador OK')
