"""A mission the operator acts on has to carry appearance, or the actions reach nothing.

test_fusion_drones.py already proves that the station uses appearance correctly, and it proves it
by building POIs with an 'emb' by hand. That is the whole blind spot: every test hands the station
what it wants, so a MISSION that ships without an appearance model is invisible to all of them.
That is not hypothetical. mision_banco_lab.py ran on both boards on 2026-10-02 with the model off,
and the bench looked healthy: contacts appeared, the fusion said '2+1', nothing errored.

Three things were silently dead, and the first section below measures the one that is worst,
because it does not look like a failure on the screen.

Run: python tests/test_misiones_operador.py

WHAT EACH SECTION PROVES
    TWO PEOPLE, STANDING STILL, A FEW METRES APART, seen by two drones. The second drone has its
    own GPS and compass bias, so it sees both of them shifted. With that bias, its reading of
    the first person lands CLOSER to the first drone's reading of the second person than to its
    own. The WRONG pin is nearer. Choosing by distance, even by the best match and not the
    first, gets it wrong: appearance has to decide. The bias is not contrived; NOTES.md has what
    share of a static target's 95 % radius is bias rather than scatter, and bias neither
    averages out over a flight nor is shared between aircraft.

    COUNTING THE PINS DETECTS NOTHING, because both cases give two. The failure is not a missing
    contact, it is that both end up in the SAME PLACE, and on the operator's map that does not
    read as an error.

    A MISSION WITH CROPS IS A MISSION MEANT FOR A PERSON TO LOOK AT AND DECIDE, and the crop is
    good for nothing else. All three actions that person has depend on the appearance vector:
    gs_mapa.plantilla_para only looks at candidates carrying one, so without it neither the
    "not it" nor the target click leaves the station, and flota.mismo_objetivo falls back to the
    weak path.

    The check is on THE CODE ONLY: a mission's docstring may name either model to explain why it
    does NOT use it, and that is not configuration. Two forms are accepted, the path written
    there and a name the mission resolves, because the appearance model costs several times more
    on one board than the other and the decision is per board. What is NOT accepted is an
    explicit None. The check is textual because importing a mission needs the GrADyS runtime and
    a camera; a mission that names a model and resolves it to None at run time slips past, and
    that is what the launcher is for, since it fails loudly without the file. The same reasoning
    and the same risk apply to the detector: a mission may name it in a variable, because the
    aerial model sees nothing indoors and the indoor one is no use flying, but no detector at all
    detects nothing and does it silently.

    THE NEAREST PIN AND NOT THE FIRST THAT FITS, with no vectors at all so that only geometry
    decides. The greedy version put a report into the pin on the left; with two aircraft that
    almost never shows, because the only pin within reach is usually the right one, but with
    three it starts to, and what decides becomes the ARRIVAL ORDER of the reports, which means
    nothing.

    THE APPEARANCE OF A FUSED PIN is whichever drone has one. The two boards of this bench do
    not compute the same things, so a fused pin has one drone with a vector and one without, and
    which won used to depend on which reported first. It is not cosmetic: a fused pin with no
    vector leaves the operator's "not it" with nothing to send, and NOTES.md records the day the
    verdict was written to disk and reached no board at all.
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
MISIONES = os.path.join(AQUI, '..', 'scripts', 'banco_embedded')
revisadas = 0
for nombre in sorted(os.listdir(MISIONES)):
    if not nombre.startswith('mision_') or not nombre.endswith('.py'):
        continue
    texto = open(os.path.join(MISIONES, nombre), encoding='utf-8').read()
    codigo = re.sub(r'"""[\s\S]*?"""', '', texto)
    recortes = re.search(r'crops\s*=\s*True', codigo) is not None
    m_reid = re.search(r'reid_model\s*=\s*([^,\n]+)', codigo)
    asignado = m_reid is not None and m_reid.group(1).strip() not in ('None', '')
    hay_ruta = re.search(r'osnet\w*\.pt', codigo) is not None
    reid = asignado and hay_ruta
    m_det = re.search(r'[^_]model\s*=\s*([^,\n]+)', codigo)
    detector = m_det is not None and m_det.group(1).strip() not in ('None', '')
    estado = ('recortes SI, modelo ' + ('SI' if reid else 'NO') + ', detector ' + ('SI' if detector else 'NO')) if recortes else 'sin recortes'
    print('  %-34s %s' % (nombre, estado))
    revisadas += 1
    assert not (recortes and not reid), (
        '%s manda recortes para que un operador decida, pero no calcula apariencia. '
        'Entonces su "no es" no sale de la estacion, su clic no puede decir quien, y la '
        'fusion entre drones cae al camino debil. O enciende reid_model, o quita crops.'
        % nombre)
    assert not (recortes and not detector), (
        '%s manda recortes y no nombra detector. Un recorte es lo que una persona mira para '
        'decidir, y sin detector no hay nada que recortar.' % nombre)
assert revisadas >= 3, 'solo se revisaron %d misiones, el patron de busqueda esta mal' % revisadas
print('  -> %d misiones revisadas' % revisadas)

print()
print('=' * 70)
print('3. LA ESTACION SE FUNDE CON EL PIN MAS CERCANO, NO CON EL PRIMERO')
print('=' * 70)
pines = fundir({'1': [poi(10.0, 10.0, None), poi(13.0, 10.0, None)],
                '2': [poi(12.4, 10.0, None)]})
fundidos = [p for p in pines if len(p['drones']) > 1]
print('  pines del dron 1 en x=10.0 y x=13.0; el dron 2 reporta x=12.4')
print('  resultado: %s' % [(round(p['x'], 2), p.get('dron', '+'.join(p['drones']))) for p in pines])
assert len(fundidos) == 1, 'el reporte del dron 2 tenia que fundirse con alguno: %r' % pines
x = fundidos[0]['x']
assert abs(x - 12.7) < 0.05, (
    'se fundio en x=%.2f. 11.2 significa que eligio el PRIMER pin que encajaba en vez del mas '
    'cercano, que es la asignacion voraz que se quito el 3oct.' % x)
print('  -> se fundio en x=%.2f, que es el pin de 13.0 y no el de 10.0' % x)

print()
print('=' * 70)
print('4. UN PIN FUNDIDO SE QUEDA CON LA APARIENCIA DEL QUE LA TENGA')
print('=' * 70)
v = np.zeros(512, dtype=np.float32); v[0] = 1.0
for primero, etiqueta in ((None, 'el que reporta primero NO tiene vector'),
                          (v, 'el que reporta primero SI tiene vector')):
    segundo = v if primero is None else None
    pines = fundir({'1': [poi(10.0, 10.0, primero)], '2': [poi(10.2, 10.0, segundo)]})
    assert len(pines) == 1, 'el montaje tenia que fundir los dos: %r' % pines
    tiene = pines[0].get('emb') is not None
    print('  %-42s -> el pin fundido %s vector' % (etiqueta, 'SI tiene' if tiene else 'NO tiene'))
    assert tiene, ('el pin fundido perdio la apariencia que uno de los dos drones si traia; '
                   'con eso el "no es" del operador no tiene plantilla que mandar')

print()
print('test_misiones_operador OK')
