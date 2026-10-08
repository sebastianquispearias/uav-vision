"""
Gate for the switchable estimator: the ground plane, or the bearings crossing each other.

Every position this system has ever reported came from the same assumption. A bearing is
intersected with a HORIZONTAL PLANE at the altitude the mission declared, and the target is the
robust centre of those intersections. On the flights that produced the 2.39 m figure the ground
really was flat at that altitude, so the assumption cost nothing and it is still the default.

It is an assumption all the same, and it fails in a way that does not look like a failure. A
target standing five metres above the declared plane -- a rooftop, a truck bed, a rise in the
terrain -- has every one of its bearings continue PAST it until they reach the plane, and they
all land further from the drone than the target is. The result is self-consistent: the
intersections agree with each other, the track matures, the pin is confident, and it is in the
wrong place. Nothing in the report says so.

Crossing the bearings with each other needs no plane at all, and that is what "rayos" does,
through fusion.ransac_fusion, which has been in the repository with its own gate since 2026-10-07
and until now was not wired to anything.

The trade is real in both directions and this file measures both. Rays that cross at a sharp
angle beat the plane; rays cast from almost the same place are almost parallel and cross
nowhere useful, which is the regime the plane handles better. That is why this is a switch.

Run with: python tests/test_rayos.py

WHAT EACH SECTION PROVES
    1. The geometry of the defect, before any estimator: a target off the declared plane makes
       every ground impact land in the same wrong place.
    2. "rayos" recovers it, and the error drops by more than an order of magnitude.
    3. THE DEFAULT DID NOT MOVE. Same sightings, ray passed and ray withheld, positions equal
       to the last bit -- and the flight-3 replay is re-run from here at its 2.39 m.
    4. The honest half: from a hovering drone the bearings barely diverge, and the ray estimate
       is the worse of the two. A switch, not a replacement.
    5. It degrades instead of disappearing: one ray, no rays, a degenerate pair.
    6. Same evidence, same answer, twice. An estimate that wandered between reports would read
       as a target that is walking.
"""
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from uav_vision.fusion import ransac_fusion
from uav_vision.identity import IncrementalIdentity

FALLOS = []


def revisar(condicion, descripcion, detalle=""):
    print("  [%s] %s%s" % ("ok  " if condicion else "FALLA", descripcion,
                           (" -- " + detalle) if detalle else ""))
    if not condicion:
        FALLOS.append(descripcion)


BLANCO = np.array([3.0, -2.0, 5.0])   # five metres ABOVE the plane the mission declares
PLANO_Z = 0.0
ALTURA = 35.0
RADIO = 20.0


def vista(angulo_deg, blanco=BLANCO, radio=RADIO, ruido_yaw_rad=0.0, rng=None):
    """One sighting: where the drone is, the unit bearing to the target, and where that
    bearing meets the declared plane."""
    a = np.radians(angulo_deg)
    pos = np.array([radio * np.cos(a), radio * np.sin(a), ALTURA])
    d = blanco - pos
    d = d / np.linalg.norm(d)
    if ruido_yaw_rad and rng is not None:
        e = rng.normal(0.0, ruido_yaw_rad)
        c, s = np.cos(e), np.sin(e)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        d = d / np.linalg.norm(d)
    # the bearing continued until it reaches z = PLANO_Z
    k = (PLANO_Z - pos[2]) / d[2]
    impacto = pos + k * d
    return pos, d, impacto[:2]


def correr(estimator, vistas, pasar_rayo=True, radio_fusion=3.5):
    """Feeds the sightings to one identity layer and returns the track's position."""
    ident = IncrementalIdentity(fusion_radius_m=radio_fusion, fps=4.0, maturity="span",
                                estimator=estimator)
    for i, (pos, d, impacto) in enumerate(vistas):
        ident.observe(frame=i + 1, track_id=1, ground_xy=tuple(impacto), conf=0.9,
                      t=i * 0.25, cls="person",
                      ray=((float(pos[0]), float(pos[1]), float(pos[2])),
                           (float(d[0]), float(d[1]), float(d[2]))) if pasar_rayo else None)
    return ident._summary(1, ident._tracks[1])["pos"]


ANGULOS = list(range(0, 360, 12))
VISTAS = [vista(a) for a in ANGULOS]

print("=" * 76)
print("1. LA GEOMETRIA DEL FALLO: UN BLANCO FUERA DEL PLANO DECLARADO")
print("=" * 76)
print("  blanco de verdad        : (%.2f, %.2f, %.2f)" % tuple(BLANCO))
print("  plano que declara la mision: z = %.1f  -> el blanco esta %.1f m POR ENCIMA"
      % (PLANO_Z, BLANCO[2] - PLANO_Z))
print("  %d miradas desde un circulo de %.0f m de radio a %.0f m de altura"
      % (len(VISTAS), RADIO, ALTURA))

impactos = np.array([v[2] for v in VISTAS])
disp = float(np.max(np.linalg.norm(impactos - impactos.mean(axis=0), axis=1)))
err_imp = float(np.linalg.norm(impactos.mean(axis=0) - BLANCO[:2]))
print("  los impactos caen en    : centro (%.2f, %.2f), dispersion maxima %.2f m"
      % (impactos.mean(axis=0)[0], impactos.mean(axis=0)[1], disp))
print("  distancia del centro de los impactos al blanco: %.2f m" % err_imp)
revisar(disp > 2.0,
        "los impactos NO coinciden entre si: el plano los abre en un anillo",
        "%.2f m de dispersion" % disp)
print("  Y ESO ES LO QUE HACE ENGANOSO EL CASO: dar la vuelta entera CANCELA casi todo")
print("  el error, porque los impactos se abren en un anillo alrededor del blanco. Pero")
print("  no lo cancela del todo (el circulo de vuelo esta centrado en el origen y el")
print("  blanco no), y sobre todo asi no se vuela: se vuela en arco.")

ARCO = [vista(a) for a in range(0, 100, 6)]
imp_arco = np.array([v[2] for v in ARCO])
err_arco = float(np.linalg.norm(np.median(imp_arco, axis=0) - BLANCO[:2]))
print()
print("  con un ARCO de 0 a 96 grados (%d miradas):" % len(ARCO))
print("    mediana de los impactos : (%.2f, %.2f)" % tuple(np.median(imp_arco, axis=0)))
print("    blanco                  : (%.2f, %.2f)" % tuple(BLANCO[:2]))
print("    error del plano         : %.2f m" % err_arco)
revisar(err_arco > 1.0,
        "el plano se equivoca, y se equivoca sin avisar: los impactos concuerdan entre si",
        "%.2f m de error" % err_arco)

print()
print("=" * 76)
print('2. "rayos" LO RECUPERA: LOS RAYOS SE CRUZAN DONDE ESTA EL BLANCO')
print("=" * 76)

p_suelo = correr("suelo", ARCO)
p_rayos = correr("rayos", ARCO)
e_suelo = float(np.linalg.norm(p_suelo - BLANCO[:2]))
e_rayos = float(np.linalg.norm(p_rayos - BLANCO[:2]))
print("  estimator='suelo' -> (%.3f, %.3f)   error %.3f m" % (p_suelo[0], p_suelo[1], e_suelo))
print("  estimator='rayos' -> (%.3f, %.3f)   error %.3f m" % (p_rayos[0], p_rayos[1], e_rayos))
print("  blanco            -> (%.3f, %.3f)" % tuple(BLANCO[:2]))
revisar(e_rayos < e_suelo / 10.0,
        "el error cae mas de un orden de magnitud al soltar el plano",
        "%.3f m contra %.3f m" % (e_rayos, e_suelo))
revisar(e_rayos < 0.2, "y el resultado es el blanco, no otro punto", "%.3f m" % e_rayos)

print()
print("  con ruido de rumbo, que es lo que tiene un dron de verdad:")
rng = np.random.default_rng(7)
ruidoso = [vista(a, ruido_yaw_rad=np.radians(1.5), rng=rng) for a in range(0, 100, 6)]
pr_s = correr("suelo", ruidoso)
pr_r = correr("rayos", ruidoso)
er_s = float(np.linalg.norm(pr_s - BLANCO[:2]))
er_r = float(np.linalg.norm(pr_r - BLANCO[:2]))
print("    1,5 grados de sigma -> suelo %.3f m, rayos %.3f m" % (er_s, er_r))
revisar(er_r < er_s,
        "con ruido el rayo sigue ganando al plano en este regimen",
        "%.3f m contra %.3f m" % (er_r, er_s))

print()
print("=" * 76)
print("3. EL DEFAULT NO SE MOVIO, Y ESA ES LA PUERTA")
print("=" * 76)

con_rayo = correr("suelo", ARCO, pasar_rayo=True)
sin_rayo = correr("suelo", ARCO, pasar_rayo=False)
print("  'suelo' con el rayo pasado : (%.17g, %.17g)" % (con_rayo[0], con_rayo[1]))
print("  'suelo' sin pasar el rayo  : (%.17g, %.17g)" % (sin_rayo[0], sin_rayo[1]))
revisar(con_rayo[0] == sin_rayo[0] and con_rayo[1] == sin_rayo[1],
        "IDENTICAS hasta el ultimo bit: pasar el rayo no toca el camino por defecto")

ident = IncrementalIdentity(fusion_radius_m=3.5, fps=4.0)
for i, (_pos, _d, impacto) in enumerate(ARCO):
    ident.observe(frame=i + 1, track_id=1, ground_xy=tuple(impacto), conf=0.9, t=i * 0.25,
                  ray=((0.0, 0.0, 35.0), (0.0, 0.0, -1.0)))
guardados = len(ident._tracks[1]["rayos"])
print("  rayos guardados por una capa en modo 'suelo': %d de %d miradas"
      % (guardados, len(ARCO)))
revisar(guardados == 0,
        "y no se guarda ni uno: el default no paga la memoria de una funcion que no usa")

print()
print("  Y LA PUERTA DE VERDAD, el replay del vuelo 3:")
r = subprocess.run([sys.executable, os.path.join(_HERE, "demo", "demo.py"), "--sin-mapa"],
                   cwd=_HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
salida = (r.stdout or "") + (r.stderr or "")
linea = [ln for ln in salida.splitlines() if "mejor POI" in ln]
if linea:
    print("   ", linea[0].strip())
    revisar("2.39 m" in linea[0],
            "demo.py --sin-mapa sigue dando 2,39 m con el estimador conmutable dentro")
else:
    # El mismo criterio que conftest.py, y la lista es LA SUYA para que no se separen: una
    # dependencia opcional que falta no es un gate roto, y un rojo que solo significa "aqui no
    # esta instalado" es un rojo que la gente aprende a ignorar. El clon pelado del CI no trae
    # el runtime de GrADyS ni el archivo de vuelos, y ahi esta puerta no se puede correr.
    sys.path.insert(0, _HERE)
    from conftest import FALTA
    porque = [f for f in FALTA if f in salida]
    ultimas = [ln for ln in salida.strip().splitlines() if ln.strip()][-3:]
    print("    demo.py no imprimio la linea. Lo ultimo que dijo:")
    for ln in ultimas:
        print("      " + ln.strip()[:100])
    if porque:
        print("    NO CORRESPONDE AQUI: %s. La puerta de 2,39 m la corre el otro job del CI,"
              % porque[0])
        print("    el que si trae el runtime, y la corre la laptop en cada sesion.")
    else:
        revisar(False,
                "demo.py --sin-mapa sigue dando 2,39 m con el estimador conmutable dentro",
                "y no es por una dependencia que falte: salio con codigo %d" % r.returncode)

print()
print("=" * 76)
print("4. LA MITAD HONESTA: EN VUELO ESTACIONARIO EL RAYO ES PEOR")
print("=" * 76)
print("  un dron que apenas se mueve da rayos casi paralelos, y rayos casi paralelos")
print("  se cruzan en cualquier lado. El plano no tiene ese problema.")

rng = np.random.default_rng(11)
QUIETO = [vista(a, radio=20.0, ruido_yaw_rad=np.radians(1.5), rng=rng)
          for a in np.linspace(0.0, 2.0, 16)]   # dos grados de arco en todo el avistamiento
base = np.array([QUIETO[0][0][0], QUIETO[0][0][1]])
recorrido = max(float(np.linalg.norm(np.array([p[0], p[1]]) - base)) for p, _, _ in QUIETO)
q_s = correr("suelo", QUIETO)
q_r = correr("rayos", QUIETO)
eq_s = float(np.linalg.norm(q_s - BLANCO[:2]))
eq_r = float(np.linalg.norm(q_r - BLANCO[:2]))
print("  el dron recorre %.2f m en todo el avistamiento" % recorrido)
print("  estimator='suelo' -> error %.3f m" % eq_s)
print("  estimator='rayos' -> error %.3f m" % eq_r)
revisar(eq_r > eq_s,
        "aqui GANA el plano, y por eso esto es un conmutador y no un reemplazo",
        "rayos %.3f m contra suelo %.3f m" % (eq_r, eq_s))

print()
print("=" * 76)
print("5. SE DEGRADA, NO DESAPARECE")
print("=" * 76)

uno = correr("rayos", ARCO[:1])
mediana_uno = np.median(np.array([ARCO[0][2]]), axis=0)
print("  con UNA sola mirada -> (%.3f, %.3f)" % tuple(uno))
revisar(np.allclose(uno, mediana_uno),
        "con un solo rayo no hay cruce posible y cae a la mediana del plano")

sin = correr("rayos", ARCO, pasar_rayo=False)
print("  con rayos pedidos pero NINGUNO pasado -> (%.3f, %.3f)" % tuple(sin))
revisar(np.allclose(sin, np.median(np.array([v[2] for v in ARCO]), axis=0)),
        "un llamador que no manda rayos obtiene el estimador de siempre, sin levantar")

print()
print("  EL CASO QUE ESTE MISMO TEST DESTAPO. Dos rayos identicos no se cruzan en")
print("  ninguna parte, y la fusion, preguntada igual, contesta LA POSICION DE LA CAMARA:")
gemelas = [vista(0.0), vista(0.0)]
crudo = ransac_fusion([((float(p[0]), float(p[1]), float(p[2])),
              (float(d[0]), float(d[1]), float(d[2]))) for p, d, _ in gemelas],
            threshold_m=3.5, rng=np.random.default_rng(0))
print("    la fusion a secas       -> (%.3f, %.3f, %.3f)" % crudo)
print("    el dron estaba en       -> (%.3f, %.3f, %.3f)" % tuple(gemelas[0][0]))
revisar(abs(crudo[0] - gemelas[0][0][0]) < 0.01 and abs(crudo[1] - gemelas[0][0][1]) < 0.01,
        "sin guarda, el blanco aterriza ENCIMA DEL DRON: finito, sin excepcion, y basura")

dos_iguales = correr("rayos", gemelas)
mediana_gemelas = np.median(np.array([v[2] for v in gemelas]), axis=0)
print("    con la guarda de divergencia -> (%.3f, %.3f)" % tuple(dos_iguales))
print("    la mediana del plano         -> (%.3f, %.3f)" % tuple(mediana_gemelas))
revisar(np.allclose(dos_iguales, mediana_gemelas),
        "la guarda lo rechaza ANTES de fusionar y devuelve el estimador del plano",
        "rayos que no divergen no se cruzan: la pregunta no tiene respuesta")

ident = IncrementalIdentity(fusion_radius_m=3.5, fps=4.0, estimator="rayos")
div_gemelas = ident._divergencia([((0, 0, 35), (0.5, 0.5, -0.707)),
                                  ((1, 0, 35), (0.5, 0.5, -0.707))])
div_arco = ident._divergencia([((float(p[0]), float(p[1]), float(p[2])),
                                (float(d[0]), float(d[1]), float(d[2])))
                               for p, d, _ in ARCO])
div_quieto = ident._divergencia([((float(p[0]), float(p[1]), float(p[2])),
                                  (float(d[0]), float(d[1]), float(d[2])))
                                 for p, d, _ in QUIETO])
print("    divergencia medida: gemelas %.3f grados, estacionario %.3f, arco %.3f"
      % (np.degrees(div_gemelas), np.degrees(div_quieto), np.degrees(div_arco)))
revisar(np.degrees(div_gemelas) < 0.5 < np.degrees(div_quieto),
        "el umbral de 0,5 grados separa 'paralelos' de 'poco separados', que no es lo mismo",
        "el caso estacionario SIGUE usando rayos, y por eso la seccion 4 sigue siendo honesta")

print()
print("=" * 76)
print("6. LA MISMA EVIDENCIA DA LA MISMA RESPUESTA")
print("=" * 76)
a1 = correr("rayos", ARCO)
a2 = correr("rayos", ARCO)
print("  corrida 1 -> (%.17g, %.17g)" % (a1[0], a1[1]))
print("  corrida 2 -> (%.17g, %.17g)" % (a2[0], a2[1]))
revisar(a1[0] == a2[0] and a1[1] == a2[1],
        "RANSAC va sembrado: un pin que se moviera sin evidencia nueva es un blanco que camina")

print()
print("=" * 76)
print("7. UN ESTIMADOR QUE NO EXISTE SE RECHAZA AL CONSTRUIR")
print("=" * 76)
try:
    IncrementalIdentity(fusion_radius_m=3.5, fps=4.0, estimator="rayitos")
    revisar(False, "tendria que haber levantado ValueError")
except ValueError as e:
    print("  IncrementalIdentity(estimator='rayitos') ->", e)
    revisar("rayitos" in str(e),
            "el error dice QUE se paso, no solo que estaba mal")

print()
print("=" * 76)
if FALLOS:
    print("FALLARON %d:" % len(FALLOS))
    for f in FALLOS:
        print("  - %s" % f)
    sys.exit(1)
print("TODO OK")
