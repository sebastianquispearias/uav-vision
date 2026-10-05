"""
Empirical gates for IncrementalIdentity, each on a synthetic scene with known ground truth.

Every section is a CONTRAST that fails if the behaviour it claims is not there. A test that
cannot fail proves nothing, so none of these merely re-reads a value that was just set.

  1. Static: one standing person under projection noise yields one candidate near the truth.
  2. Mobile: a walker is classified as mobile and its reported position must beat the lagging
     naive estimate, which is the median of the recent window.
  3. Co-occurrence veto: two people seen in the same frames stay two candidates.
  4. Twin override: duplicate boxes of one person merge into one candidate. The detector is made
     to duplicate the box half of the time.
  5. Short pass: nothing matures, but a preliminary worth verifying does appear. This is the
     SWEEP case: a short pass over a person never matures a candidate, so without preliminaries
     the system says nothing at all about somebody it tracked perfectly well. The section also
     carries the guarantee that keeps preliminaries honest -- once evidence accumulates the same
     candidate matures, and the two calls agree. It asks for the SPAN rule explicitly, because
     that is the rule the measurement was made under and the default is now looks.
  6. Fragments reinforce an existing candidate and never create one. A tracker that keeps losing
     a target leaves pieces shorter than the track gate; with reinforce_with_fragments they may
     join the candidate a lasting track opened, by the same rules as any merge. Three contrasts,
     each of which fails if the rule is wrong: the same fragment joins ONLY with the option on,
     a fragment alone opens nothing even with it on, and a fragment in the right place with
     another appearance stays out. Span is asked for explicitly here too, because a fragment is
     defined against the span rule's track gate.
  7. Looks: confirms sooner, does not open on an instant, and the radius never falls below the
     bias. Each contrast runs the same observations through BOTH modes, so it fails if the modes
     stop behaving differently where they are supposed to. The last case checks the default bar
     itself: ten seconds seen are not enough and twenty-five are.
  8. A moving target detected in bursts: the current position is fitted against the CLOCK. A
     target in frame is detected in a minority of frames, in bursts, and fitting against the
     observation index treats every sighting as one equal step while fitting against time does
     not.
  9. Radius and distance: the margin grows with how far away the thing is looked at. A heading
     error moves the impact by range times angle, so the same target seen for the same time must
     carry a wider margin from far away than from close, and at the range the error floor was
     measured at the model must give back the radius every earlier report carried.
 10. Patrol: a target that goes back and forth is described by its RECENT past. Over its whole
     life its median is the middle of the patrol and a line through its last quarter crosses the
     turns, so span reports it far from where it is and looks must report it close.
 11. Between sightings: a mover is reported where it is AT REPORT TIME. The report goes out on
     its own clock and the last sighting of a moving target is already old by then.
 12. Unseen, the margin grows: the radius of a mover says how far it can have gone. A target
     that turns back right after its last sighting ends up far off the extrapolated line, and a
     radius frozen at the last sighting claims to hold it and does not.
 13. Short pieces of something standing still: NOISE IS NOT MOTION. A standing person leaves
     short tracks whose ground points jump by metres and whose fitted speed clears the speed
     test, and because a mobile is never merged, one standing person became several points on
     the map. Here a standing person seen as eight short jittery tracks must be ONE static
     candidate, and a person who really walks must stay mobile.
 14. Age: the report says how long ago the candidate was last seen. A lost candidate keeps being
     reported at the same point, and without this nothing in the report tells a consumer it is
     old news -- an escorting drone chased points tens of metres away. The same candidate
     reported half a second and twenty seconds after its last sighting must say so, and span
     mode keeps its report unchanged.
 15. Evidence: the report says how much is MISSING to be reported, not only how much there is.
     The station used to print a count with no threshold beside it, so neither an operator nor
     this repository could see how far from reportable anything was. The fraction has to be the
     count over the threshold IN FORCE, which the third case proves: the same five looks read
     one value against a bar of 20 and another against a bar of 10, so a field that always
     returned the first would pass the first two cases and fail that one. The last assertion
     ties the bar and the verdict together: today both read one axis so the equivalence is free,
     but the moment a second axis joins the rule, evidence has to be the MINIMUM over every axis
     or a card will say a thing is reportable about something the layer refuses to report.
 16. Density: in what fraction of the frames a detection was produced in the candidate was seen.
     A static false positive is a flicker spread thin while a person being tracked is dense
     while they are in view. Sightings per second separates the same rows and is THE WRONG
     QUANTITY, because it is not scale free, and the third case is the proof: the same target,
     the same seconds, twice the frame rate, and sightings per look doubles while the density
     does not move. On a steady board a threshold set on sightings per second would never fire.

The measured numbers behind sections 5, 13, 14, 15 and 16 are in NOTES.md.

Run with: python tests/test_identity.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from uav_vision.identity import IncrementalIdentity

RNG = np.random.default_rng(7)
FPS = 5.0
SIGMA = 1.2


def emb_de(base: int) -> np.ndarray:
    v = np.random.default_rng(base).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def ruido():
    return RNG.normal(0, SIGMA, size=2)


print("=" * 64)
print("1. ESTATICO: una persona parada -> UN candidato cerca de la verdad")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
VERDAD = np.array([2.0, -3.0])
e1 = emb_de(1)
for f in range(300):
    ident.observe(f, 10, VERDAD + ruido(), 0.7, e1 + 0.05 * RNG.normal(size=512))
c = ident.candidates()
assert len(c) == 1, f"fragmento en {len(c)} candidates"
err = math.hypot(c[0]["x"] - VERDAD[0], c[0]["y"] - VERDAD[1])
print(f"  1 candidato, error {err:.2f} m, mobile={c[0]['mobile']}")
assert err < 0.5 and not c[0]["mobile"]

print()
print("=" * 64)
print("2. MOVIL: caminante a 1 m/s -> MOVIL y sin retraso")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e2 = emb_de(2)
pos_final = None
for f in range(300):
    t = f / FPS
    real = np.array([-10.0 + 1.0 * t, 4.0])
    pos_final = real
    ident.observe(f, 20, real + ruido(), 0.6, e2 + 0.05 * RNG.normal(size=512))
c = ident.candidates()
assert len(c) == 1 and c[0]["mobile"], f"no salio MOVIL: {c}"
err_fit = math.hypot(c[0]["x"] - pos_final[0], c[0]["y"] - pos_final[1])
imps = np.array([( -10.0 + (f / FPS), 4.0) for f in range(300)])
q = max(2, len(imps) // 4)
naive = np.median(imps[-q:], axis=0)
err_naive = math.hypot(naive[0] - pos_final[0], naive[1] - pos_final[1])
print(f"  MOVIL detectado; error ajuste lineal {err_fit:.2f} m "
      f"vs mediana-reciente {err_naive:.2f} m (retraso puro, sin ruido)")
assert err_fit < err_naive, "el ajuste no mejora al estimador ingenuo"
assert err_fit < 1.5

print()
print("=" * 64)
print("3. VETO: dos personas juntas 2 m aparte -> DOS candidates")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
A, B = np.array([0.0, 0.0]), np.array([2.0, 0.0])
ea, eb = emb_de(3), emb_de(4)
for f in range(300):
    ident.observe(f, 30, A + ruido(), 0.7, ea + 0.05 * RNG.normal(size=512))
    ident.observe(f, 31, B + ruido(), 0.7, eb + 0.05 * RNG.normal(size=512))
c = ident.candidates()
print(f"  candidates: {len(c)} (posiciones {[(p['x'], p['y']) for p in c]})")
assert len(c) == 2, "el veto de co-ocurrencia fallo: se fusionaron"

print()
print("=" * 64)
print("4. GEMELO: cajas duplicadas de UNA persona -> UN candidato")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
P = np.array([-1.0, 5.0])
ep = emb_de(5)
for f in range(300):
    ident.observe(f, 40, P + ruido(), 0.7, ep + 0.03 * RNG.normal(size=512))
    if f % 2 == 0:
        ident.observe(f, 41, P + ruido(), 0.5, ep + 0.03 * RNG.normal(size=512))
c = ident.candidates()
print(f"  candidates: {len(c)} con n_obs={[p['n_obs'] for p in c]}")
assert len(c) == 1, "el gemelo no se fusiono (regla A-1 rota)"

print()
print("=" * 64)
print("5. PASADA CORTA: nada mature, pero SI un preliminar que verificar")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, report_dur_s=36.0, maturity="span")
P = np.array([4.0, -2.0])
ep = emb_de(9)
n_pasada = int(20 * FPS)
for f in range(n_pasada):
    ident.observe(f, 70, P + ruido(), 0.6, ep + 0.03 * RNG.normal(size=512))

maduros = ident.candidates()
todos = ident.candidates(preliminary=True)
print(f"  maduros: {len(maduros)}   con preliminary: {len(todos)}")
assert len(maduros) == 0, "una pasada de 20 s no deberia madurar nada"
assert len(todos) == 1, "la pasada corta debe dejar UN preliminar que verificar"
assert todos[0]["mature"] is False, "el preliminar debe venir marcado mature=False"
d = float(np.linalg.norm(np.array([todos[0]["x"], todos[0]["y"]]) - P))
print(f"  preliminar en ({todos[0]['x']}, {todos[0]['y']}), a {d:.2f} m del real, "
      f"n_obs={todos[0]['n_obs']}")
assert d < 1.0, "el preliminar apunta al lugar equivocado"

for f in range(n_pasada, int(60 * FPS)):
    ident.observe(f, 70, P + ruido(), 0.6, ep + 0.03 * RNG.normal(size=512))
maduros = ident.candidates()
assert len(maduros) == 1 and maduros[0]["mature"] is True,     "con evidencia suficiente el preliminar tiene que madurar"
print(f"  tras 60 s: mature=True, n_obs={maduros[0]['n_obs']}")

print()
print("=" * 64)
print("6. FRAGMENTOS: refuerzan un candidato existente, nunca crean uno")
print("=" * 64)


def escena(refuerzo, con_pista_larga=True):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity="span",
                                reinforce_with_fragments=refuerzo)
    P, Q = np.array([0.0, 0.0]), np.array([30.0, 30.0])
    e_p, e_otro = emb_de(11), emb_de(12)
    if con_pista_larga:
        for f in range(300):
            ident.observe(f, 80, P + ruido(), 0.7, e_p + 0.03 * RNG.normal(size=512))
    for f in range(300, 320):
        ident.observe(f, 81, P + ruido(), 0.7, e_p + 0.03 * RNG.normal(size=512))
        ident.observe(f, 82, Q + ruido(), 0.7, e_p + 0.03 * RNG.normal(size=512))
    for f in range(320, 340):
        ident.observe(f, 83, P + ruido(), 0.7, e_otro + 0.03 * RNG.normal(size=512))
    return ident.candidates(preliminary=True, with_tracks=True)


apagado, encendido = escena(False), escena(True)
print(f"  sin refuerzo: {[(c['tracks'], c['n_obs']) for c in apagado]}")
print(f"  con refuerzo: {[(c['tracks'], c['n_obs']) for c in encendido]}")
assert len(apagado) == 1 and apagado[0]["tracks"] == [80], "sin la opcion un fragmento no deberia entrar"
assert len(encendido) == 1, "un fragmento aislado no puede abrir un candidato"
assert 81 in encendido[0]["tracks"], "el fragmento de la misma persona deberia reforzar"
assert 82 not in encendido[0]["tracks"], "un fragmento a 42 m no es la misma cosa"
assert 83 not in encendido[0]["tracks"], "otra apariencia en el mismo sitio no deberia entrar"
assert encendido[0]["n_obs"] == apagado[0]["n_obs"] + 20
solos = escena(True, con_pista_larga=False)
print(f"  solo fragmentos, con refuerzo: {len(solos)} candidatos")
assert solos == [], "sin una pista que dure, los fragmentos no dejan nada en el mapa"

print()
print("=" * 64)
print("7. MIRADAS: confirma pronto, no abre con un instante, y el radio no baja del sesgo")
print("=" * 64)


def seguido(maturity, segundos, hz=FPS, t0=0.0, tid=90, **kw):
    kw.setdefault("report_min_looks", 5)
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity=maturity, **kw)
    P = np.array([1.0, 1.0])
    e = emb_de(13)
    for k in range(int(segundos * hz)):
        ident.observe(k, tid, P + ruido(), 0.7, e + 0.03 * RNG.normal(size=512), t=t0 + k / hz)
    return ident, ident.candidates(preliminary=True)


_, c_span = seguido("span", 10)
_, c_look = seguido("looks", 10)
print(f"  10 s seguidos -> span: maduro={[c['mature'] for c in c_span]} | "
      f"miradas: maduro={[c['mature'] for c in c_look]} looks={[c.get('looks') for c in c_look]}")
assert len(c_span) == 1 and not c_span[0]["mature"], "span no deberia confirmar con 10 s"
assert len(c_look) == 1 and c_look[0]["mature"], "10 miradas deberian confirmar"

_, c_span = seguido("span", 2.6)
_, c_look = seguido("looks", 2.6)
print(f"  2.6 s seguidos -> span: {len(c_span)} candidatos | miradas: {len(c_look)} candidatos, "
      f"maduro={[c['mature'] for c in c_look]}")
assert c_span == [], "2.6 s no llegan a la puerta de 8.6 s"
assert len(c_look) == 1 and not c_look[0]["mature"], "3 miradas abren un preliminar, no un maduro"

_, c_look = seguido("looks", 0.5, hz=40.0)
print(f"  20 detecciones en medio segundo -> miradas: {len(c_look)} candidatos")
assert c_look == [], "un instante es una sola mirada, por muchas cajas que tenga"

i5, c5 = seguido("looks", 5)
i60, c60 = seguido("looks", 60)
suelo = 2.4477 * i60.bias_sigma_m
print(f"  radio 95%: con 5 miradas {c5[0]['radius_m']} m | con 60 miradas {c60[0]['radius_m']} m "
      f"| suelo del sesgo {suelo:.2f} m")
assert c5[0]["radius_m"] > c60[0]["radius_m"], "mas miradas tienen que achicar el radio"
assert c60[0]["radius_m"] >= round(suelo, 2) - 0.01, "el radio no puede bajar del sesgo compartido"
assert "radius_m" not in seguido("span", 60)[1][0], "el modo span no cambia el reporte"

def por_defecto(segundos):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
    e = emb_de(14)
    for k in range(int(segundos * FPS)):
        ident.observe(k, 91, np.array([1.0, 1.0]) + ruido(), 0.7, e + 0.03 * RNG.normal(size=512), t=k / FPS)
    return ident.candidates(preliminary=True)
d10, d25 = por_defecto(10), por_defecto(25)
print(f"  por defecto (20 miradas): 10 s maduro={[c['mature'] for c in d10]} | 25 s maduro={[c['mature'] for c in d25]}")
assert IncrementalIdentity(fusion_radius_m=3.5, fps=FPS).maturity == "looks"
assert len(d10) == 1 and not d10[0]["mature"], "con el defecto, 10 miradas no confirman"
assert len(d25) == 1 and d25[0]["mature"], "con el defecto, 25 miradas confirman"

print()
print("=" * 64)
print("8. BLANCO MOVIL DETECTADO A RAFAGAS: la posicion actual se ajusta contra el reloj")
print("=" * 64)
from uav_vision.identity import _current_position

vel = np.array([6.0, 0.0])
tiempos = [t0 + k * 0.2 for t0 in (0.0, 1.0, 9.0, 10.0, 17.0, 18.0, 26.0, 27.0, 34.0, 35.0) for k in range(5)]
reales = np.array([vel * t for t in tiempos])
ruido_rng = np.random.default_rng(21)
imps = reales + ruido_rng.normal(0, 0.8, size=reales.shape)
fin = reales[-1]
err_indice = float(np.linalg.norm(_current_position(imps) - fin))
err_reloj = float(np.linalg.norm(_current_position(imps, tiempos) - fin))
print(f"  error de la posicion actual: contra el indice {err_indice:.2f} m | contra el reloj {err_reloj:.2f} m")
assert err_reloj < 0.5 * err_indice, "ajustar contra el reloj tiene que corregir las rafagas"
assert err_reloj < 2.0

ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e = emb_de(31)
for k, (t, xy) in enumerate(zip(tiempos, imps)):
    ident.observe(k, 95, xy, 0.7, e + 0.03 * RNG.normal(size=512), t=t)
c = ident.candidates(preliminary=True)
movil = [x for x in c if x["mobile"]]
assert len(movil) == 1, f"el blanco a 6 m/s tiene que salir movil: {c}"
d = float(np.hypot(movil[0]["x"] - fin[0], movil[0]["y"] - fin[1]))
print(f"  por la capa de identidad con t: MOVIL, a {d:.2f} m de donde esta")
assert d < 2.0

print()
print("=" * 64)
print("9. RADIO Y DISTANCIA: el margen crece con lo lejos que se mira")
print("=" * 64)
from uav_vision.identity import RANGO_REFERENCIA_M


def radio_a(rango):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
    e = emb_de(41)
    for k in range(int(30 * FPS)):
        ident.observe(k, 97, np.array([2.0, 2.0]) + ruido(), 0.7, e + 0.03 * RNG.normal(size=512),
                      t=k / FPS, range_m=rango)
    return ident.candidates(preliminary=True)[0]["radius_m"]


r_cerca, r_ref, r_lejos, r_sin = radio_a(12.0), radio_a(RANGO_REFERENCIA_M), radio_a(90.0), radio_a(None)
print(f"  radio 95%: a 12 m {r_cerca} | a {RANGO_REFERENCIA_M:.0f} m (referencia) {r_ref} | a 90 m {r_lejos} "
      f"| sin distancia {r_sin}")
assert r_lejos > r_ref + 1.0, "desde 90 m el margen tiene que ser claramente mayor"
assert r_cerca < r_ref, "desde cerca el margen tiene que ser menor"
assert abs(r_ref - r_sin) < 0.3, "en la distancia de referencia el modelo tiene que devolver el radio medido"

print()
print("=" * 64)
print("10. PATRULLA: un blanco que va y vuelve se describe por su pasado reciente")
print("=" * 64)


def patrulla(maturity, v=6.0, quieto=False):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity=maturity)
    rng = np.random.default_rng(5)
    e = emb_de(51)
    k, real = 0, None
    for t0 in np.arange(0.0, 90.0, 3.0):
        for j in range(5):
            t = t0 + 0.2 * j
            s_ = (v * t) % 100.0
            x = -25.0 + (s_ if s_ <= 50.0 else 100.0 - s_)
            real = np.array([0.0 if quieto else x, 30.0])
            ident.observe(k, 99, real + rng.normal(0, 1.2, size=2), 0.7,
                          e + 0.03 * RNG.normal(size=512), t=t)
            k += 1
    c = ident.candidates(preliminary=True)[0]
    return c, float(np.hypot(c["x"] - real[0], c["y"] - real[1]))


c_span, e_span = patrulla("span")
c_look, e_look = patrulla("looks")
c_quieto, e_quieto = patrulla("looks", quieto=True)
print(f"  patrulla a 6 m/s: span movil={c_span['mobile']} error {e_span:.1f} m | "
      f"miradas movil={c_look['mobile']} error {e_look:.1f} m")
print(f"  blanco quieto con el mismo ruido: miradas movil={c_quieto['mobile']} error {e_quieto:.2f} m")
assert c_look["mobile"] and e_look < 3.0, "la patrulla tiene que salir movil y cerca de donde esta"
assert e_span > 3 * e_look, "el contraste con la regla vieja tiene que verse"
assert not c_quieto["mobile"], "el ruido de un blanco quieto no puede leerse como movimiento"

print()
print("=" * 64)
print("11. ENTRE AVISTAMIENTOS: un movil se reporta donde esta al reportar")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e = emb_de(61)
vel = np.array([6.0, 0.0])
for k in range(200):
    ident.observe(k, 101, vel * (0.2 * k) + RNG.normal(0, 1.0, size=2), 0.7,
                  e + 0.03 * RNG.normal(size=512), t=0.2 * k)
t_ultimo = 0.2 * 199
ahora = t_ultimo + 1.5
real = vel * ahora
sin = ident.candidates(preliminary=True)[0]
con = ident.candidates(preliminary=True, now=ahora)[0]
lejos = ident.candidates(preliminary=True, now=t_ultimo + 60.0)[0]
e_sin = float(np.hypot(sin["x"] - real[0], sin["y"] - real[1]))
e_con = float(np.hypot(con["x"] - real[0], con["y"] - real[1]))
avance_lejos = float(np.hypot(lejos["x"] - sin["x"], lejos["y"] - sin["y"]))
print(f"  reporte 1.5 s despues del ultimo avistamiento: sin 'now' a {e_sin:.1f} m | con 'now' a {e_con:.1f} m")
print(f"  reporte 60 s despues: se lleva {avance_lejos:.1f} m, no {6.0 * 60:.0f} (tope de la ventana)")
assert con["mobile"] and e_con < 2.0, "con el instante del reporte tiene que quedar cerca"
assert e_sin > 7.0, "el contraste sin extrapolar tiene que verse"
assert avance_lejos <= 6.0 * ident.motion_window_s + 2.0, "la extrapolacion no puede pasar la ventana"

quieto = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
for k in range(200):
    quieto.observe(k, 102, np.array([3.0, 3.0]) + RNG.normal(0, 1.0, size=2), 0.7,
                   e + 0.03 * RNG.normal(size=512), t=0.2 * k)
a, b = quieto.candidates(preliminary=True)[0], quieto.candidates(preliminary=True, now=500.0)[0]
assert (a["x"], a["y"]) == (b["x"], b["y"]), "un blanco quieto no se mueve por pasar el tiempo"

print()
print("=" * 64)
print("12. SIN VERLO, EL MARGEN CRECE: el radio de un movil dice cuanto puede haberse ido")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e = emb_de(71)
vel = np.array([6.0, 0.0])
for k in range(200):
    ident.observe(k, 103, vel * (0.2 * k) + RNG.normal(0, 1.0, size=2), 0.7,
                  e + 0.03 * RNG.normal(size=512), t=0.2 * k)
t_ultimo = 0.2 * 199
ahora = t_ultimo + 4.0
real = vel * t_ultimo - vel * 4.0
fijo = ident.candidates(preliminary=True)[0]
vivo = ident.candidates(preliminary=True, now=ahora)[0]
d_vivo = float(np.hypot(vivo["x"] - real[0], vivo["y"] - real[1]))
print(f"  4 s despues de dar la vuelta: radio sin 'now' {fijo['radius_m']} m | con 'now' {vivo['radius_m']} m "
      f"| el blanco a {d_vivo:.1f} m del punto reportado")
assert fijo["radius_m"] < 8.0, "el radio base de un movil es su incertidumbre, no la longitud de su recorrido"
assert vivo["radius_m"] >= fijo["radius_m"] + 6.0 * 4.0 - 0.5, "el margen tiene que crecer con rapidez x tiempo"
assert d_vivo <= vivo["radius_m"], "con el margen que crece, la verdad tiene que caer dentro"
assert d_vivo > fijo["radius_m"], "el contraste: el margen congelado no la contenia"
r_quieto = (quieto.candidates(preliminary=True)[0]["radius_m"], quieto.candidates(preliminary=True, now=500.0)[0]["radius_m"])
assert r_quieto[0] == r_quieto[1], "un blanco quieto no gana margen por pasar el tiempo"

print()
print("=" * 64)
print("13. TRAMOS CORTOS DE UNO QUIETO: el ruido no es movimiento")
print("=" * 64)


def tramos_cortos(caminando):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
    rng = np.random.default_rng(31)
    e = emb_de(95)
    k = 0
    for tramo in range(8):
        t0 = tramo * 6.0
        for j in range(15):
            t = t0 + 0.2 * j
            real = np.array([-10.0 + 1.4 * t, 3.0]) if caminando else np.array([2.0, 3.0])
            salto = rng.normal(0, 1.2, size=2) + (rng.normal(0, 2.5, size=2) if rng.random() < 0.2 else 0.0)
            ident.observe(k, 400 + tramo if not caminando else 400, real + salto, 0.6,
                          e + 0.03 * rng.normal(size=512), t=t)
            k += 1
    return ident.candidates(preliminary=True)


quieto_cortos = tramos_cortos(False)
camina = tramos_cortos(True)
print(f"  quieto en 8 tramos cortos: {len(quieto_cortos)} candidato(s), moviles {sum(c['mobile'] for c in quieto_cortos)}, "
      f"radios {[c.get('radius_m') for c in quieto_cortos]}")
print(f"  caminando 1.4 m/s (una pista): movil={camina[0]['mobile']}")
assert len(quieto_cortos) == 1 and not quieto_cortos[0]["mobile"], "una persona quieta tiene que ser un solo punto quieto"
assert camina[0]["mobile"], "el contraste: quien camina de verdad sigue siendo movil"

print()
print("=" * 64)
print("14. EDAD: el reporte dice hace cuanto se vio el candidato por ultima vez")
print("=" * 64)
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e = emb_de(97)
for k in range(50):
    ident.observe(k, 500, np.array([1.0, 1.0]) + RNG.normal(0, 0.5, size=2), 0.7, e + 0.03 * RNG.normal(size=512), t=0.2 * k)
reciente = ident.candidates(preliminary=True, now=10.3)[0]
viejo = ident.candidates(preliminary=True, now=29.8)[0]
sin_reloj = ident.candidates(preliminary=True)[0]
span = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity="span")
for k in range(50):
    span.observe(k, 500, np.array([1.0, 1.0]), 0.7, e, t=0.2 * k)
print(f"  0.5 s despues: age_s={reciente.get('age_s')} | 20 s despues: age_s={viejo.get('age_s')} | "
      f"sin 'now': {'age_s' in sin_reloj} | modo span: {'age_s' in span.candidates(preliminary=True, now=29.8)[0]}")
assert abs(reciente["age_s"] - 0.5) < 0.01 and abs(viejo["age_s"] - 20.0) < 0.01, "la edad es now menos el ultimo avistamiento"
assert "age_s" not in sin_reloj, "sin el instante del reporte no hay edad que dar"
assert "age_s" not in span.candidates(preliminary=True, now=29.8)[0], "el modo span no cambia su reporte"

print()
print("=" * 64)
print("15. EVIDENCIA: el reporte dice cuanta falta para reportar, no solo cuanta hay")
print("=" * 64)
e = emb_de(31)

def con_miradas(n_looks, min_looks=20, modo="looks"):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity=modo,
                                report_min_looks=min_looks)
    for k in range(n_looks):
        ident.observe(k, 700, np.array([2.0, -1.0]) + RNG.normal(0, 0.3, size=2), 0.7,
                      e + 0.03 * RNG.normal(size=512), t=float(k))
    return ident.candidates(preliminary=True, now=float(n_looks))[0]

pocas = con_miradas(5)
muchas = con_miradas(20)
bar_baja = con_miradas(5, min_looks=10)
span = con_miradas(20, modo="span")
print(f"   5 de 20 miradas: evidence={pocas['evidence']} looks={pocas['looks']} "
      f"looks_min={pocas['looks_min']} maduro={pocas['mature']}")
print(f"  20 de 20 miradas: evidence={muchas['evidence']} looks={muchas['looks']} "
      f"looks_min={muchas['looks_min']} maduro={muchas['mature']}")
print(f"   5 de 10 miradas: evidence={bar_baja['evidence']} looks_min={bar_baja['looks_min']} "
      f"maduro={bar_baja['mature']}")
print(f"  modo span: {'evidence' in span}, {'looks_min' in span}")
assert pocas["evidence"] == 0.25 and pocas["looks"] == 5 and pocas["looks_min"] == 20,     "cinco de veinte miradas es un cuarto de la evidencia"
assert not pocas["mature"] and muchas["mature"],     "el contraste: con un cuarto no se reporta y con la barra llena si"
assert muchas["evidence"] == 1.0, "la barra llena es exactamente 1.0, no 1.05"
assert bar_baja["evidence"] == 0.5,     "la fraccion se mide contra el umbral vigente, no contra un 20 escrito a mano"
assert "evidence" not in span and "looks_min" not in span, "el modo span no cambia su reporte"
for caso in (con_miradas(3), pocas, con_miradas(19), muchas, con_miradas(40), bar_baja,
             con_miradas(10, min_looks=10)):
    assert (caso["evidence"] >= 1.0) == caso["mature"],         "la barra llena y el veredicto tienen que decir lo mismo, siempre"

print()
print("=" * 64)
print("16. DENSIDAD: en que fraccion de los cuadros con deteccion se vio al candidato")
print("=" * 64)
e_obj, e_esc = emb_de(41), emb_de(42)

def densidad(fps_cam, parpadea, segundos=20):
    """Twenty seconds of frames, every one of them carrying a detection of something.

    The target is seen in all of them, or, when it flickers, only during one second out of
    every four, which is the shape a static false positive has: present across a long life and
    absent from most of it. The density is read inside the candidate's own span, so a target
    tracked perfectly while it is in view scores 1 however briefly it was there.
    """
    ident = IncrementalIdentity(fusion_radius_m=2.0, fps=fps_cam)
    n_frames = int(segundos * fps_cam)
    for k in range(n_frames):
        sello = k / fps_cam
        ident.observe(k, 801, np.array([50.0, 50.0]), 0.6, e_esc, t=sello)
        if (not parpadea) or int(sello) % 4 == 0:
            ident.observe(k, 800, np.array([2.0, -1.0]) + RNG.normal(0, 0.2, size=2), 0.7,
                          e_obj + 0.03 * RNG.normal(size=512), t=sello)
    return [c for c in ident.candidates(preliminary=True, now=float(segundos))
            if abs(c["x"] - 2.0) < 3.0][0]

denso = densidad(2.0, parpadea=False)
tirones = densidad(2.0, parpadea=True)
rapido = densidad(4.0, parpadea=True)
for nom, c in (("visto siempre     ", denso), ("parpadea a 2 FPS  ", tirones),
               ("parpadea a 4 FPS  ", rapido)):
    print(f"  {nom}: duty={c['duty']:<6} looks={c['looks']:<4} n_obs={c['n_obs']:<4} "
          f"obs/mirada={c['n_obs'] / c['looks']:.1f}")
assert denso["duty"] == 1.0, "visto en cada cuadro de su vida es densidad 1"
assert tirones["duty"] < 0.4,     "quien parpadea tiene que caer del lado de los fantasmas, cuyo techo medido es 0.397"
assert abs(rapido["duty"] - tirones["duty"]) < 0.02,     "la densidad no puede depender de la cadencia de la camara"
assert rapido["n_obs"] / rapido["looks"] > 1.8 * (tirones["n_obs"] / tirones["looks"]),     "el contraste: avistamientos por mirada SI dependen de la cadencia, y por eso no se usan"
assert "duty" not in span, "el modo span no cambia su reporte"

print()
print("TODO OK")
