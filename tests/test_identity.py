"""
Empirical gates for IncrementalIdentity, each on a synthetic scene with known ground truth:

  1. Static: one standing person under projection noise yields one candidate near the truth.
  2. Mobile: a walker is classified as mobile and its reported position must beat the lagging
     naive estimate (median of the recent window).
  3. Co-occurrence veto: two people seen in the same frames stay two candidates.
  4. Twin override: duplicate boxes of one person merge into one candidate.

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
SIGMA = 1.2          # ground projection noise, m


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
for f in range(300):                                   # 60 s a 5 Hz
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
    real = np.array([-10.0 + 1.0 * t, 4.0])            # 1 m/s hacia el este
    pos_final = real
    ident.observe(f, 20, real + ruido(), 0.6, e2 + 0.05 * RNG.normal(size=512))
c = ident.candidates()
assert len(c) == 1 and c[0]["mobile"], f"no salio MOVIL: {c}"
err_fit = math.hypot(c[0]["x"] - pos_final[0], c[0]["y"] - pos_final[1])
# el estimador ingenuo que reemplazamos: mediana de la ventana reciente
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
    if f % 2 == 0:      # el detector duplica la caja la mitad del tiempo
        ident.observe(f, 41, P + ruido(), 0.5, ep + 0.03 * RNG.normal(size=512))
c = ident.candidates()
print(f"  candidates: {len(c)} con n_obs={[p['n_obs'] for p in c]}")
assert len(c) == 1, "el gemelo no se fusiono (regla A-1 rota)"

print()
print("=" * 64)
print("5. PASADA CORTA: nada mature, pero SI un preliminar que verificar")
print("=" * 64)
# The sweep case, measured on flight 3: a pass of 30 s never matures a candidate. A track
# forms and then the drone is gone. Without preliminaries the system says nothing at all
# about a person it tracked perfectly well for half a minute.
# This section documents the span rule, the one that measured it; the default is now maturity by
# looks, under which a 20 s pass does mature (section 7). Asked for explicitly so it keeps testing it.
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, report_dur_s=36.0, maturity="span")
P = np.array([4.0, -2.0])
ep = emb_de(9)
n_pasada = int(20 * FPS)          # 20 s of pass, well under the 36 s report bar
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

# And the guarantee that keeps preliminaries honest: once evidence accumulates, the same
# candidate matures, and the two calls agree.
for f in range(n_pasada, int(60 * FPS)):
    ident.observe(f, 70, P + ruido(), 0.6, ep + 0.03 * RNG.normal(size=512))
maduros = ident.candidates()
assert len(maduros) == 1 and maduros[0]["mature"] is True,     "con evidencia suficiente el preliminar tiene que madurar"
print(f"  tras 60 s: mature=True, n_obs={maduros[0]['n_obs']}")

print()
print("=" * 64)
print("6. FRAGMENTOS: refuerzan un candidato existente, nunca crean uno")
print("=" * 64)
# A tracker that keeps losing a target leaves pieces shorter than track_dur_s. With
# reinforce_with_fragments they may join the candidate a lasting track opened, by the same
# rules as any merge. Three contrasts, each of which fails if the rule is wrong: the same
# fragment joins only with the option on; a fragment alone opens nothing even with it on;
# a fragment in the right place with another appearance stays out.


def escena(refuerzo, con_pista_larga=True):
    # A fragment is defined against the span rule's 8.6 s track gate; under looks, four seconds
    # of sightings are already a track. So this section asks for span explicitly.
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity="span",
                                reinforce_with_fragments=refuerzo)
    P, Q = np.array([0.0, 0.0]), np.array([30.0, 30.0])
    e_p, e_otro = emb_de(11), emb_de(12)
    if con_pista_larga:
        for f in range(300):                                 # 60 s: a lasting track
            ident.observe(f, 80, P + ruido(), 0.7, e_p + 0.03 * RNG.normal(size=512))
    for f in range(300, 320):                                # 4 s, same person, new id
        ident.observe(f, 81, P + ruido(), 0.7, e_p + 0.03 * RNG.normal(size=512))
        ident.observe(f, 82, Q + ruido(), 0.7, e_p + 0.03 * RNG.normal(size=512))
    for f in range(320, 340):                                # 4 s, right place, other look
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
# maturity="looks" counts independent looks instead of the time between first and last
# sighting. Each contrast runs the same observations through both modes, so it fails if the
# modes stop behaving differently where they are supposed to.


def seguido(maturity, segundos, hz=FPS, t0=0.0, tid=90, **kw):
    kw.setdefault("report_min_looks", 5)      # this section was written for 5; the default is checked below
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

# The default itself: 20 looks. Ten seconds seen are not enough, twenty-five are.
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
# A target in frame is detected in 4.6-38 % of frames on flight 3, in bursts. Here a boat-like
# target crosses at 6 m/s and is seen in bursts of five frames with long gaps. Fitting against
# the observation index treats every sighting as one equal step; fitting against time does not.
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
# A heading error moves the impact by range times angle. The same target, seen for the same time,
# must carry a wider margin from 90 m than from 12 m; at the range the 2.4 m floor was measured at,
# the model must give back the radius every earlier report carried.
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
# A boat patrolling x in [-25, 25] at 6 m/s, seen in bursts. Over its whole life its median is the
# middle of the patrol and a line through its last quarter crosses the turns. The same sightings
# through both modes: span reports it far from where it is, looks must report it close.


def patrulla(maturity, v=6.0, quieto=False):
    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS, maturity=maturity)
    rng = np.random.default_rng(5)
    e = emb_de(51)
    k, real = 0, None
    for t0 in np.arange(0.0, 90.0, 3.0):                 # a burst of 5 sightings every 3 s
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
# The report goes out on its own clock; the last sighting of a moving target is already old. A
# target at 6 m/s last seen 1.5 s before the report is 9 m past that sighting.
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e = emb_de(61)
vel = np.array([6.0, 0.0])
for k in range(200):                                     # 40 s at 5 Hz, straight line
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
# A target at 6 m/s turns back right after its last sighting. Four seconds later the report is 48 m
# off the extrapolated line's end; a radius frozen at the last sighting claims to hold it and does not.
ident = IncrementalIdentity(fusion_radius_m=3.5, fps=FPS)
e = emb_de(71)
vel = np.array([6.0, 0.0])
for k in range(200):
    ident.observe(k, 103, vel * (0.2 * k) + RNG.normal(0, 1.0, size=2), 0.7,
                  e + 0.03 * RNG.normal(size=512), t=0.2 * k)
t_ultimo = 0.2 * 199
ahora = t_ultimo + 4.0
real = vel * t_ultimo - vel * 4.0                        # it turned round and came back
fijo = ident.candidates(preliminary=True)[0]
vivo = ident.candidates(preliminary=True, now=ahora)[0]
d_vivo = float(np.hypot(vivo["x"] - real[0], vivo["y"] - real[1]))
print(f"  4 s despues de dar la vuelta: radio sin 'now' {fijo['radius_m']} m | con 'now' {vivo['radius_m']} m "
      f"| el blanco a {d_vivo:.1f} m del punto reportado")
assert vivo["radius_m"] >= fijo["radius_m"] + 6.0 * 4.0 - 0.5, "el margen tiene que crecer con rapidez x tiempo"
assert d_vivo <= vivo["radius_m"], "con el margen que crece, la verdad tiene que caer dentro"
assert d_vivo > fijo["radius_m"], "el contraste: el margen congelado no la contenia"
r_quieto = (quieto.candidates(preliminary=True)[0]["radius_m"], quieto.candidates(preliminary=True, now=500.0)[0]["radius_m"])
assert r_quieto[0] == r_quieto[1], "un blanco quieto no gana margen por pasar el tiempo"

print()
print("TODO OK")
