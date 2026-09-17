"""
Gate for what the station does with a point CLIP calls "probably not a person".

By default it demotes and never removes: hiding a contact the operator never saw is their decision,
and CLIP is wrong often enough that silent removal would lose people. With --clip-descarta it removes
instead, which on the 02ago flight dropped the bag and the cone and kept all five people, taking the
phantoms from two to one. Both behaviours have to be pinned, because the difference between them is a
policy about who decides, not a tuning constant.

The demoting order matters as much as the filtering: a doubted point must go last without dragging a
point that has no score at all down with it, since no score means nobody looked, not that it is doubtful.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "banco_embedded"))
import gs_mapa as G

ahora = time.time()
G.ESTADO['drones'] = {1: {'t': ahora}}
G.ESTADO['origen'] = (-22.978029946, -43.23214256266666)
G.ESTADO['pois_por_dron'] = {1: [
    {'x': 0.0, 'y': 6.0, 'n_obs': 400, 'conf': 0.8},                          # la persona, sin puntuar
    {'x': 3.0, 'y': 1.0, 'n_obs': 200, 'conf': 0.7, 'clip_no_persona': True},  # la bolsa
    {'x': 9.0, 'y': 5.0, 'n_obs': 50, 'conf': 0.6, 'clip': 2.4},               # puntuada y aprobada
]}

G.CLIP_DESCARTA = False
sin = G.pois_vigentes(ahora)
assert len(sin) == 3, "por defecto no puede sacar nada: quedaron %d de 3" % len(sin)
assert bool(sin[-1].get('clip_no_persona')), "el dudoso no quedo ultimo en la cola"
assert not sin[0].get('clip_no_persona'), "degrado a uno que no era dudoso"
print("  por defecto: los 3 siguen, el dudoso al final de la cola")

G.CLIP_DESCARTA = True
con = G.pois_vigentes(ahora)
assert len(con) == 2, "con --clip-descarta deberian quedar 2, quedaron %d" % len(con)
assert all(not q.get('clip_no_persona') for q in con), "dejo pasar al que CLIP rechazo"
assert any(abs(q['x'] - 0.0) < 1e-6 for q in con), "se llevo puesta a la persona sin puntuar"
assert any(abs(q['x'] - 9.0) < 1e-6 for q in con), "se llevo puesta a la que CLIP aprobo"
print("  con --clip-descarta: queda %d de 3, sobreviven la persona sin puntuar y la aprobada" % len(con))

# a point from a drone that went quiet is gone whatever CLIP said: the filter must not resurrect it
G.ESTADO['drones'] = {1: {'t': ahora - G.DRON_CALLADO_S - 10}}
assert not G.pois_vigentes(ahora), "mostro puntos de un dron que dejo de hablar"
print("  un dron callado no deja puntos vivos, con filtro o sin el")

G.CLIP_DESCARTA = False
print("TODO OK")
