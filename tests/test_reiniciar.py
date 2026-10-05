"""A clear that travels, because the one that did not was lying about the system.

The station's clear button hides pins on the operator's page. On 2026-10-02 the operator pressed
it, the board emptied, he reloaded the page, and everything came back. Nothing was broken: the
candidates do not live on the station, they live in each aircraft's identity layer, which forgets
nothing on purpose ("a candidate is never forgotten, so a target lost a minute ago is still
reported"). The station only ever held a copy, and a reload asked for a fresh one.

So the button was telling the truth about the screen and a lie about the system. This is the gate
for the order that reaches the aircraft, and for the one thing it must NOT undo.

Run: python tests/test_reiniciar.py

THE LAYER HAS TO STAY USABLE and not end up broken. That is what separates a reset from a
shutdown: the drone keeps flying and the next thing it sees has to come in as if nothing had
happened.

THE CONTRAST THAT JUSTIFIES THE WHOLE METHOD: without the order, TIME CLEANS NOTHING, and that
is deliberate, because a target lost a minute ago is still reported. If that section stopped
passing, the layer would have started forgetting on its own and olvidar_todo() would no longer
be the only way.

The message name is checked as TEXT, because the protocol cannot be imported without the GrADyS
runtime, which is not on the laptop. Two copies of a name is how a drone and its station stop
understanding each other in silence: the station sends, the drone ignores, and nobody sees an
error.

THE REFUSALS SURVIVE. A refusal is a judgement about the world and it cost a person's attention;
the board is just how things are being drawn right now. Clearing both with the same button would
make the drone report again exactly the point somebody already said was not a person.
"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))

from uav_vision.identity import IncrementalIdentity


def poblar(idt, n=40, x=10.0, tid=1):
    for f in range(n):
        idt.observe(f, tid, (x, 10.0), 0.9, t=f / 4.0, cls='person')


print('=' * 70)
print('1. OLVIDAR DE VERDAD, NO DEJAR DE DIBUJAR')
print('=' * 70)
idt = IncrementalIdentity(fusion_radius_m=2.0, fps=4.0, track_dur_s=1.0,
                          mobile_dur_s=10.0, report_dur_s=5.0, report_min_looks=4)
poblar(idt)
poblar(idt, x=30.0, tid=2)
antes = len(idt.candidates(now=12.0))
print('  dos objetivos observados 40 cuadros cada uno -> %d candidatos' % antes)
assert antes == 2, 'el montaje no produjo los dos candidatos: %d' % antes

idt.olvidar_todo()
despues = len(idt.candidates(now=12.0))
print('  despues de olvidar_todo()                   -> %d candidatos' % despues)
assert despues == 0, 'quedaron %d candidatos despues de olvidar: no se olvido nada' % despues

poblar(idt, x=50.0, tid=7)
reusada = idt.candidates(now=24.0)
print('  y un objetivo nuevo despues del reinicio    -> %d candidato en x=%.1f'
      % (len(reusada), reusada[0]['x'] if reusada else float('nan')))
assert len(reusada) == 1, 'la capa quedo inutilizable tras el reinicio: %r' % reusada
assert abs(reusada[0]['x'] - 50.0) < 1.0, 'el candidato nuevo trae posicion vieja: %r' % reusada[0]

print()
print('=' * 70)
print('2. UN CANDIDATO VIEJO NO SE BORRA SOLO: POR ESO HACIA FALTA LA ORDEN')
print('=' * 70)
idt2 = IncrementalIdentity(fusion_radius_m=2.0, fps=4.0, track_dur_s=1.0,
                           mobile_dur_s=10.0, report_dur_s=5.0, report_min_looks=4)
poblar(idt2)
for ahora in (12.0, 600.0, 3600.0):
    n = len(idt2.candidates(now=ahora))
    print('  ultimo avistamiento en t=9.75 s, preguntando en t=%7.1f s -> %d candidato(s)'
          % (ahora, n))
    assert n == 1, 'el candidato desaparecio solo en t=%.1f, y no deberia' % ahora
print('  -> ni una hora despues. El tiempo no limpia: la unica via es la orden del operador')

print()
print('=' * 70)
print('3. LA ESTACION Y EL PROTOCOLO SE PONEN DE ACUERDO EN EL NOMBRE')
print('=' * 70)
proto = open(os.path.join(AQUI, '..', 'uav_vision', 'vision_protocol.py'), encoding='utf-8').read()
gs = open(os.path.join(AQUI, '..', 'scripts', 'banco_embedded', 'gs_mapa.py'), encoding='utf-8').read()
for quien, texto in (('el protocolo', proto), ('la estacion', gs)):
    assert "'vision_reiniciar'" in texto or '"vision_reiniciar"' in texto, \
        '%s no nombra vision_reiniciar' % quien
    print('  %-14s nombra vision_reiniciar' % quien)
assert 'def reiniciar' in proto, 'el protocolo no tiene el metodo que atiende la orden'
assert 'def olvidar_todo' in open(os.path.join(AQUI, '..', 'uav_vision', 'identity.py'),
                                 encoding='utf-8').read(), 'identity no sabe olvidar'
assert "/reiniciar" in gs and 'btn-reiniciar' in gs, 'la estacion no expone la orden ni su boton'
print('  la estacion tiene la ruta y el boton, el protocolo el metodo, identity el olvido')

print()
print('=' * 70)
print('4. EL REINICIO NO BORRA LOS VEREDICTOS DEL OPERADOR')
print('=' * 70)
assert 'self._descartados' not in proto.split('def reiniciar')[1].split('def descartar')[0] \
    .replace('len(self._descartados)', ''), \
    'reiniciar() toca la lista de descartes; los veredictos tienen que sobrevivir'
print('  reiniciar() solo cuenta los descartes, no los toca')
assert "Los \"no es\" NO se tocan" in gs, 'la estacion no deja dicho que los veredictos sobreviven'
print('  y la estacion lo dice en su propio log, para que el operador no tenga que suponerlo')

print()
print('test_reiniciar OK')
