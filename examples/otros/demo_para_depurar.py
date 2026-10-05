"""
DEMO QUE CORRE EN TU LAPTOP Y SE PUEDE DEPURAR PASO A PASO.

Para que sirve: para ENTENDER el sistema viendolo funcionar, en vez de
leyendo diagramas. Pone un punto de interrupcion donde quieras y mira
que pasa de verdad.

COMO SE CORRE
    Abrir este archivo en VS Code y pulsar F5.
    O desde la terminal:  python examples/demo_para_depurar.py

No hace falta el dron, ni la Raspberry, ni la camara. Usa
SimulatedCamera, que inventa las detecciones con geometria pura.

DONDE PONER PUNTOS DE INTERRUPCION (clic a la izquierda del numero de
linea, sale un punto rojo). Los cuatro que ensenan mas:

    1. MiniRunner.schedule_timer   -> ves al protocolo PEDIR una alarma
    2. MiniRunner.send_communication_command -> ves lo que TRANSMITE
    3. la linea "protocolo.handle_timer(nombre)" -> el runner LLAMANDO
       al protocolo. Con F11 (Step Into) entras dentro de tu codigo.
    4. dentro de uav_vision/vision_protocol.py, en _see() -> ves como
       un pixel se convierte en un punto del suelo.

The path juggling at the top is so the imports work with ANY interpreter, whether or not
anything was installed with pip. Without it, pressing F5 in an editor can pick a different
Python and fail with "No module named 'uav_vision'".

THE RUNNER, DISTILLED TO THIRTY LINES
    On the Raspberry, the runner of gradys_embedded is a large program: it speaks HTTP to
    uav_api, drives the radio and serves a web panel. But what it essentially does to a protocol
    is only this: keep a clock, keep a diary of alarms, and call the protocol when one is due.
    Seeing that in thirty lines is the fastest way to understand what a runner is.

    The fake provider keeps the pending alarms with the nearest first, and whatever the protocol
    broadcast. Its mobility command raises ON PURPOSE: the vision protocol is observe-only, so
    if it ever tried to move the drone this blows up and the mistake is visible.

    The line that instantiates the protocol and hands it the provider is LITERALLY what the
    Raspberry's runner does.

THE CONFIGURATION is the same with_config() as the real mission. The only change is the camera:
SimulatedCamera here instead of OnboardCamera, with the person and the hovering drone placed by
hand, the yaw fixed instead of read from UavApiYaw, and short maturity thresholds so the demo
reports something within its own runtime. The JPEG crop is unreadable here and is left out.
"""

import heapq
import json
import os
import sys

_AQUI = os.path.dirname(os.path.abspath(__file__))
_UAV_VISION = os.path.dirname(_AQUI)
_LAC = os.path.dirname(_UAV_VISION)
sys.path.insert(0, _UAV_VISION)
sys.path.insert(0, os.path.join(_LAC, "gradys-embedded"))

from gradys_embedded.protocol.interface import IProvider
from gradys_embedded.protocol.messages.telemetry import Telemetry

from uav_vision.camera import SimulatedCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol


class MiniRunner(IProvider):

    def __init__(self):
        self.reloj = 0.0
        self.agenda = []
        self.transmitido = []


    def schedule_timer(self, timer, timestamp):
        """El protocolo pide: 'despertame en el instante X'."""
        print("    [runner] el protocolo pide la alarma '%s' para t=%.2f"
              % (timer, timestamp))
        heapq.heappush(self.agenda, (timestamp, timer))

    def cancel_timer(self, timer):
        self.agenda = [(t, n) for (t, n) in self.agenda if n != timer]
        heapq.heapify(self.agenda)

    def current_time(self):
        return self.reloj

    def get_id(self):
        return 1

    def send_mobility_command(self, command):
        raise AssertionError("observe-only violado: el protocolo quiso mover el dron")

    def send_communication_command(self, command):
        """El protocolo quiere transmitir por radio. Lo guardamos."""
        self.transmitido.append(json.loads(command.message))
        print("    [runner] el protocolo TRANSMITE (mensaje %d)"
              % len(self.transmitido))


    def volar(self, protocolo, segundos, posicion_dron):
        proxima_telemetria = 0.0
        while self.reloj <= segundos:
            hay_alarma = self.agenda and self.agenda[0][0] <= proxima_telemetria
            if hay_alarma:
                self.reloj, nombre = heapq.heappop(self.agenda)
                protocolo.handle_timer(nombre)
            else:
                self.reloj = proxima_telemetria
                protocolo.handle_telemetry(
                    Telemetry(current_position=posicion_dron))
                proxima_telemetria += 0.5



PERSONA = (40.0, -12.0, 0.0)
DRON    = (25.0, -12.0, 35.0)

ProtocoloDemo = VisionProtocol.with_config(
    camera=SimulatedCamera(target=PERSONA, pitch_deg=-55.0),
    pitch_deg=-55.0,
    yaw_source=lambda: 90.0,
    see_period_s=0.25,
    report_period_s=2.0,
    identity=IncrementalIdentity(
        fusion_radius_m=1.0,
        fps=4.0,
        track_dur_s=2.0,
        report_dur_s=6.0,
    ),
)


def main():
    print("=" * 62)
    print("La persona esta en (%.1f, %.1f)" % (PERSONA[0], PERSONA[1]))
    print("El dron esta en   (%.1f, %.1f) a %.0f m de altura" % DRON)
    print("=" * 62)

    runner = MiniRunner()

    protocolo = ProtocoloDemo.instantiate(runner)

    print("\n--- initialize(): el protocolo se prepara y pide sus alarmas ---")
    protocolo.initialize()

    print("\n--- 10 segundos de vuelo simulado ---")
    runner.volar(protocolo, segundos=10.0, posicion_dron=DRON)

    print("\n" + "=" * 62)
    print("TRANSMITIO %d MENSAJES" % len(runner.transmitido))
    print("=" * 62)

    for i, m in enumerate(runner.transmitido, 1):
        pois = m.get("pois", [])
        for p in pois:
            p.pop("crop", None)
            p.pop("recorte", None)
        print("\nMENSAJE %d  (t = %.1f s)" % (i, m.get("time", 0.0)))
        print("   frames vistos: %s" % m.get("frames_seen"))
        print("   latido: %s" % m.get("latido"))
        if not pois:
            print("   pois: [] <- todavia no hay evidencia suficiente")
        else:
            for p in pois:
                print("   POI en (%.2f, %.2f)  con %s observaciones"
                      % (p["x"], p["y"], p.get("n_obs")))

    print("\n" + "=" * 62)
    print("QUE ACABAS DE VER")
    print("=" * 62)
    print("1. El protocolo NUNCA corre solo: el runner lo llama.")
    print("2. Al principio calla y manda 'latido: true' -- ya vio a la")
    print("   persona, pero no tiene evidencia suficiente para afirmarlo.")
    print("3. Al final reporta el punto, cerca de donde esta la persona.")
    print("   Aca el error es de centimetros porque solo hay ruido de")
    print("   pixel. En vuelo se suman GPS y rumbo, y da ~2,3 m.")
    print("4. send_mobility_command tiene una trampa que revienta si el")
    print("   protocolo intenta mover el dron. No salto: es observe-only,")
    print("   demostrado por ejecucion y no por promesa.")


if __name__ == "__main__":
    main()
