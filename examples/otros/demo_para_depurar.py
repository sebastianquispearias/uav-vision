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
"""

import os
import sys
import json
import heapq

# Que los imports funcionen con CUALQUIER interprete, este o no instalado
# nada con pip. Sin esto, F5 en VS Code puede usar otro Python y fallar con
# "No module named 'uav_vision'".
_AQUI = os.path.dirname(os.path.abspath(__file__))          # .../uav_vision/examples
_UAV_VISION = os.path.dirname(_AQUI)                        # .../uav_vision   (el repo)
_LAC = os.path.dirname(_UAV_VISION)                         # .../lac
sys.path.insert(0, _UAV_VISION)                             # para uav_vision
sys.path.insert(0, os.path.join(_LAC, "gradys-embedded"))   # para gradys_embedded

from gradys_embedded.protocol.interface import IProvider          # del grupo
from gradys_embedded.protocol.messages.telemetry import Telemetry  # del grupo
from uav_vision.camera import SimulatedCamera                      # tuyo
from uav_vision.identity import IncrementalIdentity                # tuyo
from uav_vision.vision_protocol import VisionProtocol              # tuyo


# ===================================================================
# EL RUNNER, DESTILADO A 30 LINEAS
#
# En la Raspberry, el runner de gradys_embedded es un programa grande:
# habla HTTP con uav_api, maneja la radio, expone un panel web. Pero lo
# esencial que le hace a tu protocolo es SOLO esto: llevar un reloj,
# una agenda de alarmas, y llamarte cuando toca.
#
# Ver esto en 30 lineas es la forma mas rapida de entender que es un
# runner.
# ===================================================================

class MiniRunner(IProvider):

    def __init__(self):
        self.reloj = 0.0
        self.agenda = []        # alarmas pendientes, la mas cercana primero
        self.transmitido = []   # lo que el protocolo emitio por radio

    # ---- lo que el protocolo le PIDE al mundo ----

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
        return 1                # el numero de este dron

    def send_mobility_command(self, command):
        # TRAMPA A PROPOSITO: el protocolo de vision es observe-only.
        # Si alguna vez intentara mover el dron, esto revienta y lo ves.
        raise AssertionError("observe-only violado: el protocolo quiso mover el dron")

    def send_communication_command(self, command):
        """El protocolo quiere transmitir por radio. Lo guardamos."""
        self.transmitido.append(json.loads(command.message))
        print("    [runner] el protocolo TRANSMITE (mensaje %d)"
              % len(self.transmitido))

    # ---- el bucle: entregar posicion y disparar alarmas ----

    def volar(self, protocolo, segundos, posicion_dron):
        proxima_telemetria = 0.0
        while self.reloj <= segundos:
            hay_alarma = self.agenda and self.agenda[0][0] <= proxima_telemetria
            if hay_alarma:
                self.reloj, nombre = heapq.heappop(self.agenda)
                protocolo.handle_timer(nombre)          # <-- EL RUNNER TE LLAMA
            else:
                self.reloj = proxima_telemetria
                protocolo.handle_telemetry(             # <-- Y TAMBIEN AQUI
                    Telemetry(current_position=posicion_dron))
                proxima_telemetria += 0.5


# ===================================================================
# LA CONFIGURACION
# Es el mismo with_config() de la mision real. Solo cambia la camara:
# aca es SimulatedCamera en vez de OnboardCamera.
# ===================================================================

PERSONA = (40.0, -12.0, 0.0)      # donde esta la persona, en metros
DRON    = (25.0, -12.0, 35.0)     # donde esta el dron, quieto, a 35 m

ProtocoloDemo = VisionProtocol.with_config(
    camera=SimulatedCamera(target=PERSONA, pitch_deg=-55.0),
    pitch_deg=-55.0,
    yaw_source=lambda: 90.0,      # mirando al Este. En el dron esto es UavApiYaw
    see_period_s=0.25,            # mirar 4 veces por segundo
    report_period_s=2.0,          # transmitir cada 2 segundos
    identity=IncrementalIdentity(
        fusion_radius_m=1.0,
        fps=4.0,
        track_dur_s=2.0,
        report_dur_s=6.0,         # 6 s de evidencia para reportar algo
    ),
)


def main():
    print("=" * 62)
    print("La persona esta en (%.1f, %.1f)" % (PERSONA[0], PERSONA[1]))
    print("El dron esta en   (%.1f, %.1f) a %.0f m de altura" % DRON)
    print("=" * 62)

    runner = MiniRunner()

    # Esta linea es TEXTUALMENTE lo que hace el runner de la Raspberry:
    # crear una instancia del protocolo y darle el proveedor.
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
            p.pop("crop", None)      # el recorte JPEG es ilegible aqui
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
