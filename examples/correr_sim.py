"""
ARRANCAR LA MISION EN SIMULACION.

    python examples/correr_sim.py

No hace falta dron, ni Raspberry, ni camara. El dron vuela un cuadrado
sobre una persona simulada y va reportando donde la ve.

Fijate en la estructura: este archivo NO tiene comportamiento. Solo monta
el escenario y arranca. El comportamiento esta en mi_mision_cuadrado.py,
que es el mismo archivo que se usa en el dron real.

AQUI es donde vive el main(). El protocolo nunca lleva uno.
"""

import os
import sys

_AQUI = os.path.dirname(os.path.abspath(__file__))
_UAV_VISION = os.path.dirname(_AQUI)
_LAC = os.path.dirname(_UAV_VISION)
for _p in (_UAV_VISION, os.path.join(_LAC, "gradys-embedded")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gradysim.simulator.handler.communication import (
    CommunicationHandler, CommunicationMedium)
from gradysim.simulator.handler.mobility import (
    MobilityHandler, MobilityConfiguration)
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import (
    SimulationBuilder, SimulationConfiguration)

from uav_vision.camera import SimulatedCamera
from mi_mision_cuadrado import construir, ALTURA


PERSONA = (30.0, 30.0, 0.0)      # en el centro del cuadrado
VELOCIDAD = 5.0                  # m/s
DURACION = 120.0                 # segundos de simulacion


# ===================================================================
# UNA ESTACION DE TIERRA DE JUGUETE, PARA VER LO QUE EL DRON TRANSMITE.
# Es otro protocolo: la misma interfaz de cinco metodos. En la realidad
# esto es gs_mapa.py corriendo en la laptop, que pinta un mapa.
# ===================================================================

class EstacionTierra:
    """Escucha la radio e imprime lo que el dron reporta."""

    @classmethod
    def instantiate(cls, provider):
        obj = cls()
        obj.provider = provider
        return obj

    def initialize(self):
        self.recibidos = 0

    def handle_packet(self, message):
        import json
        self.recibidos += 1
        m = json.loads(message)
        pois = m.get("pois", [])
        t = m.get("time", 0.0)
        if not pois:
            print("  t=%6.1f s   latido: no veo a nadie todavia" % t)
        else:
            for p in pois:
                print("  t=%6.1f s   PERSONA en (%.1f, %.1f)  con %d observaciones"
                      % (t, p["x"], p["y"], p.get("n_obs", 0)))

    def handle_telemetry(self, telemetry):
        pass

    def handle_timer(self, timer):
        pass

    def finish(self):
        print("\n  La estacion de tierra recibio %d mensajes." % self.recibidos)


def main():
    print("=" * 62)
    print("  Simulacion: el dron vuela un cuadrado y busca a una persona")
    print("  Persona en (%.0f, %.0f). Cuadrado de 60 m de lado, a %.0f m."
          % (PERSONA[0], PERSONA[1], ALTURA))
    print("=" * 62)

    # ---- LO UNICO QUE CAMBIA RESPECTO DEL DRON REAL: estos dos ----
    Protocolo = construir(
        camara=SimulatedCamera(target=PERSONA, pitch_deg=-55.0),
        yaw_source=lambda: 0.0,
        velocidad=VELOCIDAD,
    )
    # ---------------------------------------------------------------

    config = SimulationConfiguration(duration=DURACION, execution_logging=False)
    builder = SimulationBuilder(config)

    # Los "handlers" son los servicios que el simulador ofrece: relojes,
    # movimiento y radio. En el dron real esos servicios los da el runner.
    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler(
        MobilityConfiguration(default_speed=VELOCIDAD)))
    builder.add_handler(CommunicationHandler(
        CommunicationMedium(transmission_range=500)))

    # add_node recibe la CLASE, no una instancia: el simulador crea el objeto.
    builder.add_node(EstacionTierra, (0.0, 0.0, 0.0))
    builder.add_node(Protocolo, (0.0, 0.0, ALTURA))

    print("\n--- lo que la estacion de tierra va recibiendo ---")

    builder.build().start_simulation()

    print("\n" + "=" * 62)
    print("  Simulacion terminada.")
    print("  Los mensajes 'vision_poi' de arriba son lo que el dron habria")
    print("  transmitido a tierra. Cada 'pois' con coordenadas es una")
    print("  persona que el sistema afirma haber encontrado.")
    print("=" * 62)


if __name__ == "__main__":
    main()
