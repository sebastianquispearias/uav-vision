"""
EL COMPORTAMIENTO: volar un cuadrado mirando el suelo.

Este archivo es EL MISMO para simulacion y para el dron real.
No lo tocas al pasar de uno al otro.

Lo unico que cambia entre los dos mundos son los DOS ARGUMENTOS de
construir(): la camara y de donde sale el rumbo. Por eso construir()
recibe esas dos cosas en vez de decidirlas el.

    simulacion ->  examples/correr_sim.py
    dron real  ->  examples/correr_uav.md
"""

import os
import sys

# Para que los imports funcionen con cualquier interprete, este o no
# instalado nada con pip.
_AQUI = os.path.dirname(os.path.abspath(__file__))
_UAV_VISION = os.path.dirname(_AQUI)
_LAC = os.path.dirname(_UAV_VISION)
for _p in (_UAV_VISION, os.path.join(_LAC, "gradys-embedded")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gradys_embedded.protocol.plugin.mission_mobility import (
    MissionMobilityPlugin,
    MissionMobilityConfiguration,
)
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol


LADO = 60.0        # metros de lado del cuadrado
ALTURA = 30.0      # metros de altura de vuelo

# Con x_axis_degrees = 0:  x = norte,  y = este.
CUADRADO = [
    (LADO,  0.0,   ALTURA),    # 60 m al norte
    (LADO,  LADO,  ALTURA),    # 60 m al este
    (0.0,   LADO,  ALTURA),    # 60 m al sur
    (0.0,   0.0,   ALTURA),    # y de vuelta al inicio
]


def construir(camara, yaw_source, velocidad=5.0):
    """Devuelve la CLASE de protocolo lista para que la corra un anfitrion.

    No devuelve un objeto: devuelve una clase. Tanto el simulador como el
    runner del dron esperan una clase y ellos crean la instancia.

    Args:
        camara: cualquier cosa que cumpla el contrato ver_alvo(pos, yaw).
                En simulacion, SimulatedCamera. En el dron, OnboardCamera.
        yaw_source: funcion sin argumentos que devuelve el rumbo en grados.
                En simulacion, algo fijo. En el dron, UavApiYaw().
        velocidad: m/s a los que recorrer el cuadrado.
    """

    # 1) La parte de VISION: se configura con la camara que nos den.
    Vision = VisionProtocol.with_config(
        camera=camara,
        yaw_source=yaw_source,
        pitch_deg=-55.0,
        see_period_s=1.0 / 3.0,
        report_period_s=2.0,
        identity=IncrementalIdentity(
            fusion_radius_m=3.5,
            fps=3.0,
            report_dur_s=6.0,
        ),
    )

    # 2) La parte de VUELO: se le agrega encima, heredando.
    class CuadradoConVision(Vision):

        def initialize(self):
            # OBLIGATORIO. Arranca toda la vision (timers, identidad...).
            # Si falta, el dron vuela CIEGO y no salta ningun error.
            super().initialize()

            self.vuelo = MissionMobilityPlugin(
                self,
                MissionMobilityConfiguration(speed=velocidad, tolerance=1.0),
            )
            self.vuelo.start_mission(CUADRADO)

    return CuadradoConVision
