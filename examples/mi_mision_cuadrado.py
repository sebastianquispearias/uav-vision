"""
EL COMPORTAMIENTO: volar un cuadrado mirando el suelo.

Este archivo es EL MISMO para simulacion y para el dron real.
No lo tocas al pasar de uno al otro.

Lo unico que cambia entre los dos mundos son los DOS ARGUMENTOS de
construir(): la camara y de donde sale el rumbo. Por eso construir()
recibe esas dos cosas en vez de decidirlas el.

    simulacion ->  examples/correr_sim.py
    dron real  ->  examples/correr_uav.md

The path juggling at the top is so the imports work with ANY interpreter, whether or not
anything was installed with pip.

With x_axis_degrees = 0 the frame is x = north, y = east, so the square is north, east, south
and back to the start.

THE MISSION IS TWO PARTS. First the VISION part, configured with whatever camera is given.
Then the FLIGHT part, added on top by inheritance. Calling the vision layer's initialize is
MANDATORY: it starts every timer and the identity layer, and leaving it out makes the drone fly
BLIND without raising anything at all.
"""

import os
import sys

_AQUI = os.path.dirname(os.path.abspath(__file__))
_UAV_VISION = os.path.dirname(_AQUI)
_LAC = os.path.dirname(_UAV_VISION)
for _p in (_UAV_VISION, os.path.join(_LAC, "gradys-embedded")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gradys_embedded.protocol.plugin.mission_mobility import (
    MissionMobilityConfiguration,
    MissionMobilityPlugin,
)

from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol

LADO = 60.0
ALTURA = 30.0

CUADRADO = [
    (LADO,  0.0,   ALTURA),
    (LADO,  LADO,  ALTURA),
    (0.0,   LADO,  ALTURA),
    (0.0,   0.0,   ALTURA),
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

    class CuadradoConVision(Vision):

        def initialize(self):
            super().initialize()

            self.vuelo = MissionMobilityPlugin(
                self,
                MissionMobilityConfiguration(speed=velocidad, tolerance=1.0),
            )
            self.vuelo.start_mission(CUADRADO)

    return CuadradoConVision
