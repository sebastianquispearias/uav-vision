"""
EJEMPLO: "el dron va 100 m al norte, luego 100 m al oeste, y aterriza"

Esto es lo que tendria que escribir una persona nueva del grupo.
Todo lo de aqui esta verificado contra el codigo real de gradys-embedded.

------------------------------------------------------------------
LO PRIMERO QUE HAY QUE SABER: NO es un archivo de comandos.

Vos imaginabas algo asi:

    start
    goto 100 north
    goto 100 west
    aterrizar

Eso NO existe en GrADyS. Lo que se escribe es una CLASE de Python con
cinco metodos que el runner llama cuando pasan cosas. La clase no se
ejecuta sola de arriba a abajo: se queda quieta esperando a que la
llamen.

Los cinco metodos (estan en gradys_embedded/protocol/interface.py:88):

    initialize()                 el runner te llama UNA vez, al arrancar
    handle_telemetry(telemetry)  te llama cada vez que llega la posicion
    handle_timer(timer)          te llama cuando vence una alarma tuya
    handle_packet(message)       te llama cuando llega un mensaje de otro dron
    finish()                     te llama una vez, al terminar

Los cinco son obligatorios. Si no usas alguno, lo dejas vacio con "pass".
------------------------------------------------------------------

MissionMobilityPlugin is a helper that ALREADY COMES with gradys-embedded, in
gradys_embedded/protocol/plugin/mission_mobility.py. Give it a list of points and it goes from
one to the next: it watches the telemetry, sees whether you arrived, and sends the next one. It
hooks itself onto the telemetry, so nothing else has to.

THE POINTS ARE METRES, not lat/lon, and the frame is fixed by whoever loads the mission through
x_axis_degrees. With x_axis_degrees = 0, verified in protocol/position.py, x is north and y is
east, so "100 m west" is y = -100.

There is NO third point for landing; see the note at the bottom of the file.
"""

from gradys_embedded.protocol.interface import IProtocol
from gradys_embedded.protocol.messages.telemetry import Telemetry
from gradys_embedded.protocol.plugin.mission_mobility import (
    MissionMobilityConfiguration,
    MissionMobilityPlugin,
)

ALTURA = 30.0


class MisionCienMetros(IProtocol):

    def initialize(self):
        self.vuelo = MissionMobilityPlugin(self, MissionMobilityConfiguration())

        self.vuelo.start_mission([
            (100.0,    0.0, ALTURA),
            (100.0, -100.0, ALTURA),
        ])


    def handle_telemetry(self, telemetry: Telemetry):
        pass

    def handle_timer(self, timer: str):
        pass

    def handle_packet(self, message: str):
        pass

    def finish(self):
        pass


"""
------------------------------------------------------------------
POR QUE NO HAY "ATERRIZAR"

Miralo vos mismo: gradys_embedded/protocol/messages/mobility.py tiene
SOLO TRES comandos de movimiento, y ninguno es aterrizar:

    GOTO_COORDS       ir a x, y, z
    GOTO_GEO_COORDS   ir a lat, lon, alt
    SET_SPEED         fijar velocidad

El aterrizaje no es algo que diga el protocolo. Lo hace el runner cuando
parás la mision. En runner/mission.py:466, dentro de stop():

    self._runner.request_return_to_launch()

O sea: cuando mandas POST /mission/stop, el runner primero apaga tu
protocolo (para que no siga mandando comandos y pelee con el retorno) y
despues pide RTL, que es "volve a casa y aterriza".

Aterrizar = parar la mision. No es una linea de tu codigo.
------------------------------------------------------------------
COMO SE CORRE ESTO

Se copia el archivo a ~/gradys_protocols/ en la Pi y despues:

    POST /mission/load   con "ejemplo_mision_100m:MisionCienMetros"
    POST /mission/setup     arma, despega, va al punto inicial
    POST /mission/start     empieza a llamar a tu protocolo
    POST /mission/stop      apaga el protocolo y manda RTL (aterriza)

------------------------------------------------------------------
SI NO QUISIERAS USAR EL PLUGIN

Ya hay un ejemplo hecho en el propio repo, que hace exactamente esto a
mano: gradys-embedded/examples/back_and_forth/protocol.py

Va y vuelve entre dos puntos. La idea es:
  - initialize(): guardas a que punto queres ir
  - handle_telemetry(): calculas la distancia al punto; si es menor que
    la tolerancia, cambias al siguiente y mandas
    send_mobility_command(GotoCoordsMobilityCommand(x, y, z))

El plugin hace justo eso por vos.
"""
