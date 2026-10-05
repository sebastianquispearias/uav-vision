"""
EJEMPLO PARA UN COLEGA DEL GRUPO
================================
"Quiero que el dron haga MI mision de vuelo, y ademas use la vision
 que ya hicieron ustedes."

Respuesta: escribis UN protocolo que hereda del de vision y le agrega
tu vuelo. Son 10 lineas nuevas. No tocas ni una linea de uav_vision.

Todo lo de este archivo esta verificado contra el codigo real.
Las referencias tipo "archivo.py:123" son reales, anda a mirarlas.

-------------------------------------------------------------------
LO PRIMERO: EN GrADyS SE ESCRIBE UNA CLASE, NO UN GUION DE COMANDOS
-------------------------------------------------------------------
No existe esto:

    start
    goto 100 north
    goto 100 west
    aterrizar

Lo que se escribe es una clase con cinco metodos que el runner llama
cuando pasan cosas. La clase no corre de arriba a abajo: espera a que
la llamen.

Los cinco (gradys_embedded/protocol/interface.py:88):

    initialize()                 UNA vez, al arrancar
    handle_telemetry(telemetry)  cada vez que llega la posicion
    handle_timer(timer)          cuando vence una alarma tuya
    handle_packet(message)       cuando llega un mensaje de otro dron
    finish()                     UNA vez, al terminar

El "runner" es el programa de gradys_embedded que corre en la Raspberry
y que llama a esos metodos. Tu protocolo nunca corre solo.

-------------------------------------------------------------------
SOLO SE CARGA UN PROTOCOLO A LA VEZ
-------------------------------------------------------------------
runner/mission.py:306 --

    async def load(self, protocol: str, ...):
        if self.state is not MissionState.IDLE:
            raise MissionError("Cannot load a mission while ...")

"protocol" en singular, y si ya hay uno cargado se niega a cargar otro.

Por eso, si queres vuelo Y vision, NO son dos protocolos: es uno solo
que hace las dos cosas. Asi:

PART 1, THE VISION, is copied exactly from scripts/banco_embedded/mision_barrido.py. Do not
touch it: those are MEASURED values, not chosen ones. with_config() returns a CLASS already
configured, with the camera mount angle, how often it looks, the fps that has to MATCH the
camera's, and how much evidence a report needs.

PART 2, THE FLIGHT, inherits from that class and adds the waypoints. That is all that is new.

CALLING THE VISION LAYER'S initialize IS MANDATORY AND IS THE CLASSIC TRAP. It starts the whole
of the vision: the see timer, the report timer, the identity layer. Forget it and the drone
FLIES BLIND with no error at all: it flies perfectly and reports nothing.
"""

from gradys_embedded.protocol.plugin.mission_mobility import (
    MissionMobilityConfiguration,
    MissionMobilityPlugin,
)

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import UavApiYaw, VisionProtocol

VisionConfigurada = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/modelos_visdrone/y960_ncnn_model",
        threshold=0.25,
        tracker=True,
        reid_model="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=3.0,
        crops=True,
    ),
    pitch_deg=-55.0,
    see_period_s=1.0 / 3.0,
    yaw_source=UavApiYaw("http://localhost:8000"),
    identity=IncrementalIdentity(
        fusion_radius_m=3.5,
        fps=3.0,
        report_dur_s=36.0,
    ),
    report_preliminary=True,
)



ALTURA = 30.0


class MiMisionConVision(VisionConfigurada):

    def initialize(self):
        super().initialize()

        self.vuelo = MissionMobilityPlugin(
            self,
            MissionMobilityConfiguration(
                speed=5.0,
                tolerance=1.0,
            ),
        )
        self.vuelo.start_mission([
            (100.0,    0.0, ALTURA),
            (100.0, -100.0, ALTURA),
        ])


"""
===================================================================
LAS COORDENADAS SON METROS, NO LAT/LON
===================================================================
El origen y la orientacion los fija quien carga la mision, con los
parametros origin_gps_coordinates y x_axis_degrees del POST
/mission/load.

Con x_axis_degrees = 0 (verificado en protocol/position.py:44-47):

        x = norte        y = este

Por eso "100 m al oeste" es y = -100.


===================================================================
NO HAY COMANDO "ATERRIZAR"
===================================================================
protocol/messages/mobility.py tiene SOLO TRES comandos, y ninguno es
aterrizar:

    GOTO_COORDS       ir a x, y, z
    GOTO_GEO_COORDS   ir a lat, lon, alt
    SET_SPEED         fijar velocidad

Aterrizar lo hace el runner cuando paras la mision. En
runner/mission.py:466, dentro de stop():

    self._runner.request_return_to_launch()

O sea: ATERRIZAR = POST /mission/stop. No es una linea de tu codigo.


===================================================================
NO HAY "QUEDARSE X SEGUNDOS EN UN PUNTO"
===================================================================
MissionMobilityConfiguration solo tiene speed, loop_mission y
tolerance (mission_mobility.py:20-36). No hay tiempo de espera en
waypoint.

Si necesitas pasar mucho tiempo sobre un punto -- y en este proyecto
hace falta: una pasada de 30 s NO detecta nada, una de 90 s detecta el
60 % -- el control es la VELOCIDAD, no una pausa. Bajas speed, o pones
waypoints mas juntos.


===================================================================
POR QUE LA VISION Y EL VUELO NO SE PISAN
===================================================================
Duda razonable: el plugin de vuelo necesita handle_telemetry para
saber si llegaste al waypoint. La vision TAMBIEN usa handle_telemetry,
para guardar la posicion del dron. Deberian chocar.

No chocan, y esta hecho a proposito. El plugin no reemplaza tu metodo:
lo ENCADENA. Textual de plugin/dispatcher.py:194 --

    "Implements a call chain for each of the protocol interface's
     methods. (...) The original method implementation is not lost."

Cuando llega la telemetria se llaman los dos, en cadena.

AVISO del propio plugin (mission_mobility.py:49): no mandes comandos de
movimiento por tu cuenta mientras hay una mision en curso, o la rompes.
El protocolo de vision es observe-only, asi que no da problema.


===================================================================
COMO SE CORRE
===================================================================
1. Copiar este archivo a ~/gradys_protocols/ en la Raspberry.
   (El runner carga desde AHI, no desde ~/uav_vision/scripts/.)

2. Cuatro peticiones HTTP al runner (puerto 8100):

   POST /mission/load    con "ejemplo_colega_vuelo_y_vision:MiMisionConVision"
                         y origin_gps_coordinates + x_axis_degrees
   POST /mission/setup   arma, despega y va al punto inicial
   POST /mission/start   empieza a llamar a tu protocolo
   POST /mission/stop    apaga el protocolo y manda RTL (aterriza)


===================================================================
SI NO QUERES USAR EL PLUGIN
===================================================================
Hay un ejemplo hecho a mano en el propio repo:
gradys-embedded/examples/back_and_forth/protocol.py

Va y vuelve entre dos puntos: en handle_telemetry calcula la distancia
al waypoint, y si es menor que la tolerancia manda el siguiente con
send_mobility_command(GotoCoordsMobilityCommand(x, y, z)).
El plugin hace exactamente eso por vos.
"""
