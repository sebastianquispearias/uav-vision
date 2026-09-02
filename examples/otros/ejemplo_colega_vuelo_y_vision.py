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
"""

from gradys_embedded.protocol.plugin.mission_mobility import (
    MissionMobilityPlugin,
    MissionMobilityConfiguration,
)
from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.vision_protocol import VisionProtocol, UavApiYaw


# ===================================================================
# PARTE 1 -- LA VISION
# Esto es copia exacta de scripts/banco_embedded/mision_barrido.py.
# No lo toques: son valores medidos, no elegidos.
# with_config() te devuelve una CLASE ya configurada.
# ===================================================================

VisionConfigurada = VisionProtocol.with_config(
    camera=OnboardCamera(
        model="/home/pi/modelos_visdrone/y960_ncnn_model",
        threshold=0.25,
        tracker=True,
        reid_model="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=3.0,
        crops=True,
    ),
    pitch_deg=-55.0,          # angulo de montaje de la camara
    see_period_s=1.0 / 3.0,   # mirar 3 veces por segundo
    yaw_source=UavApiYaw("http://localhost:8000"),
    identity=IncrementalIdentity(
        fusion_radius_m=3.5,
        fps=3.0,              # tiene que coincidir con el fps de la camara
        report_dur_s=36.0,    # cuanta evidencia hace falta para reportar
    ),
    report_preliminary=True,
)


# ===================================================================
# PARTE 2 -- TU VUELO
# Heredas de la clase de arriba y le agregas los waypoints.
# ESTO ES TODO LO NUEVO.
# ===================================================================

ALTURA = 30.0   # metros sobre el punto de despegue


class MiMisionConVision(VisionConfigurada):

    def initialize(self):
        # ---------------------------------------------------------
        # ESTA LINEA ES OBLIGATORIA Y ES LA TRAMPA CLASICA.
        # Arranca TODA la vision: los timers de ver y de reportar,
        # la identidad, todo. Si te la olvidas, el dron VUELA CIEGO
        # y no salta ningun error. Vuela perfecto y no reporta nada.
        # ---------------------------------------------------------
        super().initialize()

        # Y aca va tu vuelo.
        self.vuelo = MissionMobilityPlugin(
            self,
            MissionMobilityConfiguration(
                speed=5.0,        # m/s
                tolerance=1.0,    # a menos de 1 m, se da por llegado
            ),
        )
        self.vuelo.start_mission([
            (100.0,    0.0, ALTURA),   # 100 m al norte
            (100.0, -100.0, ALTURA),   # y desde ahi, 100 m al oeste
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
