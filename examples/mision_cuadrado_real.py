"""
LA MISMA MISION, EN EL DRON REAL.

El comportamiento esta en mi_mision_cuadrado.py y NO se toca: es el mismo
archivo que usa la simulacion. Aqui solo se eligen los DOS argumentos que
cambian entre los dos mundos.

    simulacion:  SimulatedCamera  +  lambda: 0.0
    dron real:   OnboardCamera    +  UavApiYaw()

Este archivo va a ~/gradys_protocols/ en la Raspberry, junto con
mi_mision_cuadrado.py. El runner carga desde AHI, no desde ~/uav_vision/.

Se carga con:
    POST /mission/load  {"protocol": "mision_cuadrado_real:MiProtocolo", ...}
"""

from uav_vision.camera import OnboardCamera
from uav_vision.vision_protocol import UavApiYaw

from mi_mision_cuadrado import construir


MiProtocolo = construir(
    camara=OnboardCamera(
        model="/home/pi/modelos_visdrone/y960_ncnn_model",
        threshold=0.25,
        tracker=True,
        reid_model="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
        fps=3.0,
        crops=True,
    ),
    yaw_source=UavApiYaw("http://localhost:8000"),   # rumbo REAL del Pixhawk
    velocidad=5.0,
)
