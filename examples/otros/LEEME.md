# Ejemplos apartados

Se movieron aqui para que `examples/` tenga **un solo ejemplo**: la mision del
cuadrado, en tres archivos que son el argumento entero (un comportamiento, dos
escenarios). Estos siguen sirviendo, pero no son la puerta de entrada.

| archivo | para que |
|---|---|
| `demo_para_depurar.py` | F5 en VS Code y corre. Lleva el runner destilado a 30 lineas (`MiniRunner`): el mejor sitio para poner puntos de interrupcion y ver que le hace el anfitrion al protocolo. |
| `ejemplo_colega_vuelo_y_vision.py` | **para leer, no para correr** (usa `OnboardCamera`, que necesita `picamera2` y solo existe en la Raspberry). |
| `ejemplo_mision_100m.py` | idem: para leer. |
