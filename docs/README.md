# Los documentos de este sistema

`ESTADO_SESION.md`, en la raíz, **se lee primero y siempre**: es el punto de retomada, con lo que está
en vuelo y los hilos abiertos. Estos documentos no lo reemplazan, lo complementan: son las
conclusiones sacadas del orden cronológico para poder consultarlas sueltas y mostrarlas a alguien.

| Documento | Contesta |
|---|---|
| [RESULTADOS.md](RESULTADOS.md) | **Qué hace el sistema**, con los números que se defienden y las advertencias que van con cada uno |
| [PLAN.md](PLAN.md) | **Qué sigue**, en orden, y cuándo se considera terminado cada punto |
| [DESCARTADO.md](DESCARTADO.md) | **Qué ya se probó y no funcionó**, con el número y el motivo, para no repetirlo |
| [LITERATURA.md](LITERATURA.md) | **Qué dice la literatura** y qué nos dio a nosotros, que casi nunca es lo que el paper promete |
| [ARQUITECTURA.md](ARQUITECTURA.md) | **Cómo encajan las piezas**: la estación de tierra, el replay y la herramienta de etiquetado, con el porqué de cada perilla |
| [../NOTES.md](../NOTES.md) | **De dónde sale cada número del código** y por qué se tomó cada decisión. En la raíz porque es la bitácora, no una conclusión |

La división entre los tres últimos y el fuente es deliberada: **los docstrings dicen qué es cada
cosa**, `ARQUITECTURA.md` dice cómo encaja, y `NOTES.md` dice de dónde salió el número. Un umbral
medido en un vuelo no va en un docstring: podría cambiar en otro vuelo y no describe el
funcionamiento.

## Por dónde empezar según quién seas

- **Si venís a ver si el sistema sirve:** `RESULTADOS.md`, la tabla de personas y fantasmas.
- **Si venís a leer el código:** `ARQUITECTURA.md`, y `scripts/sistema_esencial.py`, que es la
  cadena entera en un archivo que se puede correr.
- **Si venís a trabajar:** `PLAN.md`, y antes de proponer algo, `DESCARTADO.md`.
- **Si venís a evaluar el trabajo:** `DESCARTADO.md`. Los fracasos medidos dicen más del método que
  los aciertos.

## Videos

| Archivo | Qué muestra |
|---|---|
| `yolo26_vs_rfdetr.mp4` | el detector que vuela (46 %) contra uno que no puede volar (90 %), con el acumulado corriendo |
| `base_vs_adaptado.mp4` | el mismo detector antes y después de adaptarse a la escena en 89 segundos |
| `antes_despues_detector_20260916.mp4` | dos detectores sobre el mismo tramo, en el formato estándar de la estación |
| `sistema_arreglado_20260916.mp4` | el sistema completo: cámara y estación real, lado a lado |

Los videos no se commitean (`.gitignore`), solo viven en disco.
