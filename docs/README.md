# Los cuatro documentos de este sistema

`ESTADO_SESION.md`, en la raíz, es el **registro cronológico**: sirve para retomar el trabajo, no para
consultar. Estos cuatro son para consultar, y cada uno contesta una pregunta distinta.

| Documento | Contesta |
|---|---|
| [RESULTADOS.md](RESULTADOS.md) | **Qué hace el sistema**, con los números que se defienden y las advertencias que van con cada uno |
| [PLAN.md](PLAN.md) | **Qué sigue**, en orden, y cuándo se considera terminado cada punto |
| [DESCARTADO.md](DESCARTADO.md) | **Qué ya se probó y no funcionó**, con el número y el motivo, para no repetirlo |
| [LITERATURA.md](LITERATURA.md) | **Qué dice la literatura** y qué nos dio a nosotros, que casi nunca es lo que el paper promete |

## Por dónde empezar según quién seas

- **Si venís a ver si el sistema sirve:** `RESULTADOS.md`, la tabla de personas y fantasmas.
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
