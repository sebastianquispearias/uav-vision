# Notas de decisiones y mediciones

Bitácora del proyecto: de dónde sale cada número del código y por qué se
tomó cada decisión. El código no repite esta historia — los docstrings
dicen QUÉ hace cada cosa; este archivo dice POR QUÉ y CUÁNDO.

## Calibración de cámara

- `focal_px = 1407.0` y `principal_point = (945.7, 547.1)` salen de los
  `run_info.json` grabados por `onboard.py` en los tres vuelos reales
  (26-jul, 01-ago y 02-ago 2026). Calibración formal con tablero de
  ajedrez sigue pendiente.
- **02-ago-2026: cambió el montaje.** La ArduCam se giró 180° sobre el
  eje óptico y el pitch pasó de −45° a −55°. El ISP endereza la imagen
  en captura (hflip+vflip), lo que refleja el punto principal:
  `(945.7, 547.1) → (973.3, 531.9)` con la fórmula `tamaño − 1 − c`.
  Ese es el motivo de `CameraConfig.rotated_180()` y de que
  `CamaraSimulada.pitch_deg` no tenga valor por defecto: un default
  escondido con el pitch viejo produjo 6.287 m de error en su momento.

| Vuelo | focal_px | principal_point | pitch |
|---|---|---|---|
| 26-jul | 1407.0 | 945.7, 547.1 | −45° |
| 01-ago | 1407.0 | 945.7, 547.1 | −45° |
| 02-ago | 1407.0 | 973.3, 531.9 (reflejado) | −55° |

## Modelos de ruido de píxel

- `heuristic`: sigma = 5.0 / conf (el del paper).
- `visdrone_1d`: sigma = 1.13 / conf, ajuste empírico sobre
  VisDrone2019-DET-val + YOLOv8s. La ablación del paper mostró que el
  ranking de métodos no depende de cuál se use.

## Huellas de apariencia (OSNet)

- Modelo: `osnet_x0_25_msmt17` vía boxmot, CPU, un batch por imagen,
  vectores normalizados.
- Separación medida sobre el vuelo 02-ago (coseno): misma identidad
  0.63–0.87; identidades distintas 0.33–0.46. No se solapan.
  - Umbral de fusión `emb_dist_max = 0.95` = punto medio del hueco
    (coseno 0.545) convertido a distancia L2.
  - Umbral del gemelo `EMB_DIST_GEMELO = 0.70` = zona profunda de
    misma-identidad (misma ≤ 0.86, distintas ≥ 1.04 en distancia).
- 24-ago-2026: verificado que `CamaraArduCam._huellas` reproduce
  exactamente las huellas del análisis offline (coseno 1.000000).
- Decisión (23-ago-2026): la huella se calcula EN la cámara; la imagen
  no sale del módulo. Por radio viajan 512 float32 (2048 B) o 128 B con
  PCA int8 (validado offline en la tesis de handoff). Alternativa
  futura: recorte bajo demanda (2–5 KB una vez) cuando la Ground
  Station quiera verificar un POI con un detector pesado.

## Umbrales de la identidad

- Reglas y valores base validados offline sobre el vuelo 02-ago
  (`drone-geolocation/entrenamiento/correr_botsort.py`): radio de
  fusión 3.5 m, mínimos de 6/20/25 observaciones a la cadencia 0.7 FPS
  de ese análisis. En el módulo se expresan como duraciones
  (8.6 / 29 / 36 s) × tasa declarada, que reproducen esos conteos.
- **Lección C-5 (24-ago-2026, vuelos 1-2):** la tasa que importa es la
  de OBSERVACIONES reales, no la de cámara. El vuelo 1 capturaba a
  8.7 FPS pero la persona se detectaba en ~30% de los frames
  (2.45 obs/s); con fps de cámara los umbrales exigían evidencia
  imposible y salían 0 candidatos.
- Posición actual de un MÓVIL: la mediana de la ventana reciente
  retrasa al caminante media ventana (medido: 7.40 m de retraso a
  1 m/s); el ajuste lineal evaluado en la última muestra lo reduce a
  0.48 m. De ahí `_posicion_actual`.
- Umbral MÓVIL 4.0 m ≈ 1.15 × radio de fusión: una pista quieta solo
  "tiembla" por ruido de proyección; más que eso, caminó.

## Rastreador (BoT-SORT)

- Parámetros validados offline: high 0.35 / low 0.2 / new 0.4 /
  buffer 40 frames / match 0.85, con huellas externas y CMC.
- El buffer se declara en segundos porque 40 frames son 24 s a la
  cadencia del vuelo 02-ago pero solo 8 s a los 5 Hz del dron.
- CMC (compensación de movimiento de cámara) apagado por defecto en el
  dron: su costo de CPU en la Raspberry no está medido.
- BotSort con `with_reid=True` exige huellas; sin ReID configurado el
  módulo lo crea en modo solo-movimiento.

## Presupuesto de la Raspberry Pi (banco 22-ago-2026)

- torch 3.74 FPS / 258 ms · NCNN 12.47 FPS / 77 ms (3.3×).
- Térmico: 3 fps → 47 °C / 20% CPU · 5 fps → 52 °C / 33% ·
  10 fps → 58 °C / 76%.
- Con UBEC 5 A el voltaje se hunde sobre ~5 FPS y el sistema colapsa;
  por eso la tasa de visión por defecto es 5 Hz. Prueba con UBEC 7 A
  pendiente.
- 24-ago-2026: `CamaraArduCam` corrió en la Pi real a 5.58 FPS
  (contrato probado en hardware). Config: `camera_auto_detect=0` +
  `dtoverlay=imx708` en config.txt (el auto-detect no reconoce el
  IMX708 de Arducam); `numpy<2` requerido por picamera2/simplejpeg.

## Validación acumulada (vuelo 02-ago salvo indicado)

- Geometría: rayos del protocolo vs rayos del vuelo real = 0.000° de
  diferencia (268 muestras).
- Replay con identidad: operador como POI separado a 2.32 m
  (análisis offline: 2.49 m); la caja de equipos ya no roba el consenso.
- Stack con RF-DETR como detector: 2.39 m, 1106 obs, conf 0.80, caja
  ausente — empate en precisión (el piso ~2.3–2.5 m lo pone el sesgo
  GPS/yaw), mejora clara en robustez. Rol asignado: verificador en la
  Ground Station.
- Vuelos sanos 1-2 (C-5): candidato dominante limpio en las 3 corridas
  — 0.41 / 3.13 / 1.43 m (fusión simple: 1.03 / 2.08 / 0.59).
- Gates sintéticos de la identidad (`tests/test_identidad.py`):
  estático 0.09 m; móvil sin retraso; veto de co-ocurrencia mantiene 2;
  gemelas fusionan a 1.
