# Papers consultados, qué dieron y qué queda sin probar

Solo lo que se leyó y se contrastó con datos propios. Cada entrada dice **qué nos dio a nosotros**,
que suele no ser lo que el paper promete.

## Detección de objetos pequeños desde el aire

**[Slicing Aided Hyper Inference (SAHI), arXiv 2202.06934](https://arxiv.org/pdf/2202.06934)**
Reporta +6,8 / +5,1 / +5,3 de AP en VisDrone y xView, sin reentrenar, cortando la imagen en fichas a
resolución nativa.
**Qué nos dio:** la técnica funciona acá pero rinde menos, y sabemos por qué. Reemplazar el cuadro por
las fichas no paga; **sumarlas sí**: +2,3 de recall total y +2,9 en el balcón, a la misma precisión.
La diferencia con el paper se explica porque nuestro modelo está especializado en personas de unos
28 px y empeora cuando le llegan más grandes.

**[Evaluation of YOLO Models with Sliced Inference, arXiv 2203.04799](https://arxiv.org/pdf/2203.04799)**
**Qué nos dio:** confirmación del orden de magnitud esperable.

**[SAHI-Improved-YOLOv8 for UAV imagery](https://www.sciencedirect.com/science/article/pii/S2772375525004125)**
**Qué nos dio:** el mismo patrón aplicado a un caso de dron.

**[Maritime Small Object Detection with Altitude-Aware Dynamic Tiling, arXiv 2511.19728](https://arxiv.org/pdf/2511.19728)**
Fichas adaptadas a la altura del vuelo.
**Sin probar, y encaja:** usar fichas **solo cuando el dron está alto**, que es cuando pagan. Tenemos
`persona_px()` midiendo el tamaño esperado por altura, así que la regla saldría de datos propios.

## Interacción con el operador

**[UAVDB: Point-Guided Masks for UAV Detection and Segmentation, arXiv 2409.06490](https://arxiv.org/pdf/2409.06490)**
Un punto como prompt para generar máscaras en imágenes de dron.
**Qué nos dio:** el nombre de lo que propuso el usuario (detección guiada por punto) y la confirmación
de que es una línea viva. Nuestra versión barata está medida: bajar el umbral solo en la ventana del
objetivo da +17 puntos de recall sin coste de cómputo.

**[A reliable UAV tracking system with online re-detection network](https://www.sciencedirect.com/science/article/abs/pii/S0019057825004744)**
Estimación de incertidumbre y re-detección en línea.
**Sin probar, y es lo que nos falta:** que el sistema sepa **cuándo dejó de estar seguro** y vuelva a
buscar, en vez de seguir afirmando. Es exactamente el problema de G y H partidas en dos.

## Traspaso entre drones

**[Continuous Marine Tracking via Autonomous UAV Handoff, arXiv 2507.12763](https://arxiv.org/pdf/2507.12763)**
82,9 % de cobertura del objetivo a 4-5 Hz sobre Jetson Nano. El traspaso se resuelve con la posición y
**un marcador ArUco pegado al otro dron**.
**Qué nos dio:** que el traspaso funciona, y que ellos lo resuelven por geometría y no por reconocer al
objetivo. Nosotros ya mandamos la huella OSNet entre drones, así que tenemos una vía distinta a mano.

**[Multi-Drone based Single Object Tracking with Agent Sharing Network, arXiv 2003.06994](https://arxiv.org/pdf/2003.06994)**
**Sin explorar.**

## Cómputo repartido entre dron y tierra

**[Supporting UAVs with Edge Computing, arXiv 2310.11957](https://arxiv.org/html/2310.11957v1)**,
**[Offloading Deep Learning Vision Tasks from UAV, arXiv 2302.01991](https://arxiv.org/pdf/2302.01991)**,
**[Real-time UAV object detection and task offloading](https://link.springer.com/article/10.1007/s11370-026-00736-z)**

La advertencia que más nos sirvió: **descargar tareas simples degrada al dron**; lo óptimo suele ser
computar a bordo.
**Qué nos dio:** el reparto que implementamos. Un cuadro son 303 KB, o sea 7,3 Mbps a 3 FPS: mandar
todo el video no es opción. Mandar **un cuadro bajo demanda** sí, y es lo que quedó cableado.

## Ideas sin fuente todavía

- **Destilar RF-DETR a yolo26.** RF-DETR encuentra el 90,5 % donde el nuestro encuentra el 46,2 %.
  Como maestro sobre metraje **nuevo** sería la vía más prometedora. Sobre los vuelos que ya tenemos
  no sirve: triplicaría el sobreajuste, no la variedad.
- **Adaptar solo la plantilla de apariencia en vez de los pesos.** Medido: el objetivo sube de 91,1 a
  94,4 % aceptando detecciones dudosas que se le parecen, sin tocar un solo peso y con vuelta atrás
  inmediata.
- **Que el veredicto del operador enseñe.** Hoy "no es" solo borra el punto del mapa. Es la única vía
  contra el fantasma que ni CLIP ni el tamaño físico detectan.
- **Fichas solo cuando el dron está alto**, siguiendo el paper de tiling adaptativo, usando nuestra
  propia medición de tamaño por altura.
