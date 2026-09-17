# Qué sigue, en orden, con su criterio de aceptación

Cada punto dice **cuándo está listo**, para que el trabajo tenga final y no sea una cinta sin fin.
Lo que ya está medido y descartado vive en `DESCARTADO.md`; los números actuales en `RESULTADOS.md`.

## Producto

### 1. Volar otra vez, a 20-25 m, otro día y otro sitio

**Por qué es el primero y no se puede saltear:** no solo da la primera evaluación limpia. Hoy **no
tenemos un solo frame de validación en el régimen que importa**, porque todo el material alto que
existe o es el test o son sus vecinos inmediatos. Por eso la validación eligió mal tres veces
seguidas y **no podemos ni elegir entre dos modelos** antes de tocar el test.

**Listo cuando:** hay un vuelo de 10 minutos a 20-25 m, con al menos dos personas además del
operador, y al menos una posición topografiada que no sea la del operador.

**Ayuda que existe:** la página `/plan` de la herramienta de etiquetado dibuja la altura contra los
frames y dice, del tramo que elijas, cuánto es nuevo y de qué tamaño se vería la persona ahí.

### 2. Medir en la Pi todo lo que se agregó

Nada de lo nuevo corrió nunca en el aire. La Pi tarda 206 ms por cuadro y las fichas cuestan 6x en
una máquina mucho más rápida.

**Listo cuando:** `<= 330 ms/frame` medido en la Pi con el modelo congelado, las fichas encendidas
cada 5 cuadros y el cuadro bajo demanda respondiendo.

### 3. Juzgar las mejoras nuevas con el marcador de producto

La adaptación paga 12 puntos de precisión y las fichas algo de precisión también. Eso **puede
significar fantasmas nuevos**, que es lo que el operador sufre, y hoy solo está medido en cajas.

**Listo cuando:** cada mejora tiene su fila en la tabla de personas y fantasmas.
**Comando:** `scratchpad/personas_encontradas.py`.

### 4. RF-DETR en tierra, de punta a punta

La cañería está: el dron responde a `vision_mirar` mandando el cuadro que miró, y
`scripts/banco_embedded/segunda_opinion.py` lo mira con fichas. **Falta el botón**: que un click en la
estación dispare el pedido y pinte lo que vuelve.

**Listo cuando:** el operador hace click en un punto del mapa y ve, en menos de 5 s, lo que RF-DETR
encontró en ese cuadro.

### 5. Modo "objetivo fijado"

Bajar el umbral solo dentro de la ventana del objetivo: recall 43,9 → 60,7 % sobre lo que el operador
señaló, **sin un milisegundo extra de cómputo** y reversible.

**Listo cuando:** se enciende con el veredicto "es lo que busco" y se mide con el marcador de producto.

### 6. Re-unir a una persona tras un hueco

G y H aparecen como dos puntos cada una **incluso con un detector perfecto**. Es lo único que la
visión no puede arreglar, y la vía es comparar apariencia contra los candidatos ya cerrados.

**Listo cuando:** G y H aparecen como un punto cada una en el test.

## Portafolio

El objetivo declarado es conseguir trabajo como ingeniero de visión o percepción. Para eso el sistema
**no necesita estar terminado**, necesita ser defendible y entendible rápido.

### 7. Una página que se entienda en 40 segundos

Hoy todo vive en un archivo de estado de miles de líneas y videos sueltos. Falta lo de arriba contado
en una página: qué hace, con qué número, y qué no hace.

### 8. Los tres entrenamientos que perdieron, contados como resultado

`DESCARTADO.md` tiene tres entrenamientos con sus números y el motivo de cada fracaso. **Eso vale más
en una entrevista que tres que ganan**, porque muestra saber cuándo la propia mejora no funcionó. Hoy
está escrito como registro, no como argumento.

### 9. El video comparativo como pieza central

`docs/yolo26_vs_rfdetr.mp4` muestra 46 % contra 90 % con el acumulado corriendo en pantalla. Es lo más
convincente que hay y no está usado en ningún lado.

## Lo que NO hay que hacer

Está medido y agotado, el detalle en `DESCARTADO.md`:

- **Más entrenamientos con otra receta sobre los mismos datos.** Tres perdieron.
- **Más etiquetado de los vuelos que ya tenemos.** El material está exprimido: lo que queda sin
  etiquetar por encima de 12 m son vecinos inmediatos de lo ya etiquetado.
- **RF-DETR como maestro mientras no haya metraje nuevo.** Pseudo-etiquetar los mismos vuelos
  triplicaría el sobreajuste, no la variedad.
- **Adaptación de pesos en vuelo sin poder revertir.** La adaptación funciona, pero el modo de fallo
  es silencioso: hay que tener los dos modelos y poder volver al original.
