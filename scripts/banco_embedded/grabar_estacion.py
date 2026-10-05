"""Records the ground station while the system runs live, pressing its buttons for the operator.

Nothing is replayed: the page is the real one, connected to the boards that are watching the room.
The script only does what a person would do -- wait, ask one drone for a second opinion, wait for
the answer, ask the other -- and saves a screenshot twice a second. The video is made from those
screenshots, so what it shows is exactly what the operator would have seen.

The buttons it presses are found by their data attributes rather than by their labels, which is
what lets the page be translated without breaking the recording: `data-mirar-dron` is the second
opinion button in the per-drone strip. A label changes often; an attribute is a contract.

    python scripts/banco_embedded/grabar_estacion.py [seconds]

It needs selenium and a Chrome on the laptop, and the station already answering on 8300. The
frames land in capturas_vivo/ next to this file and are turned into a video separately, so a run
that is interrupted still leaves everything it captured.

GUION is the script of the recording, in seconds from the start, as (when, what to do) pairs.
It holds still long enough for the map and the cards to be seen as they are, then asks drone 1
for a second opinion, then drone 2, so that both answers end up on screen at once.
"""
import json
import os
import shutil
import sys
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.by import By

AQUI = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8300/"
CAPT = os.path.join(AQUI, "capturas_vivo")
FPS_CAPTURA = 2.0
ANCHO, ALTO = 1100, 1500

GUION = [
    (0.0, None),
    (8.0, "mirar:1"),
    (26.0, "mirar:2"),
    (46.0, None),
]
DURACION = float(sys.argv[1]) if len(sys.argv) > 1 else 56.0


def estado():
    return json.load(urllib.request.urlopen(URL + "estado", timeout=5))


shutil.rmtree(CAPT, ignore_errors=True)
os.makedirs(CAPT)

op = webdriver.ChromeOptions()
for a in ("--headless=new", "--window-size=%d,%d" % (ANCHO, ALTO), "--hide-scrollbars",
          "--force-device-scale-factor=1"):
    op.add_argument(a)
nav = webdriver.Chrome(options=op)
nav.get(URL)
time.sleep(3.0)
vw, vh = nav.execute_script("return [window.innerWidth, window.innerHeight]")
if (vw, vh) != (ANCHO, ALTO):
    nav.set_window_size(ANCHO + (ANCHO - vw), ALTO + (ALTO - vh))
print("viewport", nav.execute_script("return [window.innerWidth, window.innerHeight]"), flush=True)

d = estado()
print("drones conectados: %s | POIs: %d" % (sorted(d.get("drones", {})), len(d.get("pois", []))), flush=True)

pendientes = list(GUION)
t0 = time.time()
k = 0
hechas = []
while True:
    t = time.time() - t0
    if t > DURACION:
        break
    while pendientes and t >= pendientes[0][0]:
        _, accion = pendientes.pop(0)
        if accion and accion.startswith("mirar:"):
            dron = accion.split(":")[1]
            try:
                b = nav.find_element(By.CSS_SELECTOR, 'button[data-mirar-dron="%s"]' % dron)
                nav.execute_script("arguments[0].click()", b)
                hechas.append((round(t, 1), "clic segunda opinion dron " + dron))
                print("  t=%5.1f  clic en 'segunda opinion' del dron %s" % (t, dron), flush=True)
            except Exception as e:
                print("  t=%5.1f  NO se pudo apretar el boton del dron %s: %s"
                      % (t, dron, type(e).__name__), flush=True)
    nav.save_screenshot(os.path.join(CAPT, "c%04d.png" % k))
    k += 1
    dormir = (k / FPS_CAPTURA) - (time.time() - t0)
    if dormir > 0:
        time.sleep(dormir)

print("capturas: %d en %.1f s" % (k, time.time() - t0))
seg = estado().get("segunda", {})
for dron in sorted(seg):
    s = seg[dron]
    print("  dron %s -> %s, %s personas en %s s" % (dron, s.get("estado"), s.get("n"), s.get("espera")))
nav.quit()
print(json.dumps({"capturas": k, "fps": FPS_CAPTURA, "acciones": hechas}))
