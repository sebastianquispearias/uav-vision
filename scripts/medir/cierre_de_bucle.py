"""Reproduces the closure bug of OnboardCamera._first_frame, and the fix, side by side.

Same shape as the real loop: a capture that wedges on the first attempt and unwedges during the
second. The camera is a stub, so nothing here needs a Raspberry.
"""
import threading
import time

ESPERA = 0.3
INTENTOS = 3


class CamaraQueSeAtasca:
    """First capture blocks for a while and then returns; the rest block for ever."""

    def __init__(self, se_desatasca_en):
        self.n = 0
        self.se_desatasca_en = se_desatasca_en

    def capture_array(self):
        self.n += 1
        if self.n == 1:
            time.sleep(self.se_desatasca_en)
            return "un cuadro que llega DEMASIADO TARDE"
        threading.Event().wait()


def como_estaba(picam):
    """The loop as it was written: one cell of `listo` shared by every attempt."""
    for intento in range(1, INTENTOS + 1):
        listo = threading.Event()

        def capture():
            try:
                picam.capture_array()
            finally:
                listo.set()

        threading.Thread(target=capture, daemon=True).start()
        if listo.wait(ESPERA):
            return intento
        if intento == INTENTOS:
            raise RuntimeError("la camara no entrego un frame")
    return None


def arreglado(picam):
    """The event bound per attempt, so a late thread can only set its OWN."""
    for intento in range(1, INTENTOS + 1):
        listo = threading.Event()

        def capture(ev=listo):
            try:
                picam.capture_array()
            finally:
                ev.set()

        threading.Thread(target=capture, daemon=True).start()
        if listo.wait(ESPERA):
            return intento
        if intento == INTENTOS:
            raise RuntimeError("la camara no entrego un frame")
    return None


for nombre, f in (("como estaba", como_estaba), ("arreglado  ", arreglado)):
    try:
        r = f(CamaraQueSeAtasca(se_desatasca_en=ESPERA * 1.5))
        print("  %s -> VUELVE diciendo que el intento %d entrego un cuadro" % (nombre, r))
    except RuntimeError as e:
        print("  %s -> levanta RuntimeError: %s" % (nombre, e))
