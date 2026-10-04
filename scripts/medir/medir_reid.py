"""What the appearance model costs on this board, measured, with everything else held equal."""
import statistics as st
import sys
import time

from uav_vision.camera import OnboardCamera

N = 12
OSNET = "/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt"


def correr(reid):
    cam = OnboardCamera(model="/home/pi/yolov8n_ncnn_model", threshold=0.3, tracker=True,
                        fps=4.0, crops=True, reid_model=reid)
    ms, cajas, con_emb = [], [], 0
    try:
        for i in range(N):
            t = time.time()
            dets = cam.detect((0.0, 0.0, 30.0), 0.0)
            dt = (time.time() - t) * 1000.0
            if i == 0:
                continue          # el primero carga el modelo, no es el ritmo
            ms.append(dt)
            cajas.append(len(dets or []))
            con_emb += sum(1 for d in (dets or []) if d.get("emb") is not None)
    finally:
        try:
            cam.close()
        except Exception:
            pass
    return ms, cajas, con_emb


print("placa:", open("/proc/device-tree/model").read().strip("\x00"))
sin_ms, sin_cajas, sin_emb = correr(None)
con_ms, con_cajas, con_emb = correr(OSNET)
print()
print("  SIN apariencia: mediana %6.0f ms/cuadro   cajas/cuadro %.1f   con vector %d"
      % (st.median(sin_ms), st.mean(sin_cajas), sin_emb))
print("  CON apariencia: mediana %6.0f ms/cuadro   cajas/cuadro %.1f   con vector %d"
      % (st.median(con_ms), st.mean(con_cajas), con_emb))
d = st.median(con_ms) - st.median(sin_ms)
print()
print("  diferencia: %+.0f ms por cuadro (%+.0f %%)" % (d, 100.0 * d / st.median(sin_ms)))
if st.mean(con_cajas) > 0:
    print("  por caja:   %+.0f ms   (el docstring dice ~33 ms)" % (d / st.mean(con_cajas)))
print("  ritmo: %.2f FPS sin apariencia, %.2f FPS con" % (1000.0 / st.median(sin_ms),
                                                          1000.0 / st.median(con_ms)))
