"""What one frame of the FLIGHT configuration costs on this board, with everything it carries."""
import statistics as st, time
from uav_vision.camera import OnboardCamera
cam = OnboardCamera(model="/home/pi/modelos_visdrone/y960_ncnn_model", threshold=0.25,
                    tracker=True, reid_model="/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt",
                    fps=3.0, crops=True)
ms = []
t0 = time.time()
for i in range(8):
    t = time.time()
    d = cam.detect((0.0, 0.0, 30.0), 0.0)
    dt = (time.time() - t) * 1000.0
    if i == 0:
        print("  primer cuadro (carga todo): %.1f s" % (dt / 1000.0))
        continue
    ms.append(dt)
print("  mediana %.0f ms/cuadro  ->  %.2f FPS reales" % (st.median(ms), 1000.0 / st.median(ms)))
print("  la mision pide see_period_s = 1/3 s = 333 ms, o sea 3.00 FPS")
print("  %s" % ("CABE" if st.median(ms) < 333 else "NO CABE: el lazo va saturado"))
cam.close()
