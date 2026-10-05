"""Does this board detect, or only capture? Six frames, through the real mission's camera."""
import time
import traceback

from uav_vision.camera import OnboardCamera

cam = OnboardCamera(model="/home/pi/yolov8n_ncnn_model", threshold=0.3, tracker=True,
                    fps=4.0, crops=True)
try:
    bien = 0
    for i in range(6):
        t = time.time()
        try:
            dets = cam.detect((0.0, 0.0, 30.0), 0.0)
        except Exception:
            print("FALLA detect() en el intento %d:" % i)
            traceback.print_exc(limit=5)
            break
        bien += 1
        print("  intento %d: %d detecciones en %.0f ms  %s"
              % (i, len(dets or []), (time.time() - t) * 1000,
                 [(d.get("cls"), round(float(d.get("conf", 0)), 2)) for d in (dets or [])][:4]))
        time.sleep(0.2)
    print()
    print("DETECTA: %d de 6 cuadros pasaron por la cadena ncnn" % bien if bien == 6
          else "NO DETECTA: solo %d de 6" % bien)
finally:
    try: cam.close()
    except Exception: pass
