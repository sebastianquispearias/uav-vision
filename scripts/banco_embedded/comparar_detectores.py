"""COCO or VisDrone, on THIS camera, right now. The bench says COCO; this checks it."""
import statistics as st
import time

from uav_vision.camera import OnboardCamera

MODELOS = [("COCO yolov8n   ", "/home/pi/yolov8n_ncnn_model"),
           ("VisDrone y960  ", "/home/pi/modelos_visdrone/y960_ncnn_model"),
           ("VisDrone y1280 ", "/home/pi/modelos_visdrone/y1280_ncnn_model")]
N = 8

print("placa:", open("/proc/device-tree/model").read().strip("\x00"))
print("umbral 0.3, misma camara, misma escena, uno detras de otro\n")
for nombre, ruta in MODELOS:
    cam = OnboardCamera(model=ruta, threshold=0.3, tracker=False, fps=4.0)
    ms, n_det, conf, clases = [], [], [], {}
    try:
        for i in range(N):
            t = time.time()
            dets = cam.detect((0.0, 0.0, 30.0), 0.0) or []
            if i == 0:
                continue
            ms.append((time.time() - t) * 1000.0)
            n_det.append(len(dets))
            for d in dets:
                conf.append(float(d.get("conf", 0)))
                clases[d.get("cls")] = clases.get(d.get("cls"), 0) + 1
    except Exception as e:
        print("  %s FALLA: %s" % (nombre, type(e).__name__))
        continue
    finally:
        try:
            cam.close()
        except Exception:
            pass
    print("  %s %6.0f ms/cuadro   %.1f cajas/cuadro   conf media %.2f   %s"
          % (nombre, st.median(ms) if ms else 0, st.mean(n_det) if n_det else 0,
             st.mean(conf) if conf else 0, dict(sorted(clases.items(), key=lambda p: -p[1])[:4])))
