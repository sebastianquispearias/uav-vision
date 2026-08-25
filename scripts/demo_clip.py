"""Records a short annotated demo clip of the live chain on the Pi.

Runs the same capture + YOLO + BoT-SORT wiring as the rehearsal, draws each
detection's box, confidence and track_id on the frame, and saves the annotated
frames as JPEGs for later assembly into a video on the laptop.
"""

import json
import sys
import time

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2
from ultralytics import YOLO
from boxmot.trackers.bbox.botsort import BotSort

DUR_S = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
SALIDA = "/home/pi/demo_frames"

import os
os.makedirs(SALIDA, exist_ok=True)
for f in os.listdir(SALIDA):
    os.remove(os.path.join(SALIDA, f))

picam = Picamera2()
picam.configure(picam.create_still_configuration(
    main={"size": (1920, 1080)}, transform=Transform(hflip=1, vflip=1)))
picam.start(); time.sleep(2)

yolo = YOLO("/home/pi/yolov8n_ncnn_model")
tracker = BotSort(reid_model=None, with_reid=False, use_cmc=False,
                  track_high_thresh=0.35, track_low_thresh=0.2,
                  new_track_thresh=0.4, track_buffer=40, match_thresh=0.85)

COLORES = [(80, 200, 80), (80, 120, 255), (255, 160, 60), (200, 80, 200)]
registro = []
t0 = time.time()
i = 0
while time.time() - t0 < DUR_S:
    frame = picam.capture_array()
    r = yolo(frame, verbose=False, conf=0.3)[0]
    dts, metas = [], []
    for b in r.boxes:
        if r.names[int(b.cls[0])] not in ("person", "pedestrian", "people"):
            continue
        xyxy = b.xyxy[0].cpu().numpy()
        dts.append([*xyxy, float(b.conf[0]), 0])
        metas.append({"conf": float(b.conf[0])})
    arr = (np.array(dts, dtype="float32") if dts
           else np.empty((0, 6), dtype="float32"))
    res = np.asarray(tracker.update(arr, frame, embs=None))

    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    n_tid = 0
    for fila in res:
        det_idx = int(fila[7])
        if not (0 <= det_idx < len(dts)):
            continue
        x1, y1, x2, y2 = [int(v) for v in fila[:4]]
        tid = int(fila[4])
        n_tid += 1
        col = COLORES[tid % len(COLORES)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        etiqueta = f"id {tid}  {metas[det_idx]['conf']:.2f}"
        cv2.putText(vis, etiqueta, (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(vis, etiqueta, (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        cv2.circle(vis, ((x1 + x2) // 2, y2), 7, (0, 0, 255), -1)
    sello = f"RPi 5 | YOLOv8n NCNN + BoT-SORT | frame {i:03d}"
    cv2.putText(vis, sello, (20, 1060),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(vis, sello, (20, 1060),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.imwrite(f"{SALIDA}/f{i:04d}.jpg", vis,
                [cv2.IMWRITE_JPEG_QUALITY, 82])
    registro.append({"i": i, "dets": len(dts), "con_tid": n_tid})
    i += 1

picam.stop(); picam.close()
fps = i / (time.time() - t0)
with open(f"{SALIDA}/registro.json", "w") as f:
    json.dump({"frames": i, "fps": round(fps, 2), "reg": registro}, f)
print(f"{i} frames, {fps:.2f} FPS reales")
