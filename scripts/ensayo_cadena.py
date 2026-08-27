"""Full-chain rehearsal on the Raspberry Pi, desk scenario.

Runs the deployment chain live on the real hardware:

    ArduCam -> YOLO (VisDrone NCNN) -> BoT-SORT -> ground impact -> IncrementalIdentity

Three phases, one per optional stage, so the cost of each stage is isolated:

    A  detector only
    B  detector + BoT-SORT (track_id)
    C  detector + BoT-SORT + OSNet embeddings

Per phase it reports latency (median/p90), achieved FPS, CPU %, RAM, temperature
and throttling flags. Phases B and C also feed IncrementalIdentity and print the
candidate list at the end: with a person walking in front of the camera the
expected outcome is a stable track_id and one dominant candidate.

Desk geometry is stated, not simulated: the camera position/yaw passed to the
projection are fixed placeholders. Accuracy is NOT under test here (no GPS, no
flight geometry); only the wiring and the compute cost are.
"""

import argparse
import json
import subprocess
import time

import numpy as np

from uav_vision.camera import OnboardCamera
from uav_vision.identity import IncrementalIdentity
from uav_vision.pinhole_local import pixel_to_ray

MODELO = "/home/pi/modelos_visdrone/best_ncnn_model"
OSNET = "/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt"

# Desk-mount stand-in for the flight telemetry: camera ~1.2 m high, pointing
# north, tilted slightly down. Impacts are only meaningful relative to each
# other, which is all the identity layer needs.
POS = (0.0, 0.0, 1.2)
YAW = 0.0
PITCH = -20.0


def leer_cpu():
    with open("/proc/stat") as f:
        campos = [float(x) for x in f.readline().split()[1:]]
    idle = campos[3] + campos[4]
    return idle, sum(campos)


def cpu_pct(antes, ahora):
    didle = ahora[0] - antes[0]
    dtotal = ahora[1] - antes[1]
    return 100.0 * (1.0 - didle / dtotal) if dtotal > 0 else 0.0


def temperatura():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000.0


def ram_disponible_mb():
    with open("/proc/meminfo") as f:
        for linea in f:
            if linea.startswith("MemAvailable"):
                return int(linea.split()[1]) / 1024.0
    return -1.0


def throttled():
    return subprocess.run(["vcgencmd", "get_throttled"],
                          capture_output=True, text=True).stdout.strip()


def impacto_suelo(det, cam):
    (ox, oy, oz), (dx, dy, dz) = pixel_to_ray(
        POS, YAW, (det["px"], det["py"]), PITCH,
        cam.camera.focal_length_px, cam.camera.image_width,
        cam.camera.image_height, cam.camera.principal_point)
    if dz >= -1e-6:  # ray does not descend: no ground intersection
        return None
    t = -oz / dz
    return (ox + t * dx, oy + t * dy)


def fase(nombre, dur_s, tracker, reid, fps_esperado, modelo):
    print(f"\n===== FASE {nombre} | tracker={tracker} reid={reid is not None} "
          f"| {dur_s}s =====", flush=True)
    cam = OnboardCamera(
        model=modelo,
        threshold=0.3,
        tracker=tracker,
        fps=fps_esperado if tracker else None,
        reid_model=reid,
    )
    identity = None
    if tracker:
        # Short maturation thresholds so one 60 s phase can produce a reported
        # candidate; the flight values live in the protocol, not here.
        identity = IncrementalIdentity(
            fusion_radius_m=0.6, fps=fps_esperado,
            track_dur_s=4.0, mobile_dur_s=15.0, report_dur_s=20.0)

    t_arranque = time.time()
    cam.detect(POS, YAW)  # first call: camera start + model load
    print(f"arranque (camera + modelos): {time.time() - t_arranque:.1f} s", flush=True)

    lat, n_det, con_tid, tids = [], 0, 0, set()
    temps = []
    cpu0 = leer_cpu()
    t0 = time.time()
    frame = 0
    while time.time() - t0 < dur_s:
        t1 = time.time()
        dets = cam.detect(POS, YAW)
        lat.append((time.time() - t1) * 1000.0)
        frame += 1
        n_det += len(dets)
        for det in dets:
            tid = det.get("track_id")
            if tid is not None:
                con_tid += 1
                tids.add(tid)
                if identity is not None:
                    imp = impacto_suelo(det, cam)
                    if imp is not None:
                        identity.observe(frame, tid, imp, det["conf"],
                                           det.get("emb"))
        if frame % 25 == 0:
            temps.append(temperatura())
            print(f"  {frame:4d} frames | {np.median(lat):6.1f} ms med | "
                  f"{len(dets)} det | temp {temps[-1]:.1f}C", flush=True)
    cpu1 = leer_cpu()
    temps.append(temperatura())

    dur = time.time() - t0
    resumen = {
        "fase": nombre,
        "frames": frame,
        "fps": round(frame / dur, 2),
        "lat_med_ms": round(float(np.median(lat)), 1),
        "lat_p90_ms": round(float(np.percentile(lat, 90)), 1),
        "detecciones": n_det,
        "con_track_id": con_tid,
        "track_ids_distintos": sorted(tids),
        "cpu_pct": round(cpu_pct(cpu0, cpu1), 1),
        "temp_max_C": max(temps),
        "ram_disp_mb": round(ram_disponible_mb(), 0),
        "throttled": throttled(),
    }
    print(json.dumps(resumen, indent=2), flush=True)

    if identity is not None:
        cands = identity.candidates()
        print(f"candidates: {len(cands)}", flush=True)
        for c in cands:
            print(f"  mobile={c['mobile']} pos=({c['x']:.2f},{c['y']:.2f}) "
                  f"n_obs={c['n_obs']} conf={c['conf']:.2f}", flush=True)

    cam.close()
    time.sleep(3)  # let the sensor pipeline release cleanly between phases
    return resumen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=60.0, help="seconds per phase")
    ap.add_argument("--fases", default="ABC", help="subset of ABC to run")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="expected capture rate for tracker/identity buffers")
    ap.add_argument("--modelo", default=MODELO,
                    help="YOLO model path (NCNN dir or .pt)")
    args = ap.parse_args()

    resumenes = []
    if "A" in args.fases:
        resumenes.append(fase("A_detector", args.dur, False, None, args.fps,
                              args.modelo))
    if "B" in args.fases:
        resumenes.append(fase("B_botsort", args.dur, True, None, args.fps,
                              args.modelo))
    if "C" in args.fases:
        resumenes.append(fase("C_osnet", args.dur, True, OSNET, args.fps,
                              args.modelo))

    print("\n===== RESUMEN =====")
    for r in resumenes:
        print(f"{r['fase']:12s} fps={r['fps']:5.2f} lat={r['lat_med_ms']:6.1f} ms "
              f"cpu={r['cpu_pct']:5.1f}% temp={r['temp_max_C']:.1f}C "
              f"dets={r['detecciones']} tids={len(r['track_ids_distintos'])} "
              f"{r['throttled']}")
    with open("/home/pi/ensayo_cadena_resumen.json", "w") as f:
        json.dump(resumenes, f, indent=2)


if __name__ == "__main__":
    main()
