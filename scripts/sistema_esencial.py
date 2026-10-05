"""
The whole system in one file: from a pixel to a point on the operator's map.

This is not a simplification that leaves the hard parts out. It is the same chain the aircraft
runs, with the same constants, written end to end so it can be read in one sitting. The real code
is spread over camera.py, pinhole_local.py, identity.py and vision_protocol.py because each of
those has to handle a real camera, a real radio and a real autopilot; what is here is the part
that decides, and nothing else.

The chain, in one line per stage:

    a box in the image  ->  a bearing ray in the world  ->  a point on the ground
    points of one track  ->  a track's estimate  ->  tracks merged into a candidate
    a candidate with enough evidence  ->  a POI the ground station draws

What the system does NOT do is as important: it never turns one sighting into a point. A single
box is a guess, and a guess on a search map sends someone to the wrong place.

Run: python scripts/sistema_esencial.py

THE NUMBERS ARE DECISIONS, NOT PHYSICS
    They encode the cost of being wrong, and they change per mission. The flight they were
    fixed on flew at 20-40 m over a campus.

    UMBRAL_DETECTOR
        Below this the detector is guessing more than it is seeing.

    MIN_MEDICIONES
        Sightings before a candidate may be reported at all.

    RADIO_FUSION_M
        gps sigma plus slant range times yaw error: how far two sightings of one thing can land
        apart. Its formula is in identity.py's docstring.

    EMB_DIST_MAX
        OSNet distance above which two crops are different people.

    COOCURRENCIA_MIN
        Seen together in this many frames means two things, whatever else says.

    DUTY_MIN
        A track must be detected in at least this fraction of its own span.

    NOTES.md has the measurement behind each one.
"""
import math

import numpy as np

UMBRAL_DETECTOR = 0.25
MIN_MEDICIONES = 8
RADIO_FUSION_M = 3.5
EMB_DIST_MAX = 0.95
COOCURRENCIA_MIN = 3
DUTY_MIN = 0.10


def rotacion_mundo_a_camara(yaw_deg, pitch_deg):
    """Rows are the camera axes in world coordinates: X right, Y down, Z along the optical axis.

    'rumbo' is where the drone faces and 'derecha' is 90 degrees clockwise from it.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    rumbo = np.array([math.sin(y), math.cos(y), 0.0])
    derecha = np.array([math.cos(y), -math.sin(y), 0.0])
    eje = rumbo * math.cos(p) + np.array([0, 0, 1.0]) * math.sin(p)   # pitch -90 looks straight down
    return np.array([derecha, np.cross(eje, derecha), eje])


def pixel_a_suelo(pos_dron, yaw_deg, pixel, pitch_deg, focal_px, ancho, alto, suelo_z=0.0):
    """Where the ground point under one pixel is, given where the aircraft was and where it looked.

    A camera measures a DIRECTION, never a distance: the same pixel is a person at 20 m or a mark
    on the roof at 25 m. What closes the problem is assuming the target touches the ground, so the
    ray is intersected with the plane z = suelo_z. That assumption is why the box's BOTTOM edge is
    used as the pixel, and why a person on a balcony lands further away than they are.

    The rotation is transposed to invert it, which it is allowed to be because it is
    orthogonal. A ray that runs parallel to the ground or points up never meets the plane.
    """
    cx, cy = ancho / 2.0, alto / 2.0
    d_cam = np.array([(pixel[0] - cx) / focal_px, (pixel[1] - cy) / focal_px, 1.0])
    d = rotacion_mundo_a_camara(yaw_deg, pitch_deg).T @ d_cam    # R is orthogonal, so R.T inverts it
    d /= np.linalg.norm(d)
    if d[2] >= -1e-9:                                           # the ray never meets the ground
        return None
    t = (suelo_z - pos_dron[2]) / d[2]
    return np.array([pos_dron[0] + t * d[0], pos_dron[1] + t * d[1]])


def resumir_pista(pista):
    """One track's estimate: where it is, what it looks like, how solid the evidence is.

    The median, not the mean: one bad ray (a box on a shadow, a yaw glitch) moves a mean and does
    not move a median. The duty cycle guards against a track whose span overstates it -- five
    sightings spread over a minute are not a minute of evidence.
    """
    puntos = np.array(pista["puntos"])
    marcos = pista["frames"]
    ciclo = len(puntos) / max(1, max(marcos) - min(marcos) + 1)
    huella = np.mean(pista["embs"], axis=0)
    return {"pos": np.median(puntos, axis=0), "n": len(puntos), "frames": set(marcos),
            "emb": huella / (np.linalg.norm(huella) + 1e-9), "ciclo": ciclo,
            "cls": pista["cls"], "tid": pista["tid"]}


def fusionar(pistas):
    """Merges tracks into candidates: the rules apply in order and any one of them vetoes.

    The order is not cosmetic. Class first, because a car is not the person standing beside it
    however alike their crops look at 35 m. Co-occurrence second, because nothing appears twice in
    the same photograph: two tracks seen together are two things, whatever position and appearance
    say. Only then distance, and only then appearance.

    A track whose duty cycle is under the floor is evidence spread too thin and is dropped
    before any of that. When a track joins a candidate, the position is reweighted by how much
    evidence each side brings.
    """
    candidatos = []
    for tk in sorted((resumir_pista(p) for p in pistas), key=lambda t: -t["n"]):
        if tk["ciclo"] < DUTY_MIN:
            continue
        mejor, puntaje_min = None, math.inf
        for c in candidatos:
            if tk["cls"] is not None and c["cls"] is not None and tk["cls"] != c["cls"]:
                continue
            if len(tk["frames"] & c["frames"]) >= COOCURRENCIA_MIN:
                continue
            dp = float(np.linalg.norm(tk["pos"] - c["pos"]))
            if dp >= RADIO_FUSION_M:
                continue
            de = float(np.linalg.norm(tk["emb"] - c["emb"]))
            if de >= EMB_DIST_MAX:
                continue
            puntaje = dp / RADIO_FUSION_M + 0.5 * de / EMB_DIST_MAX
            if puntaje < puntaje_min:
                mejor, puntaje_min = c, puntaje
        if mejor is None:
            candidatos.append(dict(tk, tids=[tk["tid"]]))
            continue
        peso = mejor["n"] / (mejor["n"] + tk["n"])
        mejor["pos"] = peso * mejor["pos"] + (1 - peso) * tk["pos"]
        mejor["n"] += tk["n"]
        mejor["frames"] |= tk["frames"]
        mejor["tids"].append(tk["tid"])
    return candidatos


def volar(camara, telemetria, camara_cfg, pitch_deg, periodo_reporte_s=2.0, emitir=print):
    """Frame by frame: see, project, accumulate. Every two seconds: merge and report.

    The reporting period is not the detection period. The aircraft looks at three frames a second
    and speaks every two seconds, because what travels is a decision and not a video: one report is
    4.5 KB against the 9 Mbit/s the same view would cost as a stream.

    A detection the tracker did not claim is a guess, so it never reaches the identity layer.
    """
    pistas, proximo_reporte = {}, periodo_reporte_s
    for marco, (pos, yaw, t) in enumerate(telemetria):
        for det in camara.detectar(marco):
            if det["conf"] < UMBRAL_DETECTOR or det.get("track_id") is None:
                continue
            suelo = pixel_a_suelo(pos, yaw, det["pixel"], pitch_deg, camara_cfg["focal_px"],
                                  camara_cfg["ancho"], camara_cfg["alto"])
            if suelo is None:
                continue
            p = pistas.setdefault(det["track_id"],
                                  {"puntos": [], "embs": [], "frames": [], "cls": det.get("cls"),
                                   "tid": det["track_id"]})
            p["puntos"].append(suelo)
            p["embs"].append(det["emb"])
            p["frames"].append(marco)
        if t < proximo_reporte:
            continue
        proximo_reporte += periodo_reporte_s
        pois = [{"x": round(float(c["pos"][0]), 2), "y": round(float(c["pos"][1]), 2),
                 "n_obs": c["n"], "pistas": c["tids"]}
                for c in fusionar(pistas.values()) if c["n"] >= MIN_MEDICIONES]
        emitir({"type": "vision_poi", "t": round(t, 1), "pois": pois})
    return pistas


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    CFG = {"focal_px": 1400.0, "ancho": 1920, "alto": 1080}
    PERSONAS = {"A": np.array([0.0, 8.0]), "B": np.array([12.0, 16.0])}
    HUELLAS = {k: rng.normal(size=8) for k in PERSONAS}

    class CamaraFalsa:
        """Projects the real people back into the image, with the noise a real detector has.

        PERSONAS holds where they really are, which is what the printed answer is judged
        against. The detector is made to miss about half the frames, which is the order of what
        a real one does on this material.
        """

        def detectar(self, marco):
            pos, yaw, _ = TELEMETRIA[marco]
            salida = []
            for k, suelo in PERSONAS.items():
                R = rotacion_mundo_a_camara(yaw, PITCH)
                d = R @ (np.array([suelo[0], suelo[1], 0.0]) - pos)
                if d[2] <= 0:
                    continue
                px = CFG["focal_px"] * d[0] / d[2] + CFG["ancho"] / 2
                py = CFG["focal_px"] * d[1] / d[2] + CFG["alto"] / 2
                if not (0 <= px < CFG["ancho"] and 0 <= py < CFG["alto"]):
                    continue
                if rng.random() > 0.55:
                    continue
                h = HUELLAS[k] + rng.normal(scale=0.05, size=8)
                salida.append({"pixel": (px + rng.normal(scale=3), py + rng.normal(scale=3)),
                               "conf": 0.6, "track_id": k, "cls": "person",
                               "emb": h / np.linalg.norm(h)})
            return salida

    PITCH = -55.0
    TELEMETRIA = [((0.0, -6.0 + 0.25 * i, 25.0), 0.0, i / 3.0) for i in range(120)]
    print("vuelo de %d cuadros a 3 FPS, dos personas en el suelo" % len(TELEMETRIA))
    for k, v in PERSONAS.items():
        print("   %s esta de verdad en (%.1f, %.1f)" % (k, v[0], v[1]))
    print()
    volar(CamaraFalsa(), TELEMETRIA, CFG, PITCH,
          emitir=lambda r: print("t=%5.1f s  %s" % (r["t"], r["pois"])) if r["pois"] else None)
