"""
Gates for which crop a candidate carries to the ground station.

The crop is the one piece of the report a person judges with their eyes, and the ground station's
appearance filter scores it before they do. It has to show what the candidate mostly is. The
most confident sighting is not that: on flight 3 a candidate whose boxes were 43 non-persons, two
of the operator and one of another person travelled with a box holding two people, the filter
called it a person, and a false alert reached the operator.

Each section is a contrast between the two rules on the same scene, so it fails if the rule
under test behaves like the other one:

  1. One track with one odd, very confident sighting: "confidence" keeps the odd one,
     "appearance" keeps a sighting that looks like the rest of the track.
  2. Two tracks fused into one candidate, the minority one with the more confident crops:
     "confidence" sends the minority's crop, "appearance" the majority's.
  3. Without appearance vectors there is nothing to compare, and "appearance" falls back to
     confidence instead of dropping the crop.
  4. The crop rule decides a photograph, never a position: both rules report the same points.

Run with: python tests/test_recorte_representativo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from uav_vision.identity import IncrementalIdentity

FALLOS = []
RNG = np.random.default_rng(11)


def revisar(condicion, descripcion, detalle=""):
    marca = "ok  " if condicion else "FALLA"
    print("  [%s] %s%s" % (marca, descripcion, (" -- " + detalle) if detalle else ""))
    if not condicion:
        FALLOS.append(descripcion)


def unitario(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


A = unitario(np.random.default_rng(1).normal(size=512))
B = unitario(np.random.default_rng(2).normal(size=512))


def parecido_a(base, ruido=0.3):
    return unitario(base + ruido * RNG.normal(size=512) / np.sqrt(512) * 3)


def nueva(regla):
    return IncrementalIdentity(fusion_radius_m=3.5, fps=1.0, crop_choice=regla)


# ================================================================ 1. una pista
print("=" * 64)
print("1. Una pista con una vista rara y muy confiada")
print("=" * 64)


def escena_pista(ident):
    for f in range(30):
        ident.observe(frame=f, track_id=1, ground_xy=(10.0, 5.0), conf=0.55,
                      emb=parecido_a(A), crop=b"parecida-%d" % f, t=float(f))
        if f == 15:
            # Two people in one box: half of what the track looks like, and the detector is sure.
            ident.observe(frame=f, track_id=1, ground_xy=(10.0, 5.0), conf=0.95,
                          emb=unitario(A + B), crop=b"doble", t=float(f) + 0.5)
    return ident.candidates(preliminary=True)


vieja = escena_pista(nueva("confidence"))
nueva_ = escena_pista(nueva("appearance"))
revisar(len(vieja) == 1 and len(nueva_) == 1, "un candidato con cada regla",
        "%d y %d" % (len(vieja), len(nueva_)))
if vieja and nueva_:
    revisar(vieja[0]["crop"] == b"doble", "'confidence' manda la vista mas confiada",
            repr(vieja[0]["crop"]))
    revisar(bool(nueva_[0]["crop"]) and nueva_[0]["crop"].startswith(b"parecida-"),
            "'appearance' manda una vista parecida al resto de la pista", repr(nueva_[0]["crop"]))

# ========================================================== 2. dos pistas, un candidato
print()
print("=" * 64)
print("2. Dos pistas fusionadas: la minoria trae los recortes mas confiados")
print("=" * 64)


def escena_candidato(ident):
    for f in range(40):
        ident.observe(frame=f, track_id=1, ground_xy=(10.0, 5.0) + RNG.normal(0, 0.3, 2),
                      conf=0.50, emb=parecido_a(A), crop=b"mayoria-%d" % f, t=float(f))
    for f in range(50, 58):
        ident.observe(frame=f, track_id=2, ground_xy=(10.2, 5.1) + RNG.normal(0, 0.3, 2),
                      conf=0.95, emb=parecido_a(unitario(A + B), 0.1), crop=b"minoria-%d" % f,
                      t=float(f))
    return ident.candidates(preliminary=True, with_tracks=True)


vieja = escena_candidato(nueva("confidence"))
nueva_ = escena_candidato(nueva("appearance"))
revisar(len(vieja) == 1 and vieja[0]["tracks"] == [1, 2]
        and len(nueva_) == 1 and nueva_[0]["tracks"] == [1, 2],
        "las dos pistas fusionan en un candidato con cada regla",
        "%s y %s" % ([c["tracks"] for c in vieja], [c["tracks"] for c in nueva_]))
if vieja and nueva_:
    revisar(bool(vieja[0]["crop"]) and vieja[0]["crop"].startswith(b"minoria-"),
            "'confidence' manda el recorte de la pista minoritaria", repr(vieja[0]["crop"]))
    revisar(bool(nueva_[0]["crop"]) and nueva_[0]["crop"].startswith(b"mayoria-"),
            "'appearance' manda el de la pista que es la mayor parte de la evidencia",
            repr(nueva_[0]["crop"]))

# ============================================================ 3. sin vectores
print()
print("=" * 64)
print("3. Sin vectores de apariencia: vuelve a la confianza, no pierde el recorte")
print("=" * 64)

ident = nueva("appearance")
for f in range(40):
    ident.observe(frame=f, track_id=1, ground_xy=(1.0, 1.0), conf=0.40, crop=b"borroso", t=float(f))
for f in range(40, 80):
    ident.observe(frame=f, track_id=2, ground_xy=(1.1, 1.0), conf=0.91, crop=b"nitido", t=float(f))
sin = ident.candidates(preliminary=True)
revisar(len(sin) == 1 and sin[0]["crop"] == b"nitido",
        "gana la vista mas confiada", repr([c["crop"] for c in sin]))

# ======================================================= 4. las posiciones no cambian
print()
print("=" * 64)
print("4. La regla elige una foto, no un punto")
print("=" * 64)
RNG = np.random.default_rng(11)
vieja = escena_candidato(nueva("confidence"))
RNG = np.random.default_rng(11)
nueva_ = escena_candidato(nueva("appearance"))
sin_crop = lambda cs: [{k: v for k, v in c.items() if k not in ("crop", "emb")} for c in cs]
revisar(sin_crop(vieja) == sin_crop(nueva_), "mismo informe salvo el recorte",
        "%s" % sin_crop(nueva_))
revisar(vieja[0]["crop"] != nueva_[0]["crop"], "y el recorte si difiere, o la comparacion no prueba nada")

try:
    nueva("la-mas-linda")
    revisar(False, "una regla desconocida se rechaza")
except ValueError:
    revisar(True, "una regla desconocida se rechaza")

print()
if FALLOS:
    print("FALLARON %d:" % len(FALLOS))
    for f in FALLOS:
        print("  -", f)
    sys.exit(1)
print("TODO OK")
