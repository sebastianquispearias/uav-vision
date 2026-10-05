"""Generates the tracker comparison table, running every row instead of transcribing it.

The first nine-tracker table was assembled by hand from console output, which is the practice
``docs/RESULTADOS.md`` records as the one that goes stale silently: a table typed once stops
describing the code the moment the code changes, and nothing fails. This script runs the three
steps that produce a row -- track ids, replay, scoreboard -- for every tracker and prints the
markdown. Re-running it is the only way the table is allowed to be updated.

WHAT A ROW CANNOT BE WITHOUT, and all three were missing from the hand-made table:

CAMERA-MOTION COMPENSATION, asked for AND actually running. boxmot exposes it through four
different mechanisms and defaults it ON, so one flag reaches three of the ten trackers; the rest
either have no compensation at all or compensate no matter what is passed. The hand-made table
declared "no row ran with CMC" while three of its rows did. ``uav_vision.trackers.cmc_real``
reads the built instance, so the column says what ran rather than what was requested.

THE SETTINGS THAT DID NOT ARRIVE. A tracker that did not receive this deployment's calibration
is not being compared on the same terms as one that did, and the npz carries the lists so a row
cannot be printed without them.

WHETHER THE CONFIGURATION WORKED AT ALL. A tracker that assigned a track id to a handful of the
2637 detections is not tracking worse, it is not running, and scoring it as the worst of the zoo
is the mistake ``docs/DESCARTADO.md`` exists to prevent. The threshold below which a row is
refused a score is an explicit decision here, not a judgement made while reading the numbers.

Needs boxmot, so it runs with the training interpreter, and every step it spawns runs with the
same one:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/comparar_trackers.py

Artefacts are cached next to the flight archive, keyed by tracker AND compensation state, and a
row whose files are already there is not recomputed unless --rehacer says so. One tracker over
the whole flight is about 90 s without compensation and longer with it.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
LAC = os.path.dirname(REPO)
ARCHIVO = os.path.join(LAC, "drone-geolocation", "entrenamiento")
DATOS = os.path.join(REPO, "demo", "data")
N_DETECCIONES = 2637
PISO_ROTO = 0.10
"""Fraction of the flight's detections that must earn a track id for a row to be scored.

A DECISION, not a measurement, and it encodes which mistake is worse. Below this the run is not
a tracker performing badly, it is a configuration that did not associate, and the two look
identical in a people-and-phantoms column: both report few people. Publishing the second as the
first is how a library gets blamed for an adapter's bug, which already nearly happened here --
ocsort assigned 6 of 2637 because this project handed it a threshold that runs in the opposite
sense, and the table scored it 0 of 7 people.

0.10 sits in a gap in the measurements rather than at a round number that felt safe. The two
runs that were broken assigned 6 and 31 of 2637 detections, which is 0.2 % and 1.2 %; with the
calibration converted correctly the lowest row that tracks at all is ocsort at 437, which is
16.6 %. Any threshold between 1.2 % and 16.6 % selects exactly the same rows, so the figure
carries no weight of its own -- which is the property a decision like this should have.

IT CURRENTLY REFUSES NOTHING, and that is the point rather than a reason to delete it. The two
rows it was written for were broken by this project's own translation, and with that fixed every
tracker in the catalogue clears it. What it protects against is the next time, when the gap will
not be known in advance and the temptation will again be to read a near-zero row as a result.
If a future run lands between those two numbers, the run is what needs looking at, not this.
"""


def correr(cmd, env=None):
    """Runs one step and returns its stdout, failing loudly rather than returning a blank row.

    stderr is NOT discarded. A run of this comparison once reported every tracker as 'did not
    start' because a step's stderr went to DEVNULL and the real cause -- a missing argument --
    was thrown away with it.
    """
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=REPO)
    if p.returncode != 0:
        return None, "%s salio con %d:\n%s" % (os.path.basename(cmd[1]), p.returncode,
                                               (p.stderr or p.stdout)[-700:])
    return p.stdout, None


def una_fila(tracker, cmc, rehacer=False):
    """Everything one line of the table needs, measured, or the reason there is no line."""
    sufijo = "%s_pistas_02ago_cmc%s" % (tracker, cmc)
    npz = os.path.join(ARCHIVO, sufijo + ".npz")
    cand = os.path.join(ARCHIVO, "candidatos_%s_cmc%s_02ago.json" % (tracker, cmc))

    if rehacer or not os.path.exists(npz):
        _, err = correr([sys.executable, os.path.join(AQUI, "botsort_pistas.py"),
                         "--tracker=" + tracker, "--cmc=" + cmc])
        if err:
            return {"tracker": tracker, "cmc": cmc, "error": err}
    if not os.path.exists(npz):
        return {"tracker": tracker, "cmc": cmc,
                "error": "no se escribio %s" % os.path.basename(npz)}

    D = np.load(npz, allow_pickle=True)
    track = D["track"]
    ids, cuenta = np.unique(track[track >= 0], return_counts=True)
    fila = {"tracker": tracker, "cmc": cmc,
            "cmc_real": str(D["cmc_real"]) if "cmc_real" in D.files else "?",
            "con_id": int((track >= 0).sum()), "n": len(track), "pistas": len(ids),
            "mediana": int(np.median(cuenta)) if len(cuenta) else 0,
            "maximo": int(cuenta.max()) if len(cuenta) else 0,
            "ignora": [str(x) for x in D["ignora"]] if "ignora" in D.files else [],
            "aprox": [str(x) for x in np.atleast_1d(D["aproximados"])]
                     if "aproximados" in D.files else []}

    # La puerta que impide puntuar lo que no corrio. Va ANTES del replay y no despues de mirar
    # el marcador, porque un criterio elegido mirando el resultado no es un criterio.
    if fila["con_id"] < PISO_ROTO * fila["n"]:
        fila["roto"] = ("asigno %d de %d (%.1f %%), bajo el piso de %.0f %%"
                        % (fila["con_id"], fila["n"], 100.0 * fila["con_id"] / fila["n"],
                           100 * PISO_ROTO))
        return fila

    if rehacer or not os.path.exists(cand):
        _, err = correr([sys.executable, os.path.join(AQUI, "replay_vuelo3.py"),
                         "--pistas=" + npz, "--candidatos=" + cand],
                        env={"UAV_VISION_DATOS": DATOS})
        if err:
            fila["error"] = err
            return fila

    salida, err = correr([sys.executable, os.path.join(AQUI, "personas_encontradas.py"),
                          "--pistas", npz, "--candidatos", cand])
    if err:
        fila["error"] = err
        return fila
    m = re.search(r"personas reales reportadas: (\d+) de (\d+)", salida or "")
    f = re.search(r"FANTASMAS \(candidatos que no son nadie\): (\d+)", salida or "")
    s = re.search(r"sin juzgar \(fuera de los frames etiquetados\): (\d+)", salida or "")
    if not (m and f):
        fila["error"] = "no encontre el marcador en la salida de personas_encontradas"
        return fila
    fila.update(personas=int(m.group(1)), de=int(m.group(2)),
                fantasmas=int(f.group(1)), sin_juzgar=int(s.group(1)) if s else 0)
    return fila


def tabla(filas):
    """The markdown, with every declaration attached to the numbers rather than under them."""
    out = []
    puntuadas = [f for f in filas if "personas" in f]
    puntuadas.sort(key=lambda f: (-f["personas"], f["fantasmas"], -f["con_id"]))
    de = puntuadas[0]["de"] if puntuadas else 7

    out.append("| tracker | CMC | personas | fantasmas | sin juzgar | con id / %d | pistas | "
               "mediana/max cajas |" % N_DETECCIONES)
    out.append("|---|---|---|---|---|---|---|---|")
    for f in puntuadas:
        out.append("| %s | %s | %d de %d | %d | %d | %d | %d | %d / %d |"
                   % (f["tracker"], f["cmc_real"], f["personas"], de, f["fantasmas"],
                      f["sin_juzgar"], f["con_id"], f["pistas"], f["mediana"], f["maximo"]))

    rotas = [f for f in filas if "roto" in f]
    if rotas:
        out.append("")
        out.append("**SIN PUNTUAR, porque la corrida no asocio** (ver `PISO_ROTO` en "
                   "`scripts/comparar_trackers.py`): no son peores, no corrieron.")
        out.append("")
        out.append("| tracker | CMC | por que no se puntua | pistas |")
        out.append("|---|---|---|---|")
        for f in sorted(rotas, key=lambda f: f["tracker"]):
            out.append("| %s | %s | %s | %d |" % (f["tracker"], f["cmc_real"], f["roto"],
                                                  f["pistas"]))

    fallidas = [f for f in filas if "error" in f]
    if fallidas:
        out.append("")
        out.append("**NO CORRIERON**, con el motivo y no omitidas:")
        out.append("")
        for f in sorted(fallidas, key=lambda f: f["tracker"]):
            out.append("- `%s` (CMC %s): %s"
                       % (f["tracker"], f["cmc"], f["error"].splitlines()[-1][:160]))

    out.append("")
    out.append("**La columna CMC dice lo que CORRIO, no lo que se pidio.** Medido sobre la "
               "instancia construida con `uav_vision.trackers.cmc_real`:")
    out.append("")
    porestado = {}
    for f in filas:
        porestado.setdefault(f.get("cmc_real", "?"), set()).add(f["tracker"])
    for estado, quienes in sorted(porestado.items()):
        glosa = {"on": "compensan", "off": "la tienen y esta apagada",
                 "ninguno": "NO tienen el mecanismo, no es que este apagado"}.get(estado, "?")
        out.append("- **%s**: %s (%s)" % (estado, ", ".join(sorted(quienes)), glosa))

    conv = sorted({a for f in filas for a in f.get("aprox", []) if "opuesto" in a})
    rol = sorted({a for f in filas for a in f.get("aprox", []) if "opuesto" not in a})
    ign = sorted({i for f in filas for i in f.get("ignora", [])})
    out.append("")
    out.append("**Lo que no llego con su propio nombre.** Una fila que no recibio la "
               "calibracion no se compara en los mismos terminos que una que si:")
    out.append("")
    out.append("- renombres que **cambian el valor**: %s" % (", ".join("`%s`" % c for c in conv)
                                                             or "ninguno"))
    out.append("- renombres solo por rol: %s" % (", ".join("`%s`" % r for r in rol) or "ninguno"))
    out.append("- ajustes que **ningun** tracker de alguna fila pudo recibir: %s"
               % (", ".join("`%s`" % i for i in ign) or "ninguno"))
    return "\n".join(out)


def main():
    sys.path.insert(0, REPO)
    from uav_vision.trackers import catalogo

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trackers", default="",
                    help="comma-separated; the whole catalogue by default (%s)"
                         % ", ".join(catalogo()))
    ap.add_argument("--cmc", default="on",
                    help="comma-separated states to run: on, off, or 'on,off' for both")
    ap.add_argument("--rehacer", action="store_true",
                    help="recompute rows whose artefacts are already on disk")
    args = ap.parse_args()

    nombres = [t.strip() for t in args.trackers.split(",") if t.strip()] or list(catalogo())
    estados = [c.strip() for c in args.cmc.split(",") if c.strip()]
    filas = []
    for cmc in estados:
        for t in nombres:
            print("[%d/%d] %s, CMC %s" % (len(filas) + 1, len(nombres) * len(estados), t, cmc),
                  flush=True)
            filas.append(una_fila(t, cmc, args.rehacer))
            ultima = filas[-1]
            print("    %s" % (ultima.get("roto") or ultima.get("error", "").splitlines()[:1]
                              or "%d de %d personas, %d fantasmas, %d con id"
                              % (ultima.get("personas", -1), ultima.get("de", 0),
                                 ultima.get("fantasmas", -1), ultima.get("con_id", 0))),
                  flush=True)
    print()
    print(tabla(filas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
