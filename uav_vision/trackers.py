"""Builds any of boxmot's trackers by name, and says which of our settings it could not honour.

The chain has run one tracker since it was written, BoT-SORT, with thresholds calibrated against
the flight. boxmot ships ten, six of which use appearance, and the library already resolves them
by name, so what is missing is not a tracker zoo: it is an ADAPTER, and this is it.

WHAT AN ADAPTER IS FOR HERE, because it is the whole point of the file. Our calibration is four
numbers -- the score at which a detection may open a track, the floor under which it is not even
associated, the score at which a NEW track may be born, and how long a lost track is remembered.
Every tracker spells those differently and some do not have all four. A factory that passed them
blindly would either crash or, far worse, accept them silently and run on its own defaults while
the caller believed otherwise. So this filters our settings against the target's own signature
and RETURNS WHAT IT DROPPED. A comparison between trackers is only worth reading if it says which
of them were given the calibration and which were not.

No name from boxmot's API leaks past this module: callers speak of 'nombre' and a dict of
settings, and that is what keeps the rest of the package from depending on one library's spelling.

    tracker, aplicados, ignorados = construir("bytetrack", track_buffer=40)

DELIBERATELY NOT HERE: a base class. The package already talks to its camera by duck typing
against a documented contract, and three implementations that already answer the same update()
do not need a hierarchy on top.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

TRADUCCION: Dict[str, List[Tuple[str, str]]] = {
    "track_high_thresh": [("track_thresh", "exacta"), ("high_th", "exacta"),
                          ("det_thresh", "aproximada")],
    "track_low_thresh": [("low_th", "exacta"), ("min_conf", "exacta")],
    "new_track_thresh": [("new_track_th", "exacta")],
    "track_buffer": [("max_age", "exacta")],
    "match_thresh": [("match_th_first", "exacta"), ("iou_threshold", "aproximada")],
}
"""What each of our five calibrated settings is called elsewhere, and how safe the rename is.

Our names are BoT-SORT's, because BoT-SORT is what the chain was calibrated against, and seven
of boxmot's ten trackers do not have a parameter by any of those names. Without this table a
comparison across trackers measures OUR tuning against boxmot's own defaults, which are tuned on
ground-level pedestrian benchmarks and not on a drone at mission altitude. That is not a
comparison of trackers.

Candidates are tried in order and the first one the target accepts wins. Each carries how safe
the rename is, and the distinction is not decoration:

    exacta       the same quantity under another spelling. track_buffer and max_age are both
                 "frames a lost track survives"; high_th, track_thresh and track_high_thresh are
                 all "the score at which a detection enters the high band".
    aproximada   the same ROLE, not provably the same quantity. det_thresh gates a detection
                 before association in the SORT family, which is close to the high band but not
                 defined identically; iou_threshold is the association gate of the SORT family,
                 which plays the part BoT-SORT's match_thresh plays in its first association.

An approximate rename is worth making and worth DECLARING. A comparison that silently treats
them as identical is the quiet version of the mistake this whole module exists to prevent.
"""


def catalogo() -> Dict[str, bool]:
    """Every tracker boxmot can build, as name -> whether it needs an appearance model.

    Read off boxmot's own registry rather than written out here, so a version that adds or
    drops a tracker is reflected without this file being touched.
    """
    from boxmot.trackers.registry import TRACKER_DEFINITIONS
    return {n: bool(d.needs_reid) for n, d in sorted(TRACKER_DEFINITIONS.items())}


def _acepta(clase) -> List[str]:
    """The keyword arguments a tracker class takes BY NAME, ignoring its **kwargs.

    Ignoring **kwargs is the whole correctness of this module. Every one of boxmot's ten
    trackers declares one, so a setting it does not name is swallowed without a word and the
    tracker runs on its own default while the caller believes otherwise. The first version of
    this function treated "the class takes **kwargs" as "it accepts everything" and reported
    five settings applied for a tracker whose parameters are called something else entirely.
    A name that is not in a signature is NOT honoured, whatever **kwargs accepts.

    The WHOLE MRO is walked, and that is the other half. These trackers forward their **kwargs
    to a common base that owns det_thresh, max_age, min_hits, iou_threshold and asso_func, so
    reading the subclass alone says a tracker ignores settings it does in fact honour. The
    first version under-reported for exactly that reason.
    """
    nombres = set()
    for ancestro in inspect.getmro(clase):
        propio = ancestro.__dict__.get("__init__")
        if propio is None:
            continue
        for n, p in inspect.signature(propio).parameters.items():
            if n != "self" and p.kind not in (inspect.Parameter.VAR_KEYWORD,
                                              inspect.Parameter.VAR_POSITIONAL):
                nombres.add(n)
    return sorted(nombres)


def construir(nombre: str, embs_propias: bool = True, traducir: bool = True,
              **ajustes: Any) -> Tuple[Any, Dict[str, Any], Dict[str, Any], List[str]]:
    """One of boxmot's trackers, built by name, with as much of our calibration as it accepts.

    embs_propias says the caller computes the appearance vectors itself and hands them to
    update() through embs=. That is what this project does -- the camera owns the crops, so it
    owns the embeddings -- and it is why no ReID weights are loaded here even for the six
    trackers that use appearance.

    traducir renames our settings to whatever the target calls them, through TRADUCCION. With
    it off, a tracker that does not use BoT-SORT's spelling receives nothing and runs on its own
    defaults, which is the comparison nobody wants to publish by accident.

    Returns (tracker, aplicados, ignorados, aproximados), and the last two are not a detail to
    print once and forget. A tracker that did not receive the calibration is not being compared
    on the same terms as one that did, and a tracker that received an APPROXIMATE rename is
    being compared on terms that need stating. Anything measured across trackers has to carry
    both lists beside the numbers.
    """
    from boxmot.trackers.registry import TRACKER_DEFINITIONS, create_tracker

    if nombre not in TRACKER_DEFINITIONS:
        raise ValueError("boxmot no conoce '%s'; conoce %s"
                         % (nombre, sorted(TRACKER_DEFINITIONS)))
    definicion = TRACKER_DEFINITIONS[nombre]
    modulo, _, clase_nombre = definicion.class_path.rpartition(".")
    clase = getattr(__import__(modulo, fromlist=[clase_nombre]), clase_nombre)

    acepta = _acepta(clase)
    aplicados: Dict[str, Any] = {}
    ignorados: Dict[str, Any] = {}
    for clave, valor in ajustes.items():
        if clave in acepta:
            aplicados[clave] = valor
            continue
        destino = next(((otro, seguridad)
                        for otro, seguridad in TRADUCCION.get(clave, ())
                        if otro in acepta), None) if traducir else None
        if destino is None:
            ignorados[clave] = valor
        else:
            otro, seguridad = destino
            aplicados[otro] = valor
            if seguridad != "exacta":
                aplicados.setdefault("_aproximados", []).append("%s->%s" % (clave, otro))

    aproximados = aplicados.pop("_aproximados", [])
    tracker = create_tracker(nombre, precomputed_reid=embs_propias,
                             tracker_kwargs=aplicados or None)
    return tracker, aplicados, ignorados, aproximados
