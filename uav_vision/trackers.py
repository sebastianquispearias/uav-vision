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


MECANISMOS_CMC = ("use_cmc", "cmc_off", "cmc_method")
"""The parameter names through which boxmot's trackers expose camera-motion compensation.

CMC is the one setting that cannot be handled by renaming, and it is worth saying why, because
a comparison across trackers was already published on the wrong side of this. Measured on
boxmot 19.0.0 by walking every tracker's MRO and then reading the built instance's ``cmc``:

    use_cmc        botsort, boosttrack, occluboost     a boolean, DEFAULT True
    cmc_off        deepocsort                          the same boolean INVERTED, default False
    cmc_method     hybridsort                          no boolean at all; the factory returns
                                                       None for a method of None, so the method
                                                       IS the switch
    nothing        bytetrack, ocsort, sfsort, sam2mot  no compensation anywhere
    nothing        strongsort                          create_cmc("ecc") hardwired in __init__
                                                       and applied with no guard: ALWAYS ON

So ``use_cmc=False`` passed blindly reaches three of the ten. On deepocsort and hybridsort it
falls into 'ignored' while their compensation keeps running, and on strongsort there is no
parameter to reach at all. A table built that way says "no tracker used CMC" while three of its
rows did, which is not a milder version of the error: CMC is the single biggest confounder
measured on this flight, worth 36 track ids against 114 and two people on the scoreboard.

The defaults being ON is the other half. A tracker that silently ignores the request is not left
in the requested state, it is left in boxmot's, which is the opposite of what was asked.
"""


def _resolver_cmc(acepta: List[str], pedido: Any, metodo: Any, traducir: bool = True):
    """Camera-motion compensation translated to whichever mechanism the target actually has.

    Returns (aplicados, ignorados, notas). The notes are renames that CHANGE THE VALUE and not
    only the spelling, which is why they do not live in TRADUCCION: inverting a boolean or
    turning a switch into a method name cannot be expressed as a pair of names, and the safety
    labels there would not describe it.

    traducir off means no renaming here either, so a tracker that spells it differently receives
    nothing and the request lands in 'ignored'. That is what the flag is for: it measures how
    much of the calibration survives WITHOUT the adapter, and a CMC rename slipping through
    would flatter that measurement.
    """
    aplicados: Dict[str, Any] = {}
    ignorados: Dict[str, Any] = {}
    notas: List[str] = []

    if not traducir:
        for clave, valor in (("use_cmc", pedido), ("cmc_method", metodo)):
            if valor is None:
                continue
            (aplicados if clave in acepta else ignorados)[clave] = valor
        return aplicados, ignorados, notas

    if "use_cmc" in acepta:
        if pedido is not None:
            aplicados["use_cmc"] = bool(pedido)
        if metodo is not None and "cmc_method" in acepta:
            aplicados["cmc_method"] = metodo
        elif metodo is not None:
            ignorados["cmc_method"] = metodo
        return aplicados, ignorados, notas

    if "cmc_off" in acepta:
        if pedido is not None:
            aplicados["cmc_off"] = not bool(pedido)
            notas.append("use_cmc=%s->cmc_off=%s (invertida)" % (bool(pedido), not bool(pedido)))
        if metodo is not None:
            ignorados["cmc_method"] = metodo
        return aplicados, ignorados, notas

    if "cmc_method" in acepta:
        if pedido is False:
            aplicados["cmc_method"] = None
            notas.append("use_cmc=False->cmc_method=None (el metodo ES el interruptor)")
        elif metodo is not None:
            aplicados["cmc_method"] = metodo
        return aplicados, ignorados, notas

    if pedido is not None:
        ignorados["use_cmc"] = pedido
    if metodo is not None:
        ignorados["cmc_method"] = metodo
    return aplicados, ignorados, notas


def cmc_real(tracker: Any) -> str:
    """Whether camera-motion compensation is running on a BUILT tracker, read off the instance.

    This exists because what was asked for and what is running are different questions, and the
    gap between them is invisible in the settings report. A tracker whose request landed in
    'ignored' is not therefore uncompensated: strongsort ignores every CMC setting there is and
    compensates on every frame.

    The signal is boxmot's own ``self.cmc``, which its base helper checks for None before doing
    any work, so it is the same thing the library tests. Three answers, and the third is not the
    second: 'ninguno' means the tracker has no compensation to run, 'off' means it has one and
    it is disabled. Collapsing those two is how a tracker with no mechanism ends up counted as
    evidence that disabling it was harmless.
    """
    if not hasattr(tracker, "cmc"):
        return "ninguno"
    return "on" if tracker.cmc is not None else "off"


CONVERSIONES: Dict[Tuple[str, str], Tuple[Any, str]] = {
    ("match_thresh", "iou_threshold"): (lambda v: round(1.0 - v, 10), "exacta"),
}
"""The renames that must also change the VALUE, because the two settings run in opposite senses.

There is one, and finding it cost a whole published table. BoT-SORT's match_thresh is a limit on
a COST and boxmot's iou_threshold is a floor on an IoU, so the same number means opposite things.
Traced in boxmot 19.0.0:

    matching.py:79   cost_matrix = 1 - ious          BoT-SORT associates on 1 - IoU
    matching.py:35   lap.lapjv(..., cost_limit=thresh)   accepts cost <= match_thresh
                     => a match needs IoU >= 1 - match_thresh
    stages.py:152    if iou_matrix.max(...) <= threshold: return   no match at all
    boost.py:196     conf[iou_matrix < iou_threshold] = 0
    hybrid.py:343    a = (iou_matrix > iou_threshold).astype(np.int32)
    hybrid.py:363    if iou_matrix[m[0], m[1]] < iou_threshold: continue
    occluboost.py:956  cost[iou < self.iou_threshold] = 1e6
                     => a match needs IoU > iou_threshold

So match_thresh = 0.85, which is the most PERMISSIVE value this calibration carries (any pair
overlapping by 0.15 may associate), arrived at ocsort, deepocsort, boosttrack, occluboost and
hybridsort as the STRICTEST value possible: boxes had to overlap by 0.85 to be the same track.
For a person a few dozen pixels tall seen from 25 m, that essentially never happens. Measured on
400 frames of the 02-ago flight, ocsort went from 6 detections with a track id to 239 by undoing
nothing but this, and to 487 of 764 once its output gate was also accounted for -- against
BoT-SORT's 454 on the same frames. The first nine-tracker table scored that configuration 0 of 7
people and would have published "OC-SORT does not work for this case".

The conversion is 1 - v and it is DERIVED, not fitted: it is the value at which both expressions
admit the same pairs. The labels stay honest about the one gap left -- BoT-SORT's bound is
inclusive and boxmot's is strict, so a pair at exactly IoU = 1 - match_thresh is associated by
one and refused by the other, which is a single point and not a tuning difference.

TRADUCCION stays a table of NAMES on purpose. A conversion cannot be expressed as a pair of
names, and a safety label on a rename would not describe it, so it lives here where a reader
looking for "what did the adapter change" finds it in one place with its derivation.
"""


RUTAS_DE_RESPALDO = {
    "botsort": ("boxmot.trackers.bbox.botsort", "boxmot.trackers.box.botsort",
                "boxmot", "BotSort"),
}
"""Where to find a tracker when boxmot has no registry, for the one tracker that flies.

boxmot MOVED its trackers between the version on one board and the version on the other -- from
boxmot.trackers.bbox.X to boxmot.trackers.box.X -- and camera.py has carried a two-way import
for that reason since before this module existed. The registry this module prefers is newer
than that, so it cannot be assumed present on every board, and a tracker factory that only
works on one of the two aircraft is worse than the hardcoded line it replaces.

Only BoT-SORT is listed, and the limit is the honest one: it is the tracker the chain flies, so
it is the one that must build on every board. The other nine are for comparing on the ground,
where the registry is there.
"""


def hay_registro() -> bool:
    """Whether this boxmot has the registry that resolves trackers by name."""
    try:
        from boxmot.trackers.registry import TRACKER_DEFINITIONS  # noqa: F401
    except Exception:
        return False
    return True


def catalogo() -> Dict[str, bool]:
    """Every tracker boxmot can build, as name -> whether it needs an appearance model.

    Read off boxmot's own registry rather than written out here, so a version that adds or
    drops a tracker is reflected without this file being touched. Without a registry only the
    fallback names are offered, because a catalogue this module invented would be a guess.
    """
    if not hay_registro():
        return dict.fromkeys(RUTAS_DE_RESPALDO, True)
    from boxmot.trackers.registry import TRACKER_DEFINITIONS
    return {n: bool(d.needs_reid) for n, d in sorted(TRACKER_DEFINITIONS.items())}


def _clase(nombre: str):
    """The tracker class, through the registry when there is one and by hand when there is not."""
    if hay_registro():
        from boxmot.trackers.registry import TRACKER_DEFINITIONS
        if nombre not in TRACKER_DEFINITIONS:
            raise ValueError("boxmot no conoce '%s'; conoce %s"
                             % (nombre, sorted(TRACKER_DEFINITIONS)))
        modulo, _, cn = TRACKER_DEFINITIONS[nombre].class_path.rpartition(".")
        return getattr(__import__(modulo, fromlist=[cn]), cn)

    if nombre not in RUTAS_DE_RESPALDO:
        raise ValueError(
            "este boxmot no trae registro, asi que solo se puede construir %s por respaldo; "
            "'%s' necesita una version con boxmot.trackers.registry"
            % (sorted(RUTAS_DE_RESPALDO), nombre))
    *modulos, cn = RUTAS_DE_RESPALDO[nombre]
    for modulo in modulos:
        try:
            return getattr(__import__(modulo, fromlist=[cn]), cn)
        except (ImportError, AttributeError):
            continue
    raise ImportError("no encontre %s en ninguna de %s" % (cn, modulos))


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

    THE CLASS IS CONSTRUCTED DIRECTLY AND BOXMOT'S YAML IS NEVER LOADED, and that is a decision
    with a measurement behind it. boxmot's create_tracker merges the caller's settings over a
    per-tracker YAML, and those files carry the result of a hyperparameter search on MOT: going
    through it moved BoT-SORT's appearance_thresh from 0.25 to 0.6188818853936099 and its
    proximity_thresh from 0.5 to 0.6084297894561342 -- two appearance gates the flight was never
    calibrated with, changed silently, for parameters nobody here set. So the only values that
    are not a class default are the ones passed in. That also makes the comparison across
    trackers mean something: every one of them runs on its own defaults plus OUR calibration,
    and not on somebody's tuning for ground-level pedestrians.

    Returns (tracker, aplicados, ignorados, aproximados), and the last two are not a detail to
    print once and forget. A tracker that did not receive the calibration is not being compared
    on the same terms as one that did, and a tracker that received an APPROXIMATE rename is
    being compared on terms that need stating. Anything measured across trackers has to carry
    both lists beside the numbers.
    """
    clase = _clase(nombre)
    acepta = _acepta(clase)
    aplicados: Dict[str, Any] = {}
    ignorados: Dict[str, Any] = {}

    # CMC va aparte del lazo porque es el unico ajuste cuyo destino cambia el VALOR y no solo
    # el nombre, y porque dos trackers lo honran bajo un nombre que ningun renombre alcanza.
    cmc_ap, cmc_ig, cmc_notas = _resolver_cmc(
        acepta, ajustes.pop("use_cmc", None), ajustes.pop("cmc_method", None), traducir)
    aplicados.update(cmc_ap)
    ignorados.update(cmc_ig)

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
            convertir, seguridad = CONVERSIONES.get((clave, otro), (None, seguridad))
            aplicados[otro] = convertir(valor) if convertir else valor
            if convertir:
                aplicados.setdefault("_aproximados", []).append(
                    "%s=%s->%s=%s (sentido opuesto)" % (clave, valor, otro, aplicados[otro]))
            elif seguridad != "exacta":
                aplicados.setdefault("_aproximados", []).append("%s->%s" % (clave, otro))

    aproximados = aplicados.pop("_aproximados", []) + cmc_notas

    # El cableado del adaptador, aparte de la calibracion del llamador: no se cuenta como
    # "honrado" porque nadie lo pidio. Un informe que los sumara diria "7 de 5" y el numero
    # que el llamador quiere leer es cuanto de LO SUYO llego.
    cableado = {}
    if "with_reid" in acepta and "with_reid" not in aplicados:
        cableado["with_reid"] = embs_propias
    if "reid_model" in acepta and "reid_model" not in aplicados:
        cableado["reid_model"] = None

    tracker = clase(**aplicados, **cableado)
    return tracker, aplicados, ignorados, aproximados
