"""
Fleet rules: when two sightings are one target, and when one deserves a second look.

These live in the package, not in the ground station, because both sides need them and a
second copy of a threshold is how a drone and its station quietly stop agreeing on what
counts as the same object. The station imports them today; a drone that listens to its
neighbours imports the same ones.

Nothing here knows about HTTP, drawing, or where it runs: data in, decisions out.
"""

import base64
import math

import numpy as np

from uav_vision.identity import EMB_DIST_MAX_MEDIDO, SEPARACION_MINIMA_M, radii_by_class

RADIOS = radii_by_class(3.5)


def vector_de(poi):
    """
    The appearance vector of a POI, or None when there is none.

    Accepts both forms it comes in: the raw array, which is what a drone has in hand about its
    own targets, and the base64 the transport puts on the wire, which is how it arrives from
    anyone else. A drone comparing its own sighting against a neighbour's holds one of each.
    """
    e = poi.get('emb')
    if e is None:
        return None
    try:
        if isinstance(e, (str, bytes)):
            v = np.frombuffer(base64.b64decode(e), dtype=np.float16).astype(np.float32)
        else:
            v = np.asarray(e, dtype=np.float32).ravel()
    except Exception:
        return None
    if v.size == 0:
        return None
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def distancia_maxima(a, b):
    """
    How far apart two drones' reports of one target can plausibly be, in metres.

    Each drone carries its own GPS and heading bias, so the same target seen by two of them lands in
    two places. When both reports carry their 95 % radius, the difference between two independent
    errors falls inside sqrt(ra^2 + rb^2) 95 % of the time -- about 7 m for two 5 m radii. A fixed
    3.5 m, the old rule, is narrower than either drone's own margin, and pairs of reports of one
    target stayed two pins. The two-drone bench never showed it: both boards replayed the same
    recording, with the same bias.

    Classes with a physical minimum separation keep it as a ceiling. Two cars in adjacent bays are
    2.5 m apart, and merging them makes one disappear with nothing on the screen saying so; two pins
    for one car are resolved at a glance. Reports with no radius fall back to the class radius, as
    before.
    """
    ca = a.get('cls') or b.get('cls')
    ra, rb = a.get('radius_m'), b.get('radius_m')
    if ra is None or rb is None:
        return RADIOS.get(ca, 3.5)
    combinado = math.hypot(float(ra), float(rb))
    tope = SEPARACION_MINIMA_M.get(ca)
    return min(combinado, tope) if tope is not None else combinado


def mismo_objetivo(a, b):
    """
    Whether two POIs from different drones are one target.

    Three conditions, and no two of them are enough. Class is a veto, as it already is inside
    a drone: a person and a car are never the same thing. Distance bounds it, with the radius
    the class earns -- two cars are never parked closer than a parking space. Appearance
    settles what is left, because two people three metres apart are two people and only their
    appearance says so.

    A POI with no vector falls back to class and distance. That is weaker and it is stated
    here rather than hidden: an older drone, or one with no ReID model, still fuses, just with
    less evidence.
    """
    ca, cb = a.get('cls'), b.get('cls')
    if ca and cb and ca != cb:
        return False
    radio = distancia_maxima(a, b)
    if math.hypot(a.get('x', 0.0) - b.get('x', 0.0),
                  a.get('y', 0.0) - b.get('y', 0.0)) > radio:
        return False
    va, vb = vector_de(a), vector_de(b)
    if va is None or vb is None:
        return True
    return float(np.linalg.norm(va - vb)) <= EMB_DIST_MAX_MEDIDO


def _mejor_pin(salida, dron, poi):
    """The pin this report belongs to, or None, choosing by distance and not by arrival order.

    Until 2026-10-03 this was the FIRST pin that passed mismo_objetivo, which is a greedy
    assignment. With two aircraft it is almost always the same answer, because the only pin a
    report can reach is usually its own target's. With three it stops being: the pins are built
    in the order the reports arrive, so which aircraft reported first could decide who gets
    paired with whom.

    That matters here more than it would elsewhere, because the error between two aircraft is
    mostly BIAS and not scatter: measured on the 02ago candidates, 82 to 99.9 % of a static
    target's 95 % radius is the gps and compass offset of that airframe. Bias does not average
    out over a flight and it is not shared, so one drone's whole picture can sit metres away
    from another's, and two targets standing closer than that offset are exactly the case where
    first-match and best-match disagree.

    Still greedy across reports: each one takes its best pin without reconsidering earlier
    choices. A joint assignment over the whole frame would be the next step and needs an
    argument of its own, because it would also have to decide what to do when the appearance
    says one thing and the geometry another.
    """
    mejor, mejor_d = None, None
    for ya in salida:
        if dron in ya['drones'] or not mismo_objetivo(ya, poi):
            continue
        d = math.hypot((ya.get('x') or 0.0) - (poi.get('x') or 0.0),
                       (ya.get('y') or 0.0) - (poi.get('y') or 0.0))
        if mejor_d is None or d < mejor_d:
            mejor, mejor_d = ya, d
    return mejor


def fundir(por_dron):
    """
    One pin per target, across drones.

    Only across: two POIs from the SAME drone are left alone. That drone's identity layer
    already decided they were different objects, and it decided with the whole track in front
    of it -- overruling that from here, with one position and one vector, would be replacing
    the better judgement with the worse one.

    The fused position is weighted by how many observations each drone contributed, so a
    target one drone barely glimpsed does not drag the estimate of the one that watched it.

    Every other field needs its own rule, because a pin otherwise keeps whatever the FIRST
    report that created it happened to carry, and "first" is an accident of arrival order:

    age_s
        A target is as fresh as the drone that saw it LAST. Keeping the first drone's age
        would fade out, on the station's map, a target another drone has in view. The sighting
        instants are compared on the station's clock, which is report time minus age; a drone
        that sends no age leaves the fused pin without one, which the page reads as "cannot be
        aged" rather than as old.

    evidence, looks, looks_min, duty
        Evidence travels as a trio -- how much, out of how much, as a fraction -- and the trio
        is taken WHOLE from the drone that has more of it, rather than each field being
        maximised on its own, which could pair one drone's count with another's threshold.
        Looks are not added up: two drones watching the same target at the same time are
        counting the same seconds, so the sum would invent evidence.

    emb
        The appearance of a fused pin is whichever drone's has one. This is not cosmetic: the
        operator's "not it" only leaves the station when there is a template to send, because
        gs_mapa.plantilla_para filters on emb, so a fused pin with no vector turns the
        operator's button into a button that does nothing. The two vectors are NOT averaged:
        they belong to different cameras with different exposures, and the mean of two
        appearances is the appearance of nothing.

    radius_m
        The margin of a fused pin is the SMALLER of the two, not the first drone's. A target
        seen from close and from far does not have one uncertainty, it has the closer drone's,
        and keeping whichever arrived first was reporting the worse of two answers for no
        reason; NOTES.md has the measurement. What is NOT done here, and would need an
        argument first, is averaging the two biases DOWN. Two aircraft have independent
        compasses, so the yaw half really is independent and sqrt(2) of it would be honest,
        but their GPS error is partly common, so the gps half is not. Claiming the whole bias
        averages would invent precision.
    """
    salida = []
    for dron, lista in por_dron.items():
        for poi in lista:
            ya = _mejor_pin(salida, str(dron), poi)
            if ya is None:
                nuevo = dict(poi)
                nuevo['drones'] = [str(dron)]
                salida.append(nuevo)
            else:
                na, nb = ya.get('n_obs') or 1, poi.get('n_obs') or 1
                ya['x'] = round((ya['x'] * na + poi['x'] * nb) / (na + nb), 2)
                ya['y'] = round((ya['y'] * na + poi['y'] * nb) / (na + nb), 2)
                ya['n_obs'] = na + nb
                ya['mature'] = bool(ya.get('mature') or poi.get('mature'))
                ea, eb = ya.get('age_s'), poi.get('age_s')
                if ea is None or eb is None:
                    ya['age_s'] = None
                elif (poi.get('t') or 0.0) - eb > (ya.get('t') or 0.0) - ea:
                    ya['age_s'], ya['t'] = eb, poi.get('t', ya.get('t'))
                if (poi.get('evidence') or 0.0) > (ya.get('evidence') or 0.0):
                    for campo in ('evidence', 'looks', 'looks_min', 'duty'):
                        ya[campo] = poi.get(campo)
                if ya.get('emb') is None and poi.get('emb') is not None:
                    ya['emb'] = poi['emb']
                ra, rb = ya.get('radius_m'), poi.get('radius_m')
                if rb is not None and (ra is None or rb < ra):
                    ya['radius_m'] = rb
                ya['drones'].append(str(dron))
                ya['dron'] = '+'.join(ya['drones'])
    return salida


def pedidos_de_verificacion(pois, drones, ahora, vivo_s=10.0):
    """
    Which targets are worth a second look, and by whom.

    The station proposes; it does not command. A drone that flies because another one sent it
    a JSON is not something you can put in the air here: the 4G link carries data and the
    stick stays with the pilot (DECEA). So this returns a list an operator reads, with the
    point to fly to already computed.

    A target earns a request when it is unconfirmed, when only one drone has seen it, and when
    some other drone is alive to go. Unconfirmed and seen by two is not a request: the second
    look already happened and the answer was still 'not sure', which is a different problem.

    'puede_ir' names every idle drone rather than the nearest one. The nearest would be better,
    but the station does not know where the others are: a report carries the target's position,
    not the drone's. Naming them all and letting the operator choose is honest; guessing would
    not be.
    """
    vivos = [d for d, f in drones.items() if ahora - f.get('t', 0) < vivo_s]
    if len(vivos) < 2:
        return []
    salida = []
    for p in pois:
        if p.get('mature'):
            continue
        vistos = p.get('drones') or [str(p.get('dron'))]
        if len(vistos) > 1:
            continue
        libres = [d for d in vivos if d not in vistos]
        if not libres:
            continue
        salida.append({
            'cls': p.get('cls'),
            'x': p.get('x'), 'y': p.get('y'),
            'lat': p.get('lat'), 'lng': p.get('lng'),
            'n_obs': p.get('n_obs'),
            'visto_por': vistos[0],
            'puede_ir': sorted(libres),
        })
    return salida
