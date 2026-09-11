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
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from uav_vision.identity import EMB_DIST_MAX_MEDIDO, radii_by_class

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
    radio = RADIOS.get(ca or cb, 3.5)
    # Plain metres. separacion_m() is for lat/lng pairs; handing it x/y in metres reads them
    # as degrees and every pair comes back impossibly far apart.
    if math.hypot(a.get('x', 0.0) - b.get('x', 0.0),
                  a.get('y', 0.0) - b.get('y', 0.0)) > radio:
        return False
    va, vb = vector_de(a), vector_de(b)
    if va is None or vb is None:
        return True
    return float(np.linalg.norm(va - vb)) <= EMB_DIST_MAX_MEDIDO


def fundir(por_dron):
    """
    One pin per target, across drones.

    Only across: two POIs from the SAME drone are left alone. That drone's identity layer
    already decided they were different objects, and it decided with the whole track in front
    of it -- overruling that from here, with one position and one vector, would be replacing
    the better judgement with the worse one.

    The fused position is weighted by how many observations each drone contributed, so a
    target one drone barely glimpsed does not drag the estimate of the one that watched it.
    """
    salida = []
    for dron, lista in por_dron.items():
        for poi in lista:
            for ya in salida:
                if str(dron) in ya['drones'] or not mismo_objetivo(ya, poi):
                    continue
                na, nb = ya.get('n_obs') or 1, poi.get('n_obs') or 1
                ya['x'] = round((ya['x'] * na + poi['x'] * nb) / (na + nb), 2)
                ya['y'] = round((ya['y'] * na + poi['y'] * nb) / (na + nb), 2)
                ya['n_obs'] = na + nb
                ya['mature'] = bool(ya.get('mature') or poi.get('mature'))
                ya['drones'].append(str(dron))
                ya['dron'] = '+'.join(ya['drones'])
                break
            else:
                nuevo = dict(poi)
                nuevo['drones'] = [str(dron)]
                salida.append(nuevo)
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
            # The nearest idle drone would be better, but the station does not know where the
            # others are: a report carries the target's position, not the drone's. Naming them
            # all and letting the operator choose is honest; guessing would not be.
            'puede_ir': sorted(libres),
        })
    return salida
