# -*- coding: utf-8 -*-
"""Replays a recorded flight into the ground station, so the whole chain can be watched.

Between the identity layer and the map there are three things that have only ever been tested
apart: that the candidates come out right, that the message on the wire says what the ground
station expects, and that the page draws them where they belong. A flight proves all three at
once, and this replays one without a drone, a Raspberry or a battery.

What is real here: the observations (every tracked detection of flight 3, already projected to
the ground), the timing between them, the identity layer, the message format, and the HTTP
transport. What is not: the camera and the radio. It is the same code path the drone runs, fed
from a file instead of a lens.

Watching it is the point. The pins appear amber -- POR VERIFICAR -- while evidence is thin, and
turn green when the identity layer is finally sure. That transition is the whole design of the
sweep path, and it is easier to trust after seeing it happen than after reading a table.

    # in one terminal
    python gs_mapa.py --fondo ... --georef ... --origen=-22.978029946,-43.23214256266666
    # in another
    python reproducir_vuelo_gs.py --velocidad 20 --preliminares

--velocidad 20 plays fifteen minutes of flight in forty-five seconds.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
from uav_vision.identity import IncrementalIdentity

# The recording lives in the companion data repository, a sibling of this one. Overridable with
# UAV_DATOS so the replay runs wherever the flight data was put.
_DATOS = os.environ.get('UAV_DATOS',
                        os.path.join(os.path.dirname(_REPO), 'drone-geolocation'))
ENTREN = os.path.join(_DATOS, 'entrenamiento')
VUELO3 = os.path.join(_DATOS, 'data', 'flight_02ago', '20260802_133309')


def enviar(url, mensaje, dron):
    """The wire format the fleet's data plane uses, and the one gs_mapa.py expects."""
    cuerpo = json.dumps({'message': json.dumps(mensaje), 'source': dron}).encode()
    req = urllib.request.Request(url, data=cuerpo,
                                 headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=3).read()
        return True
    except Exception as e:
        print('  no se pudo enviar (%s): esta corriendo gs_mapa.py?' % e, flush=True)
        return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gs', default='http://localhost:8300')
    ap.add_argument('--stream', default=os.path.join(ENTREN, 'obs_stream_1280.npz'))
    ap.add_argument('--velocidad', type=float, default=20.0,
                    help='multiplicador de tiempo: 20 = 15 min de vuelo en 45 s')
    ap.add_argument('--periodo-reporte', type=float, default=2.0,
                    help='segundos DE VUELO entre reportes, como el protocolo real')
    ap.add_argument('--preliminares', action='store_true',
                    help='reportar tambien las pistas que aun no maduraron (modo barrido)')
    ap.add_argument('--dron', type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.stream):
        raise SystemExit('no existe %s -- generalo con barrido_pasada.py' % args.stream)
    D = np.load(args.stream)
    obs = D['obs']                      # t, k, track_id, x, y, conf, idx_emb
    embs = np.load(os.path.join(ENTREN, 'y1280_embs.npy'))

    # The cadence frames arrive at, not frames over the wall-clock span: the flight-3 span
    # includes 573 s of the drone on the ground, and dividing by it shrinks every maturity
    # threshold by almost 3x.
    import csv
    FR = os.path.join(VUELO3, 'frames.csv')
    t_air = np.array([float(r['t_mono']) for r in csv.DictReader(open(FR))
                      if float(r['alt_agl']) > 3.0])
    dt = np.diff(t_air)
    fps = float(1.0 / np.median(dt[dt > 0]))

    ident = IncrementalIdentity(fusion_radius_m=3.5, fps=fps)
    T = obs[:, 0]
    dur = float(T.max())
    print('vuelo de %.0f s, %d observaciones, cadencia %.2f Hz' % (dur, len(obs), fps))
    print('reproduciendo a %.0fx -> %.0f s reales. GS: %s'
          % (args.velocidad, dur / args.velocidad, args.gs))
    print('preliminares: %s' % ('SI (modo barrido)' if args.preliminares else 'no (merodeo)'))

    i = 0
    t_reporte = 0.0
    n_env = 0
    inicio = time.time()
    while i < len(obs):
        t_vuelo = (time.time() - inicio) * args.velocidad
        if t_vuelo > dur:
            t_vuelo = dur
        # feed everything that had happened by now
        while i < len(obs) and T[i] <= t_vuelo:
            _t, k, tid, x, y, cf, idx = obs[i]
            ident.observe(frame=int(k), track_id=int(tid), ground_xy=(x, y),
                           conf=float(cf), emb=embs[int(idx)])
            i += 1
        if t_vuelo - t_reporte >= args.periodo_reporte:
            t_reporte = t_vuelo
            pois = ident.candidates(preliminary=args.preliminares)
            mensaje = {'type': 'vision_poi', 'sender': args.dron,
                       'time': round(t_vuelo, 2), 'frames_seen': int(i), 'pois': pois}
            if enviar(args.gs, mensaje, args.dron):
                n_env += 1
            maduros = sum(1 for p in pois if p.get('mature'))
            print('  t=%6.1f s | %d POI (%d maduros, %d por verificar) | obs %d'
                  % (t_vuelo, len(pois), maduros, len(pois) - maduros, i), flush=True)
        if t_vuelo >= dur:
            break
        time.sleep(0.05)

    print('\nfin: %d reportes enviados en %.0f s reales' % (n_env, time.time() - inicio))
