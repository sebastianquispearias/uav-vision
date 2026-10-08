"""Ground station with a map: the points the drone finds, on the ground they were found on.

The laptop has been the ground station since the first end-to-end run -- oyente_gs.py already
received real POIs over the fleet's own HTTP transport, from a drone with a Pixhawk attached.
What it did with them was print lines. This is the same receiver with a face: a live map, so
the thing the whole project exists for -- a person is out there, and a pin appears where they
are -- can be looked at instead of read.

Two kinds of pin, and the difference is the point:

    CONFIRMADO   a mature candidate. The identity layer tracked it long enough to be sure.
    POR VERIFICAR a preliminary candidate. A track formed but never earned a report -- what a
                 sweep produces, measured on flight 3: a 30 s pass over a person NEVER matures
                 a candidate. Showing these as finds would be lying to the operator; hiding
                 them would be worse, because in a sweep they are all there is.

Each pin also carries the class the detector gave it, and the panel can hide classes: a sweep
over a car park reports dozens of cars, and one person among them is the thing worth seeing.

Everything is served by one process with no external dependencies: no CDN, no npm, nothing
that needs a network in the field. The page polls a JSON endpoint and draws.

    python gs_mapa.py --puerto 8300 \
        --fondo ../../drone-geolocation/entrenamiento/satelite_zona.png \
        --georef ../../drone-geolocation/entrenamiento/satelite_georef.txt \
        --origen -22.978029946,-43.23214256266666

Without --fondo it draws a metric grid, which works anywhere and needs no imagery.
Then open http://localhost:8300 in a browser.

--demo injects moving fake POIs so the page can be seen without flying.

--clip adds a CLIP score to each person candidate's crop (filtro_clip.py). Doubtful ones go to the
end of the list with a "probable no persona" mark; nothing is hidden. torch and open_clip live
only in the training venv, which also runs everything else this file imports:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/banco_embedded/gs_mapa.py --clip

Without open_clip, --clip prints a warning and the station runs without scores.

How the station is put together, what lives in its state, and every knob the operator
has: docs/ARQUITECTURA.md.
"""
import argparse
import base64
import json
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from http import server

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from uav_vision.camera import EMB_DIST_OBJETIVO
    from uav_vision.flota import fundir, pedidos_de_verificacion
    FUSION_DISPONIBLE = True
except Exception:
    FUSION_DISPONIBLE = False
    EMB_DIST_OBJETIVO = None

    def fundir(por_dron):
        return [q for lista in por_dron.values() for q in lista]

    def pedidos_de_verificacion(pois, drones, ahora, vivo_s=10.0):
        return []

R_TIERRA = 6378137.0

ESTADO = {
    'pois': [],
    'historia': [],
    'drones': {},
    'veredictos_dir': 'veredictos',
    'origen': None,
    'origen_cli': None,
    'desacuerdo': None,
    'celdas': None,

    'georef': None,
    'fondo': None,
    'arranque': time.time(),

    'buscar': None,
    'buscar_v': 0,
    'buscar_epoca': uuid.uuid4().hex[:8],
    'nodos': {},

    'pois_por_dron': {},

    'rastros': {},
    'frame_actual': None,
    'frames_dir': None,

    'objetivo': None,

    'segunda': {},
}
CANDADO = threading.Lock()

SEGUNDA = None
PYTHON_RFDETR = os.path.join('..', 'drone-geolocation', 'entrenamiento', 'venv', 'Scripts',
                             'python.exe')


class SegundaOpinion:
    """Keeps RF-DETR loaded in its own process so that a click costs a third of a second.

    Measured on this laptop: loading the model takes 17.3 s, the first frame 1.5 s while CUDA warms
    up, and every frame after that 0.29 s. Starting a process per request would therefore answer a
    click in twenty seconds. The acceptance criterion for this feature is five, so the process is
    started once, when the station boots, and the operator never waits for the model.

    It speaks the line protocol of segunda_opinion.py --servidor. A worker that dies (no venv, no
    GPU, no rfdetr) is reported as an error on the page rather than crashing the station: the map
    has to keep working for an operator who has no second detector at all.
    """

    def __init__(self, python, carpeta):
        self.python = python
        self.carpeta = carpeta
        self.proceso = None
        self.listo = False
        self.error = None
        self.candado = threading.Lock()

    def arrancar(self):
        import subprocess
        guion = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segunda_opinion.py')
        if not os.path.exists(self.python):
            self.error = 'no existe %s' % self.python
            print('segunda opinion NO disponible: %s' % self.error, flush=True)
            return
        os.makedirs(self.carpeta, exist_ok=True)
        try:
            self.proceso = subprocess.Popen(
                [self.python, guion, '--servidor'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
        except Exception as e:
            self.error = str(e)
            print('segunda opinion NO disponible: %s' % e, flush=True)
            return
        threading.Thread(target=self._esperar_listo, daemon=True).start()

    def _esperar_listo(self):
        linea = self.proceso.stdout.readline()
        try:
            d = json.loads(linea)
        except Exception:
            self.error = 'el detector de tierra no arranco'
            print('segunda opinion NO disponible: %s' % (linea or '(sin salida)'), flush=True)
            return
        self.listo = True
        print('segunda opinion lista: RF-DETR cargado en %.1f s' % d.get('segundos', 0), flush=True)

    def disponible(self):
        return self.proceso is not None and self.proceso.poll() is None

    def mirar(self, datos, quien):
        """Hands one frame over and waits for the answer. Called from the thread that received it."""
        if not self.disponible():
            return {'error': self.error or 'el detector de tierra no esta corriendo', 'n': 0}
        destino = os.path.join(self.carpeta, 'marco_%s.jpg' % quien)
        dibujo = os.path.join(self.carpeta, 'mirada_%s.jpg' % quien)
        with open(destino, 'wb') as f:
            f.write(datos)
        with self.candado:
            try:
                self.proceso.stdin.write(json.dumps(
                    {'id': str(quien), 'archivo': destino, 'dibujar': dibujo}) + '\n')
                self.proceso.stdin.flush()
                r = json.loads(self.proceso.stdout.readline())
            except Exception as e:
                return {'error': str(e), 'n': 0}
        r['dibujo'] = dibujo if os.path.exists(dibujo) else None
        return r


def empujar_mensaje(nodos, mensaje):
    """Puts one packet on every drone's data plane, without waiting for any of them.

    The envelope is the one a GrADyS node uses to talk to another -- {"message", "source"} POSTed
    to /message -- so the drone receives it in handle_packet exactly as it receives a neighbour's
    report. Each send runs in its own thread: the operator's click must not wait on a radio link.

    'source' is 0 because the embedded runtime types it as an int, and no mission numbers a
    drone 0.
    """
    if not nodos:
        return
    import threading as _t
    import urllib.request
    cuerpo = json.dumps({'message': json.dumps(mensaje),
                         'source': 0}).encode('utf-8')

    def uno(nodo, direccion):
        try:
            urllib.request.urlopen(urllib.request.Request(
                'http://%s/message' % direccion, data=cuerpo,
                headers={'Content-Type': 'application/json'}), timeout=2).read()
            print('  %s entregado al dron %s' % (mensaje.get('type'), nodo), flush=True)
        except Exception as e:
            print('  %s NO llego al dron %s (%s)' % (mensaje.get('type'), nodo, e), flush=True)

    for nodo, direccion in nodos.items():
        _t.Thread(target=uno, args=(nodo, direccion), daemon=True).start()

CLIP = None


def a_latlng(x, y, origen):
    """Local metres east/north back to coordinates, given the mission origin."""
    if origen is None:
        return None, None
    lat0, lng0 = origen
    lat = lat0 + math.degrees(y / R_TIERRA)
    lng = lng0 + math.degrees(x / (R_TIERRA * math.cos(math.radians(lat0))))
    return round(lat, 7), round(lng, 7)


def ficha(ahora, mensaje):
    """What the ground station knows about a drone from its last message.

    fps_real and slots_perdidos are here because a drone that cannot keep up does not look
    any different from one that can: same pins, same cadence of reports, fewer looks taken.
    Until 2026-08-25 that gap was only findable with a stopwatch.

    'salud' is what the board says about its own current and temperature, and it is kept beside
    those two because it is the same kind of thing: the health of the aircraft, not a find. It
    is here because a board spent four hours browning out and nothing on this screen showed it;
    NOTES.md has that story.

    'bateria_v' is the pack voltage, which answers a question 'salud' cannot: the firmware
    reports the computer's rail, not the battery feeding it, so a pack on its way down looks
    perfectly healthy until the rail collapses all at once. None means the drone was configured
    without a battery source, which is every bench and replay run, and is not a fault.

    'temp_c' and 'fps_pedido' exist for the health tab, and they are the two things that were
    NOT already arriving when that tab was planned. 'salud' carries the thermal throttling bit
    but not the degrees, so a board at 84 C and a board at 50 C looked identical until the
    moment one of them started slowing down. And 'fps_pedido' is set per board ON the board,
    through BANCO_FPS, so the station had no way to say whether a rate was good: on 2026-10-02 a
    visitor read '4 FPS' here and asked why a Pi 4 and a Pi 5 reported the same number. They did
    not -- the screen was showing what both had been ASKED for.

    'pos' is where the drone is and, through the trail, where it has been. The trail is what
    shows an operator whether the drone is working the area or hovering, which is the difference
    between rays that cross and rays that do not.

    'buscando' is what the camera is REALLY searching for, and whether it took the last order.
    The station's own record of the request says nothing about a drone out of range.
    """
    return {
        't': ahora,
        'pos': mensaje.get('pos'),
        'frames_seen': mensaje.get('frames_seen'),
        'fps_real': mensaje.get('fps_real'),
        'slots_perdidos': mensaje.get('slots_perdidos'),
        'slots_perdidos_total': mensaje.get('slots_perdidos_total'),
        'salud': mensaje.get('salud'),
        'bateria_v': mensaje.get('bateria_v'),
        'temp_c': mensaje.get('temp_c'),
        'fps_pedido': mensaje.get('fps_pedido'),
        'buscando': mensaje.get('buscando'),
    }


def empujar_orden(nodos, clases, v, epoca):
    """
    Sends the search order to every drone on the fleet's data plane, without waiting.

    The envelope is the one a GrADyS node uses to talk to another -- {"message", "source"}
    POSTed to /message -- so the drone receives it in handle_packet exactly as it receives a
    neighbour's report. Each send runs in its own thread: the operator's click must not wait on
    a radio link, and a drone out of range still gets the order on its next poll, if it polls.
    """
    empujar_mensaje(nodos, {'type': 'vision_buscar', 'clases': clases, 'v': v, 'epoca': epoca})


def recibir_marco(mensaje, fuente):
    """Takes the frame a drone sent and has the ground look at it again.

    Handled in its own thread because the answer takes about a third of a second and this runs on
    the same handler that receives the fleet's reports: blocking here would hold up the map while
    the ground thinks. The frame is what the aircraft's own detector judged, not a fresh capture,
    which is what makes the comparison mean anything.
    """
    quien = str(mensaje.get('sender', fuente))
    try:
        datos = base64.b64decode(mensaje.get('jpeg') or '')
    except Exception:
        datos = b''
    if not datos:
        with CANDADO:
            ESTADO['segunda'][quien] = {'estado': 'error', 't': time.time(),
                                        'error': 'el marco llego vacio'}
        return
    with CANDADO:
        ESTADO['segunda'][quien] = {'estado': 'mirando', 't': time.time()}

    def trabajo():
        t0 = time.time()
        r = SEGUNDA.mirar(datos, quien) if SEGUNDA is not None else {
            'error': 'la estacion arranco sin segunda opinion', 'n': 0}
        fila = {'t': time.time(), 'espera': round(time.time() - t0, 2),
                'dibujo': r.get('dibujo')}
        if r.get('error'):
            fila.update(estado='error', error=r['error'])
        else:
            fila.update(estado='listo', n=r.get('n', 0), segundos=r.get('segundos'),
                        personas=r.get('personas', []))
            print('[%s] segunda opinion sobre el dron %s: %d personas en %.2f s' %
                  (datetime.now().strftime('%H:%M:%S'), quien, fila['n'], fila['espera']),
                  flush=True)
        with CANDADO:
            ESTADO['segunda'][quien] = fila

    threading.Thread(target=trabajo, daemon=True).start()


def separacion_m(a, b):
    """Ground distance between two lat/lng pairs, flat-earth: only used for small gaps."""
    dlat = math.radians(b[0] - a[0]) * R_TIERRA
    dlng = math.radians(b[1] - a[1]) * R_TIERRA * math.cos(math.radians(a[0]))
    return math.hypot(dlat, dlng)


def adoptar_origen(mensaje):
    """
    Takes the frame from the drone, and says so when it contradicts the operator.

    The drone's origin is not an opinion: it is the frame its metres are actually measured
    in. A value typed on the ground is a guess about that frame, and when the mission is
    loaded without an origin the runner resolves one from the GPS fix, which no operator can
    know in advance. So the drone wins -- but silently overriding would hide the very mistake
    worth catching, so the disagreement is recorded and shown.

    A disagreement under a metre is not reported: a metre is well under the system's own error
    and far above float noise.
    """
    origen = mensaje.get('origen_gps')
    if not origen or len(origen) < 2:
        return
    nuevo = (float(origen[0]), float(origen[1]))
    if ESTADO['origen_cli'] is not None:
        d = separacion_m(ESTADO['origen_cli'], nuevo)
        ESTADO['desacuerdo'] = round(d, 1) if d > 1.0 else None
    if ESTADO['origen'] != nuevo:
        ESTADO['origen'] = nuevo
        print('origen tomado del dron: %.7f, %.7f%s' % (
            nuevo[0], nuevo[1],
            '' if not ESTADO['desacuerdo']
            else '  <-- NO COINCIDE con --origen, %s m' % ESTADO['desacuerdo']), flush=True)


def _rastro(fuente, pos):
    """Keeps the last stretch of a drone's path. Called with the lock already held.

    The trail is bounded on purpose: a whole flight drawn at once is a scribble, and the
    question an operator has is where the drone went lately, not where it took off.
    """
    if not pos:
        return
    r = ESTADO['rastros'].setdefault(str(fuente), [])
    if not r or (abs(r[-1][0] - pos[0]) + abs(r[-1][1] - pos[1])) > 0.5:
        r.append([pos[0], pos[1]])
        del r[:-200]


CLIP_DESCARTA = False
DRON_CALLADO_S = 15.0
OBJETIVO_RADIO_M = 3.0
RODEO_RADIO_M = 30.0
RODEO_ALTURA_M = 25.0
RODEO_PUNTOS = 1
EN_BANCO = False


def dron_para_rodear(nodos, dron_que_vio):
    """Which aircraft to send. Another one if there is another one.

    The point of flying is a direction nobody has yet, and the drone that reported the target is
    standing in the direction we already have. With a single aircraft the answer is that one: it
    can still move, and refusing would be a worse answer than an imperfect one. Returns None when
    there is nobody to send, which the caller has to say out loud rather than pretend it ordered
    something.
    """
    ids = [str(d) for d in nodos]
    if not ids:
        return None
    otros = [d for d in ids if d != str(dron_que_vio)]
    return sorted(otros)[0] if otros else ids[0]


def plantilla_para(x, y, vigentes):
    """The appearance of the candidate the operator meant, or None when there is none near.

    The click says where and who. The who is the embedding this station already received with the
    candidate, handed straight back the way it arrived, base64 of float16: nothing is decoded
    here, because nothing here reads it. A click with no candidate within OBJETIVO_RADIO_M sends
    only the position, exactly as it did before, rather than reaching further and handing the
    drone the appearance of somebody standing next to the person who was pointed at.

    The sort key is the distance ALONE. Two candidates exactly as far from the click is not a
    corner case to shrug at: comparing the pairs would compare the dicts and raise, inside the
    handler that answers the operator's click.
    """
    cerca = sorted(((math.hypot(p['x'] - x, p['y'] - y), p) for p in vigentes
                    if p.get('emb') is not None
                    and p.get('x') is not None and p.get('y') is not None),
                   key=lambda par: par[0])
    if cerca and cerca[0][0] <= OBJETIVO_RADIO_M:
        return cerca[0][1]['emb']
    return None


def descartes_por_dron(x, y, ahora):
    """Which drone to tell, and in ITS OWN coordinates, that the operator refused this point.

    Until 2026-10-03 one refusal was built from the fused pin and pushed identically to every
    aircraft. Two things made that not work, and the second is the subtle one.

    A fused pin sits at the weighted mean of what each drone reported, so its position is not
    any drone's own. The drone matches a refusal against its own candidates with
    mismo_objetivo, which bounds by distance, so a refusal carrying the fused position can land
    outside the radius of the very candidate it was meant to silence. Measured on the bench: the
    refusal was delivered to both drones and neither one's POI count moved.

    And the appearance has to be that drone's own too. Two aircraft photograph the same person
    through different lenses at different exposures; handing drone 2 the vector drone 1 computed
    asks it to match its own crops against somebody else's camera.

    So this returns one refusal per drone, built from that drone's own report, and nothing at
    all for a drone with no candidate near the click: it did not see what was refused, and
    silencing a point it never reported would be silencing whatever it finds there next.
    """
    salida = {}
    with CANDADO:
        por_dron = {k: list(v) for k, v in ESTADO['pois_por_dron'].items()}
        drones = dict(ESTADO['drones'])
    for dron, pois in por_dron.items():
        if ahora - drones.get(dron, {}).get('t', float('-inf')) > DRON_CALLADO_S:
            continue
        cerca = sorted(((math.hypot(p['x'] - x, p['y'] - y), p) for p in pois
                        if p.get('emb') is not None
                        and p.get('x') is not None and p.get('y') is not None),
                       key=lambda par: par[0])
        if cerca and cerca[0][0] <= OBJETIVO_RADIO_M:
            p = cerca[0][1]
            salida[dron] = {'x': p['x'], 'y': p['y'], 'cls': p.get('cls'),
                            'plantilla': p['emb']}
    return salida


def pois_vigentes(ahora):
    """
    The fused targets of the drones that are still talking, with their coordinates.

    A silent drone's targets are kept, not deleted, so they return the moment it reports again.
    They are only left out while it is silent, so a live drone's find is shown as its own instead
    of as corroborated by a drone that may have fallen out of the sky. Measured on the
    two-Raspberry bench before this existed: fifteen seconds after one runner was killed, the
    station still showed the target as seen by 1+2.

    The fusion works in metres and knows nothing about the globe, so the coordinates are put
    back on each merged position here; otherwise the pin and its GPS reading would part.

    What CLIP doubts goes to the END of the queue and nothing else moves: the sort is stable,
    and a POI without a score is never demoted. With --clip-descarta it is dropped instead of
    demoted. That is off by default for two reasons: hiding a point the operator never saw is
    their call and not the system's, and CLIP still misses the harder clutter. NOTES.md has
    what dropping buys and what it still lets through.
    """
    vivos = {k: v for k, v in ESTADO['pois_por_dron'].items()
             if ahora - ESTADO['drones'].get(k, {}).get('t', float('-inf')) <= DRON_CALLADO_S}
    pois = fundir(vivos)
    for q in pois:
        q['lat'], q['lng'] = a_latlng(q['x'], q['y'], ESTADO['origen'])
    if CLIP_DESCARTA:
        pois = [q for q in pois if not q.get('clip_no_persona')]
    pois.sort(key=lambda q: bool(q.get('clip_no_persona')))
    return pois


def registrar(mensaje, fuente):
    """Files one drone's report: its own pin list, its trail, and the fused map the page reads.

    AN EMPTY BEAT says "still here", not "there is nothing". Touching the pin list on one would
    wipe the map every time a target left the frame for a second, so a heartbeat updates the
    drone's record and the trail and returns.

    The drone is heard BEFORE the fusion runs, or its own report would find it silent and leave
    its targets out. The per-drone lists are concatenated and not merged beyond what fundir
    decides: two drones seeing the same person still produce two pins until something decides
    they are the same target, and showing both is the honest state of affairs while hiding one
    would be a claim nobody has made yet.

    How each field of a POI is read off the wire, where it is not obvious:

    mature
        A POI with no 'mature' field predates the sweep work, or came from the RANSAC fallback
        that fires before any candidate exists. It is treated as UNCONFIRMED: assuming the
        safer reading is what keeps a maybe from being shown as a find.

    cls
        None when the drone never named it, which is older firmware or a camera with no class.
        Shown as 'sin clase' rather than guessed at.

    looks, looks_min, evidence, duty, radius_m
        How much independent evidence there is, how much it takes to be reported, the fraction
        of the way there, and how far off the point may be, when the drone matures by looks.
        Absent from older reports, and then simply not drawn: a bar with no threshold behind it
        would be a guess dressed as a measurement.

    age_s
        Seconds between the drone's last sighting of the target and this report. The page adds
        the time elapsed since the report and fades live contacts by it.

    emb
        Kept so the station can decide whether two drones are looking at one target. NEVER
        drawn: it is evidence, not something an operator reads.

    CLIP scores outside the station's lock, because the drone already has its answer and a
    model call must not stall the page's reads. Each distinct crop goes through the model once.
    """
    ahora = time.time()
    with CANDADO:
        adoptar_origen(mensaje)
    if mensaje.get('latido'):
        with CANDADO:
            ESTADO['drones'][str(fuente)] = ficha(ahora, mensaje)
        _rastro(fuente, mensaje.get('pos'))
        return
    pois = []
    for p in mensaje.get('pois', []):
        lat, lng = a_latlng(p.get('x', 0.0), p.get('y', 0.0), ESTADO['origen'])
        pois.append({
            'x': p.get('x'), 'y': p.get('y'),
            'lat': lat, 'lng': lng,
            'n_obs': p.get('n_obs'),
            'cls': p.get('cls'),
            'conf': p.get('conf', p.get('conf_mean')),
            'mobile': p.get('mobile'),
            'mature': bool(p.get('mature', False)),
            'crop': p.get('crop'),
            'looks': p.get('looks'),
            'looks_min': p.get('looks_min'),
            'evidence': p.get('evidence'),
            'duty': p.get('duty'),
            'radius_m': p.get('radius_m'),
            'age_s': p.get('age_s'),
            'emb': p.get('emb'),
            'dron': fuente,
            't': ahora,
        })
    if CLIP is not None:
        for q in pois:
            CLIP.anotar(q)
    with CANDADO:
        ESTADO['pois_por_dron'][str(fuente)] = pois
        ESTADO['historia'].append({'t': ahora, 'n': len(pois),
                                   'frames': mensaje.get('frames_seen')})
        ESTADO['historia'][:] = ESTADO['historia'][-500:]
        ESTADO['drones'][str(fuente)] = ficha(ahora, mensaje)
        ESTADO['pois'] = pois_vigentes(ahora)
        _rastro(fuente, mensaje.get('pos'))
    hora = datetime.now().strftime('%H:%M:%S')
    print('[%s] dron %s | %d POI(s) | frames %s' %
          (hora, fuente, len(pois), mensaje.get('frames_seen')), flush=True)


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _responder(self, cuerpo, tipo='application/json', codigo=200):
        self.send_response(codigo)
        self.send_header('Content-Type', tipo)
        self.send_header('Content-Length', str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_POST(self):
        """Everything that arrives at the station: the fleet's own reports, and the operator's
        clicks.

        The fleet data plane is unchanged from oyente_gs.py. What was added is the operator's
        side of the control plane, and each endpoint earns its own paragraph:

        /buscar
            What the operator wants the drones to look for. Answering before any drone has
            polled is deliberate: the order is STORED, not routed, so the station never blocks
            on a link it does not control.

        /objetivo
            "Es lo que busco". Not only a verdict kept on disk: it is the one thing the
            operator knows that the drone cannot work out, and it buys recall for free, because
            sending it back lets the camera lower its threshold over that square of the image
            where the detector had already scored the boxes it was discarding.

        /mirar
            The operator asks the GROUND to look again at what one drone is seeing right now.
            Nothing is computed here: the request goes out, the frame comes back on the data
            plane like any other packet, and the answer appears when it appears. The click must
            not block on a radio link, or on a detector that takes a second and a half.

        /reiniciar
            A clean board, for real. The clear button used to hide pins and a page reload
            brought them all back, because the candidates do not live here: they live in each
            aircraft's identity layer, which forgets nothing on purpose. So the only honest
            clear is one that TRAVELS. The station empties its own copy too and says how many
            it dropped, because a button that clears and reports nothing reads as broken.

        /rodear
            The operator sends an aircraft to look at one target from a side nobody has looked
            from. THIS IS THE ONLY REQUEST IN THE STATION THAT MAKES SOMETHING FLY, so it says
            out loud which aircraft went, and refuses rather than pretend when there is nobody
            to send. The radius and the altitude travel with the order, because the layer that
            flies will not invent them. A walker is refused HERE, on the ground, so the
            operator is told why instead of the order vanishing; the drone refuses it too,
            which is the line that matters, but a refusal nobody explains reads as a broken
            button.

        /veredicto
            The operator's verdict on a point, kept on disk with the crop it was given on. The
            drone's own signals cannot tell a person from an object the detector keeps
            confusing with one; the operator can, and every "no es" is exactly the hard
            negative a detector trained on public aerial data is missing. Keeping them turns
            using the system into labelling it.

            A "no es" used to STOP here, on disk, and the drone kept reporting the same wrong
            point every two seconds for the rest of the flight over a 4G dongle, while the next
            session started knowing nothing. Now the refusal goes back with the appearance of
            what was refused. Position alone is never sent as a refusal: see
            VisionProtocol.descartar for why, and without an appearance nothing travels and the
            verdict keeps working exactly as it did.

            One refusal PER DRONE, carrying the position and the appearance THAT drone
            reported. The pin the operator pointed at can be the mean of two drones, and that
            mean belongs to neither: sent as it is, it falls outside the radius of the very
            candidate it was meant to silence. See descartes_por_dron. When no drone can obey,
            the station says so out loud, because the case exists and used to be silence: a
            board with no appearance model publishes no emb, and with no emb there is nothing
            to send.
        """
        largo = int(self.headers.get('Content-Length', 0))
        crudo = self.rfile.read(largo)
        if self.path.split('?')[0] == '/frame_actual':
            try:
                n = int(json.loads(crudo).get('n'))
            except Exception:
                self._responder(b'{"error": "n"}', codigo=400)
                return
            with CANDADO:
                ESTADO['frame_actual'] = n
            self._responder(b'{"status": "ok"}')
            return
        if self.path.split('?')[0] == '/buscar':
            try:
                clases = json.loads(crudo).get('clases')
            except Exception:
                self._responder(b'{"error": "json"}', codigo=400)
                return
            with CANDADO:
                ESTADO['buscar'] = list(clases) if clases else None
                ESTADO['buscar_v'] += 1
                v = ESTADO['buscar_v']
                epoca = ESTADO['buscar_epoca']
                nodos = dict(ESTADO['nodos'])
            print('[%s] el operador pide buscar: %s' %
                  (datetime.now().strftime('%H:%M:%S'), clases or '(lo de siempre)'),
                  flush=True)
            empujar_orden(nodos, list(clases) if clases else None, v, epoca)
            self._responder(json.dumps({'clases': clases, 'v': v,
                                        'epoca': epoca}).encode('utf-8'))
            return
        if self.path.split('?')[0] == '/objetivo':
            try:
                d = json.loads(crudo)
                apagar = bool(d.get('off'))
                x, y = (None, None) if apagar else (float(d['x']), float(d['y']))
            except Exception:
                self._responder(b'{"error": "objetivo"}', codigo=400)
                return
            with CANDADO:
                nodos = dict(ESTADO['nodos'])
                vigentes = [] if apagar else pois_vigentes(time.time())
                ESTADO['objetivo'] = None if apagar else {'x': x, 'y': y, 't': time.time()}
            print('[%s] objetivo %s' % (datetime.now().strftime('%H:%M:%S'),
                                        'liberado' if apagar else 'fijado en (%.1f, %.1f)' % (x, y)),
                  flush=True)
            plantilla = None if apagar else plantilla_para(x, y, vigentes)
            mensaje = {'type': 'vision_objetivo', 'x': x, 'y': y}
            if plantilla is not None and EMB_DIST_OBJETIVO is not None:
                mensaje['plantilla'] = plantilla
                mensaje['emb_dist'] = EMB_DIST_OBJETIVO
            empujar_mensaje(nodos, mensaje)
            self._responder(json.dumps({'objetivo': None if apagar else {'x': x, 'y': y}}).encode('utf-8'))
            return
        if self.path.split('?')[0] == '/mirar':
            try:
                dron = str(json.loads(crudo)['dron'])
            except Exception:
                self._responder(b'{"error": "dron"}', codigo=400)
                return
            with CANDADO:
                nodos = dict(ESTADO['nodos'])
                ESTADO['segunda'][dron] = {'estado': 'pedido', 't': time.time()}
            print('[%s] el operador pide segunda opinion al dron %s' %
                  (datetime.now().strftime('%H:%M:%S'), dron), flush=True)
            empujar_mensaje(nodos, {'type': 'vision_mirar', 'para': dron})
            listo = SEGUNDA is not None and SEGUNDA.listo
            self._responder(json.dumps({'dron': dron, 'detector_listo': listo}).encode('utf-8'))
            return
        if self.path.split('?')[0] == '/reiniciar':
            with CANDADO:
                nodos = dict(ESTADO['nodos'])
                cuantos = len(ESTADO['pois'])
                ESTADO['pois'] = []
                ESTADO['historia'] = []
            empujar_mensaje(nodos, {'type': 'vision_reiniciar'})
            print('[%s] reinicio pedido por el operador: %d POI borrados aqui y la orden sale a '
                  '%d dron(es). Los "no es" NO se tocan.'
                  % (datetime.now().strftime('%H:%M:%S'), cuantos, len(nodos)), flush=True)
            self._responder(json.dumps({'status': 'ok', 'borrados': cuantos,
                                        'drones': len(nodos)}).encode('utf-8'))
            return
        if self.path.split('?')[0] == '/rodear':
            try:
                d = json.loads(crudo)
                x, y = float(d['x']), float(d['y'])
            except Exception:
                self._responder(b'{"error": "rodear"}', codigo=400)
                return
            with CANDADO:
                nodos = dict(ESTADO['nodos'])
            if d.get('movil'):
                self._responder(json.dumps({'error':
                    'ese contacto se esta moviendo: cuando el dron llegue ya no va a estar ahi'
                    }).encode('utf-8'), codigo=409)
                return
            elegido = dron_para_rodear(nodos, d.get('dron'))
            if elegido is None:
                self._responder(b'{"error": "ningun dron conectado"}', codigo=409)
                return
            radio = float(d.get('radio_m') or RODEO_RADIO_M)
            altura = float(d.get('altura_m') or RODEO_ALTURA_M)
            empujar_mensaje({elegido: nodos[elegido]},
                            {'type': 'vision_rodear', 'x': x, 'y': y,
                             'radio_m': radio, 'altura_m': altura,
                             'puntos': int(d.get('puntos') or RODEO_PUNTOS)})
            print('[%s] el operador manda al dron %s a mirar (%.1f, %.1f) desde otro lado, '
                  'a %.0f m de radio y %.0f m de altura'
                  % (datetime.now().strftime('%H:%M:%S'), elegido, x, y, radio, altura),
                  flush=True)
            self._responder(json.dumps({'dron': elegido, 'radio_m': radio,
                                        'altura_m': altura}).encode('utf-8'))
            return
        if self.path.split('?')[0] == '/veredicto':
            try:
                d = json.loads(crudo)
                if d.get('v') not in ('si', 'no'):
                    raise ValueError('v')
                x, y = float(d['x']), float(d['y'])
            except Exception:
                self._responder(b'{"error": "veredicto"}', codigo=400)
                return
            import base64
            carpeta = ESTADO['veredictos_dir']
            os.makedirs(carpeta, exist_ok=True)
            sello = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            archivo = None
            if d.get('crop'):
                archivo = '%s_%s.jpg' % (sello, d['v'])
                with open(os.path.join(carpeta, archivo), 'wb') as f:
                    f.write(base64.b64decode(d['crop']))
            fila = {'t': sello, 'v': d['v'], 'x': x, 'y': y, 'cls': d.get('cls'),
                    'dron': d.get('dron'), 'n_obs': d.get('n_obs'), 'looks': d.get('looks'),
                    'radius_m': d.get('radius_m'), 'crop': archivo}
            with CANDADO:
                with open(os.path.join(carpeta, 'veredictos.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(fila) + '\n')
            print('[%s] veredicto del operador: %s en (%.1f, %.1f)%s' %
                  (datetime.now().strftime('%H:%M:%S'), d['v'], x, y,
                   ' con recorte' if archivo else ''), flush=True)
            if d['v'] == 'no':
                with CANDADO:
                    nodos = dict(ESTADO['nodos'])
                rechazos = descartes_por_dron(x, y, time.time())
                for dron, cuerpo in rechazos.items():
                    if dron not in nodos:
                        continue
                    empujar_mensaje({dron: nodos[dron]}, dict(cuerpo, type='vision_descarte'))
                if rechazos:
                    print('        y deja de reportarlo en: %s'
                          % ', '.join('dron %s (%.1f, %.1f)' % (k, v['x'], v['y'])
                                      for k, v in sorted(rechazos.items())), flush=True)
                else:
                    print('        PERO NINGUN DRON PUEDE OBEDECERLO: ninguno reporta ese punto '
                          'con apariencia. Una placa sin reid_model no puede honrar un veredicto.',
                          flush=True)
            self._responder(json.dumps({'status': 'ok', 'crop': archivo}).encode('utf-8'))
            return
        self._responder(b'{"status": "ok"}')
        try:
            payload = json.loads(crudo)
            mensaje = json.loads(payload['message'])
        except Exception:
            print('mensaje no-JSON:', crudo[:200], flush=True)
            return
        if mensaje.get('type') == 'vision_poi':
            registrar(mensaje, payload.get('source'))
        elif mensaje.get('type') == 'vision_marco':
            recibir_marco(mensaje, payload.get('source'))
        else:
            print('[%s] mensaje: %s' % (datetime.now().strftime('%H:%M:%S'), mensaje),
                  flush=True)

    def do_GET(self):
        """Everything the operator's page reads: itself, the state, and the pictures.

        /estado recomputes the fused map on every READ as well as on every report. When every
        drone goes silent no report arrives to refresh it, and a silent map must not keep
        claiming corroboration. It also carries 'banco', which says whether these boards are on
        a desk: the page uses it to say the metres mean nothing instead of printing a number
        nobody should believe, and it travels in the state rather than being baked into the page
        so that the page stays a static file.

        /segunda.jpg is served from disk instead of travelling inside /estado, because the
        annotated frame is the size of a photograph and the state is polled once a second.
        """
        ruta = self.path.split('?')[0]
        if ruta == '/':
            self._responder(PAGINA.encode('utf-8'), 'text/html; charset=utf-8')
        elif ruta == '/estado':
            with CANDADO:
                ESTADO['pois'] = pois_vigentes(time.time())
                d = {
                    'objetivo': ESTADO['objetivo'],
                    'segunda': ESTADO['segunda'],
                    'pois': ESTADO['pois'],
                    'banco': EN_BANCO,
                    'drones': ESTADO['drones'],
                    'rastros': ESTADO['rastros'],
                    'origen': ESTADO['origen'],
                    'origen_cli': ESTADO['origen_cli'],
                    'desacuerdo': ESTADO['desacuerdo'],
                    'celdas': ESTADO['celdas'],
                    'georef': ESTADO['georef'],
                    'pedidos': pedidos_de_verificacion(
                        ESTADO['pois'], ESTADO['drones'], time.time()),
                    'tiene_fondo': ESTADO['fondo'] is not None,
                    'frame_actual': ESTADO['frame_actual'],
                    'ahora': time.time(),
                    'reportes': len(ESTADO['historia']),
                }
            self._responder(json.dumps(d).encode('utf-8'))
        elif ruta == '/frame_actual_n':
            with CANDADO:
                n = ESTADO['frame_actual']
            self._responder(json.dumps({'n': n}).encode('utf-8'))
        elif ruta == '/segunda.jpg':
            consulta = self.path.split('?', 1)[1] if '?' in self.path else ''
            dron = dict(par.split('=', 1) for par in consulta.split('&') if '=' in par).get('dron', '')
            with CANDADO:
                ruta_jpg = (ESTADO['segunda'].get(dron) or {}).get('dibujo')
            if not ruta_jpg or not os.path.exists(ruta_jpg):
                self._responder(b'{"error": "sin imagen"}', codigo=404)
                return
            with open(ruta_jpg, 'rb') as fh:
                self._responder(fh.read(), 'image/jpeg')
        elif ruta == '/frame':
            with CANDADO:
                n, base = ESTADO['frame_actual'], ESTADO['frames_dir']
            if n is None or not base:
                self._responder(b'sin frame', 'text/plain', 404)
                return
            ruta_img = os.path.join(base, 'frame_%04d.jpg' % n)
            if not os.path.exists(ruta_img):
                self._responder(b'no existe', 'text/plain', 404)
                return
            with open(ruta_img, 'rb') as fh:
                self._responder(fh.read(), 'image/jpeg')
        elif ruta == '/pedidos':
            with CANDADO:
                d = pedidos_de_verificacion(
                    ESTADO['pois'], ESTADO['drones'], time.time())
            self._responder(json.dumps({'pedidos': d}).encode('utf-8'))
        elif ruta == '/buscar':
            with CANDADO:
                d = {'clases': ESTADO['buscar'], 'v': ESTADO['buscar_v'],
                     'epoca': ESTADO['buscar_epoca']}
            self._responder(json.dumps(d).encode('utf-8'))
        elif ruta == '/fondo':
            if not ESTADO['fondo']:
                self._responder(b'sin fondo', 'text/plain', 404)
                return
            with open(ESTADO['fondo'], 'rb') as fh:
                self._responder(fh.read(), 'image/png')
        else:
            self._responder(b'no', 'text/plain', 404)


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Ground Station</title>
<style>
  :root { --fondo:#12141a; --panel:#1b1f28; --linea:#2c3240; --texto:#e6e9ef;
          --tenue:#8b93a5; --ok:#4ade80; --duda:#fbbf24; --mobile:#60a5fa; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--fondo); color:var(--texto);
         font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid var(--linea); }
  header h1 { margin:0; font-size:16px; font-weight:600; letter-spacing:.02em; }
  .estado { margin-left:auto; display:flex; gap:18px; color:var(--tenue); font-size:13px; }
  .punto { display:inline-block; width:9px; height:9px; border-radius:50%;
           margin-right:6px; background:#555; }
  .vivo { background:var(--ok); box-shadow:0 0 8px var(--ok); }
  main { display:grid; grid-template-columns:1fr 340px; height:calc(100vh - 49px); }
  #lienzo { width:100%; height:100%; display:block; background:#0b0d12; }
  aside { background:var(--panel); border-left:1px solid var(--linea);
          overflow-y:auto; padding:14px; }
  aside h2 { margin:0 0 10px; font-size:12px; text-transform:uppercase;
             letter-spacing:.08em; color:var(--tenue); font-weight:600; }
  .poi { border:1px solid var(--linea); border-left-width:4px; border-radius:6px;
         padding:10px 12px; margin-bottom:10px; background:#171b23; }
  .poi.ok { border-left-color:var(--ok); }
  .poi.duda { border-left-color:var(--duda); }
  .poi .tit { font-weight:600; display:flex; align-items:center; gap:8px; }
  .chip { font-size:11px; padding:1px 7px; border-radius:99px; font-weight:600; }
  .chip.ok { background:rgba(74,222,128,.15); color:var(--ok); }
  .chip.duda { background:rgba(251,191,36,.15); color:var(--duda); }
  .chip.mobile { background:rgba(96,165,250,.15); color:var(--mobile); }
  .chip.clase { background:rgba(230,233,239,.10); color:var(--texto); }
  .chip.noper { background:rgba(248,113,113,.15); color:#f87171; }
  #btn-fondo, #btn-limpiar, #btn-historial {
               font:600 11px system-ui; padding:3px 10px; border-radius:99px;
               cursor:pointer; border:1px solid var(--linea); background:#171b23;
               color:var(--tenue); }
  #btn-fondo.on, #btn-historial.on { background:var(--texto); border-color:var(--texto);
                                     color:#0b0e14; }
  .chip.viejo { background:rgba(139,147,165,.15); color:var(--tenue); }
  #filtro { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }
  #filtro button { font:600 11px system-ui; padding:3px 9px; border-radius:99px;
                   cursor:pointer; border:1px solid var(--linea); background:#171b23;
                   color:var(--texto); }
  #filtro button.off { color:var(--tenue); text-decoration:line-through; opacity:.55; }
  #buscar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:10px; }
  #buscar button { font:600 11px system-ui; padding:3px 9px; border-radius:99px;
                   cursor:pointer; border:1px solid var(--linea); background:#171b23;
                   color:var(--tenue); }
  #buscar button.on { background:var(--ok); border-color:var(--ok); color:#0b0e14; }
  #buscar .etq { font:600 10px system-ui; letter-spacing:.08em; text-transform:uppercase;
                 color:var(--tenue); margin-right:2px; }
  #camara { margin-bottom:12px; display:none; }
  #camara img { width:100%; border-radius:6px; display:block; border:1px solid var(--linea); }
  #camara .cab { font:600 10px system-ui; letter-spacing:.08em; text-transform:uppercase;
                 color:var(--tenue); margin-bottom:6px; }
  #pedidos { margin-bottom:12px; }
  #pedidos .cab { font:600 10px system-ui; letter-spacing:.08em; text-transform:uppercase;
                  color:var(--duda); margin-bottom:6px; }
  #pedidos .item { border-left:3px solid var(--duda); background:rgba(251,191,36,.07);
                   padding:7px 10px; border-radius:0 6px 6px 0; margin-bottom:6px;
                   font-size:12px; line-height:1.5; }
  #pedidos .coord { color:var(--tenue); font-size:11px; }
  .poi dl { margin:8px 0 0; display:grid; grid-template-columns:auto 1fr;
            gap:2px 10px; font-size:13px; }
  .poi dt { color:var(--tenue); }
  .poi dd { margin:0; font-variant-numeric:tabular-nums; }
  .poi dd.nosirve { color:var(--tenue); font-style:italic; font-variant-numeric:normal; }
  .poi .barra { margin:8px 0 0; height:6px; border-radius:99px; overflow:hidden;
                background:rgba(255,255,255,.08); }
  .poi .barra i { display:block; height:100%; border-radius:99px; background:var(--duda); }
  .poi.ok .barra i { background:var(--ok); }
  .poi .cuenta { margin:4px 0 0; font-size:11px; color:var(--tenue);
                 font-variant-numeric:tabular-nums; }
  .veredicto button { font:600 11px system-ui; padding:3px 9px; margin:8px 6px 0 0;
                      border-radius:99px; border:1px solid var(--linea); background:transparent;
                      color:inherit; cursor:pointer; }
  .crop { display:block; margin:10px 0 0; width:128px; max-width:100%;
             border-radius:4px; border:1px solid var(--linea); background:#0b0d12; }
  .sinrecorte { margin:8px 0 0; font-size:12px; color:var(--tenue); font-style:italic; }
  #buscar button.mas { opacity:.55; font-style:italic; }
  #buscar button.vista { font-size:11px; padding:1px 8px; margin-left:6px; }
  #buscar .segunda { margin:4px 0 0; }
  #buscar .mirada { max-width:340px; }
  .segunda { margin:8px 0 0; font-size:12px; color:var(--tenue); }
  .segunda.hallazgo { color:var(--ok); font-weight:600; }
  .segunda.fallo { color:#f87171; }
  .mirada { display:block; margin:6px 0 0; width:100%; border-radius:6px; }
  .vacio { color:var(--tenue); font-style:italic; padding:20px 0; text-align:center; }
  .nota { color:var(--tenue); font-size:12px; margin-top:14px;
          padding-top:12px; border-top:1px solid var(--linea); }
  #alarma { background:#7f1d1d; color:#fee2e2; padding:9px 16px; font-size:13px;
            font-weight:600; border-bottom:1px solid #991b1b; }
  /* The two panels of the aside. A tab and not a second column: the map is the point, and the
     health of the fleet is what you go and look at, not what you watch. */
  #solapas { display:flex; gap:4px; margin:0 0 12px; }
  #solapas button { flex:1; background:#171b23; color:var(--tenue); border:1px solid var(--linea);
                    border-radius:6px; padding:6px 0; font-size:11px; font-weight:600;
                    text-transform:uppercase; letter-spacing:.07em; cursor:pointer; }
  #solapas button.on { background:var(--panel); color:var(--texto); border-color:#4b5563; }
  #solapas button .mal { color:#f87171; }
  #diag table { width:100%; border-collapse:collapse; font-size:12px; }
  #diag th { text-align:left; color:var(--tenue); font-weight:600; font-size:10px;
             text-transform:uppercase; letter-spacing:.06em; padding:0 6px 6px 0; }
  #diag td { padding:7px 6px 7px 0; border-top:1px solid var(--linea); font-variant-numeric:tabular-nums; }
  #diag .dato { font-weight:600; }
  #diag .bien { color:var(--ok); }
  #diag .ojo  { color:var(--duda); }
  #diag .mal  { color:#f87171; }
  #diag .nada { color:var(--tenue); font-style:italic; font-weight:400; }
  #diag .porque { color:var(--tenue); font-size:11px; padding:0 0 9px; border:0; }
</style></head>
<body>
<div id="alarma" style="display:none"></div>
<header>
  <h1>Ground Station</h1>
  <div class="estado">
    <span><span class="punto" id="luz"></span><span id="enlace">waiting for a drone</span></span>
    <span id="cuenta">0 POI</span>
    <span id="ritmo"></span>
    <span id="reportes">0 reports</span>
    <button id="btn-limpiar"
            title="hide every unconfirmed contact until it is seen again">clear</button>
    <button id="btn-historial" title="show, dimmed, what the layers hid">history</button>
    <button id="btn-reiniciar"
            title="tell every drone to forget what it has found and start over; the operator's verdicts are kept">reset all</button>
    <button id="btn-fondo" hidden>satellite</button>
  </div>
</header>
<main>
  <canvas id="lienzo"></canvas>
  <aside>
    <div id="solapas">
      <button data-solapa="contactos" class="on">Contacts</button>
      <button data-solapa="diag">Health</button>
    </div>
    <div id="panel-contactos">
      <div id="camara"><div class="cab">what the camera sees</div><img alt=""></div>
      <div id="pedidos"></div>
      <div id="buscar"></div>
      <div id="filtro"></div>
      <div id="lista"><div class="vacio">Nothing yet.</div></div>
    </div>
    <div id="panel-diag" hidden>
      <div id="diag"><div class="vacio">No drone has reported yet.</div></div>
    </div>
  </aside>
</main>
<script>
const lienzo = document.getElementById('lienzo');
const ctx = lienzo.getContext('2d');
let fondo = null, estado = null;
// Classes hidden by the operator, and every class seen since the page opened. The second is
// kept so a button does not vanish the moment its last POI leaves the frame -- it would take
// the operator's filter with it, silently.
let ocultas = new Set(), clasesVistas = [];
// Declared here, not next to its button: redimensionar() draws on load, before
// the button block runs, and a `let` read from the temporal dead zone throws and
// takes the whole script with it -- empty counters, no controls, black canvas.
let verFondo = false;
// The frame is fitted to the POIs, which with a single pin is 33 m across: tight
// enough to place it against the grid, too tight for the imagery to say where on
// earth this is. The wheel widens it without moving the fit.
let zoom = 1;
// What is actually drawn: the POIs surviving the filter. Kept apart from estado.pois so
// dibujar(), which also runs on resize and when the imagery loads, never has to re-filter.
let visibles = [];

function claseDe(p) { return p.cls || 'no class'; }

// -- the two layers ------------------------------------------------------------
// A POI is a claim about where something was when the drone last saw it, and that claim loses
// value by the second. Tactical displays treat an unrefreshed track the same way: it fades, then
// leaves the screen. Two layers follow from that.
//
// LIVE contacts are the unconfirmed, unverified ones. They are drawn at full opacity for
// FRESCO_S, fade linearly to ALFA_MIN at TOPE_VIVO_S, and are not drawn at all past it -- neither
// the pin, nor its 95 % circle, nor its card.
//
// PERSISTENT ones are those the identity layer confirmed (mature) or the operator marked "es lo
// que busco". They stay at their last position regardless of age, and say how long ago that was.
//
// FRESCO_S is the identity layer's extrapolation cap (identity.py, extrapolation_max_s): up to it
// the reported point has been carried forward to the report instant, past it the position is
// frozen while the target keeps moving. That is where the claim starts to degrade.
//
// TOPE_VIVO_S is chosen from data. Reports arrive every 2 s and a person in view is detected in
// ~28 % of frames, in bursts, so a target in view goes unseen for several seconds at a time; a cap
// shorter than those gaps makes a real contact blink on and off. Measured on the flight-3 replay
// (preliminaries on), the gaps after which a POI was sighted again were, in seconds:
// median 3.7, and the longest ones 9.4, 10.0, 10.0, 11.4, 11.5, 12.7 -- then nothing until 16.0,
// 17.4, 22.5, 29.5 and 49-55, which are targets that left the view and came back. 15 s is 7.5 report
// periods and sits in that gap: it covers every in-view dropout measured and hides what has
// really gone. A contact that returns after longer simply reappears, which is the honest display.
const FRESCO_S = 3;
const TOPE_VIVO_S = 15;
const ALFA_MIN = 0.25;
// How what the layers hid is drawn when the operator asks for the history.
const ALFA_HISTORIAL = 0.3;
// Cards never go fainter than this: the card is where the crop is judged, and a crop drawn at a
// quarter opacity cannot be.
const ALFA_TARJETA_MIN = 0.45;
let verHistorial = false;
// Lo ultimo que un boton tiene para decir. Se borra con el reporte siguiente: es la respuesta a un
// clic, no un estado del sistema.
let aviso = '';
// What "limpiar" hid: where each live contact was and how old it was at that moment. Kept by
// position, like the verdicts, because a POI carries no id that survives a report.
let limpiados = [];

// Seconds since the last sighting, on the station's clock: the age the drone reported plus the
// time since that report arrived, so a contact keeps ageing while its drone is silent. None for a
// POI the drone did not age (older firmware): there is nothing to fade it by.
function edadDe(p) {
  if (p.age_s == null) return null;
  const desde = (estado && estado.ahora != null && p.t != null) ? Math.max(0, estado.ahora - p.t) : 0;
  return p.age_s + desde;
}

function persistente(p) { return !!p.mature || veredictoDe(p) === 'si'; }

// Whether POI p is the contact recorded at cleared place v: the same reach as an operator's verdict,
// never the POI's own 95 % radius, which for a mobile contact spans hundreds of metres and would hide
// contacts that appeared after the click.
function cercaDe(p, v) {
  return veredictoAplica(v, p);
}

// A cleared contact comes back with a newer sighting: an age smaller than the one it had when it
// was cleared. The record is then dropped, so ageing again inside the cap does not re-hide it. A
// record whose place no POI occupies any more is dropped too, or it would hide whatever arrives
// there next.
function podarLimpiados(pois) {
  limpiados = limpiados.filter(v =>
    pois.some(p => cercaDe(p, v))
    && !pois.some(p => cercaDe(p, v) && edadDe(p) != null && edadDe(p) < v.edad));
}

function ocultoPorCapas(p) {
  // Clearing wins over being confirmed, and only clearing does. The operator asked for a clear
  // button that clears, and a confirmed point that survived it would make the button look broken.
  // Fading does NOT win: a confirmed contact still stays put and says how long ago it was seen,
  // because that is staleness and not a decision anybody made.
  if (limpiados.some(v => cercaDe(p, v))) return true;
  if (persistente(p)) return false;
  const e = edadDe(p);
  return e != null && e >= TOPE_VIVO_S;
}

// The opacity a POI is drawn with, pin, circle and card alike. 0 means not drawn.
function alfaDe(p) {
  if (persistente(p)) return 1;
  if (ocultoPorCapas(p)) return verHistorial ? ALFA_HISTORIAL : 0;
  const e = edadDe(p);
  if (e == null || e <= FRESCO_S) return 1;
  return 1 - (1 - ALFA_MIN) * (e - FRESCO_S) / (TOPE_VIVO_S - FRESCO_S);
}

// "visto hace N s", only once the age has passed the cap: before it the fading already says so.
function vistoHace(p) {
  const e = edadDe(p);
  return (e != null && e > TOPE_VIVO_S) ? `seen ${Math.round(e)} s ago` : '';
}

// Hides every live contact on screen until it is seen again. A POI without an age is left alone:
// nothing would ever say it had been seen again, so it would be hidden for good.
function limpiar() {
  if (!estado) return;
  let ocultados = 0, confirmados = 0;
  for (const p of estado.pois) {
    const e = edadDe(p);
    if (e == null || ocultas.has(claseDe(p)) || ocultoPorCapas(p)) continue;
    if (persistente(p)) confirmados++;
    limpiados.push({ x: p.x, y: p.y, r: p.radius_m || 0, edad: e });
    ocultados++;
  }
  // It used to refuse to hide confirmed points, on the grounds that something the system stands
  // behind should not vanish because the operator tidied up. The operator disagreed, and they are
  // the one pressing it: a button called clear that leaves the screen full reads as broken. So it
  // clears everything and SAYS how many of those were confirmed, and 'historial' brings them all
  // back, which is what makes clearing safe rather than destructive.
  aviso = ocultados
    ? 'cleared ' + ocultados + (confirmados ? ', ' + confirmados + ' of them confirmed' : '')
    : 'nothing to clear';
  pintar();
}

function alternarHistorial() {
  verHistorial = !verHistorial;
  const b = document.getElementById('btn-historial');
  b.className = verHistorial ? 'on' : '';
  const tapados = estado ? estado.pois.filter(ocultoPorCapas).length : 0;
  aviso = tapados ? '' : 'nothing hidden to show';
  if (estado) pintar();
}

// The area drawn, in metres around the mission origin. Redrawn to fit whatever arrives, so a
// POI never lands outside the view.
let vista = {e0:-25, e1:25, n0:-25, n1:25};

function redimensionar() {
  const r = lienzo.getBoundingClientRect();
  const d = window.devicePixelRatio || 1;
  lienzo.width = r.width * d; lienzo.height = r.height * d;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  dibujar();
}
window.addEventListener('resize', redimensionar);

function aPantalla(e, n) {
  const r = lienzo.getBoundingClientRect();
  return [ (e - vista.e0) / (vista.e1 - vista.e0) * r.width,
           (1 - (n - vista.n0) / (vista.n1 - vista.n0)) * r.height ];
}

function ajustarVista(pois) {
  // The drones count towards the frame as much as the targets do. Fitting to the targets
  // alone sends the aircraft off the edge, which is when an operator most wants to see it.
  const puntos = pois.slice();
  if (estado) for (const d of Object.values(estado.drones || {}))
    if (d.pos) puntos.push({x: d.pos[0], y: d.pos[1]});
  if (!puntos.length) return;
  let e0=1e9,e1=-1e9,n0=1e9,n1=-1e9;
  for (const p of puntos) { e0=Math.min(e0,p.x); e1=Math.max(e1,p.x);
                            n0=Math.min(n0,p.y); n1=Math.max(n1,p.y); }
  const m = (Math.max(12, (e1-e0), (n1-n0)) * 0.7 + 8) * zoom;
  const ce=(e0+e1)/2, cn=(n0+n1)/2;
  vista = {e0:ce-m, e1:ce+m, n0:cn-m, n1:cn+m};
}

// The aircraft, and where it has been. Drawn under the targets on purpose: the drone is
// context for the find, not the find. The trail is what tells an operator whether the drone
// is working the area or hovering over it -- and a drone that hovers gives rays that barely
// cross, which is the commonest cause of a bad fix and is invisible from the pins alone.
const COLOR_DRON = ['#60a5fa', '#f472b6', '#a3e635', '#fbbf24'];

function dibujarDrones() {
  const ids = Object.keys(estado.drones || {}).sort();
  ids.forEach((id, i) => {
    const col = COLOR_DRON[i % COLOR_DRON.length];
    const rastro = (estado.rastros || {})[id] || [];
    if (rastro.length > 1) {
      ctx.strokeStyle = col; ctx.globalAlpha = 0.35; ctx.lineWidth = 2;
      ctx.beginPath();
      rastro.forEach((q, k) => {
        const [x, y] = aPantalla(q[0], q[1]);
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    const d = estado.drones[id];
    if (!d.pos) return;
    const [x, y] = aPantalla(d.pos[0], d.pos[1]);
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 6.2832); ctx.fill();
    ctx.strokeStyle = '#0b0d12'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = col; ctx.font = '600 11px system-ui';
    ctx.fillText('drone ' + id + (d.pos[2] != null ? '  ' + d.pos[2].toFixed(0) + ' m' : ''),
                 x + 10, y - 8);
  });
}

// The satellite image is georeferenced by its corners, so a POI in metres becomes a fraction
// of the image once it is coordinates. Without an image a metric grid stands in: the pins are
// still in the right place relative to each other and to the origin.
function dibujarFondo() {
  const r = lienzo.getBoundingClientRect();
  ctx.fillStyle = '#0b0d12'; ctx.fillRect(0, 0, r.width, r.height);
  if (verFondo && fondo && estado && estado.georef && estado.origen) {
    const [lat0, lon0, lat1, lon1] = estado.georef;
    const [olat, olon] = estado.origen;
    const R = 6378137.0, gr = Math.PI/180;
    const fx = e => { const lng = olon + (e/(R*Math.cos(olat*gr)))/gr;
                      return (lng - lon0)/(lon1 - lon0); };
    const fy = n => { const lat = olat + (n/R)/gr;
                      return (lat - lat0)/(lat1 - lat0); };
    const sx0 = fx(vista.e0)*fondo.width, sx1 = fx(vista.e1)*fondo.width;
    const sy0 = fy(vista.n1)*fondo.height, sy1 = fy(vista.n0)*fondo.height;
    try {
      ctx.drawImage(fondo, sx0, sy0, sx1-sx0, sy1-sy0, 0, 0, r.width, r.height);
    } catch (e) { /* crop fuera de la imagen: queda el fondo liso */ }
  }
  // metric grid, every 5 m, drawn over the imagery too: scale is what makes a map readable
  ctx.strokeStyle = 'rgba(255,255,255,.07)'; ctx.lineWidth = 1;
  ctx.fillStyle = 'rgba(255,255,255,.35)'; ctx.font = '11px system-ui';
  for (let e = Math.ceil(vista.e0/5)*5; e <= vista.e1; e += 5) {
    const [x] = aPantalla(e, 0);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, r.height); ctx.stroke();
    ctx.fillText(e + ' m', x + 3, r.height - 5);
  }
  for (let n = Math.ceil(vista.n0/5)*5; n <= vista.n1; n += 5) {
    const [, y] = aPantalla(0, n);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(r.width, y); ctx.stroke();
    ctx.fillText(n + ' m', 4, y - 4);
  }
  // the mission origin: the point every metre in this view is counted from
  const [ox, oy] = aPantalla(0, 0);
  ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(ox-7, oy); ctx.lineTo(ox+7, oy);
  ctx.moveTo(ox, oy-7); ctx.lineTo(ox, oy+7); ctx.stroke();
}

function dibujar() {
  dibujarFondo();
  if (!estado) return;
  dibujarDrones();
  for (const p of visibles) {
    const [x, y] = aPantalla(p.x, p.y);
    const col = p.mature ? '#4ade80' : '#fbbf24';
    // Everything this POI draws, its 95 % circle included, shares one opacity: a faded pin under
    // a full-strength circle would still claim the area.
    ctx.globalAlpha = alfaDe(p);
    // A halo sized by nothing but legibility: this is not an uncertainty ellipse and must not
    // be read as one. The real uncertainty is a few metres and would swallow the pin.
    ctx.beginPath(); ctx.arc(x, y, 22, 0, 6.2832);
    ctx.fillStyle = col + '22'; ctx.fill();
    if (p.radius_m != null) {
      // This one IS the uncertainty: the drone's own 95 % circle, in metres on the ground.
      // Where to send someone is inside it, not at the pin.
      const [xr] = aPantalla(p.x + p.radius_m, p.y);
      ctx.beginPath(); ctx.setLineDash([6, 5]); ctx.arc(x, y, Math.abs(xr - x), 0, 6.2832);
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.2832);
    ctx.fillStyle = col; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#12141a'; ctx.stroke();
    if (p.mobile) {
      ctx.beginPath(); ctx.arc(x, y, 15, 0, 6.2832);
      ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2; ctx.stroke();
    }
    ctx.fillStyle = '#e6e9ef'; ctx.font = '600 12px system-ui';
    ctx.fillText(veredictoDe(p) === 'si' ? 'VERIFICADO'
                 : (p.mature ? 'CONFIRMADO' : 'POR VERIFICAR'), x + 16, y - 12);
    if (p.cls) {
      // The class under the verdict, dimmer: the verdict decides whether to look, the class
      // decides whether it is what you are looking for.
      ctx.fillStyle = 'rgba(230,233,239,.65)'; ctx.font = '12px system-ui';
      ctx.fillText(p.cls, x + 16, y + 3);
    }
    if (vistoHace(p)) {
      ctx.fillStyle = 'rgba(230,233,239,.65)'; ctx.font = 'italic 12px system-ui';
      ctx.fillText(vistoHace(p), x + 16, y + 18);
    }
    ctx.globalAlpha = 1;
  }
}

function pintarLista(pois) {
  const cont = document.getElementById('lista');
  if (!pois.length) { cont.innerHTML = '<div class="vacio">Nothing yet.</div>'; return; }
  cont.innerHTML = pois.map((p, i) => `
    <div class="poi ${p.mature ? 'ok' : 'duda'}"
         style="opacity:${Math.max(alfaDe(p), ALFA_TARJETA_MIN).toFixed(2)}">
      <div class="tit">#${i+1}
        ${vistoHace(p) ? `<span class="chip viejo">${vistoHace(p)}</span>` : ''}
        <span class="chip ${p.mature ? 'ok' : 'duda'}" title="${p.mature
          ? 'identity tracked this long enough to stand behind it'
          : 'a track formed but never matured: a request to verify, not a finding'}"
              >${p.mature ? 'CONFIRMED' : 'UNCONFIRMED'}</span>
        ${p.mobile ? '<span class="chip mobile">MOVING</span>' : ''}
        ${p.cls ? `<span class="chip clase">${p.cls}</span>` : ''}
        ${veredictoDe(p) === 'si' ? '<span class="chip ok">OPERATOR CONFIRMED</span>' : ''}
        ${p.clip_no_persona
          ? `<span class="chip noper">probably not a person (CLIP ${p.clip.toFixed(2)})</span>` : ''}
      </div>
      ${p.crop
        ? `<img class="crop" src="data:image/jpeg;base64,${p.crop}" alt="what the drone saw">`
        : (p.mature ? '' : '<div class="sinrecorte">no crop: nothing to judge by</div>')}
      ${p.evidence != null && p.looks_min
        ? `<div class="barra"><i style="width:${Math.round(p.evidence * 100)}%"></i></div>
           <div class="cuenta">seen in ${p.looks} of the ${p.looks_min} looks it takes${
             p.evidence >= 1 ? ' &middot; enough to report' : ''}</div>` : ''}
      <dl>
        <dt>evidence</dt><dd>${p.n_obs} sightings${p.conf != null ? ', confidence ' + p.conf : ''}</dd>
        <dt>drone</dt><dd>${p.dron}</dd>
        ${(estado && estado.banco)
          ? '<dt>where</dt><dd class="nosirve">on the bench: no autopilot, the metres mean nothing</dd>'
          : `<dt>where</dt><dd>${p.x} m E, ${p.y} m N${
               p.lat != null ? ` &middot; ${p.lat}, ${p.lng}` : ''}</dd>`
            + (p.radius_m != null
               ? `<dt>margin</dt><dd>&plusmn;${p.radius_m} m (95 %)</dd>` : '')}
      </dl>
      ${segundaDe(p)}
      <div class="veredicto">${veredictoDe(p) === 'si' ? '' :
        `<button data-v="si" data-i="${i}" title="keep it on the map and stop asking">that is what I am looking for</button>`
        + `<button data-v="no" data-i="${i}" title="the drone stops reporting this point">not it</button>`}
        <button data-mirar="${i}" title="asks drone ${p.dron} for the frame it is looking at and runs a bigger detector on the ground, about 1.4 s">second opinion</button>
        <button data-rodear="${i}" title="sends another drone to a point this target has not been seen from, and that picture comes back">look from another angle</button></div>
    </div>`).join('');
  for (const b of cont.querySelectorAll('button[data-v]')) {
    b.onclick = () => marcar(pois[+b.dataset.i], b.dataset.v);
  }
  for (const b of cont.querySelectorAll('button[data-mirar]')) {
    b.onclick = () => pedirSegunda(pois[+b.dataset.mirar].dron);
  }
  for (const b of cont.querySelectorAll('button[data-rodear]')) {
    b.onclick = () => pedirRodeo(pois[+b.dataset.rodear]);
  }
}

// The block a card shows about its drone's second opinion. Keyed by DRONE, not by point: the
// aircraft sends the frame it is looking at, which answers "is there anybody here", not "is this
// particular point real". Saying otherwise would promise a link between the boxes and the POI that
// nothing in the exchange establishes.
// What the ground made of one drone's frame. Keyed by DRONE because that is what the exchange
// answers: the aircraft sends the frame it is looking at, which says whether there is anybody
// there, not whether one particular point is real.
function bloqueSegunda(dron) {
  const d = estado.segunda && estado.segunda[String(dron)];
  if (!d) return '';
  const clase = d.estado === 'error' ? 'fallo' : (d.estado === 'listo' && d.n ? 'hallazgo' : '');
  const viejo = d.estado === 'listo' && estado.ahora && (estado.ahora - d.t) > SEGUNDA_VIEJA_S
    ? ` (${Math.round(estado.ahora - d.t)} s ago)` : '';
  return `<div class="segunda ${clase}">${textoSegunda(d)}${viejo}</div>`
    + (d.estado === 'listo' && d.dibujo
       ? `<img class="mirada" src="/segunda.jpg?dron=${encodeURIComponent(dron)}&t=${d.t}"
               alt="lo que RF-DETR encontro en ese cuadro">` : '');
}

function segundaDe(p) {
  return bloqueSegunda(p.dron);
}

function pintarFiltro(pois) {
  for (const p of pois) {
    const c = claseDe(p);
    if (!clasesVistas.includes(c)) clasesVistas.push(c);
  }
  clasesVistas.sort();
  const cont = document.getElementById('filtro');
  // Nothing to choose between while only one class has ever arrived.
  if (clasesVistas.length < 2) { cont.innerHTML = ''; return; }
  cont.innerHTML = '<span class="etq" title="these buttons only hide: the POI keeps arriving '
    + 'and comes back with a click">showing</span>'
    + clasesVistas.map(c =>
      `<button data-c="${c}" class="${ocultas.has(c) ? 'off' : ''}">${c}</button>`).join('');
  for (const b of cont.querySelectorAll('button')) {
    b.onclick = () => {
      const c = b.dataset.c;
      if (ocultas.has(c)) ocultas.delete(c); else ocultas.add(c);
      pintar();
    };
  }
}

// What the operator decided about a point after looking at its crop. The drone cannot tell a
// person from an object it keeps confusing with one -- measured, none of its signals separate
// them -- so this is not a convenience, it is the step that makes a report actionable.
//
// Kept by position, never by list index: the list is rebuilt on every report, reorders as
// evidence grows, and a POI carries no id that survives it. Neither the crop nor the appearance
// vector is one either: both are replaced as evidence arrives.
//
// A verdict belongs to one POI. In each report it goes to the POI nearest to where it was given,
// of the same class, and only if that POI is within VEREDICTO_ALCANCE_M. The reach is the chain's
// report-to-report jitter, never a POI's 95 % radius: that radius says how far off a point may
// be, not how far apart two things are, and a mobile POI carries one of hundreds of metres. On
// the flight-3 replay a static POI moved at most 1.12 m between consecutive reports and a mobile
// one 3.44 m, while two distinct static POIs came as close as 0.98 m -- which is why the nearest
// one wins and a neighbour inside the reach does not inherit the verdict.
const VEREDICTO_ALCANCE_M = 3;
const veredictos = [];

function mismaClase(a, b) {
  return a.cls == null || b.cls == null || a.cls === b.cls;
}

// Whether verdict v belongs to POI p in the current report.
function veredictoAplica(v, p) {
  if (!mismaClase(v, p)) return false;
  const d = Math.hypot(p.x - v.x, p.y - v.y);
  if (d > Math.min(v.r || VEREDICTO_ALCANCE_M, VEREDICTO_ALCANCE_M)) return false;
  return !estado.pois.some(q => q !== p && mismaClase(v, q)
                               && Math.hypot(q.x - v.x, q.y - v.y) < d);
}

function veredictoDe(p) {
  // The latest verdict wins, so an operator who changes their mind is obeyed.
  for (let i = veredictos.length - 1; i >= 0; i--) {
    if (veredictoAplica(veredictos[i], p)) return veredictos[i].v;
  }
  return null;
}

function marcar(p, v) {
  veredictos.push({ x: p.x, y: p.y, r: p.radius_m || 0, cls: p.cls, v });
  pintar();
  // "Es lo que busco" also tells the drone where to look harder. The verdict is the only thing the
  // operator knows that the aircraft cannot, and acting on it costs no computing: the detector had
  // already scored the boxes it was throwing away under that square.
  if (v === 'si') {
    fetch('/objetivo', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({x: p.x, y: p.y})}).catch(() => {});
  }
  // Kept on the station's disk too, with the crop: the page forgets on reload, the data must not.
  fetch('/veredicto', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({v, x: p.x, y: p.y, cls: p.cls, dron: p.dron, n_obs: p.n_obs,
                          looks: p.looks, radius_m: p.radius_m, crop: p.crop})})
    .catch(() => {});
}

// -- the ground's second opinion ------------------------------------------------------------
// What flies is what fits in the power budget, and it finds fewer people than a detector that does
// not have to fit: on the 02ago flight, where the drone is high, the aircraft found 46 % of the
// people and RF-DETR on the laptop found 90 %. RF-DETR takes 1.4 s per frame against 35 ms, so it
// will never fly; this button is what lets the operator borrow it for one frame.
function pedirSegunda(dron) {
  fetch('/mirar', {method: 'POST', headers: {'Content-Type': 'application/json'},
                   body: JSON.stringify({dron: String(dron)})})
    .then(() => refrescar()).catch(() => {});
}

// The only click in this page that makes an aircraft fly. It asks the ground which drone to
// send, because the station knows who is connected and the card only knows who reported.
function pedirRodeo(p) {
  fetch('/rodear', {method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x: p.x, y: p.y, dron: String(p.dron),
                                          movil: !!p.mobile})})
    .then(r => r.json()).then(d => {
      // Same reasoning as the clear button: one that says nothing when pressed reads as broken,
      // and this one is sending an aircraft somewhere, so it has to say which one went.
      aviso = (d && d.dron)
        ? 'drone ' + d.dron + ' is going to look at (' + p.x + ', ' + p.y + ') from another side'
        : 'no drone to send: ' + ((d && d.error) || 'no answer');
      pintar();
    }).catch(() => { aviso = 'no se pudo mandar la orden'; pintar(); });
}

function textoSegunda(d) {
  if (!d) return '';
  if (d.estado === 'pedido') return 'asking the drone for the frame...';
  if (d.estado === 'mirando') return 'RF-DETR looking at the frame...';
  if (d.estado === 'error') return 'no second opinion: ' + (d.error || 'it failed');
  if (d.estado === 'listo') {
    return d.n + (d.n === 1 ? ' person' : ' people')
      + ' on the ground, in ' + (d.espera != null ? d.espera.toFixed(2) : '?') + ' s';
  }
  return '';
}

// A drone whose frame was judged more than this long ago is showing an old answer, and an old
// answer about a moving scene is worse than none: the card says when it was taken.
const SEGUNDA_VIEJA_S = 60;

// -- the health tab ------------------------------------------------------------
// ORDERED BY HOW SOON EACH THING ENDS THE FLIGHT, which is the rule the alarm banner already
// follows: the electrical warning goes before the one about the origin because it is the only
// one that predicts a loss. Cell voltage is minutes of warning, a brown-out is seconds, heat
// degrades the product now, and a rate below the one asked for degrades it quietly.
//
// NO GRAPHS, on purpose. The group's own direction on 2026-10-02 was that it did not want to
// recreate Grafana and would start with its own plots; a line chart of five numbers from two
// boards is a worse way to answer "can this aircraft finish the mission" than five numbers
// with thresholds on them.
//
// VOLTS PER CELL AND NOT PERCENT. Measured on this airframe, the autopilot reported 93 % at
// 3.79 V per cell: a LiPo's discharge curve is flat across most of its range, so the percentage
// is an interpolation along a plateau while the voltage is the measurement. 3.50 V is turn back
// and 3.30 V is land now, which are the usual numbers for the chemistry and not something this
// screen derived.
const CELDA_VOLVER = 3.5, CELDA_ATERRIZAR = 3.3;
const TEMP_OJO = 70, TEMP_MAL = 80;

// THE CELL COUNT IS TOLD IF THE OPERATOR TOLD IT, AND ONLY INFERRED OTHERWISE. Nothing
// upstream sends it -- uav_api forwards a pack voltage and no count -- so --celdas on the
// station is the exact answer and this is the fallback.
//
// The window is 2.80 to 4.25 V a cell, which is the whole life of the chemistry: 4.20 is a full
// charge and 4.25 is as high as one goes. THAT UPPER BOUND IS LOAD-BEARING and a looser one was
// wrong: at 4.35 a 13.0 V pack came out as a 3S at 4.33 V a cell, which reads as a FULL pack in
// green, when the same 13.0 V over 4S is 3.25 and means land now. One number, two readings, and
// the wrong one is the reassuring one.
//
// It still cannot resolve everything, and the residual case is real: 12.0 V is a 3S charged to
// 4.00 a cell and equally a 4S down to 3.00. There is no threshold that separates those, which
// is the reason --celdas exists. The count is printed next to the voltage so an operator who
// knows their airframe sees a wrong guess at once.
function celdasDe(v, dichas) {
  if (dichas) return dichas;
  let mejor = null;
  for (let n = 2; n <= 6; n++) {
    const porCelda = v / n;
    if (porCelda >= 2.8 && porCelda <= 4.25
        && (mejor === null || Math.abs(porCelda - 3.8) < Math.abs(v / mejor - 3.8))) mejor = n;
  }
  return mejor;
}

function celda(texto, clase, bajo) {
  return `<td class="dato ${clase || ''}">${texto}`
    + (bajo ? `<br><span class="nada">${bajo}</span>` : '') + `</td>`;
}

function sinDato(texto) { return `<td class="nada">${texto}</td>`; }

function filaDiag(etiqueta, celdas, porque) {
  return `<tr><th>${etiqueta}</th>${celdas.join('')}</tr>`
    + `<tr><td class="porque" colspan="${celdas.length + 1}">${porque}</td></tr>`;
}

function pintarDiagnostico() {
  const caja = document.getElementById('diag');
  const ids = Object.keys((estado && estado.drones) || {}).sort();
  if (!ids.length) {
    caja.innerHTML = '<div class="vacio">No drone has reported yet.</div>';
    return;
  }
  const ds = ids.map(id => estado.drones[id]);

  const voltios = ds.map(d => {
    if (d.bateria_v == null) return sinDato('no battery source');
    const dichas = estado.celdas || null;
    const n = celdasDe(d.bateria_v, dichas);
    if (!n) return sinDato(`${d.bateria_v} V, not a LiPo`);
    const c = d.bateria_v / n;
    const clase = c < CELDA_ATERRIZAR ? 'mal' : (c < CELDA_VOLVER ? 'ojo' : 'bien');
    const que = c < CELDA_ATERRIZAR ? ' land now' : (c < CELDA_VOLVER ? ' turn back' : '');
    return celda(`${c.toFixed(2)} V/cell${que}`, clase,
                 `${d.bateria_v} V over ${n}S` + (dichas ? '' : ', guessed'));
  });

  const caidas = ds.map(d => {
    if (!d.salud) return sinDato('not a Raspberry');
    const ahora = (d.salud.ahora || []).indexOf('bajo_voltaje') >= 0;
    const antes = (d.salud.alguna_vez || []).indexOf('bajo_voltaje') >= 0;
    if (ahora) return celda('BROWNING OUT NOW', 'mal', 'check its supply');
    if (antes) return celda('has browned out', 'ojo', 'not now, but it did');
    return celda('clean', 'bien');
  });

  const calor = ds.map(d => {
    if (d.temp_c == null) return sinDato('not reported');
    const bits = (d.salud && d.salud.ahora) || [];
    const frena = bits.indexOf('limite_termico') >= 0 || bits.indexOf('acelerador') >= 0;
    const clase = (frena || d.temp_c >= TEMP_MAL) ? 'mal'
                : (d.temp_c >= TEMP_OJO ? 'ojo' : 'bien');
    return celda(`${d.temp_c.toFixed(1)} °C`, clase, frena ? 'slowing itself down' : '');
  });

  const tasa = ds.map(d => {
    if (d.fps_real == null) return sinDato('no rate yet');
    if (d.fps_pedido == null) return celda(`${d.fps_real} FPS`, '', 'asked rate unknown');
    const frac = d.fps_pedido > 0 ? d.fps_real / d.fps_pedido : 1;
    const clase = frac < 0.6 ? 'mal' : (frac < 0.9 ? 'ojo' : 'bien');
    return celda(`${d.fps_real} of ${d.fps_pedido} FPS`, clase,
                 `${(frac * 100).toFixed(0)} % of what it was asked for`);
  });

  const slots = ds.map(d => {
    if (d.slots_perdidos == null) return sinDato('not reported');
    const clase = d.slots_perdidos > 0 ? 'ojo' : 'bien';
    return celda(String(d.slots_perdidos), clase,
                 d.slots_perdidos_total == null ? '' : `${d.slots_perdidos_total} since takeoff`);
  });

  caja.innerHTML = '<table><tr><th></th>'
    + ids.map(id => `<th>drone ${id}</th>`).join('') + '</tr>'
    + filaDiag('Cell voltage', voltios,
               `minutes of warning. ${CELDA_VOLVER.toFixed(2)} V turn back, `
               + `${CELDA_ATERRIZAR.toFixed(2)} V land now. Volts and not percent, `
               + `because the discharge curve is flat.`)
    + filaDiag('Supply dips', caidas,
               'seconds of warning, and STICKY: a dip lasts an instant and the bit for "now" is '
               + 'already off by the time anyone looks. A board that browns out does not warn, '
               + 'it disappears.')
    + filaDiag('Temperature', calor,
               'degrades the product now, before anything is lost. A board at 84 C is taking '
               + 'fewer looks than the one beside it and nothing else on this screen says so.')
    + filaDiag('Rate delivered', tasa,
               'against the rate the board was ASKED for, which is set on the board itself. '
               + 'Fewer looks taken is fewer chances to find anyone.')
    + filaDiag('Slots missed', slots, 'in the last report interval, and since takeoff.')
    + '</table>';
}

// Which panel the aside is showing. The health tab carries a count of what is red on it, so a
// tab nobody has open can still get the operator to open it.
let solapa = 'contactos';

// Asked of the container and not of the document, which is the way every other control on this
// page finds its buttons. It is also what keeps the switcher out of the way of a caller that
// has no document to query: the gates run this script against a stub DOM.
function botonesSolapa() {
  return document.getElementById('solapas').querySelectorAll('button');
}

function pintarSolapas() {
  const malas = (document.getElementById('diag').innerHTML.match(/class="dato mal"/g) || []).length;
  botonesSolapa().forEach(b => {
    b.className = b.dataset.solapa === solapa ? 'on' : '';
    if (b.dataset.solapa === 'diag') {
      b.innerHTML = 'Health' + (malas ? ` <span class="mal">● ${malas}</span>` : '');
    }
  });
  document.getElementById('panel-contactos').hidden = solapa !== 'contactos';
  document.getElementById('panel-diag').hidden = solapa !== 'diag';
}

botonesSolapa().forEach(b => b.addEventListener('click', () => {
  solapa = b.dataset.solapa;
  pintarSolapas();
}));

function pintar() {
  podarLimpiados(estado.pois);
  const recibidos = estado.pois.filter(p => !ocultas.has(claseDe(p)));
  const vigentes = recibidos.filter(p => veredictoDe(p) !== 'no');
  const descartados = recibidos.length - vigentes.length;
  // The layers hide by age and by "limpiar"; a "no es" is a verdict, not staleness, and the
  // history does not bring it back.
  const tapados = vigentes.filter(ocultoPorCapas).length;
  // What CLIP doubts is shown last, never hidden. The station already sends them in that order;
  // sorting here too keeps the page honest with a station that does not. Array sort is stable.
  visibles = vigentes.filter(p => verHistorial || !ocultoPorCapas(p))
    .sort((a, b) => !!a.clip_no_persona - !!b.clip_no_persona);
  pintarFiltro(estado.pois);
  document.getElementById('cuenta').textContent = visibles.length + ' POI'
    + (visibles.length === estado.pois.length ? '' : ` de ${estado.pois.length}`)
    + (descartados ? ` \u00b7 ${descartados} discarded` : '')
    + (tapados ? ` \u00b7 ${tapados} ${verHistorial ? 'in history' : 'hidden'}` : '')
    + (aviso ? ` \u00b7 ${aviso}` : '');
  ajustarVista(visibles);
  pintarLista(visibles);
  pintarDiagnostico();
  pintarSolapas();
  dibujar();
}

async function refrescar() {
  try {
    const r = await fetch('/estado');
    estado = await r.json();
    aviso = '';
    if (estado.tiene_fondo && !fondo) {
      fondo = new Image();
      fondo.onload = dibujar;
      fondo.src = '/fondo';
    }
    const drones = Object.values(estado.drones);
    const ultimo = drones.length ? Math.max(...drones.map(d => d.t)) : 0;
    const edad = estado.ahora - ultimo;
    const vivo = drones.length && edad < 10;
    document.getElementById('luz').className = 'punto' + (vivo ? ' vivo' : '');
    pintarBotonFondo();
    pintarPedidos(estado.pedidos || []);
    ultimoEstado = estado;
    pintarBuscar();

    document.getElementById('enlace').textContent = !drones.length
      ? 'waiting for a drone'
      : (vivo ? `drone live (${edad.toFixed(0)} s ago)`
              : `no signal for ${edad.toFixed(0)} s`);
    const al = document.getElementById('alarma');
    // La corriente va PRIMERO, antes que el origen, porque es lo unico en esta pantalla que
    // predice que una aeronave va a desaparecer. El 3oct una placa alimentada por bateria estuvo
    // cuatro horas avisando de bajo voltaje en su propio kernel; nada aqui lo mostraba, y murio a
    // mitad de una linea. "alguna vez" y no solo "ahora": una caida de tension dura un instante y
    // el bit de ahora ya se apago cuando el operador mira.
    const flacos = Object.entries(estado.drones || {})
      .filter(([, d]) => d.salud && (d.salud.alguna_vez || []).length)
      .map(([id, d]) => {
        const ahora = (d.salud.ahora || []).length ? ' AHORA MISMO' : '';
        return `drone ${id}: ${d.salud.alguna_vez.join(', ')}${ahora}`;
      });
    if (flacos.length) {
      al.textContent = 'POWER: ' + flacos.join(' \u00b7 ')
        + ' \u00b7 a board that browns out does not warn, it disappears. Check its supply.';
      al.style.display = 'block';
    } else if (estado.desacuerdo) {
      // Silence here would be the expensive kind: every pin lands somewhere plausible and
      // wrong, and nothing on the page looks broken.
      al.textContent = `The origin the drone declares is ${estado.desacuerdo} m from the one `
        + `passed in --origen. The drone's wins; check the one on the ground.`;
      al.style.display = 'block';
    } else { al.style.display = 'none'; }
    // The rate the drone is actually managing. A saturated drone looks exactly like a
    // healthy one -- same pins, same reports, fewer looks taken -- unless it is shown.
    const d0 = drones[0];
    const rit = document.getElementById('ritmo');
    if (d0 && d0.fps_real != null) {
      const perdidas = d0.slots_perdidos || 0;
      rit.textContent = `${d0.fps_real} FPS` + (perdidas ? ` · ${perdidas} perdidas` : '');
      rit.style.color = perdidas ? 'var(--duda)' : '';
    } else { rit.textContent = ''; }
    document.getElementById('reportes').textContent = estado.reportes + ' reportes';
    pintar();
  } catch (e) { /* la GS se cayo: la pagina se queda con lo ultimo que vio */ }
}
redimensionar();
refrescar();
// -- what the camera saw ----------------------------------------------------
// Next to the map, not instead of it. A pin on a grid is an assertion; the frame behind it is
// what the assertion was made from, and putting the two side by side is the difference between
// being told the system works and watching it work.
//
// The frame number is in the query string so the browser fetches a new image when it changes
// and reuses the cached one when it does not.
// Polled on its own clock, faster than the rest. The map redraws once a second because a POI
// does not move faster than that; the frame under it changes several times in that same
// second, and hanging it off the slow poll turned a sampled view into a slideshow.
//
// It stays a sampled view either way: the camera records about 8.7 frames a second and the
// chain processes under 2 of them. What this shows is the frame the chain is working on, not
// the recording -- for the recording there is scripts/render_clip.py.
let frameVisto = null;

setInterval(async () => {
  try {
    const r = await fetch('/frame_actual_n');
    pintarCamara((await r.json()).n);
  } catch (e) { /* la estacion se fue; el mapa ya lo dice */ }
}, 250);

function pintarCamara(n) {
  const c = document.getElementById('camara');
  if (n == null) { c.style.display = 'none'; return; }
  c.style.display = 'block';
  if (n !== frameVisto) {
    frameVisto = n;
    c.querySelector('img').src = '/frame?n=' + n;
  }
}

// -- what the station used to propose -------------------------------------
// This panel is gone, and the reason is worth keeping. It listed targets that only one drone had
// seen, with the drone that could go and look, and at first it only proposed because the station
// could not command. When the station learned to send an aircraft, the obvious fix looked like
// adding a button to each line.
//
// With three such targets on screen the user read this:
//
//     Nobody has corroborated this person. Only drone 1 has seen it.
//     Nobody has corroborated this person. Only drone 1 has seen it.
//     Nobody has corroborated this person. Only drone 1 has seen it.
//
// Three different targets described identically, so there is no way to tell which button sends a
// drone where. Adding the target's number would have fixed the sentence and not the problem: every
// one of those targets is already a card in the list, and every card already carries the same
// button next to the picture the operator decides by. The panel was duplication with worse
// information. flota.pedidos_de_verificacion still computes the suggestion and the station still
// serves it; nothing reads it on the page.
function pintarPedidos(pedidos) {
  const c = document.getElementById('pedidos');
  if (c) c.innerHTML = '';
}

// -- the imagery ------------------------------------------------------------
// Off by default. The metric grid is the honest view: it shows where the pins
// are with respect to each other and to the origin, and it is right anywhere.
// The imagery answers a different question -- where on earth this is -- and it
// is only as good as its georeference, so it goes behind a switch the operator
// can turn off when comparing positions.
function pintarBotonFondo() {
  const b = document.getElementById('btn-fondo');
  const hay = !!(estado && estado.tiene_fondo && estado.georef && estado.origen);
  b.hidden = !hay;
  b.className = verFondo ? 'on' : '';
  b.textContent = verFondo ? 'satellite' : 'grid';
  b.title = verFondo ? 'drop the image and go back to the metric grid'
                     : 'put the satellite image behind the map';
}

lienzo.addEventListener('wheel', ev => {
  ev.preventDefault();
  zoom = Math.min(40, Math.max(0.6, zoom * (ev.deltaY > 0 ? 1.25 : 0.8)));
  // pintar() is the one that reframes; dibujar() would redraw the old window.
  if (estado) pintar(); else dibujar();
}, {passive: false});

document.getElementById('btn-limpiar').onclick = limpiar;
// clear hides; this one travels. The pins live in each aircraft's identity layer, so a board
// cleared only here comes back with the next report, which is what a reload used to show.
document.getElementById('btn-reiniciar').onclick = async () => {
  const b = document.getElementById('btn-reiniciar');
  b.disabled = true;
  try {
    const r = await fetch('/reiniciar', {method: 'POST', body: '{}'});
    const d = await r.json();
    aviso = `reset: ${d.borrados} contacts dropped, order sent to ${d.drones} drone(s)`;
    limpiados = [];
    verHistorial = false;
  } catch (e) {
    aviso = 'reset failed: ' + e;
  }
  b.disabled = false;
  pintar();
};
document.getElementById('btn-historial').onclick = alternarHistorial;

document.getElementById('btn-fondo').onclick = () => {
  verFondo = !verFondo;
  pintarBotonFondo();
  dibujar();
};

// -- the control plane -------------------------------------------------------
// Kept apart from the display filter on purpose: these buttons travel to the
// drone. The list is what the detector emits, not what has been seen so far --
// a class you have never received is exactly the one you may want to ask for.
const BUSCABLES = ['person', 'car', 'truck', 'bus', 'boat'];
const ALIAS_PERSONA = ['person', 'pedestrian', 'people'];
let buscando = null;
// Whether the operator asked to see everything the detector knows, beyond the mission's classes.
let verTodas = false;
let orden = {v: null, epoca: null};
let ultimoEstado = null;

// The buttons offered: the classes of THIS mission that the drones say their detector can
// actually emit. A detector trained on a public dataset knows eighty things, most of which have no
// business on a search map -- offering broccoli and teddy bears next to "person" reads as a demo,
// and worse, it buries the four buttons an operator will ever press. The rest stay one click away
// rather than removed, because the detector really can emit them and hiding that would be a lie.
// The names a model uses for people ('pedestrian', 'people') collapse into the one an operator
// types; the drone translates it back.
function buscables(drones) {
  const nombres = new Set(Object.values(drones || {})
    .flatMap(d => (d.buscando && d.buscando.conocidas) || []));
  if (!nombres.size) return {mision: BUSCABLES, resto: []};
  const tiene = c => nombres.has(c) || (c === 'person' && ALIAS_PERSONA.some(a => nombres.has(a)));
  const mision = BUSCABLES.filter(tiene);
  const resto = [...nombres].filter(c => !ALIAS_PERSONA.includes(c) && !BUSCABLES.includes(c)).sort();
  return {mision: mision.length ? mision : BUSCABLES, resto};
}

// One line per drone with what its camera is really doing. The station's record of the request
// is no evidence that a drone out of range, or one whose model lacks the class, complied.
function estadoBusqueda(drones, ord) {
  return Object.entries(drones || {}).map(([id, d]) => {
    const b = d.buscando;
    if (!b) return `drone ${id}: does not say what it is looking for`;
    const pendiente = ord.v != null && ord.v > 0 && (b.v !== ord.v || b.epoca !== ord.epoca);
    if (pendiente) return `drone ${id}: has not taken the order yet`;
    // The reason for a refusal carries the whole list of what the detector knows, which is
    // eighty names and buries the page. Only what matters is shown: that it refused, and what
    // it was asked for.
    if (b.rechazo) return `drone ${id}: refused the order (${String(b.rechazo).split(';')[0]})`;
    const sordo = b.apariencia === false
      ? ' · cannot act on a verdict: no appearance model' : '';
    return `drone ${id}: looking for ${b.clases ? b.clases.join(', ') : 'the usual'}${sordo}`;
  });
}

function pintarBuscar() {
  const sel = new Set(buscando || []);
  const cont = document.getElementById('buscar');
  const drones = (ultimoEstado && ultimoEstado.drones) || {};
  const {mision, resto} = buscables(drones);
  // Anything already being searched for stays visible even if it is not a mission class: the
  // operator must always see what they asked for.
  const extra = resto.filter(c => sel.has(c));
  const visibles = verTodas ? mision.concat(resto) : mision.concat(extra);
  const boton = c => `<button data-b="${c}" class="${sel.has(c) ? 'on' : ''}">${c}</button>`;
  // One line per drone, each with its own button. A drone that reports no points still has a
  // camera and can still be asked what it is looking at: putting the button only inside a point's
  // card made the second opinion unreachable for exactly the drone an operator would wonder about.
  const ids = Object.keys(drones).sort();
  const lineas = estadoBusqueda(drones, orden);
  cont.innerHTML = '<span class="etq" title="this travels to the drone and changes what it '
    + 'detects, live, without reloading the model">searching</span>' + visibles.map(boton).join('')
    + (resto.length ? `<button data-todas="1" class="mas">${verTodas
        ? 'fewer' : '+ ' + resto.length + ' more classes'}</button>` : '')
    + lineas.map((t, i) => `<div class="acuse">${t}`
        + (ids[i] != null
           ? ` <button class="vista" data-mirar-dron="${ids[i]}">second opinion</button>` : '')
        + (ids[i] != null ? bloqueSegunda(ids[i]) : '') + '</div>').join('');
  for (const b of cont.querySelectorAll('button[data-mirar-dron]')) {
    b.onclick = () => pedirSegunda(b.dataset.mirarDron);
  }
  const mas = cont.querySelectorAll('button[data-todas]')[0];
  if (mas) mas.onclick = () => { verTodas = !verTodas; pintarBuscar(); };
  // Solo los botones de CLASE llevan el manejador de busqueda. Atarlo a todos los botones de la
  // barra hacia que "segunda opinion" y "+ N clases mas" mandaran una orden con la clase undefined,
  // y el dron la rechazaba con "the detector does not emit [None]".
  for (const b of cont.querySelectorAll('button[data-b]')) {
    b.onclick = async () => {
      const c = b.dataset.b;
      if (sel.has(c)) sel.delete(c); else sel.add(c);
      const clases = [...sel];
      try {
        const r = await fetch('/buscar', {method: 'POST',
                                          headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify({clases})});
        const d = await r.json();
        orden = {v: d.v, epoca: d.epoca};
        buscando = clases;
      } catch (e) { /* the station is the one that just answered us; ignore */ }
      pintarBuscar();
    };
  }
}

fetch('/buscar').then(r => r.json()).then(d => {
  buscando = d.clases || [];
  orden = {v: d.v, epoca: d.epoca};
  pintarBuscar();
}).catch(() => pintarBuscar());

setInterval(refrescar, 1000);
</script>
</body></html>"""


def demo():
    """Fake POIs so the page can be checked without a drone: one settles, one walks."""
    import random
    t0 = time.time()
    frames = 0
    while True:
        time.sleep(2.0)
        frames += 8
        t = time.time() - t0
        pois = [{'x': round(-1.3 + random.uniform(-0.3, 0.3), 2),
                 'y': round(8.8 + random.uniform(-0.3, 0.3), 2),
                 'n_obs': int(40 + t * 4), 'conf': 0.83, 'cls': 'car',
                 'mobile': False, 'mature': t > 20}]
        if t > 8:
            pois.append({'x': round(6.0 + 0.5 * t % 14 - 7, 2),
                         'y': round(2.0 + math.sin(t / 6) * 4, 2),
                         'n_obs': int(15 + t * 2), 'conf': 0.61, 'cls': 'pedestrian',
                         'mobile': True, 'mature': t > 40})
        registrar({'type': 'vision_poi', 'pois': pois, 'frames_seen': frames}, 'demo')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--puerto', type=int, default=8300)
    ap.add_argument('--frames', default=None,
                    help='carpeta con frame_NNNN.jpg, para ver lo que vio la camara')
    ap.add_argument('--fondo', default=None, help='PNG georeferenciado (opcional)')
    ap.add_argument('--georef', default=None,
                    help='archivo con lat0,lon0,lat1,lon1[,zoom] de las esquinas del PNG')
    ap.add_argument('--celdas', type=int, default=None,
                    help='celdas en serie del pack (4 para un 4S). Sin esto la pagina las '
                         'infiere de la tension, y hay packs que la inferencia no puede '
                         'distinguir: 12,0 V son un 3S cargado o un 4S agotado')
    ap.add_argument('--origen', default=None,
                    help='lat,lon del origen de la mision: convierte los metros a coordenadas')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--callado-s', type=float, default=DRON_CALLADO_S,
                    help='segundos sin reportar tras los que un dron deja de contar en el mapa')
    ap.add_argument('--nodos', default=None,
                    help='drones a los que empujar la orden de busqueda, como en node_ip_dict: '
                         '1=192.168.1.120:8200,3=192.168.1.123:8200')
    ap.add_argument('--veredictos', default='veredictos',
                    help='carpeta donde se guardan los veredictos del operador y sus recortes')
    ap.add_argument('--clip', action='store_true',
                    help='puntua cada crop con CLIP: los "probable no persona" van al final de la '
                         'lista, marcados, sin ocultarse. Necesita open_clip (venv de entrenamiento)')
    ap.add_argument('--clip-descarta', action='store_true',
                    help='ademas de marcarlos, SACA de la lista los "probable no persona". Medido sobre '
                         'el vuelo del 02ago: tira la bolsa y el cono, los fantasmas pasan de 2 a 1 y no '
                         'se pierde ninguna de las 5 personas. Opcional a proposito: esconder algo que el '
                         'operador no vio es una decision suya, no del sistema')
    ap.add_argument('--segunda-opinion', action='store_true',
                    help='arranca RF-DETR en tierra para el boton de segunda opinion')
    ap.add_argument('--python-rfdetr', default=PYTHON_RFDETR,
                    help='el interprete que tiene rfdetr (el venv de entrenamiento)')
    ap.add_argument('--clip-umbral', type=float, default=None,
                    help='umbral del puntaje CLIP (por defecto 1.496, fijado con los vuelos del 01ago)')
    ap.add_argument('--radio-rodeo', type=float, default=RODEO_RADIO_M,
                    help='metros desde el objetivo al mirar desde otro lado (por defecto %.0f)'
                         % RODEO_RADIO_M)
    ap.add_argument('--altura-rodeo', type=float, default=RODEO_ALTURA_M,
                    help='metros de altura al mirar desde otro lado (por defecto %.0f)'
                         % RODEO_ALTURA_M)
    ap.add_argument('--puntos-rodeo', type=int, default=RODEO_PUNTOS,
                    help='paradas de la vuelta: 1 es una foto desde otro angulo, 12 es rodear '
                         '(por defecto %d)' % RODEO_PUNTOS)
    ap.add_argument('--banco', action='store_true',
                    help='las placas estan en un escritorio: la pagina dice que los metros no '
                         'significan nada en vez de imprimir un numero que nadie deberia creer')
    args = ap.parse_args()
    DRON_CALLADO_S = args.callado_s
    RODEO_RADIO_M = float(args.radio_rodeo)
    RODEO_ALTURA_M = float(args.altura_rodeo)
    RODEO_PUNTOS = max(1, int(args.puntos_rodeo))
    EN_BANCO = bool(args.banco)
    CLIP_DESCARTA = bool(args.clip_descarta)
    if args.clip:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import filtro_clip
        print('cargando CLIP ViT-B-32-quickgelu/openai...', flush=True)
        CLIP = filtro_clip.cargar(args.clip_umbral if args.clip_umbral is not None
                                  else filtro_clip.UMBRAL)
        if CLIP is not None:
            print('CLIP listo: umbral %.3f' % CLIP.umbral, flush=True)
    ESTADO['veredictos_dir'] = args.veredictos
    if args.segunda_opinion:
        print('arrancando la segunda opinion (RF-DETR tarda ~17 s en cargar)...', flush=True)
        SEGUNDA = SegundaOpinion(args.python_rfdetr,
                                 os.path.join(args.veredictos, 'segunda_opinion'))
        SEGUNDA.arrancar()
    if args.nodos:
        ESTADO['nodos'] = dict(par.split('=', 1) for par in args.nodos.split(','))

    if args.frames and os.path.isdir(args.frames):
        ESTADO['frames_dir'] = args.frames
    elif args.frames:
        print('AVISO: no existe la carpeta %s; sin vista de camara.' % args.frames)
    if args.fondo and os.path.exists(args.fondo):
        ESTADO['fondo'] = args.fondo
        if args.georef and os.path.exists(args.georef):
            v = [float(x) for x in open(args.georef).read().split(',')]
            ESTADO['georef'] = v[:4]
        else:
            print('AVISO: hay --fondo pero no --georef; la imagen no se puede ubicar y '
                  'solo se dibuja la cuadricula.')
            ESTADO['fondo'] = None
    elif args.fondo:
        print('AVISO: no existe %s; se dibuja solo la cuadricula.' % args.fondo)

    # El operador sabe cuantas celdas tiene su pack y la pagina no puede saberlo siempre: 12,0 V
    # son un 3S cargado a 4,00 V por celda y tambien un 4S agotado a 3,00, y las dos lecturas son
    # fisicamente plausibles. Decirlo aqui saca la fila mas urgente de la pestana de salud del
    # terreno de la adivinanza.
    if args.celdas:
        ESTADO['celdas'] = args.celdas

    if args.origen:
        ESTADO['origen_cli'] = tuple(float(x) for x in args.origen.split(','))
        ESTADO['origen'] = ESTADO['origen_cli']
    elif ESTADO['fondo']:
        print('AVISO: sin --origen no se puede ubicar el fondo ni dar coordenadas.')
        ESTADO['fondo'] = None

    if args.demo:
        threading.Thread(target=demo, daemon=True).start()
        print('modo DEMO: inyectando POIs falsos')

    print('Ground Station en http://localhost:%d  (POST del enjambre en el mismo puerto)'
          % args.puerto)
    try:
        import urllib.request
        urllib.request.urlopen('http://127.0.0.1:%d/estado' % args.puerto,
                               timeout=1).read()
    except Exception:
        pass
    else:
        print('YA HAY UNA ESTACION EN EL PUERTO %d.' % args.puerto)
        print('Arrancar otra encima parte los reportes entre las dos y el mapa')
        print('se queda vacio sin decir por que. Cerra la anterior:')
        print('  netstat -ano | findstr LISTENING | findstr :%d' % args.puerto)
        print('  taskkill /PID <numero> /F')
        raise SystemExit(1)

    server.ThreadingHTTPServer(('0.0.0.0', args.puerto), Handler).serve_forever()
