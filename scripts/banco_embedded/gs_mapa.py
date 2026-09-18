# -*- coding: utf-8 -*-
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

    python gs_mapa.py --puerto 8300 \\
        --fondo ../../drone-geolocation/entrenamiento/satelite_zona.png \\
        --georef ../../drone-geolocation/entrenamiento/satelite_georef.txt \\
        --origen -22.978029946,-43.23214256266666

Without --fondo it draws a metric grid, which works anywhere and needs no imagery.
Then open http://localhost:8300 in a browser.

--demo injects moving fake POIs so the page can be seen without flying.

--clip adds a CLIP score to each person candidate's crop (filtro_clip.py). Doubtful ones go to the
end of the list with a "probable no persona" mark; nothing is hidden. torch and open_clip live
only in the training venv, which also runs everything else this file imports:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/banco_embedded/gs_mapa.py --clip

Without open_clip, --clip prints a warning and the station runs without scores.
"""
import argparse
import uuid
import base64
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from http import server

# Cross-drone fusion needs the same numbers the drone uses to fuse its own tracks. Imported
# rather than copied: two drones disagreeing with the station about what counts as the same
# object is a bug nobody would think to look for. Without the package -- this file is meant to
# be droppable anywhere -- the station still runs and simply shows both drones' targets.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from uav_vision.flota import fundir, pedidos_de_verificacion
    FUSION_DISPONIBLE = True
except Exception:
    FUSION_DISPONIBLE = False

    def fundir(por_dron):
        return [q for lista in por_dron.values() for q in lista]

    def pedidos_de_verificacion(pois, drones, ahora, vivo_s=10.0):
        return []

# POIs arrive in local metres (x east, y north) from the mission origin. With the origin's
# coordinates the same points become lat/lng -- the conversion that was supposedly blocked on
# agreeing a format with the group. It is not: we own both ends of this link.
R_TIERRA = 6378137.0

ESTADO = {
    'pois': [],            # last list received, annotated
    'historia': [],        # every report, for the trail
    'drones': {},          # id -> last seen
    'veredictos_dir': 'veredictos',   # where the operator's verdicts and their crops are kept
    'origen': None,        # in use: the drone's, if it declares one
    'origen_cli': None,    # what the operator typed, kept to check the drone against
    'desacuerdo': None,    # metres between the two, when they disagree

    'georef': None,
    'fondo': None,
    'arranque': time.time(),

    # What the operator asks the drone to look for. This is NOT the display
    # filter below it: hiding a class only stops drawing it, while this changes
    # what the detector reports at all. None means "whatever the drone booted
    # with". The counter lets the drone notice a change without diffing lists.
    'buscar': None,
    'buscar_v': 0,
    # Which run of this station issued the order. A restarted station counts from zero again,
    # and without this a drone that took version 7 would ignore every new order below it.
    'buscar_epoca': uuid.uuid4().hex[:8],
    # Data-plane addresses of the drones, from --nodos, to push each order to. Empty means
    # the drones learn it by polling /buscar.
    'nodos': {},

    # One list per drone. A single shared list was replaced on every report, so a second
    # drone erased the first one's targets and the map flickered between the two views.
    # What the operator sees is the concatenation, kept in 'pois'.
    'pois_por_dron': {},

    # Which frame the bench replay is on, and where the frames live. Only the bench sends
    # this: a real drone ships coordinates, not pictures, and the link could not carry them.
    # It exists so a demo can put what the camera saw next to what the map made of it.
    'rastros': {},
    'frame_actual': None,
    'frames_dir': None,

    # The target the operator fixed with "es lo que busco", in metres, or None.
    'objetivo': None,

    # The ground's second opinion, one entry per drone: what the operator asked for and what came
    # back. Never the picture itself, which is served from disk by /segunda.jpg: a 1920x1080 frame
    # is 300 KB, and carrying it inside a state poll that runs every second would be four megabits
    # of the same image for as long as the operator looks at it.
    'segunda': {},
}
CANDADO = threading.Lock()

# Where the second opinion runs. It is a separate process on purpose: RF-DETR lives in the training
# venv and this station has to stay droppable on a laptop with nothing installed.
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
    """
    if not nodos:
        return
    import threading as _t
    import urllib.request
    cuerpo = json.dumps({'message': json.dumps(mensaje),
                         # The embedded runtime types the source as an int, and no mission numbers a drone 0.
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

# The optional CLIP second opinion on each crop (filtro_clip.Anotador), set by --clip. None means
# the station never started one, and then no POI carries a score field at all.
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
    """
    return {
        't': ahora,
        # Where it is, and where it has been. The trail is what shows an operator whether the
        # drone is working the area or hovering, which is the difference between rays that
        # cross and rays that do not.
        'pos': mensaje.get('pos'),
        'frames_seen': mensaje.get('frames_seen'),
        'fps_real': mensaje.get('fps_real'),
        'slots_perdidos': mensaje.get('slots_perdidos'),
        'slots_perdidos_total': mensaje.get('slots_perdidos_total'),
        # What the camera is really searching for, and whether it took the last order. The
        # station's own record of the request says nothing about a drone out of range.
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
    """
    origen = mensaje.get('origen_gps')
    if not origen or len(origen) < 2:
        return
    nuevo = (float(origen[0]), float(origen[1]))
    if ESTADO['origen_cli'] is not None:
        d = separacion_m(ESTADO['origen_cli'], nuevo)
        # A metre is well under the system's own error and far above float noise.
        ESTADO['desacuerdo'] = round(d, 1) if d > 1.0 else None
    if ESTADO['origen'] != nuevo:
        ESTADO['origen'] = nuevo
        print('origen tomado del dron: %.7f, %.7f%s' % (
            nuevo[0], nuevo[1],
            '' if not ESTADO['desacuerdo']
            else '  <-- NO COINCIDE con --origen, %s m' % ESTADO['desacuerdo']), flush=True)


def _rastro(fuente, pos):
    """Keeps the last stretch of a drone's path. Called with the lock already held."""
    if not pos:
        return
    r = ESTADO['rastros'].setdefault(str(fuente), [])
    if not r or (abs(r[-1][0] - pos[0]) + abs(r[-1][1] - pos[1])) > 0.5:
        r.append([pos[0], pos[1]])
        # Bounded on purpose: a whole flight drawn at once is a scribble, and the question an
        # operator has is where it went lately, not where it took off.
        del r[:-200]


# A drone that has not reported for this long is not seeing anything now. Its targets stop counting
# towards the map: a pin labelled "seen by 1+2" must not outlive one of the two going silent. It is
# the same window the drones apply to what they hear from each other. --callado-s changes it.
CLIP_DESCARTA = False          # --clip-descarta: sacar de la lista lo que CLIP llama no-persona
DRON_CALLADO_S = 15.0


def pois_vigentes(ahora):
    """
    The fused targets of the drones that are still talking, with their coordinates.

    A silent drone's targets are kept, not deleted, so they return the moment it reports again.
    They are only left out while it is silent, so a live drone's find is shown as its own instead
    of as corroborated by a drone that may have fallen out of the sky. Measured on the
    two-Raspberry bench before this existed: fifteen seconds after one runner was killed, the
    station still showed the target as seen by 1+2.
    """
    vivos = {k: v for k, v in ESTADO['pois_por_dron'].items()
             if ahora - ESTADO['drones'].get(k, {}).get('t', float('-inf')) <= DRON_CALLADO_S}
    pois = fundir(vivos)
    for q in pois:
        q['lat'], q['lng'] = a_latlng(q['x'], q['y'], ESTADO['origen'])
    # What CLIP doubts goes to the end of the queue, and nothing else moves: the sort is stable
    # and a POI without a score is never demoted. With --clip-descarta it is dropped instead of
    # demoted, which on the 02ago flight removed the bag and the cone and left every person standing:
    # phantoms 2 -> 1 with 5 of 5 people kept. Off by default because hiding a point the operator never
    # saw is their call and not the system's, and because CLIP still misses the harder clutter -- the
    # red object on that flight scored 1.81, above the threshold, and looks like a person by size too.
    if CLIP_DESCARTA:
        pois = [q for q in pois if not q.get('clip_no_persona')]
    pois.sort(key=lambda q: bool(q.get('clip_no_persona')))
    return pois


def registrar(mensaje, fuente):
    ahora = time.time()
    with CANDADO:
        adoptar_origen(mensaje)
    if mensaje.get('latido'):
        # An empty beat says "still here", not "there is nothing". Touching the pin list on
        # one would wipe the map every time a target left the frame for a second.
        with CANDADO:
            ESTADO['drones'][str(fuente)] = ficha(ahora, mensaje)
        _rastro(fuente, mensaje.get('pos'))
        return
    pois = []
    for p in mensaje.get('pois', []):
        lat, lng = a_latlng(p.get('x', 0.0), p.get('y', 0.0), ESTADO['origen'])
        # A POI with no 'mature' field predates the sweep work, or came from the RANSAC
        # fallback that fires before any candidate exists. Treat it as unconfirmed: assuming
        # the safer reading is what keeps a maybe from being shown as a find.
        pois.append({
            'x': p.get('x'), 'y': p.get('y'),
            'lat': lat, 'lng': lng,
            'n_obs': p.get('n_obs'),
            # None when the drone never named it: older firmware, or a camera with no
            # class. Shown as 'sin clase' rather than guessed at.
            'cls': p.get('cls'),
            'conf': p.get('conf', p.get('conf_mean')),
            'mobile': p.get('mobile'),
            'mature': bool(p.get('mature', False)),
            'crop': p.get('crop'),
            # How much independent evidence, and how far off the point may be, when the drone
            # matures by looks. Absent from older reports, and then simply not drawn.
            'looks': p.get('looks'),
            'radius_m': p.get('radius_m'),
            # Seconds between the drone's last sighting of the target and this report. The page
            # adds the time elapsed since the report ('t') and fades live contacts by it.
            'age_s': p.get('age_s'),
            # Kept so the station can decide whether two drones are looking at one target.
            # Never drawn: it is evidence, not something an operator reads.
            'emb': p.get('emb'),
            'dron': fuente,
            't': ahora,
        })
    # Scored outside the station's lock: the drone already has its answer, and a model call must
    # not stall the page's reads. Each distinct crop goes through the model once.
    if CLIP is not None:
        for q in pois:
            CLIP.anotar(q)
    with CANDADO:
        ESTADO['pois_por_dron'][str(fuente)] = pois
        # Concatenated, not merged: two drones seeing the same person still produce two pins
        # until something decides they are the same target. Showing both is the honest state
        # of affairs; hiding one would be a claim nobody has made yet.
        ESTADO['historia'].append({'t': ahora, 'n': len(pois),
                                   'frames': mensaje.get('frames_seen')})
        ESTADO['historia'][:] = ESTADO['historia'][-500:]
        # Heard before fusing, or the drone's own report would find it silent.
        ESTADO['drones'][str(fuente)] = ficha(ahora, mensaje)
        # The fusion works in metres and knows nothing about the globe; pois_vigentes puts the
        # coordinates back on each merged position, or the pin and its GPS reading would part.
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

    # -- the fleet data plane: unchanged from oyente_gs.py -----------------
    def do_POST(self):
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
            # The operator's side of the control plane. Answering before the
            # drone has polled is deliberate: the order is stored, not routed,
            # so the station never blocks on a link it does not control.
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
            # "Es lo que busco" is not only a verdict kept on disk: it is the one thing the operator
            # knows that the drone cannot work out, and it buys recall for free. Sending it back
            # lets the camera lower its threshold over that square of the image, where the detector
            # had already scored the boxes it was discarding.
            try:
                d = json.loads(crudo)
                apagar = bool(d.get('off'))
                x, y = (None, None) if apagar else (float(d['x']), float(d['y']))
            except Exception:
                self._responder(b'{"error": "objetivo"}', codigo=400)
                return
            with CANDADO:
                nodos = dict(ESTADO['nodos'])
                ESTADO['objetivo'] = None if apagar else {'x': x, 'y': y, 't': time.time()}
            print('[%s] objetivo %s' % (datetime.now().strftime('%H:%M:%S'),
                                        'liberado' if apagar else 'fijado en (%.1f, %.1f)' % (x, y)),
                  flush=True)
            empujar_mensaje(nodos, {'type': 'vision_objetivo', 'x': x, 'y': y})
            self._responder(json.dumps({'objetivo': None if apagar else {'x': x, 'y': y}}).encode('utf-8'))
            return
        if self.path.split('?')[0] == '/mirar':
            # The operator asks the ground to look again at what one drone is seeing right now.
            # Nothing is computed here: the request goes out, the frame comes back on the data
            # plane like any other packet, and the answer appears when it appears. The click must
            # not block on a radio link or on a detector that takes a second and a half.
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
        if self.path.split('?')[0] == '/veredicto':
            # The operator's verdict on a point, kept on disk with the crop it was given on. The
            # drone's own signals cannot tell a person from an object the detector keeps
            # confusing with one; the operator can, and every "no es" is exactly the hard
            # negative a detector trained on public aerial data is missing. Keeping them turns
            # using the system into labelling it.
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

    # -- the operator's side ----------------------------------------------
    def do_GET(self):
        ruta = self.path.split('?')[0]
        if ruta == '/':
            self._responder(PAGINA.encode('utf-8'), 'text/html; charset=utf-8')
        elif ruta == '/estado':
            with CANDADO:
                # Recomputed on every read too: when every drone goes silent no report arrives
                # to refresh the map, and a silent map must not keep claiming corroboration.
                ESTADO['pois'] = pois_vigentes(time.time())
                d = {
                    'objetivo': ESTADO['objetivo'],
                    'segunda': ESTADO['segunda'],
                    'pois': ESTADO['pois'],
                    'drones': ESTADO['drones'],
                    'rastros': ESTADO['rastros'],
                    'origen': ESTADO['origen'],
                    'origen_cli': ESTADO['origen_cli'],
                    'desacuerdo': ESTADO['desacuerdo'],
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
            # Served from disk instead of travelling inside /estado: the annotated frame is the
            # size of a photograph and the state is polled once a second.
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
  .veredicto button { font:600 11px system-ui; padding:3px 9px; margin:8px 6px 0 0;
                      border-radius:99px; border:1px solid var(--linea); background:transparent;
                      color:inherit; cursor:pointer; }
  .crop { display:block; margin:10px 0 0; width:128px; max-width:100%;
             border-radius:4px; border:1px solid var(--linea); background:#0b0d12; }
  .sinrecorte { margin:8px 0 0; font-size:12px; color:var(--tenue); font-style:italic; }
  .segunda { margin:8px 0 0; font-size:12px; color:var(--tenue); }
  .segunda.hallazgo { color:var(--ok); font-weight:600; }
  .segunda.fallo { color:#f87171; }
  .mirada { display:block; margin:6px 0 0; width:100%; border-radius:6px; }
  .vacio { color:var(--tenue); font-style:italic; padding:20px 0; text-align:center; }
  .nota { color:var(--tenue); font-size:12px; margin-top:14px;
          padding-top:12px; border-top:1px solid var(--linea); }
  #alarma { background:#7f1d1d; color:#fee2e2; padding:9px 16px; font-size:13px;
            font-weight:600; border-bottom:1px solid #991b1b; }
</style></head>
<body>
<div id="alarma" style="display:none"></div>
<header>
  <h1>Ground Station</h1>
  <div class="estado">
    <span><span class="punto" id="luz"></span><span id="enlace">esperando al dron</span></span>
    <span id="cuenta">0 POI</span>
    <span id="ritmo"></span>
    <span id="reportes">0 reportes</span>
    <button id="btn-limpiar"
            title="ocultar los contactos no confirmados hasta que se vuelvan a ver">limpiar</button>
    <button id="btn-historial" title="mostrar, atenuado, lo que las capas ocultaron">historial</button>
    <button id="btn-fondo" hidden>satelite</button>
  </div>
</header>
<main>
  <canvas id="lienzo"></canvas>
  <aside>
    <h2>Detecciones</h2>
    <div id="camara"><div class="cab">lo que ve la camara</div><img alt=""></div>
    <div id="pedidos"></div>
    <div id="buscar"></div>
    <div id="filtro"></div>
    <div id="lista"><div class="vacio">Nada todavia.</div></div>
    <div class="nota">
      <b style="color:var(--ok)">CONFIRMADO</b>: la capa de identity lo siguio lo
      suficiente.<br>
      <b style="color:var(--duda)">POR VERIFICAR</b>: se formo una pista pero no alcanzo a
      madurar. Es lo que produce una pasada corta. No es un hallazgo: es un pedido de
      verificacion.<br>
      La rueda del raton aleja y acerca el mapa.<br>
      <b>BUSCANDO</b> cambia lo que el dron detecta, en caliente y sin recargar el modelo.
      Los botones de debajo solo <b>ocultan</b>: el POI sigue llegando y vuelve con un clic.
      Uno manda sobre el sensor; el otro, sobre el dibujo.<br>
      Lo no confirmado se apaga mientras pasa tiempo sin verse y desaparece al rato;
      CONFIRMADO y VERIFICADO quedan en su ultima posicion y dicen hace cuanto se vieron.
      <b>limpiar</b> oculta lo no confirmado hasta un avistamiento nuevo; <b>historial</b>
      muestra lo oculto.
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

function claseDe(p) { return p.cls || 'sin clase'; }

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
  if (persistente(p)) return false;
  const e = edadDe(p);
  if (e != null && e >= TOPE_VIVO_S) return true;
  return limpiados.some(v => cercaDe(p, v));
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
  return (e != null && e > TOPE_VIVO_S) ? `visto hace ${Math.round(e)} s` : '';
}

// Hides every live contact on screen until it is seen again. A POI without an age is left alone:
// nothing would ever say it had been seen again, so it would be hidden for good.
function limpiar() {
  if (!estado) return;
  for (const p of estado.pois) {
    const e = edadDe(p);
    if (e == null || ocultas.has(claseDe(p)) || persistente(p) || ocultoPorCapas(p)) continue;
    limpiados.push({ x: p.x, y: p.y, r: p.radius_m || 0, edad: e });
  }
  pintar();
}

function alternarHistorial() {
  verHistorial = !verHistorial;
  const b = document.getElementById('btn-historial');
  b.className = verHistorial ? 'on' : '';
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
    ctx.fillText('dron ' + id + (d.pos[2] != null ? '  ' + d.pos[2].toFixed(0) + ' m' : ''),
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
  if (!pois.length) { cont.innerHTML = '<div class="vacio">Nada todavia.</div>'; return; }
  cont.innerHTML = pois.map((p, i) => `
    <div class="poi ${p.mature ? 'ok' : 'duda'}"
         style="opacity:${Math.max(alfaDe(p), ALFA_TARJETA_MIN).toFixed(2)}">
      <div class="tit">#${i+1}
        ${vistoHace(p) ? `<span class="chip viejo">${vistoHace(p)}</span>` : ''}
        <span class="chip ${p.mature ? 'ok' : 'duda'}">${p.mature ? 'CONFIRMADO' : 'POR VERIFICAR'}</span>
        ${p.mobile ? '<span class="chip mobile">MOVIL</span>' : ''}
        ${p.cls ? `<span class="chip clase">${p.cls}</span>` : ''}
        ${veredictoDe(p) === 'si' ? '<span class="chip ok">VERIFICADO</span>' : ''}
        ${p.clip_no_persona
          ? `<span class="chip noper">probable no persona (CLIP ${p.clip.toFixed(2)})</span>` : ''}
      </div>
      <dl>
        <dt>local</dt><dd>${p.x} m E, ${p.y} m N</dd>
        ${p.lat != null ? `<dt>coords</dt><dd>${p.lat}, ${p.lng}</dd>` : ''}
        <dt>evidencia</dt><dd>${p.n_obs} obs${p.conf != null ? ', conf ' + p.conf : ''}</dd>
        <dt>dron</dt><dd>${p.dron}</dd>
        ${p.radius_m != null
          ? `<dt>margen</dt><dd>&plusmn;${p.radius_m} m (95 %), ${p.looks} miradas</dd>` : ''}
      </dl>
      ${p.crop
        ? `<img class="crop" src="data:image/jpeg;base64,${p.crop}" alt="lo que vio el dron">`
        : (p.mature ? '' : '<div class="sinrecorte">sin crop: no se puede verificar</div>')}
      ${segundaDe(p)}
      <div class="veredicto">${veredictoDe(p) === 'si' ? '' :
        `<button data-v="si" data-i="${i}">es lo que busco</button>`
        + `<button data-v="no" data-i="${i}">no es</button>`}
        <button data-mirar="${i}">segunda opinion</button></div>
    </div>`).join('');
  for (const b of cont.querySelectorAll('button[data-v]')) {
    b.onclick = () => marcar(pois[+b.dataset.i], b.dataset.v);
  }
  for (const b of cont.querySelectorAll('button[data-mirar]')) {
    b.onclick = () => pedirSegunda(pois[+b.dataset.mirar].dron);
  }
}

// The block a card shows about its drone's second opinion. Keyed by DRONE, not by point: the
// aircraft sends the frame it is looking at, which answers "is there anybody here", not "is this
// particular point real". Saying otherwise would promise a link between the boxes and the POI that
// nothing in the exchange establishes.
function segundaDe(p) {
  const d = estado.segunda && estado.segunda[String(p.dron)];
  if (!d) return '';
  const clase = d.estado === 'error' ? 'fallo' : (d.estado === 'listo' && d.n ? 'hallazgo' : '');
  const viejo = d.estado === 'listo' && estado.ahora && (estado.ahora - d.t) > SEGUNDA_VIEJA_S
    ? ` (hace ${Math.round(estado.ahora - d.t)} s)` : '';
  return `<div class="segunda ${clase}">${textoSegunda(d)}${viejo}</div>`
    + (d.estado === 'listo' && d.dibujo
       ? `<img class="mirada" src="/segunda.jpg?dron=${encodeURIComponent(p.dron)}&t=${d.t}"
               alt="lo que RF-DETR encontro en ese cuadro">` : '');
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
  cont.innerHTML = clasesVistas.map(c =>
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

function textoSegunda(d) {
  if (!d) return '';
  if (d.estado === 'pedido') return 'pidiendole el cuadro al dron...';
  if (d.estado === 'mirando') return 'RF-DETR mirando el cuadro...';
  if (d.estado === 'error') return 'sin segunda opinion: ' + (d.error || 'fallo');
  if (d.estado === 'listo') {
    return d.n + (d.n === 1 ? ' persona' : ' personas')
      + ' en tierra, en ' + (d.espera != null ? d.espera.toFixed(2) : '?') + ' s';
  }
  return '';
}

// A drone whose frame was judged more than this long ago is showing an old answer, and an old
// answer about a moving scene is worse than none: the card says when it was taken.
const SEGUNDA_VIEJA_S = 60;

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
    + (descartados ? ` \u00b7 ${descartados} descartados` : '')
    + (tapados ? ` \u00b7 ${tapados} ${verHistorial ? 'en historial' : 'ocultos'}` : '');
  ajustarVista(visibles);
  pintarLista(visibles);
  dibujar();
}

async function refrescar() {
  try {
    const r = await fetch('/estado');
    estado = await r.json();
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
      ? 'esperando al dron'
      : (vivo ? `dron activo (hace ${edad.toFixed(0)} s)`
              : `sin señal hace ${edad.toFixed(0)} s`);
    const al = document.getElementById('alarma');
    if (estado.desacuerdo) {
      // Silence here would be the expensive kind: every pin lands somewhere plausible and
      // wrong, and nothing on the page looks broken.
      al.textContent = `El origen que declara el dron esta a ${estado.desacuerdo} m del que se `
        + `paso en --origen. Manda el del dron; revisa el de tierra.`;
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

// -- what the station proposes ----------------------------------------------
// A suggestion with the point already computed, never an order. The link carries data; the
// stick stays with the pilot. Writing it as a command here would be writing a behaviour that
// cannot legally fly.
function pintarPedidos(pedidos) {
  const c = document.getElementById('pedidos');
  if (!pedidos.length) { c.innerHTML = ''; return; }
  c.innerHTML = '<div class="cab">verificacion sugerida</div>' + pedidos.map(p => `
    <div class="item">
      <b>${p.cls || 'sin clase'}</b> sin confirmar, visto solo por el dron ${p.visto_por}.
      Podria ir: ${p.puede_ir.join(', ')}.
      <div class="coord">${p.x} m E, ${p.y} m N${
        p.lat != null ? ` &middot; ${p.lat.toFixed(6)}, ${p.lng.toFixed(6)}` : ''}</div>
    </div>`).join('');
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
  b.textContent = verFondo ? 'satelite' : 'cuadricula';
  b.title = verFondo ? 'quitar la imagen y volver a la cuadricula metrica'
                     : 'poner la imagen de satelite de fondo';
}

lienzo.addEventListener('wheel', ev => {
  ev.preventDefault();
  zoom = Math.min(40, Math.max(0.6, zoom * (ev.deltaY > 0 ? 1.25 : 0.8)));
  // pintar() is the one that reframes; dibujar() would redraw the old window.
  if (estado) pintar(); else dibujar();
}, {passive: false});

document.getElementById('btn-limpiar').onclick = limpiar;
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
let orden = {v: null, epoca: null};
let ultimoEstado = null;

// The buttons offered: the classes the drones say their detector can emit, once they say it.
// The fixed list is only the fallback before any drone has reported. The names a model uses for
// people ('pedestrian', 'people') collapse into the one an operator types; the drone translates
// it back.
function buscables(drones) {
  const nombres = new Set(Object.values(drones || {})
    .flatMap(d => (d.buscando && d.buscando.conocidas) || []));
  if (!nombres.size) return BUSCABLES;
  const lista = [...nombres].filter(c => !ALIAS_PERSONA.includes(c)).sort();
  if (ALIAS_PERSONA.some(c => nombres.has(c))) lista.unshift('person');
  return lista;
}

// One line per drone with what its camera is really doing. The station's record of the request
// is no evidence that a drone out of range, or one whose model lacks the class, complied.
function estadoBusqueda(drones, ord) {
  return Object.entries(drones || {}).map(([id, d]) => {
    const b = d.buscando;
    if (!b) return `dron ${id}: no informa que busca`;
    const pendiente = ord.v != null && ord.v > 0 && (b.v !== ord.v || b.epoca !== ord.epoca);
    if (pendiente) return `dron ${id}: todavia no tomo la orden`;
    if (b.rechazo) return `dron ${id}: rechazo la orden (${b.rechazo})`;
    return `dron ${id}: busca ${b.clases ? b.clases.join(', ') : 'lo de siempre'}`;
  });
}

function pintarBuscar() {
  const sel = new Set(buscando || []);
  const cont = document.getElementById('buscar');
  const drones = (ultimoEstado && ultimoEstado.drones) || {};
  cont.innerHTML = '<span class="etq">buscando</span>' + buscables(drones).map(c =>
    `<button data-b="${c}" class="${sel.has(c) ? 'on' : ''}">${c}</button>`).join('')
    + estadoBusqueda(drones, orden).map(t => `<div class="acuse">${t}</div>`).join('');
  for (const b of cont.querySelectorAll('button')) {
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
    args = ap.parse_args()
    DRON_CALLADO_S = args.callado_s
    CLIP_DESCARTA = bool(args.clip_descarta)
    if args.clip:
        # Imported only when asked for: the station stays droppable anywhere without torch.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import filtro_clip
        print('cargando CLIP ViT-B-32-quickgelu/openai...', flush=True)
        CLIP = filtro_clip.cargar(args.clip_umbral if args.clip_umbral is not None
                                  else filtro_clip.UMBRAL)
        if CLIP is not None:
            print('CLIP listo: umbral %.3f' % CLIP.umbral, flush=True)
    ESTADO['veredictos_dir'] = args.veredictos
    if args.segunda_opinion:
        # Started before the first report, never on the first click: loading RF-DETR takes 17 s and
        # the criterion for this feature is that the operator waits less than five.
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

    # Kept apart from the origin actually in use: --origen is what the operator believes,
    # and the whole point is to be able to tell the two apart once a drone declares its own.
    # Until one speaks, the typed value is all there is, so it seeds the one in use.
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
    # Windows lets a second station bind a port that already has one, and then
    # the reports go to whichever socket accepts first. The symptom is a map that
    # stays empty while the flight clearly runs, and it has cost two sessions.
    # Refuse instead of guessing.
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
