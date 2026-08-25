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

Everything is served by one process with no external dependencies: no CDN, no npm, nothing
that needs a network in the field. The page polls a JSON endpoint and draws.

    python gs_mapa.py --puerto 8300 \\
        --fondo ../../drone-geolocation/entrenamiento/satelite_zona.png \\
        --georef ../../drone-geolocation/entrenamiento/satelite_georef.txt \\
        --origen -22.978029946,-43.23214256266666

Without --fondo it draws a metric grid, which works anywhere and needs no imagery.
Then open http://localhost:8300 in a browser.

--demo injects moving fake POIs so the page can be seen without flying.
"""
import argparse
import json
import math
import os
import threading
import time
from datetime import datetime
from http import server

# POIs arrive in local metres (x east, y north) from the mission origin. With the origin's
# coordinates the same points become lat/lng -- the conversion that was supposedly blocked on
# agreeing a format with the group. It is not: we own both ends of this link.
R_TIERRA = 6378137.0

ESTADO = {
    'pois': [],            # last list received, annotated
    'historia': [],        # every report, for the trail
    'drones': {},          # id -> last seen
    'origen': None,        # in use: the drone's, if it declares one
    'origen_cli': None,    # what the operator typed, kept to check the drone against
    'desacuerdo': None,    # metres between the two, when they disagree

    'georef': None,
    'fondo': None,
    'arranque': time.time(),
}
CANDADO = threading.Lock()


def a_latlng(x, y, origen):
    """Local metres east/north back to coordinates, given the mission origin."""
    if origen is None:
        return None, None
    lat0, lng0 = origen
    lat = lat0 + math.degrees(y / R_TIERRA)
    lng = lng0 + math.degrees(x / (R_TIERRA * math.cos(math.radians(lat0))))
    return round(lat, 7), round(lng, 7)


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


def registrar(mensaje, fuente):
    ahora = time.time()
    with CANDADO:
        adoptar_origen(mensaje)
    if mensaje.get('latido'):
        # An empty beat says "still here", not "there is nothing". Touching the pin list on
        # one would wipe the map every time a target left the frame for a second.
        with CANDADO:
            ESTADO['drones'][str(fuente)] = {
                't': ahora, 'frames_seen': mensaje.get('frames_seen')}
        return
    pois = []
    for p in mensaje.get('pois', []):
        lat, lng = a_latlng(p.get('x', 0.0), p.get('y', 0.0), ESTADO['origen'])
        # A POI with no 'maduro' field predates the sweep work, or came from the RANSAC
        # fallback that fires before any candidate exists. Treat it as unconfirmed: assuming
        # the safer reading is what keeps a maybe from being shown as a find.
        pois.append({
            'x': p.get('x'), 'y': p.get('y'),
            'lat': lat, 'lng': lng,
            'n_obs': p.get('n_obs'),
            'conf': p.get('conf', p.get('conf_mean')),
            'movil': p.get('movil'),
            'maduro': bool(p.get('maduro', False)),
            'recorte': p.get('recorte'),
            'dron': fuente,
            't': ahora,
        })
    with CANDADO:
        ESTADO['pois'] = pois
        ESTADO['historia'].append({'t': ahora, 'n': len(pois),
                                   'frames': mensaje.get('frames_seen')})
        ESTADO['historia'][:] = ESTADO['historia'][-500:]
        ESTADO['drones'][str(fuente)] = {
            't': ahora, 'frames_seen': mensaje.get('frames_seen')}
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
        self._responder(b'{"status": "ok"}')
        try:
            payload = json.loads(crudo)
            mensaje = json.loads(payload['message'])
        except Exception:
            print('mensaje no-JSON:', crudo[:200], flush=True)
            return
        if mensaje.get('type') == 'vision_poi':
            registrar(mensaje, payload.get('source'))
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
                d = {
                    'pois': ESTADO['pois'],
                    'drones': ESTADO['drones'],
                    'origen': ESTADO['origen'],
                    'origen_cli': ESTADO['origen_cli'],
                    'desacuerdo': ESTADO['desacuerdo'],
                    'georef': ESTADO['georef'],
                    'tiene_fondo': ESTADO['fondo'] is not None,
                    'ahora': time.time(),
                    'reportes': len(ESTADO['historia']),
                }
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
          --tenue:#8b93a5; --ok:#4ade80; --duda:#fbbf24; --movil:#60a5fa; }
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
  .chip.movil { background:rgba(96,165,250,.15); color:var(--movil); }
  .poi dl { margin:8px 0 0; display:grid; grid-template-columns:auto 1fr;
            gap:2px 10px; font-size:13px; }
  .poi dt { color:var(--tenue); }
  .poi dd { margin:0; font-variant-numeric:tabular-nums; }
  .recorte { display:block; margin:10px 0 0; width:128px; max-width:100%;
             border-radius:4px; border:1px solid var(--linea); background:#0b0d12; }
  .sinrecorte { margin:8px 0 0; font-size:12px; color:var(--tenue); font-style:italic; }
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
    <span id="reportes">0 reportes</span>
  </div>
</header>
<main>
  <canvas id="lienzo"></canvas>
  <aside>
    <h2>Detecciones</h2>
    <div id="lista"><div class="vacio">Nada todavia.</div></div>
    <div class="nota">
      <b style="color:var(--ok)">CONFIRMADO</b>: la capa de identidad lo siguio lo
      suficiente.<br>
      <b style="color:var(--duda)">POR VERIFICAR</b>: se formo una pista pero no alcanzo a
      madurar. Es lo que produce una pasada corta. No es un hallazgo: es un pedido de
      verificacion.
    </div>
  </aside>
</main>
<script>
const lienzo = document.getElementById('lienzo');
const ctx = lienzo.getContext('2d');
let fondo = null, estado = null;

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
  if (!pois.length) return;
  let e0=1e9,e1=-1e9,n0=1e9,n1=-1e9;
  for (const p of pois) { e0=Math.min(e0,p.x); e1=Math.max(e1,p.x);
                          n0=Math.min(n0,p.y); n1=Math.max(n1,p.y); }
  const m = Math.max(12, (e1-e0), (n1-n0)) * 0.7 + 8;
  const ce=(e0+e1)/2, cn=(n0+n1)/2;
  vista = {e0:ce-m, e1:ce+m, n0:cn-m, n1:cn+m};
}

// The satellite image is georeferenced by its corners, so a POI in metres becomes a fraction
// of the image once it is coordinates. Without an image a metric grid stands in: the pins are
// still in the right place relative to each other and to the origin.
function dibujarFondo() {
  const r = lienzo.getBoundingClientRect();
  ctx.fillStyle = '#0b0d12'; ctx.fillRect(0, 0, r.width, r.height);
  if (fondo && estado && estado.georef && estado.origen) {
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
    } catch (e) { /* recorte fuera de la imagen: queda el fondo liso */ }
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
  for (const p of estado.pois) {
    const [x, y] = aPantalla(p.x, p.y);
    const col = p.maduro ? '#4ade80' : '#fbbf24';
    // A halo sized by nothing but legibility: this is not an uncertainty ellipse and must not
    // be read as one. The real uncertainty is a few metres and would swallow the pin.
    ctx.beginPath(); ctx.arc(x, y, 22, 0, 6.2832);
    ctx.fillStyle = col + '22'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.2832);
    ctx.fillStyle = col; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#12141a'; ctx.stroke();
    if (p.movil) {
      ctx.beginPath(); ctx.arc(x, y, 15, 0, 6.2832);
      ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2; ctx.stroke();
    }
    ctx.fillStyle = '#e6e9ef'; ctx.font = '600 12px system-ui';
    ctx.fillText(p.maduro ? 'CONFIRMADO' : 'POR VERIFICAR', x + 16, y - 12);
  }
}

function pintarLista(pois) {
  const cont = document.getElementById('lista');
  if (!pois.length) { cont.innerHTML = '<div class="vacio">Nada todavia.</div>'; return; }
  cont.innerHTML = pois.map((p, i) => `
    <div class="poi ${p.maduro ? 'ok' : 'duda'}">
      <div class="tit">#${i+1}
        <span class="chip ${p.maduro ? 'ok' : 'duda'}">${p.maduro ? 'CONFIRMADO' : 'POR VERIFICAR'}</span>
        ${p.movil ? '<span class="chip movil">MOVIL</span>' : ''}
      </div>
      <dl>
        <dt>local</dt><dd>${p.x} m E, ${p.y} m N</dd>
        ${p.lat != null ? `<dt>coords</dt><dd>${p.lat}, ${p.lng}</dd>` : ''}
        <dt>evidencia</dt><dd>${p.n_obs} obs${p.conf != null ? ', conf ' + p.conf : ''}</dd>
        <dt>dron</dt><dd>${p.dron}</dd>
      </dl>
      ${p.recorte
        ? `<img class="recorte" src="data:image/jpeg;base64,${p.recorte}" alt="lo que vio el dron">`
        : (p.maduro ? '' : '<div class="sinrecorte">sin recorte: no se puede verificar</div>')}
    </div>`).join('');
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
    document.getElementById('cuenta').textContent = estado.pois.length + ' POI';
    document.getElementById('reportes').textContent = estado.reportes + ' reportes';
    ajustarVista(estado.pois);
    pintarLista(estado.pois);
    dibujar();
  } catch (e) { /* la GS se cayo: la pagina se queda con lo ultimo que vio */ }
}
redimensionar();
refrescar();
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
                 'n_obs': int(40 + t * 4), 'conf': 0.83,
                 'movil': False, 'maduro': t > 20}]
        if t > 8:
            pois.append({'x': round(6.0 + 0.5 * t % 14 - 7, 2),
                         'y': round(2.0 + math.sin(t / 6) * 4, 2),
                         'n_obs': int(15 + t * 2), 'conf': 0.61,
                         'movil': True, 'maduro': t > 40})
        registrar({'type': 'vision_poi', 'pois': pois, 'frames_seen': frames}, 'demo')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--puerto', type=int, default=8300)
    ap.add_argument('--fondo', default=None, help='PNG georeferenciado (opcional)')
    ap.add_argument('--georef', default=None,
                    help='archivo con lat0,lon0,lat1,lon1[,zoom] de las esquinas del PNG')
    ap.add_argument('--origen', default=None,
                    help='lat,lon del origen de la mision: convierte los metros a coordenadas')
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()

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
    server.ThreadingHTTPServer(('0.0.0.0', args.puerto), Handler).serve_forever()
