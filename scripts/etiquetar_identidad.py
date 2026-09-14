"""Hand-labelling of identities on the 02-ago flight: the ground truth for IDF1 and ID switches.

An identity metric needs to know who each detection really is, and that is the one thing the
system under test must not decide. Labels proposed by its tracker or by its identity layer would
score the system against itself, and the number would prove nothing. So the boxes are grouped here
by a rule neither of the trackers being measured uses -- a box continues a segment only when it
overlaps that segment's last box (IoU >= 0.3) in the very next processed frame, less than a second
later -- and every segment is shown to a person, who names it.

The rule is deliberately timid. It splits far more often than a tracker does, which costs a few
extra keystrokes, and it almost never glues two people together, which would cost a wrong label.
When it does, any single crop can be relabelled on its own.

Nothing leaves the machine: the page is served on localhost, crops are cut from the flight frames
on request, and the labels are written next to the flight data, outside the repository, which does
not publish flight imagery.

    python scripts/etiquetar_identidad.py        # then open http://127.0.0.1:8411
    http://127.0.0.1:8411/?desde=3000&hasta=3700  # only the segments starting in that frame window

Most segments are a single box, because the chain processed a frame about every half second and a
box rarely overlaps its predecessor across that gap. Loosening the overlap rule barely helps --
measured, IoU 0.05 instead of 0.3 leaves 1228 segments instead of 1564 while segments that span two
BoT-SORT ids go from 7 to 18 -- so the speed comes from the page instead: Enter on an empty field
repeats the last label, and the keys 1-9 apply the identities in the gallery.
"""
import argparse
import csv
import json
import os
import threading
from functools import lru_cache
from http import server
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

LAC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATOS = os.path.join(LAC, "uav_vision", "demo", "data")
FRAMES = os.path.join(LAC, "drone-geolocation", "data", "flight_02ago", "20260802_133309", "frames")
SALIDA = os.path.join(LAC, "drone-geolocation", "entrenamiento", "identidad_gt_02ago.json")
CONF_MIN = 0.25
IOU_MIN = 0.3
SALTO_MAX_S = 1.0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def segmentos(dets, t_de):
    """
    Groups detections into short segments by overlap between consecutive processed frames.

    Matching is greedy by IoU, highest first, and one-to-one, so two boxes in the same frame can
    never continue the same segment. A frame farther than SALTO_MAX_S from the previous one breaks
    every segment: the flight landed for nine minutes, and overlap across that gap is coincidence.
    """
    por_frame = {}
    for i, d in enumerate(dets):
        por_frame.setdefault(int(d[0]), []).append(i)
    frames = sorted(por_frame)
    seg_de = -np.ones(len(dets), dtype=int)
    grupos = []
    previo = None
    for f in frames:
        actuales = por_frame[f]
        libres = set(actuales)
        if previo is not None and t_de(f) - t_de(previo) <= SALTO_MAX_S:
            pares = sorted(((iou(dets[i][2:6], dets[j][2:6]), i, j)
                            for i in actuales for j in por_frame[previo]), reverse=True)
            usados = set()
            for v, i, j in pares:
                if v < IOU_MIN:
                    break
                if i in libres and j not in usados:
                    seg_de[i] = seg_de[j]
                    grupos[seg_de[j]].append(i)
                    libres.discard(i)
                    usados.add(j)
        for i in sorted(libres):
            seg_de[i] = len(grupos)
            grupos.append([i])
        previo = f
    return grupos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puerto", type=int, default=8411)
    args = ap.parse_args()

    dets_all = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))["dets"]
    dets = dets_all[dets_all[:, 1] >= CONF_MIN]
    poses = {int(r["frame"]): float(r["t_mono"]) for r in
             csv.DictReader(open(os.path.join(DATOS, "frames.csv")))}
    grupos = segmentos(dets, lambda f: poses.get(f, 0.0))
    catalogo = [{"id": k, "t": poses.get(int(dets[g[0]][0]), 0.0),
                 "dets": [{"i": int(i), "frame": int(dets[i][0]), "conf": round(float(dets[i][1]), 2)}
                          for i in g]} for k, g in enumerate(grupos)]
    print("%d detecciones en %d segmentos (mediana %d cajas, maximo %d)" % (
        len(dets), len(grupos), int(np.median([len(g) for g in grupos])),
        max(len(g) for g in grupos)), flush=True)

    candado = threading.Lock()
    etiquetas = {}
    if os.path.exists(SALIDA):
        etiquetas = json.load(open(SALIDA, encoding="utf-8")).get("etiquetas", {})
        print("retomando %d cajas ya etiquetadas de %s" % (len(etiquetas), SALIDA), flush=True)

    @lru_cache(maxsize=48)
    def imagen(frame):
        return cv2.imread(os.path.join(FRAMES, "frame_%04d.jpg" % frame))

    def jpeg(img):
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    class Manejador(server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def responder(self, cuerpo, tipo="application/json", codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self.responder(PAGINA.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/segmentos":
                desde = int(q.get("desde", ["0"])[0])
                hasta = int(q.get("hasta", ["999999"])[0])
                ventana = [c for c in catalogo if desde <= c["dets"][0]["frame"] <= hasta]
                self.responder(json.dumps(ventana).encode("utf-8"))
            elif u.path == "/etiquetas":
                with candado:
                    self.responder(json.dumps(etiquetas).encode("utf-8"))
            elif u.path in ("/recorte", "/frame"):
                i = int(q["i"][0])
                img = imagen(int(dets[i][0]))
                if img is None:
                    self.responder(b"sin frame", "text/plain", 404)
                    return
                x1, y1, x2, y2 = [float(v) for v in dets[i][2:6]]
                if u.path == "/frame":
                    vista = img.copy()
                    cv2.rectangle(vista, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                    self.responder(jpeg(vista), "image/jpeg")
                    return
                mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
                h, w = img.shape[:2]
                c = img[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))]
                escala = 110.0 / max(1, c.shape[0])
                c = cv2.resize(c, (max(1, int(c.shape[1] * escala)), 110))
                self.responder(jpeg(c), "image/jpeg")
            else:
                self.responder(b"no", "text/plain", 404)

        def do_POST(self):
            cambios = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with candado:
                for k, v in cambios.items():
                    if v in (None, ""):
                        etiquetas.pop(str(k), None)
                    else:
                        etiquetas[str(k)] = str(v)
                tmp = SALIDA + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"vuelo": "02ago", "conf_min": CONF_MIN, "n_detecciones": len(dets),
                               "etiquetas": etiquetas}, fh, indent=0)
                os.replace(tmp, SALIDA)
                n = len(etiquetas)
            self.responder(json.dumps({"guardadas": n}).encode("utf-8"))

    print("abri http://127.0.0.1:%d   (Ctrl+C para salir; se guarda en cada cambio)" % args.puerto,
          flush=True)
    server.ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador).serve_forever()


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Etiquetar identidades</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#15171a;color:#e8e6e1}
header{position:sticky;top:0;background:#1f2226;padding:10px 16px;border-bottom:1px solid #333;z-index:2}
h1{font-size:15px;margin:0 0 6px}
.ayuda{color:#a9a69e;font-size:12px}
#galeria{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.id{display:flex;align-items:center;gap:4px;background:#2a2e33;border:1px solid #444;border-radius:6px;padding:2px 6px;cursor:pointer}
.id img{height:44px}
.seg{margin:10px 16px;padding:8px;border:1px solid #333;border-radius:8px;background:#1b1d20}
.seg.actual{border-color:#e0b84a;box-shadow:0 0 0 1px #e0b84a}
.seg.hecho{opacity:.55}
.cab{display:flex;gap:12px;align-items:center;margin-bottom:6px;font-size:12px;color:#a9a69e}
.cab input{width:70px;font:14px monospace;background:#0f1113;color:#fff;border:1px solid #555;border-radius:4px;padding:2px 6px}
.tira{display:flex;flex-wrap:wrap;gap:3px}
.caja{position:relative}
.caja img{height:110px;display:block;border:2px solid transparent}
.caja .lbl{position:absolute;left:2px;top:2px;background:#000c;color:#e0b84a;font:12px monospace;padding:0 3px}
.caja.distinta img{border-color:#d9534f}
#visor{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:5}
#visor img{max-width:96vw;max-height:92vh}
</style></head><body>
<header>
  <h1>Etiquetar identidades, vuelo 02ago <span id="progreso"></span></h1>
  <div class="ayuda">Escribi una etiqueta (A, B, C... una por persona real) y Enter: se aplica al segmento
  entero y pasa al siguiente sin etiquetar. <b>Enter con el campo vacio</b> repite la ultima etiqueta.
  <b>1-9</b> con el campo vacio aplica la ficha de la galeria con ese numero.
  <b>x</b> = no es una persona. <b>?</b> = no se distingue.
  Clic en una caja: le pone otra etiqueta solo a esa caja. Clic derecho: ver el frame entero.
  Las fichas de arriba son tus etiquetas con su primera caja, para mantenerlas coherentes.</div>
  <div id="galeria"></div>
</header>
<div id="lista"></div>
<div id="visor" onclick="this.style.display='none'"><img alt=""></div>
<script>
let segs = [], etq = {}, actual = 0, ultima = '', galeria = [];

async function cargar() {
  segs = await (await fetch('/segmentos' + location.search)).json();
  etq = await (await fetch('/etiquetas')).json();
  pintar();
  irA(siguienteLibre(0));
}

async function guardar(cambios) {
  Object.entries(cambios).forEach(([k, v]) => { if (v) etq[k] = v; else delete etq[k]; });
  await fetch('/etiquetas', {method: 'POST', body: JSON.stringify(cambios)});
  pintarProgreso(); pintarGaleria();
}

function etiquetaDe(s) {
  const vals = s.dets.map(d => etq[d.i]).filter(Boolean);
  return vals.length ? vals[0] : '';
}
function completo(s) { return s.dets.every(d => etq[d.i]); }
function siguienteLibre(desde) {
  for (let k = desde; k < segs.length; k++) if (!completo(segs[k])) return k;
  return Math.min(desde, segs.length - 1);
}

function pintarProgreso() {
  const total = segs.reduce((n, s) => n + s.dets.length, 0);
  const hechas = segs.reduce((n, s) => n + s.dets.filter(d => etq[d.i]).length, 0);
  document.getElementById('progreso').textContent =
    `-- ${hechas}/${total} cajas, ${segs.filter(completo).length}/${segs.length} segmentos`;
}

function pintarGaleria() {
  const primera = {};
  segs.forEach(s => s.dets.forEach(d => { const e = etq[d.i]; if (e && !(e in primera)) primera[e] = d.i; }));
  galeria = Object.keys(primera).filter(e => e !== 'x' && e !== '?').sort();
  document.getElementById('galeria').innerHTML = galeria.map((e, n) =>
    `<span class="id" onclick="aplicar('${e}')"><img src="/recorte?i=${primera[e]}" loading="lazy">`
    + `${n < 9 ? '<b>' + (n + 1) + '</b> ' : ''}${e}</span>`).join('');
}

function pintar() {
  document.getElementById('lista').innerHTML = segs.map((s, k) => `
    <div class="seg" id="s${k}">
      <div class="cab">segmento ${k} &middot; ${s.dets.length} cajas &middot; frames ${s.dets[0].frame}-${s.dets[s.dets.length-1].frame}
        <input id="in${k}" value="${etiquetaDe(s)}" onfocus="irA(${k}, true)"
               onkeydown="tecla(event, this)"></div>
      <div class="tira">${s.dets.map(d => `
        <div class="caja" id="c${d.i}" onclick="unaCaja(${d.i})"
             oncontextmenu="event.preventDefault();verFrame(${d.i})">
          <img src="/recorte?i=${d.i}" loading="lazy" title="det ${d.i}, conf ${d.conf}">
          <span class="lbl">${etq[d.i] || ''}</span></div>`).join('')}</div>
    </div>`).join('');
  pintarProgreso(); pintarGaleria(); marcar();
}

function marcar() {
  segs.forEach((s, k) => {
    const el = document.getElementById('s' + k);
    el.className = 'seg' + (k === actual ? ' actual' : '') + (completo(s) ? ' hecho' : '');
    const base = etiquetaDe(s);
    s.dets.forEach(d => {
      const c = document.getElementById('c' + d.i);
      c.querySelector('.lbl').textContent = etq[d.i] || '';
      c.className = 'caja' + (etq[d.i] && etq[d.i] !== base ? ' distinta' : '');
    });
  });
}

function tecla(ev, campo) {
  if (ev.key === 'Enter') { ev.preventDefault(); aplicar(campo.value); return; }
  if (!campo.value && /^[1-9]$/.test(ev.key) && galeria[+ev.key - 1]) {
    ev.preventDefault();
    aplicar(galeria[+ev.key - 1]);
  }
}

function irA(k, sinFoco) {
  actual = k;
  const campo = document.getElementById('in' + k);
  if (campo) campo.placeholder = ultima;
  marcar();
  const el = document.getElementById('s' + k);
  if (!sinFoco) { el.scrollIntoView({block: 'center'}); document.getElementById('in' + k).focus(); }
}

async function aplicar(valor) {
  const v = (valor || '').trim() || ultima;
  if (!v) return;
  ultima = v;
  const s = segs[actual], cambios = {};
  s.dets.forEach(d => { cambios[d.i] = v; });
  await guardar(cambios);
  document.getElementById('in' + actual).value = v;
  marcar();
  if (v) irA(siguienteLibre(actual + 1));
}

async function unaCaja(i) {
  const v = prompt('Etiqueta solo para esta caja (vacio = quitar):', etq[i] || '');
  if (v === null) return;
  await guardar({[i]: v.trim()});
  marcar();
}

function verFrame(i) {
  const v = document.getElementById('visor');
  v.querySelector('img').src = '/frame?i=' + i;
  v.style.display = 'flex';
}

cargar();
</script></body></html>
"""

if __name__ == "__main__":
    main()
