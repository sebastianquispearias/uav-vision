"""Hand-labelling of identities on the 02-ago flight: the ground truth for IDF1 and ID switches.

An identity metric needs to know who each detection really is, and that is the one thing the
system under test must not decide. Labels proposed by its tracker or by its identity layer would
score the system against itself, and the number would prove nothing. So the boxes are grouped here
by a rule neither of the trackers being measured uses -- a box continues a segment only when it
overlaps that segment's last box (IoU >= 0.3) in the very next processed frame, less than a second
later -- and every segment is shown to a person, who names it.

The rule is deliberately timid. It splits far more often than a tracker does, which costs a few
extra keystrokes, and it almost never glues two people together, which would cost a wrong label.
Loosening it barely helps: measured, IoU 0.05 instead of 0.3 leaves 1228 segments instead of 1564
while segments that span two BoT-SORT ids go from 7 to 18. So the speed comes from the page.

The page is built so that looking never changes a label. Clicking a label in the gallery shows the
segments that carry it; labels are applied only by typing, or by the keys 1-9. Every change is
appended to a history file next to the labels, and the last ones can be undone. A review panel
lists what deserves a second look -- a label that is not a single letter, one letter on two boxes
of the same frame that are not the detector's duplicate, a jump on the ground no person makes,
segments with mixed letters, boxes left unlabelled -- so errors are found by the tool, not by
scrolling.

Nothing leaves the machine: the page is served on localhost, crops are cut from the flight frames
on request, and the labels are written next to the flight data, outside the repository, which does
not publish flight imagery.

    python scripts/etiquetar_identidad.py        # then open http://127.0.0.1:8411
    http://127.0.0.1:8411/?desde=3000&hasta=3700  # only the segments starting in that frame window
"""
import argparse
import csv
import json
import os
import re
import threading
import time
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
# Two boxes of one letter in the same frame overlapping this much are the detector boxing one
# person twice, which is correct to label alike. Below it, one of the two letters is wrong.
DUPLICADO_IOU = 0.5
# Ground distance a person does not cover between two processed frames less than a second apart.
# Not proof of an error -- a person on a terrace projects badly onto the ground -- a reason to look.
SALTO_SOSPECHOSO_M = 5.0
NO_PERSONA = {"X", "x", "?"}
UNA_LETRA = re.compile(r"^[A-Z]$")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def impacto(fila):
    """Ground point of a detection, from the ray stored with it (origin, then direction)."""
    o, w = fila[6:9], fila[9:12]
    s = o[2] / -w[2]
    return np.array([o[0] + s * w[0], o[1] + s * w[1]])


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


def problemas(dets, t_de, grupos, etiquetas):
    """
    What in the given segments deserves a second look, each item pointing at one box.

    None of these decides a label. A jump can be a person on a terrace; two letters in one segment
    can be a correct cut. They are listed so that finding them costs a click instead of a search.
    """
    cajas = [i for g in grupos for i in g]
    out = []

    def item(tipo, caja, texto):
        out.append({"tipo": tipo, "caja": int(caja), "frame": int(dets[caja][0]), "texto": texto})

    for i in cajas:
        lab = etiquetas.get(str(i))
        if lab and lab not in NO_PERSONA and not UNA_LETRA.match(lab):
            item("etiqueta rara", i, "'%s' no es una letra sola, X ni ?" % lab)

    por_frame = {}
    for i in cajas:
        lab = etiquetas.get(str(i))
        if lab and lab not in NO_PERSONA:
            por_frame.setdefault((int(dets[i][0]), lab), []).append(i)
    for (f, lab), ix in sorted(por_frame.items()):
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                v = iou(dets[ix[a]][2:6], dets[ix[b]][2:6])
                if v < DUPLICADO_IOU:
                    item("misma letra dos veces", ix[b],
                         "%s en dos cajas del frame %d que no se solapan (IoU %.2f)" % (lab, f, v))

    por_letra = {}
    for i in cajas:
        lab = etiquetas.get(str(i))
        if lab and lab not in NO_PERSONA:
            por_letra.setdefault(lab, []).append(i)
    for lab, ix in sorted(por_letra.items()):
        ix.sort(key=lambda j: dets[j][0])
        for a, b in zip(ix, ix[1:]):
            dt = t_de(int(dets[b][0])) - t_de(int(dets[a][0]))
            dm = float(np.linalg.norm(impacto(dets[b]) - impacto(dets[a])))
            if 0 < dt < SALTO_MAX_S and dm > SALTO_SOSPECHOSO_M:
                item("salto en el suelo", b, "%s salta %.1f m en %.2f s" % (lab, dm, dt))

    for g in grupos:
        letras = {etiquetas.get(str(i)) for i in g if etiquetas.get(str(i))}
        if len(letras) > 1:
            item("letras mezcladas", g[0], "un segmento con %s" % ", ".join(sorted(letras)))

    for g in grupos:
        sin = [i for i in g if str(i) not in etiquetas]
        if sin:
            item("sin etiquetar", sin[0], "%d de %d cajas del segmento sin letra" % (len(sin), len(g)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puerto", type=int, default=8411)
    ap.add_argument("--salida", default=SALIDA,
                    help="archivo de etiquetas; el historial se escribe a su lado (.historial.jsonl)")
    args = ap.parse_args()
    salida = args.salida
    historial = os.path.splitext(salida)[0] + ".historial.jsonl"

    dets_all = np.load(os.path.join(DATOS, "examen_v3_datos.npz"))["dets"]
    dets = dets_all[dets_all[:, 1] >= CONF_MIN]
    poses = {int(r["frame"]): float(r["t_mono"]) for r in
             csv.DictReader(open(os.path.join(DATOS, "frames.csv")))}
    t_de = lambda f: poses.get(f, 0.0)
    grupos = segmentos(dets, t_de)
    catalogo = [{"id": k, "t": t_de(int(dets[g[0]][0])),
                 "dets": [{"i": int(i), "frame": int(dets[i][0]), "conf": round(float(dets[i][1]), 2)}
                          for i in g]} for k, g in enumerate(grupos)]
    print("%d detecciones en %d segmentos (mediana %d cajas, maximo %d)" % (
        len(dets), len(grupos), int(np.median([len(g) for g in grupos])),
        max(len(g) for g in grupos)), flush=True)

    candado = threading.Lock()
    etiquetas = {}
    if os.path.exists(salida):
        etiquetas = json.load(open(salida, encoding="utf-8")).get("etiquetas", {})
        print("retomando %d cajas ya etiquetadas de %s" % (len(etiquetas), salida), flush=True)

    def ventana(q):
        desde = int(q.get("desde", ["0"])[0])
        hasta = int(q.get("hasta", ["999999"])[0])
        return [k for k, g in enumerate(grupos) if desde <= int(dets[g[0]][0]) <= hasta]

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

        def json(self, obj):
            self.responder(json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self.responder(PAGINA.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/segmentos":
                self.json([catalogo[k] for k in ventana(q)])
            elif u.path == "/etiquetas":
                with candado:
                    self.json(etiquetas)
            elif u.path == "/problemas":
                with candado:
                    self.json(problemas(dets, t_de, [grupos[k] for k in ventana(q)], dict(etiquetas)))
            elif u.path == "/historial":
                n = int(q.get("n", ["30"])[0])
                lineas = open(historial, encoding="utf-8").read().splitlines() if os.path.exists(historial) else []
                self.json([json.loads(x) for x in lineas[-n:]])
            elif u.path in ("/recorte", "/frame"):
                i = int(q["i"][0])
                f = int(dets[i][0])
                img = imagen(f)
                if img is None:
                    self.responder(b"sin frame", "text/plain", 404)
                    return
                if u.path == "/frame":
                    vista = img.copy()
                    with candado:
                        rotulos = dict(etiquetas)
                    for j in np.where(dets[:, 0].astype(int) == f)[0]:
                        x1, y1, x2, y2 = [int(v) for v in dets[j][2:6]]
                        color = (0, 255, 255) if j == i else (255, 255, 255)
                        cv2.rectangle(vista, (x1, y1), (x2, y2), color, 5 if j == i else 2)
                        cv2.putText(vista, "%s #%d" % (rotulos.get(str(j), "-"), j), (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                    self.responder(jpeg(vista), "image/jpeg")
                    return
                x1, y1, x2, y2 = [float(v) for v in dets[i][2:6]]
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
            ahora = time.strftime("%Y-%m-%d %H:%M:%S")
            with candado:
                registro = []
                for k, v in cambios.items():
                    antes = etiquetas.get(str(k))
                    despues = None if v in (None, "") else str(v)
                    if antes == despues:
                        continue
                    if despues is None:
                        etiquetas.pop(str(k), None)
                    else:
                        etiquetas[str(k)] = despues
                    registro.append({"t": ahora, "caja": int(k), "antes": antes, "despues": despues})
                tmp = salida + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"vuelo": "02ago", "conf_min": CONF_MIN, "n_detecciones": len(dets),
                               "etiquetas": etiquetas}, fh, indent=0)
                os.replace(tmp, salida)
                if registro:
                    with open(historial, "a", encoding="utf-8") as fh:
                        for r in registro:
                            fh.write(json.dumps(r) + "\n")
                n = len(etiquetas)
            self.json({"guardadas": n, "cambios": len(registro)})

    print("abri http://127.0.0.1:%d   (Ctrl+C para salir; se guarda en cada cambio)" % args.puerto,
          flush=True)
    server.ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador).serve_forever()


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Etiquetar identidades</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#15171a;color:#e8e6e1}
header{position:sticky;top:0;background:#1f2226;padding:10px 16px;border-bottom:1px solid #333;z-index:3}
h1{font-size:15px;margin:0 0 6px}
.ayuda{color:#a9a69e;font-size:12px;line-height:1.5}
.barra{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
button{font:13px system-ui;background:#2a2e33;color:#e8e6e1;border:1px solid #555;border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:#e0b84a;color:#111;border-color:#e0b84a}
#ir{width:120px;font:13px monospace;background:#0f1113;color:#fff;border:1px solid #555;border-radius:4px;padding:4px 6px}
#galeria{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.id{display:flex;align-items:center;gap:5px;background:#2a2e33;border:1px solid #444;border-radius:6px;padding:2px 8px 2px 2px;cursor:pointer}
.id.on{border-color:#e0b84a;box-shadow:0 0 0 1px #e0b84a}
.id img{height:44px}
.id .n{color:#a9a69e;font-size:11px}
.seg{margin:10px 16px;padding:8px;border:1px solid #333;border-radius:8px;background:#1b1d20}
.seg.actual{border-color:#e0b84a;box-shadow:0 0 0 1px #e0b84a}
.seg.hecho{opacity:.55}
.seg.oculto{display:none}
.cab{display:flex;gap:12px;align-items:center;margin-bottom:6px;font-size:12px;color:#a9a69e}
.cab input{width:70px;font:14px monospace;background:#0f1113;color:#fff;border:1px solid #555;border-radius:4px;padding:2px 6px}
.tira{display:flex;flex-wrap:wrap;gap:3px}
.caja{position:relative}
.caja img{height:110px;display:block;border:2px solid transparent}
.caja.foco img{border-color:#e0b84a}
.caja .lbl{position:absolute;left:2px;top:2px;background:#000c;color:#e0b84a;font:12px monospace;padding:0 3px}
.caja.distinta img{border-color:#d9534f}
#visor{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9}
#visor img{max-width:96vw;max-height:92vh}
#panel{position:fixed;right:0;top:0;bottom:0;width:360px;background:#1f2226;border-left:1px solid #444;z-index:5;overflow:auto;padding:12px;display:none}
#panel .p{padding:6px;border-bottom:1px solid #333;cursor:pointer;font-size:12px}
#panel .p:hover{background:#2a2e33}
#panel .tipo{color:#e0b84a}
#aviso{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);background:#2a2e33;border:1px solid #e0b84a;border-radius:6px;padding:6px 14px;display:none;z-index:8}
</style></head><body>
<header>
  <h1>Etiquetar identidades, vuelo 02ago <span id="progreso"></span></h1>
  <div class="ayuda">
    <b>Etiquetar:</b> escribi una letra (una por persona real) y <b>Enter</b>. <b>Enter vacio</b> repite la ultima.
    <b>1-9</b> con el campo vacio aplican la ficha con ese numero. <b>X</b> = no es persona, <b>?</b> = no se distingue.
    <b>Moverse sin tocar nada:</b> flechas arriba/abajo con el campo vacio.
    <b>Ver:</b> clic en una ficha muestra solo sus segmentos (no cambia nada). Clic en una caja: letra solo para esa caja.
    Clic derecho: frame entero con todas las letras. <b>Ctrl+Z</b> deshace.
  </div>
  <div class="barra">
    <button id="f-todos" class="on" onclick="filtrar('todos')">Todos</button>
    <button id="f-sin" onclick="filtrar('sin')">Sin etiquetar</button>
    <input id="ir" placeholder="#caja o fframe" onkeydown="if(event.key==='Enter'){irAlDato(this.value)}">
    <button id="b-revisar" onclick="panel()">Revisar</button>
    <button onclick="deshacer()">Deshacer</button>
  </div>
  <div id="galeria"></div>
</header>
<div id="lista"></div>
<div id="panel"><b>Para revisar</b> <button onclick="cargarProblemas()">actualizar</button>
  <button onclick="panel()">cerrar</button><div id="problemas"></div></div>
<div id="visor" onclick="this.style.display='none'"><img alt=""></div>
<div id="aviso"></div>
<script>
let segs = [], etq = {}, actual = 0, ultima = '', galeria = [], pila = [];
let filtro = {modo: 'todos', letra: null};
const NO_PERSONA = ['X', 'x', '?'];

async function cargar() {
  segs = await (await fetch('/segmentos' + location.search)).json();
  etq = await (await fetch('/etiquetas')).json();
  pintar();
  irA(siguienteVisible(0, true));
  cargarProblemas();
}

function normalizar(v) {
  v = (v || '').trim();
  if (v.length === 1 && v !== '?') v = v.toUpperCase();
  return v;
}

async function guardar(cambios, deshacible) {
  const previo = {};
  Object.keys(cambios).forEach(k => { previo[k] = etq[k] || ''; });
  Object.entries(cambios).forEach(([k, v]) => { if (v) etq[k] = v; else delete etq[k]; });
  const r = await (await fetch('/etiquetas', {method: 'POST', body: JSON.stringify(cambios)})).json();
  if (deshacible && r.cambios) pila.push(previo);
  pintarProgreso(); pintarGaleria(); marcar();
  cargarProblemas();
  return r.cambios;
}

function avisar(texto) {
  const a = document.getElementById('aviso');
  a.textContent = texto; a.style.display = 'block';
  clearTimeout(avisar.t); avisar.t = setTimeout(() => { a.style.display = 'none'; }, 2500);
}

function etiquetaDe(s) {
  const vals = s.dets.map(d => etq[d.i]).filter(Boolean);
  return vals.length ? vals[0] : '';
}
function completo(s) { return s.dets.every(d => etq[d.i]); }
function visible(s) {
  if (filtro.modo === 'sin') return !completo(s);
  if (filtro.modo === 'letra') return s.dets.some(d => etq[d.i] === filtro.letra);
  return true;
}
function siguienteVisible(desde, libre) {
  for (let k = desde; k < segs.length; k++) if (visible(segs[k]) && (!libre || !completo(segs[k]))) return k;
  for (let k = desde; k < segs.length; k++) if (visible(segs[k])) return k;
  return Math.max(0, Math.min(desde, segs.length - 1));
}
function anteriorVisible(desde) {
  for (let k = desde; k >= 0; k--) if (visible(segs[k])) return k;
  return actual;
}

function pintarProgreso() {
  const total = segs.reduce((n, s) => n + s.dets.length, 0);
  const hechas = segs.reduce((n, s) => n + s.dets.filter(d => etq[d.i]).length, 0);
  document.getElementById('progreso').textContent =
    `-- ${hechas}/${total} cajas, ${segs.filter(completo).length}/${segs.length} segmentos`;
}

function pintarGaleria() {
  const primera = {}, cuenta = {};
  segs.forEach(s => s.dets.forEach(d => {
    const e = etq[d.i];
    if (!e) return;
    cuenta[e] = (cuenta[e] || 0) + 1;
    if (!(e in primera)) primera[e] = d.i;
  }));
  galeria = Object.keys(primera).filter(e => !NO_PERSONA.includes(e)).sort();
  const otras = Object.keys(primera).filter(e => NO_PERSONA.includes(e)).sort();
  document.getElementById('galeria').innerHTML = galeria.concat(otras).map(e => {
    const n = galeria.indexOf(e);
    const on = filtro.modo === 'letra' && filtro.letra === e ? ' on' : '';
    return `<span class="id${on}" title="clic: ver solo sus segmentos" onclick="filtrarLetra('${e}')">`
      + `<img src="/recorte?i=${primera[e]}" loading="lazy">`
      + `${n >= 0 && n < 9 ? '<b>' + (n + 1) + '</b> ' : ''}${e} <span class="n">${cuenta[e]}</span></span>`;
  }).join('');
}

function pintar() {
  document.getElementById('lista').innerHTML = segs.map((s, k) => `
    <div class="seg" id="s${k}">
      <div class="cab">segmento ${k} &middot; ${s.dets.length} cajas &middot; frames ${s.dets[0].frame}-${s.dets[s.dets.length-1].frame}
        <input id="in${k}" value="${etiquetaDe(s)}" onfocus="irA(${k}, true)" onkeydown="tecla(event, this)"></div>
      <div class="tira">${s.dets.map(d => `
        <div class="caja" id="c${d.i}" onclick="unaCaja(${d.i})"
             oncontextmenu="event.preventDefault();verFrame(${d.i})">
          <img src="/recorte?i=${d.i}" loading="lazy" title="caja #${d.i}, frame ${d.frame}, conf ${d.conf}">
          <span class="lbl">${etq[d.i] || ''}</span></div>`).join('')}</div>
    </div>`).join('');
  pintarProgreso(); pintarGaleria(); marcar();
}

function marcar() {
  segs.forEach((s, k) => {
    const el = document.getElementById('s' + k);
    el.className = 'seg' + (k === actual ? ' actual' : '') + (completo(s) ? ' hecho' : '')
      + (visible(s) ? '' : ' oculto');
    const base = etiquetaDe(s);
    const campo = document.getElementById('in' + k);
    if (campo && document.activeElement !== campo) campo.value = base;
    s.dets.forEach(d => {
      const c = document.getElementById('c' + d.i);
      c.querySelector('.lbl').textContent = etq[d.i] || '';
      c.className = 'caja' + (etq[d.i] && etq[d.i] !== base ? ' distinta' : '');
    });
  });
  document.getElementById('f-todos').className = filtro.modo === 'todos' ? 'on' : '';
  document.getElementById('f-sin').className = filtro.modo === 'sin' ? 'on' : '';
}

function filtrar(modo) { filtro = {modo, letra: null}; pintarGaleria(); marcar(); irA(siguienteVisible(0)); }
function filtrarLetra(e) {
  if (filtro.modo === 'letra' && filtro.letra === e) return filtrar('todos');
  filtro = {modo: 'letra', letra: e};
  pintarGaleria(); marcar();
  irA(siguienteVisible(0));
  avisar(`mostrando solo los segmentos con ${e} (clic otra vez para verlos todos)`);
}

function tecla(ev, campo) {
  if (ev.key === 'Enter') { ev.preventDefault(); aplicar(campo.value); return; }
  if (!campo.value && ev.key === 'ArrowDown') { ev.preventDefault(); irA(siguienteVisible(actual + 1)); return; }
  if (!campo.value && ev.key === 'ArrowUp') { ev.preventDefault(); irA(anteriorVisible(actual - 1)); return; }
  if (!campo.value && /^[1-9]$/.test(ev.key) && galeria[+ev.key - 1]) {
    ev.preventDefault();
    aplicar(galeria[+ev.key - 1]);
  }
}

document.addEventListener('keydown', ev => {
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') { ev.preventDefault(); deshacer(); }
});

function irA(k, sinFoco) {
  actual = k;
  const campo = document.getElementById('in' + k);
  if (campo) campo.placeholder = ultima;
  marcar();
  const el = document.getElementById('s' + k);
  if (el && !sinFoco) { el.scrollIntoView({block: 'center'}); campo.focus(); }
}

function irAlDato(texto) {
  const t = (texto || '').trim().toLowerCase();
  let k = -1, caja = null;
  if (t.startsWith('#')) {
    caja = +t.slice(1);
    k = segs.findIndex(s => s.dets.some(d => d.i === caja));
  } else if (t.startsWith('f')) {
    const f = +t.slice(1);
    k = segs.findIndex(s => s.dets[s.dets.length - 1].frame >= f);
  }
  if (k < 0) { avisar(`no esta en esta ventana: ${texto}`); return; }
  if (!visible(segs[k])) filtrar('todos');
  irA(k);
  document.querySelectorAll('.caja.foco').forEach(c => c.classList.remove('foco'));
  if (caja !== null) { const c = document.getElementById('c' + caja); if (c) c.classList.add('foco'); }
}

async function aplicar(valor) {
  const v = normalizar(valor) || ultima;
  if (!v) return;
  if (v.length > 1 && v !== '?' &&
      !confirm(`"${v}" tiene mas de una letra. Una persona es una letra (A, B...), X no es persona. Guardar igual?`)) return;
  ultima = v;
  const s = segs[actual], cambios = {};
  s.dets.forEach(d => { cambios[d.i] = v; });
  const n = await guardar(cambios, true);
  if (n) avisar(`${n} caja${n > 1 ? 's' : ''} -> ${v}   (Ctrl+Z deshace)`);
  irA(siguienteVisible(actual + 1, true));
}

async function unaCaja(i) {
  const v = prompt('Letra solo para esta caja (vacio = quitar):', etq[i] || '');
  if (v === null) return;
  const n = normalizar(v);
  const cambios = await guardar({[i]: n}, true);
  if (cambios) avisar(`caja #${i} -> ${n || '(sin letra)'}   (Ctrl+Z deshace)`);
}

async function deshacer() {
  const previo = pila.pop();
  if (!previo) { avisar('nada que deshacer en esta sesion'); return; }
  const n = await guardar(previo, false);
  avisar(`deshecho: ${n} caja${n === 1 ? '' : 's'} vuelven a su letra anterior`);
}

function verFrame(i) {
  const v = document.getElementById('visor');
  v.querySelector('img').src = '/frame?i=' + i + '&v=' + Date.now();
  v.style.display = 'flex';
}

function panel() {
  const p = document.getElementById('panel');
  p.style.display = p.style.display === 'block' ? 'none' : 'block';
  if (p.style.display === 'block') cargarProblemas();
}

async function cargarProblemas() {
  const lista = await (await fetch('/problemas' + location.search)).json();
  const sinEtiq = lista.filter(p => p.tipo === 'sin etiquetar').length;
  document.getElementById('b-revisar').textContent = `Revisar (${lista.length - sinEtiq} + ${sinEtiq} sin letra)`;
  document.getElementById('problemas').innerHTML = lista.map(p =>
    `<div class="p" onclick="irAlDato('#${p.caja}')"><span class="tipo">${p.tipo}</span>`
    + ` &middot; caja #${p.caja}, frame ${p.frame}<br>${p.texto}</div>`).join('') || '<p>Nada que revisar.</p>';
}

cargar();
</script></body></html>
"""

if __name__ == "__main__":
    main()
