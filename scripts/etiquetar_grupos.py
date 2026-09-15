"""
Labels person / not-person by groups of similar-looking boxes instead of one box at a time.

Measured on flight 3, window 3000-3700: its 852 boxes take 852 clicks one by one, 225 one track at
a time, and 33 as 30 groups of OSNet appearance, with only 3 boxes disagreeing with their group.
Appearance does not transfer between flights -- a classifier trained on one rejects the people of
another -- but inside one flight it separates well, which is exactly what labelling needs.

Each group is shown as a grid of crops. One click labels the whole group; "mezcla" splits it in two
by the same clustering, until every group is clean. Labels are written after every click, so a
closed page loses nothing and reopening resumes.

What it cannot label: people the detector never boxed. Those need boxes proposed by another
source, reviewed the same way.

With --identidad it labels WHO instead of WHAT: one capital letter per real person, X for not a
person, ? for cannot tell. Measured on the hand-labelled window 3000-3700 of flight 3, labelling
identities this way costs 45 clicks for 852 boxes at 30 groups, with 98.2 % of the boxes in a group
whose majority is their own person; the confusions were between two people who look alike from 40 m.
That is the risk of the method, and it is a biased one: the groups come from the same appearance
network the identity layer uses, so what the network confuses is what a hurried labeller would
copy. A group that shows two people must be split, and a click on a crop opens the whole frame with
the box drawn, to decide from context rather than from the crop.

    python scripts/etiquetar_grupos.py \\
        --cajas  ../drone-geolocation/entrenamiento/valida_ident_cajas_vuelo2a.csv \\
        --embs   ../drone-geolocation/entrenamiento/valida_ident_embs_vuelo2a.npy \\
        --frames ../drone-geolocation/data/flight_01ago/20260801_184259/frames \\
        --salida ../drone-geolocation/entrenamiento/grupos_vuelo2a.json
    -> http://127.0.0.1:8412/

The recorded flights have a shortcut, so the command fits on one line:

    python scripts/etiquetar_grupos.py --vuelo 2a

The output keys are row indices of the --cajas CSV, so each label points back at a frame and a box.

Every thumbnail and every frame view draws the box being labelled with a thick border, and the other
boxes of the same frame that fall inside the picture with a thin one. With --contexto those include
the boxes of another CSV, typically the flight's own detections, each with its label (a column
"etiqueta", or a JSON given with --contexto-etiquetas keyed by row of that CSV) or its confidence. That
is what makes a lost person decidable: a person the flight already boxed was not lost.

    python scripts/etiquetar_grupos.py ... \\
        --contexto ../drone-geolocation/entrenamiento/identidad_cajas_02ago.csv \\
        --contexto-etiquetas ../drone-geolocation/entrenamiento/identidad_gt_02ago.json
"""
import argparse
import csv
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ETIQUETAS = ("persona", "no")

_ENT = os.path.join("..", "drone-geolocation", "entrenamiento")
_DATOS = os.path.join("..", "drone-geolocation", "data", "flight_01ago")
# Flights whose boxes and embeddings are already on disk: --vuelo fills the four paths.
VUELOS = {
    "2a": (os.path.join(_ENT, "valida_ident_cajas_vuelo2a.csv"), os.path.join(_ENT, "valida_ident_embs_vuelo2a.npy"),
           os.path.join(_DATOS, "20260801_184259", "frames"), os.path.join(_ENT, "grupos_vuelo2a.json")),
    "2b": (os.path.join(_ENT, "valida_ident_cajas_vuelo2b.csv"), os.path.join(_ENT, "valida_ident_embs_vuelo2b.npy"),
           os.path.join(_DATOS, "20260801_185326", "frames"), os.path.join(_ENT, "grupos_vuelo2b.json")),
}
MUESTRA = 24          # crops shown per group: enough to see what it is, few enough to load fast
LADO = 128            # crop side, px: at 96 the text of a context box is a smudge of a few pixels
GRUESA = (0, 230, 255)    # BGR yellow: the box being labelled
CONTEXTO = (255, 0, 255)  # BGR magenta: boxes of --contexto, the flight's detections
VECINA = (255, 255, 0)    # BGR cyan: other boxes of the CSV being labelled, in the same frame


def leer_contexto(ruta, etiquetas=None):
    """The boxes of a context CSV grouped by frame, as (x1, y1, x2, y2, text).

    The text is the row's "etiqueta" column if the CSV has one, else the label of that row in the
    JSON of --contexto-etiquetas (a dict under "etiquetas", keyed by row index), else its confidence.
    """
    letras = {}
    if etiquetas:
        d = json.load(open(etiquetas, encoding="utf-8"))
        letras = {int(k): str(v) for k, v in d.get("etiquetas", d).items()}
    por_frame = {}
    for j, r in enumerate(csv.DictReader(open(ruta, encoding="utf-8"))):
        texto = (r.get("etiqueta") or letras.get(j) or "%.2f" % float(r.get("conf") or 0)).upper()
        por_frame.setdefault(int(r["frame"]), []).append(
            tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2")) + (texto,))
    return por_frame


def agrupar(emb, k):
    """Agglomerative clustering on cosine distance, the grouping the labelling cost was measured with."""
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    k = max(1, min(int(k), len(emb)))
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)


class Sesion:
    """The boxes, their groups and the labels given so far, with every change written to disk."""

    def __init__(self, cajas, emb, frames, salida, k, desde=None, hasta=None, identidad=False,
                 contexto=None, contexto_etiquetas=None):
        self.identidad = identidad
        self.contexto = leer_contexto(contexto, contexto_etiquetas) if contexto else {}
        filas = list(csv.DictReader(open(cajas, encoding="utf-8")))
        if len(filas) != len(emb):
            raise ValueError("%d cajas pero %d embeddings" % (len(filas), len(emb)))
        self.orig = [i for i, r in enumerate(filas)
                     if (desde is None or int(r["frame"]) >= desde)
                     and (hasta is None or int(r["frame"]) <= hasta)]
        self.filas = [filas[i] for i in self.orig]
        e = np.asarray(emb, dtype=np.float64)[self.orig]
        self.emb = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        self.cajas, self.frames, self.salida = cajas, frames, salida
        self.por_frame = {}
        for i, r in enumerate(self.filas):
            self.por_frame.setdefault(int(r["frame"]), []).append(i)
        self.lock = threading.Lock()
        self.recortes = {}
        self.etiquetas = {}                                   # local index -> label
        if os.path.exists(salida):
            previas = json.load(open(salida, encoding="utf-8"))["etiquetas"]
            local = {o: j for j, o in enumerate(self.orig)}
            self.etiquetas = {local[int(i)]: v for i, v in previas.items() if int(i) in local}
        self.grupos = {}
        for i, g in enumerate(agrupar(self.emb, k)):
            self.grupos.setdefault(int(g), []).append(i)

    def estado(self):
        out = []
        for g, miembros in self.grupos.items():
            vals = [self.etiquetas.get(i) for i in miembros]
            if all(v is None for v in vals):
                etiqueta = None
            elif len(set(vals)) == 1:
                etiqueta = vals[0]
            else:
                etiqueta = "parcial"
            paso = max(1, len(miembros) // MUESTRA)
            out.append({"g": g, "n": len(miembros), "etiqueta": etiqueta,
                        "muestra": miembros[::paso][:MUESTRA]})
        # Unlabelled first, then the largest: the next click is always the one that labels most.
        out.sort(key=lambda d: (d["etiqueta"] not in (None, "parcial"), -d["n"]))
        return {"grupos": out, "cajas": len(self.filas), "etiquetadas": len(self.etiquetas),
                "identidad": self.identidad, "letras": sorted(set(self.etiquetas.values()))}

    def validar(self, v):
        """The label as stored, or ValueError. In identity mode a lower-case x is the same X."""
        if not isinstance(v, str):
            raise ValueError("etiqueta invalida")
        if self.identidad:
            v = v.strip().upper()
            if len(v) != 1 or not ("A" <= v <= "Z" or v == "?"):
                raise ValueError("una letra por persona (A-Z), X no es persona, ? no se distingue")
            return v
        if v not in ETIQUETAS:
            raise ValueError("etiqueta invalida")
        return v

    def marcar(self, g, v):
        v = self.validar(v)
        if g not in self.grupos:
            raise ValueError("grupo invalido")
        for i in self.grupos[g]:
            self.etiquetas[i] = v
        self._guardar()

    def partir(self, g):
        miembros = self.grupos.get(g)
        if not miembros or len(miembros) < 2:
            raise ValueError("no hay nada que partir")
        lab = agrupar(self.emb[miembros], 2)
        nuevo = max(self.grupos) + 1
        self.grupos[g] = [m for m, l in zip(miembros, lab) if l == 0]
        self.grupos[nuevo] = [m for m, l in zip(miembros, lab) if l == 1]
        return [g, nuevo]

    def _guardar(self):
        datos = {"cajas": os.path.abspath(self.cajas), "clave": "fila del CSV de cajas",
                 "etiquetas": {str(self.orig[i]): v for i, v in sorted(self.etiquetas.items())}}
        tmp = self.salida + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=1)
        os.replace(tmp, self.salida)

    def _caja(self, i):
        return tuple(float(self.filas[i][c]) for c in ("x1", "y1", "x2", "y2"))

    def _dibujar(self, img, i, ox, oy, esc, gruesa, fina, letra):
        """Draws the boxes of row i's frame on an image that is that frame shifted by (ox, oy) and
        scaled by esc: row i with a thick border, every other box that falls inside with a thin one
        and its text."""
        import cv2
        h, w = img.shape[:2]

        def punto(x, y):
            return int(round((x - ox) * esc)), int(round((y - oy) * esc))

        f = int(self.filas[i]["frame"])
        otras = [self._caja(j) + (None, VECINA) for j in self.por_frame.get(f, []) if j != i]
        otras += [c[:4] + (c[4], CONTEXTO) for c in self.contexto.get(f, [])]
        for x1, y1, x2, y2, texto, color in otras:
            p1, p2 = punto(x1, y1), punto(x2, y2)
            if p2[0] < 0 or p2[1] < 0 or p1[0] >= w or p1[1] >= h:
                continue
            cv2.rectangle(img, p1, p2, color, fina)
            if texto:
                cv2.putText(img, texto, (max(1, p1[0]), max(int(24 * letra), p1[1] - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, letra, color, fina)
        x1, y1, x2, y2 = self._caja(i)
        cv2.rectangle(img, punto(x1, y1), punto(x2, y2), GRUESA, gruesa)

    def recorte(self, i):
        if i in self.recortes:
            return self.recortes[i]
        import cv2
        r = self.filas[i]
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % int(r["frame"])))
        if img is None:
            raise FileNotFoundError(r["frame"])
        x1, y1, x2, y2 = self._caja(i)
        # A margin, because a box drawn tight at altitude cuts off the context that tells a
        # person from a post.
        mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
        h, w = img.shape[:2]
        ox, oy = max(0, int(x1 - mx)), max(0, int(y1 - my))
        c = img[oy:min(h, int(y2 + my)), ox:min(w, int(x2 + mx))]
        esc = LADO / float(max(c.shape[:2]))
        c = cv2.resize(c, (max(1, int(c.shape[1] * esc)), max(1, int(c.shape[0] * esc))))
        # Drawn after resizing, so border widths are thumbnail pixels whatever the size of the box.
        self._dibujar(c, i, ox, oy, esc, gruesa=3, fina=1, letra=0.35)
        lienzo = np.full((LADO, LADO, 3), 24, np.uint8)
        y0, x0 = (LADO - c.shape[0]) // 2, (LADO - c.shape[1]) // 2
        lienzo[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
        ok, buf = cv2.imencode(".jpg", lienzo, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.recortes[i] = buf.tobytes()
        return self.recortes[i]

    def frame(self, i):
        """The whole frame a box came from, with that box thick and the other boxes thin: the
        context a crop cuts away."""
        import cv2
        r = self.filas[i]
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % int(r["frame"])))
        if img is None:
            raise FileNotFoundError(r["frame"])
        esc = min(1.0, 1280.0 / img.shape[1])
        img = cv2.resize(img, (int(img.shape[1] * esc), int(img.shape[0] * esc)))
        self._dibujar(img, i, 0, 0, esc, gruesa=3, fina=1, letra=0.5)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


PAGINA = r"""<!doctype html><meta charset="utf-8"><title>Etiquetar por grupos</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:16px; }
  .grupo { border:1px solid #2a2e38; border-radius:8px; padding:10px; margin-bottom:12px; }
  .grupo.persona { border-left:5px solid #4ade80; } .grupo.no { border-left:5px solid #f87171; }
  .grupo.parcial { border-left:5px solid #fbbf24; }
  .rejilla { display:flex; flex-wrap:wrap; gap:4px; margin:8px 0; }
  .rejilla img { width:128px; height:128px; border-radius:4px; }
  button { font:600 13px system-ui; padding:5px 14px; margin-right:6px; border-radius:99px;
           border:1px solid #3a3f4b; background:#1c1f27; color:#e6e9ef; cursor:pointer; }
</style>
<h2>Etiquetar por grupos</h2>
<p id="cuenta"></p>
<p id="ayuda">Un clic etiqueta TODO el grupo. Si en la rejilla hay de las dos cosas, "mezcla" lo parte en dos.</p>
<p id="cajas">Borde <b style="color:#ffe600">amarillo grueso</b> = esta caja, la que se etiqueta.
Borde <b style="color:#ff00ff">magenta fino</b> = las detecciones del vuelo (--contexto), con su letra o su confianza:
si la persona ya tiene una caja fina del vuelo, el vuelo no la perdio y esta caja es <b>no</b>.
Borde <b style="color:#00ffff">cian fino</b> = otras cajas de esta misma lista en el mismo frame.
<b>Clic en un recorte</b>: el frame entero con las mismas cajas.</p>
<div id="grupos"></div>
<script>
async function cargar() {
  const e = await (await fetch('/estado')).json();
  document.getElementById('cuenta').textContent =
    `${e.etiquetadas} de ${e.cajas} cajas etiquetadas, ${e.grupos.length} grupos`;
  if (e.identidad) document.getElementById('ayuda').innerHTML =
    'Una <b>letra por persona real</b> para todo el grupo y <b>Enter</b> (o clic en una letra ya usada). ' +
    '<b>X</b> = no es persona, <b>?</b> = no se distingue. Si el grupo tiene a dos personas: <b>mezcla: partir</b>. ' +
    '<b>Clic en un recorte</b>: abre el frame entero con la caja marcada.';
  const botones = g => e.identidad
    ? `<input size="3" maxlength="1" placeholder="letra" onkeydown="if(event.key==='Enter'){enviar('/marcar', {g: ${g.g}, v: this.value})}">
       ${e.letras.map(l => `<button onclick="enviar('/marcar', {g: ${g.g}, v: '${l}'})">${l}</button>`).join('')}`
    : `<button onclick="enviar('/marcar', {g: ${g.g}, v: 'persona'})">persona</button>
       <button onclick="enviar('/marcar', {g: ${g.g}, v: 'no'})">no es persona</button>`;
  document.getElementById('grupos').innerHTML = e.grupos.map(g => `
    <div class="grupo ${g.etiqueta === 'persona' || g.etiqueta === 'no' || g.etiqueta === 'parcial' ? g.etiqueta : (g.etiqueta ? 'persona' : '')}">
      <b>grupo ${g.g}</b> · ${g.n} cajas · ${g.etiqueta || 'sin etiquetar'}
      <div class="rejilla">${g.muestra.map(i => `<a href="/frame/${i}" target="_blank"><img loading="lazy" src="/recorte/${i}"></a>`).join('')}</div>
      ${botones(g)}
      ${g.n > 1 ? `<button onclick="enviar('/partir', {g: ${g.g}})">mezcla: partir</button>` : ''}
    </div>`).join('');
}
async function enviar(ruta, cuerpo) {
  const r = await fetch(ruta, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cuerpo)});
  if (!r.ok) alert((await r.json()).error);
  cargar();
}
cargar();
</script>
"""


def servir(sesion, puerto):
    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _responder(self, cuerpo, tipo="application/json", codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            if self.path == "/":
                self._responder(PAGINA.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/estado":
                with sesion.lock:
                    self._responder(json.dumps(sesion.estado()).encode("utf-8"))
            elif self.path.startswith("/recorte/"):
                try:
                    self._responder(sesion.recorte(int(self.path.rsplit("/", 1)[1])), "image/jpeg")
                except Exception:
                    self._responder(b'{"error": "recorte"}', codigo=404)
            elif self.path.startswith("/frame/"):
                try:
                    self._responder(sesion.frame(int(self.path.rsplit("/", 1)[1])), "image/jpeg")
                except Exception:
                    self._responder(b'{"error": "frame"}', codigo=404)
            else:
                self._responder(b'{"error": "ruta"}', codigo=404)

        def do_POST(self):
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with sesion.lock:
                    if self.path == "/marcar":
                        sesion.marcar(int(d["g"]), d.get("v"))
                        r = {"status": "ok"}
                    elif self.path == "/partir":
                        r = {"grupos": sesion.partir(int(d["g"]))}
                    else:
                        raise ValueError("ruta")
                self._responder(json.dumps(r).encode("utf-8"))
            except Exception as e:
                self._responder(json.dumps({"error": str(e)}).encode("utf-8"), codigo=400)

    ThreadingHTTPServer(("127.0.0.1", puerto), Manejador).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vuelo", choices=sorted(VUELOS), help="atajo: rellena --cajas --embs --frames --salida")
    ap.add_argument("--cajas", help="CSV con frame,conf,x1,y1,x2,y2")
    ap.add_argument("--embs", help="embeddings OSNet, una fila por caja del CSV")
    ap.add_argument("--frames", help="carpeta con frame_NNNN.jpg")
    ap.add_argument("--salida", help="JSON de etiquetas; si existe, se retoma")
    ap.add_argument("--grupos", type=int, default=30)
    ap.add_argument("--desde", type=int, default=None)
    ap.add_argument("--hasta", type=int, default=None)
    ap.add_argument("--puerto", type=int, default=8412)
    ap.add_argument("--contexto", help="CSV frame,conf,x1,y1,x2,y2[,etiqueta] dibujado en fino (p. ej. las detecciones del vuelo)")
    ap.add_argument("--contexto-etiquetas", help="JSON {etiquetas: {fila del CSV de contexto: letra}}")
    ap.add_argument("--identidad", action="store_true",
                    help="una letra por persona real (X no es persona, ? no se distingue) en vez de persona/no")
    args = ap.parse_args()
    if args.vuelo:
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for campo, ruta in zip(("cajas", "embs", "frames", "salida"), VUELOS[args.vuelo]):
            if getattr(args, campo) is None:
                setattr(args, campo, os.path.normpath(os.path.join(raiz, ruta)))
    faltan = [c for c in ("cajas", "embs", "frames", "salida") if getattr(args, c) is None]
    if faltan:
        ap.error("faltan %s (o usa --vuelo %s)" % (", ".join("--" + c for c in faltan), "/".join(sorted(VUELOS))))
    s = Sesion(args.cajas, np.load(args.embs), args.frames, args.salida, args.grupos, args.desde, args.hasta,
               identidad=args.identidad, contexto=args.contexto, contexto_etiquetas=args.contexto_etiquetas)
    print("%d cajas en %d grupos, %d ya etiquetadas -> http://127.0.0.1:%d/"
          % (len(s.filas), len(s.grupos), len(s.etiquetas), args.puerto), flush=True)
    servir(s, args.puerto)


if __name__ == "__main__":
    main()
