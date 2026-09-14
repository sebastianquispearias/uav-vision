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

    python scripts/etiquetar_grupos.py \\
        --cajas  ../drone-geolocation/entrenamiento/valida_ident_cajas_vuelo2a.csv \\
        --embs   ../drone-geolocation/entrenamiento/valida_ident_embs_vuelo2a.npy \\
        --frames ../drone-geolocation/data/flight_01ago/20260801_184259/frames \\
        --salida ../drone-geolocation/entrenamiento/grupos_vuelo2a.json
    -> http://127.0.0.1:8412/

The recorded flights have a shortcut, so the command fits on one line:

    python scripts/etiquetar_grupos.py --vuelo 2a

The output keys are row indices of the --cajas CSV, so each label points back at a frame and a box.
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
LADO = 96             # crop side, px


def agrupar(emb, k):
    """Agglomerative clustering on cosine distance, the grouping the labelling cost was measured with."""
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    k = max(1, min(int(k), len(emb)))
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)


class Sesion:
    """The boxes, their groups and the labels given so far, with every change written to disk."""

    def __init__(self, cajas, emb, frames, salida, k, desde=None, hasta=None):
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
        out.sort(key=lambda d: (d["etiqueta"] in ETIQUETAS, -d["n"]))
        return {"grupos": out, "cajas": len(self.filas), "etiquetadas": len(self.etiquetas)}

    def marcar(self, g, v):
        if v not in ETIQUETAS or g not in self.grupos:
            raise ValueError("grupo o etiqueta invalidos")
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

    def recorte(self, i):
        if i in self.recortes:
            return self.recortes[i]
        import cv2
        r = self.filas[i]
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % int(r["frame"])))
        if img is None:
            raise FileNotFoundError(r["frame"])
        x1, y1, x2, y2 = (float(r[c]) for c in ("x1", "y1", "x2", "y2"))
        # A margin, because a box drawn tight at altitude cuts off the context that tells a
        # person from a post.
        mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
        h, w = img.shape[:2]
        c = img[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))]
        esc = LADO / float(max(c.shape[:2]))
        c = cv2.resize(c, (max(1, int(c.shape[1] * esc)), max(1, int(c.shape[0] * esc))))
        lienzo = np.full((LADO, LADO, 3), 24, np.uint8)
        y0, x0 = (LADO - c.shape[0]) // 2, (LADO - c.shape[1]) // 2
        lienzo[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
        ok, buf = cv2.imencode(".jpg", lienzo, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.recortes[i] = buf.tobytes()
        return self.recortes[i]


PAGINA = r"""<!doctype html><meta charset="utf-8"><title>Etiquetar por grupos</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:16px; }
  .grupo { border:1px solid #2a2e38; border-radius:8px; padding:10px; margin-bottom:12px; }
  .grupo.persona { border-left:5px solid #4ade80; } .grupo.no { border-left:5px solid #f87171; }
  .grupo.parcial { border-left:5px solid #fbbf24; }
  .rejilla { display:flex; flex-wrap:wrap; gap:4px; margin:8px 0; }
  .rejilla img { width:96px; height:96px; border-radius:4px; }
  button { font:600 13px system-ui; padding:5px 14px; margin-right:6px; border-radius:99px;
           border:1px solid #3a3f4b; background:#1c1f27; color:#e6e9ef; cursor:pointer; }
</style>
<h2>Etiquetar por grupos</h2>
<p id="cuenta"></p>
<p>Un clic etiqueta TODO el grupo. Si en la rejilla hay de las dos cosas, "mezcla" lo parte en dos.</p>
<div id="grupos"></div>
<script>
async function cargar() {
  const e = await (await fetch('/estado')).json();
  document.getElementById('cuenta').textContent =
    `${e.etiquetadas} de ${e.cajas} cajas etiquetadas, ${e.grupos.length} grupos`;
  document.getElementById('grupos').innerHTML = e.grupos.map(g => `
    <div class="grupo ${g.etiqueta || ''}">
      <b>grupo ${g.g}</b> · ${g.n} cajas · ${g.etiqueta || 'sin etiquetar'}
      <div class="rejilla">${g.muestra.map(i => `<img loading="lazy" src="/recorte/${i}">`).join('')}</div>
      <button onclick="enviar('/marcar', {g: ${g.g}, v: 'persona'})">persona</button>
      <button onclick="enviar('/marcar', {g: ${g.g}, v: 'no'})">no es persona</button>
      ${g.n > 1 ? `<button onclick="enviar('/partir', {g: ${g.g}})">mezcla: partir</button>` : ''}
    </div>`).join('');
}
async function enviar(ruta, cuerpo) {
  await fetch(ruta, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cuerpo)});
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
    args = ap.parse_args()
    if args.vuelo:
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for campo, ruta in zip(("cajas", "embs", "frames", "salida"), VUELOS[args.vuelo]):
            if getattr(args, campo) is None:
                setattr(args, campo, os.path.normpath(os.path.join(raiz, ruta)))
    faltan = [c for c in ("cajas", "embs", "frames", "salida") if getattr(args, c) is None]
    if faltan:
        ap.error("faltan %s (o usa --vuelo %s)" % (", ".join("--" + c for c in faltan), "/".join(sorted(VUELOS))))
    s = Sesion(args.cajas, np.load(args.embs), args.frames, args.salida, args.grupos, args.desde, args.hasta)
    print("%d cajas en %d grupos, %d ya etiquetadas -> http://127.0.0.1:%d/"
          % (len(s.filas), len(s.grupos), len(s.etiquetas), args.puerto), flush=True)
    servir(s, args.puerto)


if __name__ == "__main__":
    main()
