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

A second page, /frames, reviews whole frames, which is what a detector needs: a frame can be used for
training only when every person in it has exactly one box. Grouping labels boxes, not frames, and on
flight 3 it left 104 of 764 lost-person labels on boxes that were shifted or doubled copies of a person
already boxed. The review page draws every candidate of the frame coloured by its label, flags a person
box that covers half of another person box, and lets the labeller change a label with a click, mark a
box as a duplicate, draw the box of a person no detector proposed, and mark the frame as reviewed. A
magnifier follows the cursor, because a person at 25 m is some 30 px tall. Those corrections are written
to <salida>_frames.json and win over the group labels without rewriting them, so grouping again later
never undoes a reviewed frame.

--lista-frames names the frames to review, one number per line, as proponer_cajas.py writes them. It
matters for the frames with no candidate at all: a person missed by every detector can only be drawn
on a frame that is shown.
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
# Flights with detector candidates from proponer_cajas.py: --vuelo also fills --lista-frames and --nombre,
# and the labels go to etiquetas_detector_<flight>.json, apart from any other labelling of that flight.
_RAIZ_DATOS = os.path.join("..", "drone-geolocation", "data")
for _nombre, _frames in (("26jul", os.path.join(_RAIZ_DATOS, "20260726_195524", "frames")),
                         ("01ago_2a", os.path.join(_DATOS, "20260801_184259", "frames")),
                         ("01ago_2b", os.path.join(_DATOS, "20260801_185326", "frames"))):
    VUELOS[_nombre] = (os.path.join(_ENT, "candidatas_%s.csv" % _nombre), os.path.join(_ENT, "candidatas_%s_embs.npy" % _nombre),
                       _frames, os.path.join(_ENT, "etiquetas_detector_%s.json" % _nombre),
                       os.path.join(_ENT, "candidatas_%s_frames.txt" % _nombre))
MUESTRA = 24          # crops shown per group: enough to see what it is, few enough to load fast
LADO = 128            # crop side, px: at 96 the text of a context box is a smudge of a few pixels
GRUESA = (0, 230, 255)    # BGR yellow: the box being labelled
CONTEXTO = (255, 0, 255)  # BGR magenta: boxes of --contexto, the flight's detections
VECINA = (255, 255, 0)    # BGR cyan: other boxes of the CSV being labelled, in the same frame
# Labels of the frame review. "duplicado" is a second box on a person who has one: dropped. "ignorar" is
# something that cannot be called either way (a lone foot, a person cut to a sliver by the frame edge):
# the export blanks it out, so the detector is neither rewarded nor punished for finding it.
REVISION = ("persona", "no", "duplicado", "ignorar")
LADO_MIN_NUEVA = 4        # px: a drawn box smaller than this is a slip of the mouse, not a person


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


def solape_menor(a, b):
    """Intersection of two boxes over the area of the smaller one: 1 when one lies inside the other.

    IoU would miss the case that matters, a box on the legs of a person inside the box of that person.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / menor if menor > 0 else 0.0


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
                 contexto=None, contexto_etiquetas=None, lista_frames=None, nombre=None):
        self.identidad = identidad
        # Shown on both pages, so two flights open in two tabs cannot be told apart only by their frames.
        self.nombre = nombre or os.path.splitext(os.path.basename(salida))[0]
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
        # Frame review: every frame with a candidate, plus the listed ones that have none.
        dentro = lambda f: (desde is None or f >= desde) and (hasta is None or f <= hasta)
        self.lista = sorted({int(f) for f in list(lista_frames or []) + list(self.por_frame) if dentro(int(f))})
        self.ruta_revision = os.path.splitext(salida)[0] + "_frames.json"
        self.revisados, self.correcciones, self.nuevas = set(), {}, {}
        if os.path.exists(self.ruta_revision):
            r = json.load(open(self.ruta_revision, encoding="utf-8"))
            local = {o: j for j, o in enumerate(self.orig)}
            self.revisados = {int(f) for f in r.get("revisados", [])}
            self.correcciones = {local[int(i)]: v for i, v in r.get("correcciones", {}).items() if int(i) in local}
            self.nuevas = {int(f): [[float(x) for x in c] for c in v] for f, v in r.get("nuevas", {}).items()}

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
        return {"grupos": out, "cajas": len(self.filas), "etiquetadas": len(self.etiquetas), "nombre": self.nombre,
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

    def final(self, i):
        """The label of box i: the frame review's correction if there is one, else the group's."""
        return self.correcciones.get(i, self.etiquetas.get(i))

    def frames_estado(self):
        frames = []
        for f in self.lista:
            d = self.frame_cajas(f)
            frames.append({"f": f, "revisado": f in self.revisados, "n": len(d["cajas"]),
                           "personas": sum(b["etiqueta"] == "persona" for b in d["cajas"]) + len(d["nuevas"]),
                           "doble": any(b["doble"] for b in d["cajas"] + d["nuevas"]),
                           "sin": sum(b["etiqueta"] is None for b in d["cajas"])})
        return {"frames": frames,
                "revisados": len(self.revisados & set(self.lista)), "nombre": self.nombre}

    def frame_cajas(self, f):
        """Everything the review page draws for frame f.

        A person box, proposed or drawn, is flagged "doble" when another person box of the same frame
        covers half of the smaller of the two: the same person boxed twice, or two people so close
        that the labeller should look again.
        """
        idx = self.por_frame.get(f, [])
        personas = [self._caja(i) for i in idx if self.final(i) == "persona"]
        personas += [tuple(c) for c in self.nuevas.get(f, [])]

        def doble(b):
            return sum(solape_menor(b, p) >= 0.5 for p in personas) > 1   # the box itself counts once

        cajas = [{"i": i, "caja": self._caja(i), "etiqueta": self.final(i), "corregida": i in self.correcciones,
                  "conf": float(self.filas[i].get("conf") or 0), "fuentes": self.filas[i].get("fuentes") or "",
                  "doble": self.final(i) == "persona" and doble(self._caja(i))} for i in idx]
        nuevas = [{"k": k, "caja": c, "doble": doble(tuple(c))} for k, c in enumerate(self.nuevas.get(f, []))]
        return {"f": f, "cajas": cajas, "nuevas": nuevas, "revisado": f in self.revisados,
                "contexto": [list(c) for c in self.contexto.get(f, [])]}

    def corregir(self, i, v):
        if v not in REVISION:
            raise ValueError("etiqueta invalida en la revision: persona, no o duplicado")
        if not 0 <= i < len(self.filas):
            raise ValueError("caja invalida")
        self.correcciones[i] = v
        self._guardar_revision()

    def nueva(self, f, caja):
        """Adds the box of a person no candidate covers, in pixels of the original frame."""
        if f not in self.lista:
            raise ValueError("frame fuera de la lista")
        x1, y1, x2, y2 = (float(v) for v in caja)
        if not np.all(np.isfinite([x1, y1, x2, y2])) or min(x1, y1) < 0 \
                or x2 - x1 < LADO_MIN_NUEVA or y2 - y1 < LADO_MIN_NUEVA:
            raise ValueError("caja invalida: x1 < x2, y1 < y2 y al menos %d px de lado" % LADO_MIN_NUEVA)
        self.nuevas.setdefault(f, []).append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        self._guardar_revision()
        return len(self.nuevas[f]) - 1

    def borrar(self, f, k):
        if not 0 <= k < len(self.nuevas.get(f, [])):
            raise ValueError("no hay caja dibujada %d en el frame %d" % (k, f))
        self.nuevas[f].pop(k)
        if not self.nuevas[f]:
            del self.nuevas[f]
        self._guardar_revision()

    def marcar_revisado(self, f, v):
        """Marks frame f as reviewed, which it can only be when no box of it is left without a label."""
        if f not in self.lista:
            raise ValueError("frame fuera de la lista")
        sin = [i for i in self.por_frame.get(f, []) if self.final(i) is None]
        if v and sin:
            raise ValueError("quedan %d cajas amarillas sin etiquetar en el frame %d" % (len(sin), f))
        if v:
            self.revisados.add(f)
        else:
            self.revisados.discard(f)
        self._guardar_revision()

    def _guardar_revision(self):
        datos = {"cajas": os.path.abspath(self.cajas), "clave": "fila del CSV de cajas",
                 "revisados": sorted(self.revisados),
                 "correcciones": {str(self.orig[i]): v for i, v in sorted(self.correcciones.items())},
                 "nuevas": {str(f): v for f, v in sorted(self.nuevas.items())}}
        tmp = self.ruta_revision + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(datos, fh, indent=1)
        os.replace(tmp, self.ruta_revision)

    def revisar_lote(self, frames):
        """Marks several frames reviewed at once, the mosaic's button; a frame that cannot be (a box
        left without a label, or not in the list) is returned with its reason and the rest still count."""
        hechos, rechazados = [], []
        for f in frames:
            try:
                self.marcar_revisado(int(f), True)
                hechos.append(int(f))
            except ValueError as e:
                rechazados.append({"f": int(f), "error": str(e)})
        return {"hechos": hechos, "rechazados": rechazados}

    def miniatura(self, f, ancho=480):
        """A small copy of frame f with its person boxes drawn thick green, the rest thin red and the
        unlabelled yellow: enough to see at a glance whether someone is left without a box."""
        import cv2
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % f))
        if img is None:
            raise FileNotFoundError(f)
        esc = ancho / float(img.shape[1])
        img = cv2.resize(img, (ancho, int(img.shape[0] * esc)))
        d = self.frame_cajas(f)
        cajas = [(b["caja"], b["etiqueta"]) for b in d["cajas"]] + [(tuple(n["caja"]), "persona") for n in d["nuevas"]]
        for (x1, y1, x2, y2), e in cajas:
            color, grosor = {"persona": ((80, 220, 80), 2), None: ((0, 220, 255), 2)}.get(e, ((60, 60, 230), 1))
            cv2.rectangle(img, (int(x1 * esc), int(y1 * esc)), (int(x2 * esc), int(y2 * esc)), color, grosor)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def imagen(self, f):
        """The frame file as it is on disk: the page draws the boxes itself, so a click redraws at once."""
        with open(os.path.join(self.frames, "frame_%04d.jpg" % f), "rb") as fh:
            return fh.read()

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
<p>Despues de los grupos, dos pasos cortos:
<b>1.</b> <a href="/frames?solo=dobles" style="color:#93c5fd">frames donde una persona tiene dos cajas</a> (apretar D en la caja chica) &middot;
<b>2.</b> <a href="/mosaico" style="color:#93c5fd">mosaico del resto</a> (mirar 24 a la vez; clic solo si falta o sobra algo) &middot;
<a href="/frames" style="color:#93c5fd">todos, uno por uno</a></p>
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
  document.title = e.nombre + ' - grupos';
  document.getElementById('cuenta').textContent =
    `${e.nombre}: ${e.etiquetadas} de ${e.cajas} cajas etiquetadas, ${e.grupos.length} grupos`;
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


FRAMES = r"""<!doctype html><meta charset="utf-8"><title>Revisar frames</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:0; padding:12px; display:flex; gap:12px; }
  #izq { flex:1; min-width:0; } #cv { display:block; cursor:crosshair; max-width:100%; }
  #der { width:300px; flex:none; }
  #lupa { width:300px; height:300px; border:1px solid #2a2e38; border-radius:6px; display:block; margin:8px 0; }
  button { font:600 13px system-ui; padding:6px 12px; margin:3px 3px 3px 0; border-radius:99px;
           border:1px solid #3a3f4b; background:#1c1f27; color:#e6e9ef; cursor:pointer; }
  button.principal { background:#166534; border-color:#22c55e; }
  .k { display:inline-block; min-width:1.3em; padding:0 4px; border:1px solid #3a3f4b; border-radius:4px;
       text-align:center; font:12px monospace; }
  #barra { height:6px; background:#2a2e38; border-radius:3px; margin:6px 0; }
  #progreso { height:100%; width:0; background:#22c55e; border-radius:3px; }
  .ley span { display:inline-block; width:14px; height:10px; margin:0 4px 0 8px; vertical-align:middle; }
  #aviso { color:#fb923c; min-height:1.2em; } a { color:#93c5fd; }
</style>
<div id="izq"><canvas id="cv"></canvas></div>
<div id="der">
  <div><b id="titulo">frame</b> <span id="donde"></span></div>
  <div id="modo" style="color:#fb923c"></div>
  <div id="barra"><div id="progreso"></div></div>
  <div id="cuenta"></div>
  <canvas id="lupa" width="300" height="300"></canvas>
  <div id="aviso"></div>
  <button class="principal" onclick="terminar()">revisado y siguiente <span class="k">Enter</span></button><br>
  <button onclick="ir(pos - 1)"><span class="k">&larr;</span></button>
  <button onclick="ir(pos + 1)"><span class="k">&rarr;</span></button>
  <button onclick="irSinRevisar()">sin revisar <span class="k">U</span></button>
  <label><input type="checkbox" id="ocultarNo" onchange="pintar()"> ocultar las "no"</label>
  <p class="ley"><span style="background:#22c55e"></span>persona<span style="background:#ef4444"></span>no
    <span style="background:#9ca3af"></span>duplicado<br><span style="background:#60a5fa"></span>ignorar
    <span style="background:#facc15"></span>sin etiquetar<span style="background:#fb923c"></span>&iquest;doble?
    <span style="background:#ff00ff"></span>vuelo</p>
  <p><b>Clic</b> en una caja: persona &harr; no.<br>
    Con el raton encima: <span class="k">P</span> persona <span class="k">N</span> no <span class="k">D</span> duplicado
    <span class="k">I</span> ignorar.<br>
    <b>Arrastrar</b>: dibuja la caja de una persona que ninguna caja cubre.<br>
    <b>Clic derecho</b> en una caja dibujada: la borra. <span class="k">R</span> desmarca revisado.</p>
  <p>Un frame esta <b>revisado</b> cuando cada persona tiene UNA caja verde y nada mas es verde.
    Una caja corrida (piernas, sombra, media persona) de alguien que ya tiene la suya es <b>duplicado</b>.</p>
  <p><a href="/">&larr; grupos</a> &middot; <a href="/frames?solo=dobles">solo dobles</a> &middot; <a href="/mosaico">mosaico</a></p>
</div>
<script>
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
const lupa = document.getElementById('lupa'), lx = lupa.getContext('2d');
const COLOR = {persona: '#22c55e', no: '#ef4444', duplicado: '#9ca3af', ignorar: '#60a5fa'};
const SIGUIENTE = {persona: 'no', no: 'persona', duplicado: 'persona', ignorar: 'persona'};
const TECLA = {p: 'persona', n: 'no', d: 'duplicado', i: 'ignorar'};
let lista = [], pos = 0, datos = null, img = new Image(), abajo = null, raton = null, nombre = '';

async function pedir(ruta, cuerpo) {
  const r = await fetch(ruta, cuerpo === undefined ? {} :
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cuerpo)});
  const d = await r.json();
  if (!r.ok) { aviso(d.error); throw new Error(d.error); }
  return d;
}
function aviso(t) { document.getElementById('aviso').textContent = t || ''; }
const ocultarNo = () => document.getElementById('ocultarNo').checked;

async function iniciar() {
  const e = await pedir('/frames/estado');
  // ?solo=dobles: only the frames where a person has two boxes, the ones that need a decision.
  const soloDobles = new URLSearchParams(location.search).get('solo') === 'dobles';
  lista = soloDobles ? e.frames.filter(x => x.doble) : e.frames;
  nombre = e.nombre; document.title = nombre + (soloDobles ? ' - dobles' : ' - frames');
  if (soloDobles) document.getElementById('modo').textContent = 'solo frames con una persona en dos cajas: ' + lista.length;
  if (!lista.length) return aviso('no hay frames');
  let p = lista.findIndex(x => x.f === parseInt(location.hash.slice(1)));
  if (p < 0) p = lista.findIndex(x => !x.revisado);
  ir(p < 0 ? 0 : p);
}
async function ir(p) {
  pos = Math.max(0, Math.min(lista.length - 1, p));
  const f = lista[pos].f;
  history.replaceState(null, '', '#' + f);
  const carga = new Promise(r => { img.onload = r; img.onerror = r; });
  img.src = '/imagen/' + f;
  const [d] = await Promise.all([pedir('/frames/' + f), carga]);
  datos = d; aviso(''); pintar();
  if (pos + 1 < lista.length) new Image().src = '/imagen/' + lista[pos + 1].f;   // the next one loads ahead
}
async function recargar() { datos = await pedir('/frames/' + datos.f); pintar(); }

function dibujar(g, ox, oy, s, w, h) {
  g.fillStyle = '#12141a'; g.fillRect(0, 0, w, h);
  g.drawImage(img, ox, oy, w / s, h / s, 0, 0, w, h);
  const rect = (b, color, ancho, trazos, texto) => {
    const x = (b[0] - ox) * s, y = (b[1] - oy) * s;
    g.setLineDash(trazos || []); g.strokeStyle = color; g.lineWidth = ancho;
    g.strokeRect(x, y, (b[2] - b[0]) * s, (b[3] - b[1]) * s); g.setLineDash([]);
    if (texto) { g.fillStyle = color; g.font = '12px system-ui'; g.fillText(texto, x, Math.max(12, y - 3)); }
  };
  datos.contexto.forEach(b => rect(b, '#ff00ff', 1, [2, 3], b[4]));
  datos.cajas.forEach(b => {
    const e = b.etiqueta;
    if (ocultarNo() && e === 'no') return;
    if (b.doble) return rect(b.caja, '#fb923c', 3, null, '¿doble?');
    rect(b.caja, e ? COLOR[e] : '#facc15', e === 'persona' || !e ? 2 : 1,
         e && e !== 'persona' ? [4, 3] : null, {duplicado: 'dup', ignorar: 'ign'}[e] || null);
  });
  datos.nuevas.forEach(b => rect(b.caja, b.doble ? '#fb923c' : '#22c55e', b.doble ? 3 : 2, null, '+'));
  if (abajo && abajo.arrastrando && raton)
    rect([Math.min(abajo.x, raton.x), Math.min(abajo.y, raton.y), Math.max(abajo.x, raton.x), Math.max(abajo.y, raton.y)],
         '#22c55e', 1, [3, 2]);
}
function pintar() {
  if (!datos || !img.naturalWidth) return;
  const ancho = Math.min(img.naturalWidth, document.getElementById('izq').clientWidth);
  if (cv.width !== ancho) { cv.width = ancho; cv.height = Math.round(img.naturalHeight * ancho / img.naturalWidth); }
  dibujar(cx, 0, 0, cv.width / img.naturalWidth, cv.width, cv.height);
  document.getElementById('titulo').textContent = nombre + ' · frame ' + datos.f;
  document.getElementById('donde').textContent = (pos + 1) + ' / ' + lista.length + (datos.revisado ? ' · revisado' : '');
  const hechos = lista.filter(x => x.revisado).length;
  document.getElementById('progreso').style.width = (100 * hechos / lista.length) + '%';
  const c = {persona: 0, no: 0, duplicado: 0, ignorar: 0, sin: 0};
  datos.cajas.forEach(b => c[b.etiqueta || 'sin']++);
  document.getElementById('cuenta').textContent = `${hechos} de ${lista.length} frames revisados · aqui: ` +
    `${c.persona + datos.nuevas.length} persona (${datos.nuevas.length} dibujadas), ${c.no} no, ` +
    `${c.duplicado} duplicado, ${c.ignorar} ignorar, ${c.sin} sin etiquetar`;
  const dobles = datos.cajas.filter(b => b.doble).length + datos.nuevas.filter(b => b.doble).length;
  if (dobles) aviso(dobles + ' cajas de persona se pisan: ¿la misma persona dos veces?');
  else if (c.sin) aviso(c.sin + ' cajas sin etiquetar en este frame');
  pintarLupa();
}
function pintarLupa() {
  lx.fillStyle = '#12141a'; lx.fillRect(0, 0, 300, 300);
  if (!raton || !datos) return;
  const s = 3;
  dibujar(lx, raton.x - 150 / s, raton.y - 150 / s, s, 300, 300);
  lx.strokeStyle = '#ffffff'; lx.lineWidth = 1; lx.beginPath();
  lx.moveTo(144, 150); lx.lineTo(156, 150); lx.moveTo(150, 144); lx.lineTo(150, 156); lx.stroke();
}

function punto(ev) {
  const r = cv.getBoundingClientRect();
  return {x: (ev.clientX - r.left) * img.naturalWidth / r.width, y: (ev.clientY - r.top) * img.naturalHeight / r.height};
}
function bajo(p) {
  const dentro = b => p.x >= b[0] && p.x <= b[2] && p.y >= b[1] && p.y <= b[3];
  const area = b => (b[2] - b[0]) * (b[3] - b[1]);
  let mejor = null;   // the smallest box under the cursor: a person inside a larger box stays clickable
  datos.nuevas.forEach(b => { if (dentro(b.caja) && (!mejor || area(b.caja) < area(mejor.caja))) mejor = {tipo: 'nueva', k: b.k, caja: b.caja}; });
  datos.cajas.forEach(b => {
    if (ocultarNo() && b.etiqueta === 'no') return;
    if (dentro(b.caja) && (!mejor || area(b.caja) < area(mejor.caja))) mejor = {tipo: 'caja', i: b.i, caja: b.caja, etiqueta: b.etiqueta};
  });
  return mejor;
}
async function poner(i, v) { try { await pedir('/frames/caja', {i, v}); } catch (e) {} recargar(); }
async function terminar() {
  try { await pedir('/frames/revisado', {f: datos.f, v: true}); } catch (e) { return; }   // the refusal is already in the notice
  lista[pos].revisado = true;
  if (pos + 1 < lista.length) ir(pos + 1); else { recargar(); aviso('ultimo frame de la lista'); }
}
function irSinRevisar() {
  for (let k = 1; k <= lista.length; k++) { const q = (pos + k) % lista.length; if (!lista[q].revisado) return ir(q); }
  aviso('todos los frames estan revisados');
}

cv.addEventListener('mousedown', ev => { if (ev.button === 0 && datos) { const p = punto(ev); abajo = {x: p.x, y: p.y, arrastrando: false}; } });
cv.addEventListener('mousemove', ev => {
  raton = punto(ev);
  if (abajo && Math.hypot(raton.x - abajo.x, raton.y - abajo.y) * cv.width / img.naturalWidth > 5) abajo.arrastrando = true;
  if (abajo && abajo.arrastrando) pintar(); else pintarLupa();
});
cv.addEventListener('mouseleave', () => { raton = null; pintarLupa(); });
window.addEventListener('mouseup', async ev => {
  if (!abajo || ev.button !== 0) return;
  const a = abajo, p = raton || a;
  abajo = null;
  if (a.arrastrando) {
    try { await pedir('/frames/nueva', {f: datos.f, caja: [Math.min(a.x, p.x), Math.min(a.y, p.y), Math.max(a.x, p.x), Math.max(a.y, p.y)]}); } catch (e) {}
    return recargar();
  }
  const b = bajo(a);
  if (b && b.tipo === 'caja') poner(b.i, SIGUIENTE[b.etiqueta] || 'persona');
});
cv.addEventListener('contextmenu', async ev => {
  ev.preventDefault();
  const b = bajo(punto(ev));
  if (b && b.tipo === 'nueva') { await pedir('/frames/borrar', {f: datos.f, k: b.k}); recargar(); }
});
document.addEventListener('keydown', async ev => {
  if (ev.target.tagName === 'INPUT' || !datos) return;
  const k = ev.key.toLowerCase();
  if (k === 'enter' || k === ' ') { ev.preventDefault(); terminar(); }
  else if (k === 'arrowright') ir(pos + 1);
  else if (k === 'arrowleft') ir(pos - 1);
  else if (k === 'u') irSinRevisar();
  else if (k === 'r') { await pedir('/frames/revisado', {f: datos.f, v: false}); lista[pos].revisado = false; recargar(); }
  else if (raton && TECLA[k]) {
    const b = bajo(raton);
    if (b && b.tipo === 'caja') poner(b.i, TECLA[k]);
  }
});
window.addEventListener('resize', pintar);
iniciar();
</script>
"""


MOSAICO = r"""<!doctype html><meta charset="utf-8"><title>Mosaico</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:0; padding:12px 16px; }
  .rejilla { display:grid; grid-template-columns:repeat(auto-fill, minmax(360px, 1fr)); gap:8px; margin:10px 0; }
  .t { position:relative; } .t img { width:100%; display:block; border-radius:4px; border:2px solid #2a2e38; }
  .t a:hover img { border-color:#fb923c; }
  .t span { position:absolute; left:4px; top:4px; background:#000b; padding:1px 6px; border-radius:4px; font-size:12px; }
  button { font:600 14px system-ui; padding:8px 16px; border-radius:99px; border:1px solid #22c55e;
           background:#166534; color:#e6e9ef; cursor:pointer; }
  a { color:#93c5fd; } #aviso { color:#fb923c; }
</style>
<h2 id="titulo">Mosaico</h2>
<p><b>Que mirar en cada miniatura</b>: <b style="color:#50dc50">verde</b> = caja de persona, <span style="color:#e63c3c">rojo</span> = no es persona.
Esta bien si <b>cada persona tiene UNA caja verde entera</b> y ninguna verde esta sobre algo que no es persona.<br>
Si una miniatura tiene algo mal (una persona sin caja verde, una verde sobre un cono, una verde que corta a la persona por la mitad):
<b>clic en ella</b>, se abre ese frame en otra pestana, lo arreglas alli y apretas Enter. Si ves amarillo, falta etiquetar esa caja.<br>
Cuando todas las de la pagina estan bien: el boton de abajo las marca revisadas y trae las siguientes.</p>
<p id="cuenta"></p>
<div class="rejilla" id="rejilla"></div>
<button id="boton" onclick="aceptar()">todas estas estan bien: marcar revisadas y ver las siguientes</button>
<p id="aviso"></p>
<p><a href="/">&larr; grupos</a> &middot; <a href="/frames?solo=dobles">solo dobles</a></p>
<script>
const POR_PAGINA = 24;
let pagina = [];
async function cargar() {
  const e = await (await fetch('/frames/estado')).json();
  document.title = e.nombre + ' - mosaico';
  document.getElementById('titulo').textContent = e.nombre + ' - mosaico';
  const dobles = e.frames.filter(x => !x.revisado && x.doble).length;
  const pendientes = e.frames.filter(x => !x.revisado && !x.doble);
  pagina = pendientes.slice(0, POR_PAGINA);
  document.getElementById('cuenta').textContent = `${e.revisados} de ${e.frames.length} frames revisados · ` +
    `quedan ${pendientes.length} para el mosaico` + (dobles ? ` y ${dobles} con dos cajas en una persona (van por "solo dobles")` : '');
  document.getElementById('rejilla').innerHTML = pagina.map(x =>
    `<div class="t"><a href="/frames#${x.f}" target="_blank"><img loading="lazy" src="/miniatura/${x.f}?v=${Date.now()}"></a>` +
    `<span>frame ${x.f} · ${x.personas} persona${x.personas === 1 ? '' : 's'}${x.sin ? ' · ' + x.sin + ' sin etiquetar' : ''}</span></div>`).join('');
  document.getElementById('boton').hidden = !pagina.length;
  if (!pendientes.length) document.getElementById('aviso').textContent = 'no quedan frames para el mosaico';
}
async function aceptar() {
  const r = await (await fetch('/frames/revisados_lote', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frames: pagina.map(x => x.f)})})).json();
  document.getElementById('aviso').textContent = r.rechazados && r.rechazados.length
    ? r.rechazados.length + ' no se marcaron: ' + r.rechazados.map(x => x.error).join('; ') : '';
  window.scrollTo(0, 0);
  cargar();
}
window.addEventListener('focus', cargar);   // back from fixing a frame in the other tab: redraw what changed
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
            elif self.path == "/frames" or self.path.startswith("/frames?"):
                self._responder(FRAMES.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/mosaico":
                self._responder(MOSAICO.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/miniatura/"):
                try:
                    f = int(self.path.split("?")[0].rsplit("/", 1)[1])
                    with sesion.lock:
                        cuerpo = sesion.miniatura(f)
                    self._responder(cuerpo, "image/jpeg")
                except Exception:
                    self._responder(b'{"error": "miniatura"}', codigo=404)
            elif self.path == "/frames/estado":
                with sesion.lock:
                    self._responder(json.dumps(sesion.frames_estado()).encode("utf-8"))
            elif self.path.startswith("/frames/"):
                try:
                    f = int(self.path.rsplit("/", 1)[1])
                    with sesion.lock:
                        cuerpo = json.dumps(sesion.frame_cajas(f)).encode("utf-8")
                    self._responder(cuerpo)
                except ValueError:
                    self._responder(b'{"error": "frame"}', codigo=404)
            elif self.path.startswith("/imagen/"):
                try:
                    self._responder(sesion.imagen(int(self.path.rsplit("/", 1)[1])), "image/jpeg")
                except Exception:
                    self._responder(b'{"error": "imagen"}', codigo=404)
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
                    elif self.path == "/frames/caja":
                        sesion.corregir(int(d["i"]), d.get("v"))
                        r = {"status": "ok"}
                    elif self.path == "/frames/nueva":
                        r = {"k": sesion.nueva(int(d["f"]), d["caja"])}
                    elif self.path == "/frames/borrar":
                        sesion.borrar(int(d["f"]), int(d["k"]))
                        r = {"status": "ok"}
                    elif self.path == "/frames/revisados_lote":
                        r = sesion.revisar_lote(d.get("frames") or [])
                    elif self.path == "/frames/revisado":
                        sesion.marcar_revisado(int(d["f"]), bool(d.get("v", True)))
                        r = {"status": "ok"}
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
    ap.add_argument("--lista-frames", help="frames a revisar en /frames, uno por linea (proponer_cajas.py la escribe)")
    ap.add_argument("--nombre", help="nombre del vuelo en las paginas (por defecto, el del archivo de salida)")
    args = ap.parse_args()
    if args.vuelo:
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for campo, ruta in zip(("cajas", "embs", "frames", "salida", "lista_frames"), VUELOS[args.vuelo]):
            if getattr(args, campo) is None:
                setattr(args, campo, os.path.normpath(os.path.join(raiz, ruta)))
        if args.nombre is None and len(VUELOS[args.vuelo]) > 4:
            args.nombre = args.vuelo
    faltan = [c for c in ("cajas", "embs", "frames", "salida") if getattr(args, c) is None]
    if faltan:
        ap.error("faltan %s (o usa --vuelo %s)" % (", ".join("--" + c for c in faltan), "/".join(sorted(VUELOS))))
    s = Sesion(args.cajas, np.load(args.embs), args.frames, args.salida, args.grupos, args.desde, args.hasta,
               identidad=args.identidad, contexto=args.contexto, contexto_etiquetas=args.contexto_etiquetas,
               lista_frames=[int(l) for l in open(args.lista_frames) if l.strip()] if args.lista_frames else None,
               nombre=args.nombre)
    print("%d cajas en %d grupos, %d ya etiquetadas -> http://127.0.0.1:%d/  (frame por frame: /frames, %d de %d revisados)"
          % (len(s.filas), len(s.grupos), len(s.etiquetas), args.puerto, len(s.revisados), len(s.lista)), flush=True)
    servir(s, args.puerto)


if __name__ == "__main__":
    main()
