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
import random
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
                         ("01ago_2b", os.path.join(_DATOS, "20260801_185326", "frames")),
                         # The test flight, converted by scripts/convertir_02ago.py so it can be reviewed too.
                         ("02ago", os.path.join(_RAIZ_DATOS, "flight_02ago", "20260802_133309", "frames"))):
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
REPASO = 0.05             # share of the reviewed frames the blind re-check asks about again
SEMILLA_REPASO = 1234     # fixed, so reopening the tool asks about the same frames
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


def etiqueta_nueva(c):
    """The label of a drawn box: person unless it was marked to be ignored."""
    return c[4] if len(c) > 4 else "persona"


def contenida(a, b):
    """Fraction of box a that lies inside box b: 1 when a is entirely inside b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / ar if ar > 0 else 0.0


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
                 contexto=None, contexto_etiquetas=None, lista_frames=None, nombre=None, sospechas=None):
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
        # Suspicions from the audit: olvidadas_<flight>.csv next to the boxes, if it was ever run.
        self.ruta_sospechas = sospechas or os.path.join(os.path.dirname(os.path.abspath(cajas)),
                                                        "olvidadas_%s.csv" % self.nombre)
        # Frame review: every frame with a candidate, plus the listed ones that have none.
        dentro = lambda f: (desde is None or f >= desde) and (hasta is None or f <= hasta)
        self.lista = sorted({int(f) for f in list(lista_frames or []) + list(self.por_frame) if dentro(int(f))})
        self.ruta_revision = os.path.splitext(salida)[0] + "_frames.json"
        self.revisados, self.correcciones, self.nuevas, self.repaso, self.ajustes = set(), {}, {}, {}, {}
        if os.path.exists(self.ruta_revision):
            r = json.load(open(self.ruta_revision, encoding="utf-8"))
            local = {o: j for j, o in enumerate(self.orig)}
            self.revisados = {int(f) for f in r.get("revisados", [])}
            self.correcciones = {local[int(i)]: v for i, v in r.get("correcciones", {}).items() if int(i) in local}
            # A drawn box is [x1, y1, x2, y2] (a person) or [x1, y1, x2, y2, "ignorar"].
            self.nuevas = {int(f): [[float(x) for x in c[:4]] + list(c[4:5]) for c in v] for f, v in r.get("nuevas", {}).items()}
            self.repaso = {int(f): int(n) for f, n in r.get("repaso", {}).items()}
            # A box of the CSV whose corners were dragged: the label is of the box, so the box has to be
            # fixable too, or a detection that covers only the legs stays wrong for ever.
            self.ajustes = {local[int(i)]: [float(x) for x in c] for i, c in r.get("ajustes", {}).items()
                            if int(i) in local}

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
            # Each crop carries its own final label, so a single box corrected apart from its group shows it.
            out.append({"g": g, "n": len(miembros), "etiqueta": etiqueta,
                        "muestra": [{"i": i, "etiqueta": self.final(i), "corregida": i in self.correcciones}
                                    for i in miembros[::paso][:MUESTRA]]})
        # Unlabelled first, then the largest: the next click is always the one that labels most.
        out.sort(key=lambda d: (d["etiqueta"] not in (None, "parcial"), -d["n"]))
        return {"grupos": out, "cajas": len(self.filas), "etiquetadas": len(self.etiquetas), "nombre": self.nombre,
                "catalogo": self.catalogo() if self.identidad else [],
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
        """The box of row i as it stands: the adjusted one if its corners were dragged, else the CSV's."""
        if i in self.ajustes:
            return tuple(self.ajustes[i])
        return tuple(float(self.filas[i][c]) for c in ("x1", "y1", "x2", "y2"))

    def ajustar(self, i, caja):
        """Moves the corners of a box of the CSV; caja = None leaves it as it came, which is what undo needs."""
        if not 0 <= i < len(self.filas):
            raise ValueError("caja invalida")
        if caja is None:
            self.ajustes.pop(i, None)
        else:
            self.ajustes[i] = self._caja_valida(caja)
        self._guardar_revision()

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
        personas += [tuple(c[:4]) for c in self.nuevas.get(f, []) if etiqueta_nueva(c) == "persona"]

        def doble(b):
            return sum(solape_menor(b, p) >= 0.5 for p in personas) > 1   # the box itself counts once

        cajas = [{"i": i, "caja": self._caja(i), "etiqueta": self.final(i), "corregida": i in self.correcciones,
                  "ajustada": i in self.ajustes,
                  "conf": float(self.filas[i].get("conf") or 0), "fuentes": self.filas[i].get("fuentes") or "",
                  "doble": self.final(i) == "persona" and doble(self._caja(i))} for i in idx]
        nuevas = [{"k": k, "caja": c[:4], "etiqueta": etiqueta_nueva(c),
                   "doble": etiqueta_nueva(c) == "persona" and doble(tuple(c[:4]))}
                  for k, c in enumerate(self.nuevas.get(f, []))]
        return {"f": f, "cajas": cajas, "nuevas": nuevas, "revisado": f in self.revisados,
                "contexto": [list(c) for c in self.contexto.get(f, [])]}

    def corregir(self, i, v):
        """Sets the label of one box apart from its group; v = None drops the correction, which is
        what undo needs to leave a box exactly as the group had left it."""
        if v is not None and v not in REVISION:
            raise ValueError("etiqueta invalida en la revision: persona, no, duplicado o ignorar")
        if not 0 <= i < len(self.filas):
            raise ValueError("caja invalida")
        if v is None:
            self.correcciones.pop(i, None)
        else:
            self.correcciones[i] = v
        self._guardar_revision()

    @staticmethod
    def _caja_valida(caja):
        x1, y1, x2, y2 = (float(v) for v in caja)
        if not np.all(np.isfinite([x1, y1, x2, y2])) or min(x1, y1) < 0 \
                or x2 - x1 < LADO_MIN_NUEVA or y2 - y1 < LADO_MIN_NUEVA:
            raise ValueError("caja invalida: x1 < x2, y1 < y2 y al menos %d px de lado" % LADO_MIN_NUEVA)
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    def nueva(self, f, caja):
        """Adds the box of a person no candidate covers, in pixels of the original frame."""
        if f not in self.lista:
            raise ValueError("frame fuera de la lista")
        self.nuevas.setdefault(f, []).append(self._caja_valida(caja))
        self._guardar_revision()
        return len(self.nuevas[f]) - 1

    def mover(self, f, k, caja):
        """Replaces a drawn box, which is what dragging one of its corners does: a box that came out
        too big or too small is fixed without drawing it again."""
        if not 0 <= k < len(self.nuevas.get(f, [])):
            raise ValueError("no hay caja dibujada %d en el frame %d" % (k, f))
        self.nuevas[f][k] = self._caja_valida(caja) + list(self.nuevas[f][k][4:5])
        self._guardar_revision()

    def nueva_etiqueta(self, f, k, v):
        """A drawn box is a person by default; "ignorar" is for what cannot be decided (a lone foot,
        a blur), so the export blanks it instead of teaching it as background."""
        if v not in ("persona", "ignorar"):
            raise ValueError("una caja dibujada es persona o ignorar")
        if not 0 <= k < len(self.nuevas.get(f, [])):
            raise ValueError("no hay caja dibujada %d en el frame %d" % (k, f))
        self.nuevas[f][k] = self.nuevas[f][k][:4] + ([] if v == "persona" else ["ignorar"])
        self._guardar_revision()

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
                 "nuevas": {str(f): v for f, v in sorted(self.nuevas.items())},
                 "repaso": {str(f): n for f, n in sorted(self.repaso.items())},
                 "ajustes": {str(self.orig[i]): c for i, c in sorted(self.ajustes.items())}}
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

    def personas_en(self, f):
        """How many people the labels say frame f has: the answer the blind re-check compares against."""
        return (sum(self.final(i) == "persona" for i in self.por_frame.get(f, []))
                + sum(etiqueta_nueva(c) == "persona" for c in self.nuevas.get(f, [])))

    def sospechas(self):
        """Boxes a stronger detector found where no label of ours lies, read from the audit CSV
        (olvidadas_<flight>.csv next to the boxes). They are the places where a person could have been
        left without a box: the check is worth nothing if it is never looked at, so the tool shows it.
        """
        if not self.ruta_sospechas or not os.path.exists(self.ruta_sospechas):
            return {"nombre": self.nombre, "archivo": self.ruta_sospechas, "hay_archivo": False, "frames": []}
        filas = []
        for r in csv.DictReader(open(self.ruta_sospechas, encoding="utf-8")):
            f = int(r["frame"])
            b = tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2"))
            etiquetadas = [self._caja(i) for i in self.por_frame.get(f, [])] + [tuple(c[:4]) for c in self.nuevas.get(f, [])]
            if any(solape_menor(b, c) >= 0.3 for c in etiquetadas):
                continue          # already covered by a label, most likely drawn after the audit ran
            filas.append({"f": f, "conf": float(r["conf"]), "caja": list(b), "revisado": f in self.revisados})
        filas.sort(key=lambda d: -d["conf"])
        return {"nombre": self.nombre, "archivo": self.ruta_sospechas, "hay_archivo": True, "frames": filas}

    def recorte_caja(self, f, caja, lado=200):
        """A crop around an arbitrary box of frame f, with that box drawn: what the suspicions page shows."""
        import cv2
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % f))
        if img is None:
            raise FileNotFoundError(f)
        for i in self.por_frame.get(f, []):
            b = self._caja(i)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 2)
        for c in self.nuevas.get(f, []):
            cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (80, 220, 80), 2)
        x1, y1, x2, y2 = caja
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 255), 3)
        cx, cy, L = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 60) * 3
        h, w = img.shape[:2]
        X, Y = int(max(0, min(w - L, cx - L / 2))), int(max(0, min(h - L, cy - L / 2)))
        c = cv2.resize(img[Y:Y + int(L), X:X + int(L)], (lado, lado))
        return cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def repaso_estado(self):
        """The blind re-check: a fixed sample of the reviewed frames, asked about again without showing
        their boxes. It measures agreement with oneself, which is the honest way to say whether the
        labelling is consistent; the count of a frame is only revealed once it has been answered.
        """
        rev = sorted(self.revisados & set(self.lista))
        muestra = []
        if rev:
            n = max(1, int(round(REPASO * len(rev))))
            muestra = sorted(random.Random(SEMILLA_REPASO).sample(rev, min(n, len(rev))))
        filas = [{"f": f, "dicho": self.repaso.get(f),
                  "tenia": self.personas_en(f) if f in self.repaso else None} for f in muestra]
        hechas = [x for x in filas if x["dicho"] is not None]
        return {"nombre": self.nombre, "frames": filas, "contestadas": len(hechas),
                "acuerdo": sum(x["dicho"] == x["tenia"] for x in hechas),
                "pendiente": next((x["f"] for x in filas if x["dicho"] is None), None)}

    def repasar(self, f, n):
        if f not in self.revisados:
            raise ValueError("el frame %d no esta revisado" % f)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("cuantas personas ves: un numero de 0 en adelante")
        self.repaso[f] = n
        self._guardar_revision()
        return {"f": f, "dicho": n, "tenia": self.personas_en(f)}

    def catalogo(self):
        """In identity mode, every letter in use with the first crop labelled with it and how many
        boxes carry it: choosing a person by picture, not by memory, is what keeps one person from
        collecting two letters."""
        out = {}
        for i, v in sorted(self.etiquetas.items()):
            d = out.setdefault(v, {"letra": v, "i": i, "n": 0})
            d["n"] += 1
        return sorted(out.values(), key=lambda d: d["letra"])

    def renombrar(self, de, a):
        """Renames a letter, or merges two: every box labelled `de` ends up labelled `a`."""
        if not self.identidad:
            raise ValueError("renombrar letras es del modo --identidad")
        de, a = self.validar(de), self.validar(a)
        cambiadas = [i for i, v in self.etiquetas.items() if v == de]
        if not cambiadas:
            raise ValueError("no hay cajas con la letra %s" % de)
        for i in cambiadas:
            self.etiquetas[i] = a
        self._guardar()
        return {"de": de, "a": a, "cajas": len(cambiadas)}

    def chequeos(self):
        """The checks a reviewer would otherwise have to run by hand, over the whole flight.

        "medias" are person boxes lying inside another person box, the half-body duplicates; "dobles"
        are person boxes that cover half of another one; "huecos" are frames with nobody between two
        frames that do have somebody, where a person was probably left without a box.
        """
        dobles, medias, personas = [], [], {}
        for f in self.lista:
            d = self.frame_cajas(f)
            P = [b["caja"] for b in d["cajas"] if b["etiqueta"] == "persona"] + [tuple(n["caja"]) for n in d["nuevas"]]
            personas[f] = len(P)
            if any(b["doble"] for b in d["cajas"] + d["nuevas"]):
                dobles.append(f)
            if any(contenida(a, b) >= 0.8 for a in P for b in P if a is not b):
                medias.append(f)
        huecos = [self.lista[k] for k in range(1, len(self.lista) - 1)
                  if personas[self.lista[k]] == 0 and personas[self.lista[k - 1]] > 0 and personas[self.lista[k + 1]] > 0]
        return {"nombre": self.nombre, "total": len(self.lista), "revisados": len(self.revisados & set(self.lista)),
                "personas": sum(personas.values()), "dobles": dobles, "medias": medias, "huecos": huecos,
                "sin_revisar": [f for f in self.lista if f not in self.revisados],
                "sin_etiquetar": [f for f in self.lista if any(self.final(i) is None for i in self.por_frame.get(f, []))]}

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
        cajas = [(b["caja"], b["etiqueta"]) for b in d["cajas"]] + [(tuple(n["caja"]), n["etiqueta"]) for n in d["nuevas"]]
        for (x1, y1, x2, y2), e in cajas:
            color, grosor = {"persona": ((80, 220, 80), 2), None: ((0, 220, 255), 2),
                             "ignorar": ((250, 170, 90), 1)}.get(e, ((60, 60, 230), 1))
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
  .rejilla img { width:128px; height:128px; border-radius:4px; display:block; border:3px solid transparent; }
  .rejilla a.persona img { border-color:#22c55e; } .rejilla a.no img { border-color:#ef4444; }
  .rejilla a.duplicado img { border-color:#9ca3af; } .rejilla a.ignorar img { border-color:#60a5fa; }
  .rejilla a { position:relative; } .rejilla a.corregida::after { content:'corregida'; position:absolute; left:4px;
    bottom:4px; background:#000b; padding:0 4px; border-radius:3px; font-size:11px; }
  .k { display:inline-block; min-width:1.3em; padding:0 4px; border:1px solid #3a3f4b; border-radius:4px;
       text-align:center; font:12px monospace; }
  button { font:600 13px system-ui; padding:5px 14px; margin-right:6px; border-radius:99px;
           border:1px solid #3a3f4b; background:#1c1f27; color:#e6e9ef; cursor:pointer; }
</style>
<h2>Etiquetar por grupos</h2>
<p>Despues de los grupos, dos pasos cortos:
<b>1.</b> <a href="/frames?solo=dobles" style="color:#93c5fd">frames donde una persona tiene dos cajas</a> (apretar D en la caja chica) &middot;
<b>2.</b> <a href="/mosaico" style="color:#93c5fd">mosaico del resto</a> (mirar 24 a la vez; clic solo si falta o sobra algo) &middot;
<a href="/frames" style="color:#93c5fd">todos, uno por uno</a> &middot; <a href="/chequeos" style="color:#93c5fd">chequeos</a> &middot;
<a href="/video" style="color:#93c5fd">video de las etiquetas</a> &middot; <a href="/sospechas" style="color:#93c5fd">sospechas del modelo</a></p>
<p id="sueltas">Si en un grupo hay UN recorte mal, no hace falta partirlo: con el <b>raton encima de ese recorte</b>,
<span class="k">P</span> persona <span class="k">N</span> no <span class="k">D</span> duplicado <span class="k">I</span> ignorar.
Corrige esa caja sola, queda con el borde de su color y el grupo no se toca. <span class="k">Z</span> deshace.</p>
<p id="cuenta"></p>
<p id="ayuda">Un clic etiqueta TODO el grupo. Si en la rejilla hay de las dos cosas, "mezcla" lo parte en dos.</p>
<p id="cajas">Borde <b style="color:#ffe600">amarillo grueso</b> = esta caja, la que se etiqueta.
Borde <b style="color:#ff00ff">magenta fino</b> = las detecciones del vuelo (--contexto), con su letra o su confianza:
si la persona ya tiene una caja fina del vuelo, el vuelo no la perdio y esta caja es <b>no</b>.
Borde <b style="color:#00ffff">cian fino</b> = otras cajas de esta misma lista en el mismo frame.
<b>Clic en un recorte</b>: el frame entero con las mismas cajas.</p>
<div id="catalogo"></div>
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
  // Identity mode: the letters in use, each with the first crop labelled with it, and renaming, which
  // is also how two letters given to the same person are merged into one.
  document.getElementById('catalogo').innerHTML = !e.catalogo.length ? '' :
    `<div class="grupo"><b>personas de este vuelo</b> (elegir por la foto, no de memoria)
      <div class="rejilla">${e.catalogo.map(c => `<a href="/frame/${c.i}" target="_blank" title="${c.n} cajas">
        <img src="/recorte/${c.i}"><div style="text-align:center">${c.letra} &middot; ${c.n}</div></a>`).join('')}</div>
      renombrar o fundir: <input id="de" size="2" maxlength="1" placeholder="de"> &rarr;
      <input id="a" size="2" maxlength="1" placeholder="a">
      <button onclick="enviar('/letras/renombrar', {de: document.getElementById('de').value, a: document.getElementById('a').value})">cambiar</button>
      <span style="color:#9ca3af">(todas las cajas de la primera letra pasan a la segunda)</span></div>`;
  e.grupos.forEach(g => g.muestra.forEach(m => { etiquetaDe[m.i] = m.corregida ? m.etiqueta : undefined; }));
  document.getElementById('grupos').innerHTML = e.grupos.map(g => `
    <div class="grupo ${g.etiqueta === 'persona' || g.etiqueta === 'no' || g.etiqueta === 'parcial' ? g.etiqueta : (g.etiqueta ? 'persona' : '')}">
      <b>grupo ${g.g}</b> · ${g.n} cajas · ${g.etiqueta || 'sin etiquetar'}
      <div class="rejilla">${g.muestra.map(m => `<a href="/frame/${m.i}" target="_blank" class="${m.etiqueta || ''} ${m.corregida ? 'corregida' : ''}"
           onmouseenter="raton = ${m.i}" onmouseleave="raton = null"><img loading="lazy" src="/recorte/${m.i}"></a>`).join('')}</div>
      ${botones(g)}
      ${g.n > 1 ? `<button onclick="enviar('/partir', {g: ${g.g}})">mezcla: partir</button>` : ''}
    </div>`).join('');
}
async function enviar(ruta, cuerpo) {
  const r = await fetch(ruta, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cuerpo)});
  if (!r.ok) alert((await r.json()).error);
  cargar();
}
// One crop at a time, without touching its group: the label of that box alone, and Z to undo.
let raton = null, deshacer = [], etiquetaDe = {};
const TECLA = {p: 'persona', n: 'no', d: 'duplicado', i: 'ignorar'};
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT') return;
  const k = ev.key.toLowerCase();
  if (raton !== null && TECLA[k]) {
    deshacer.push({i: raton, v: etiquetaDe[raton] === undefined ? null : etiquetaDe[raton]});
    enviar('/frames/caja', {i: raton, v: TECLA[k]});
  } else if (k === 'z' && deshacer.length) {
    const u = deshacer.pop();
    enviar('/frames/caja', {i: u.i, v: u.v});
  }
});
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
  <button onclick="reproducir()" id="bplay">reproducir <span class="k">V</span></button>
  <label><input type="checkbox" id="ocultarNo" onchange="pintar()"> ocultar las "no"</label>
  <p>ir al frame <input id="saltar" size="6" placeholder="numero"
       onkeydown="if (event.key === 'Enter') saltar(this.value)"> <span class="k">G</span></p>
  <p class="ley"><span style="background:#22c55e"></span>persona<span style="background:#ef4444"></span>no
    <span style="background:#9ca3af"></span>duplicado<br><span style="background:#60a5fa"></span>ignorar
    <span style="background:#facc15"></span>sin etiquetar<span style="background:#fb923c"></span>&iquest;doble?
    <span style="background:#ff00ff"></span>vuelo</p>
  <p><b>Clic</b> en una caja: persona &harr; no.<br>
    Con el raton encima: <span class="k">P</span> persona <span class="k">N</span> no <span class="k">D</span> duplicado
    <span class="k">I</span> ignorar.<br>
    <b>Arrastrar</b>: dibuja la caja de una persona que ninguna caja cubre.<br>
    <b>Arrastrar una esquina</b> de cualquier caja (dibujada o del detector): la corrige, por ejemplo cuando
    cubre solo las piernas. <span class="k">V</span> reproduce los frames como video y vuelve a parar.<br>
    <b>Clic derecho</b> en una caja dibujada: la borra. Con el raton encima de una dibujada,
    <span class="k">I</span> la pasa a ignorar y <span class="k">P</span> la devuelve a persona.
    <span class="k">R</span> desmarca revisado. <span class="k">Z</span> deshace lo ultimo.</p>
  <p>Un frame esta <b>revisado</b> cuando cada persona tiene UNA caja verde y nada mas es verde.
    Una caja corrida (piernas, sombra, media persona) de alguien que ya tiene la suya es <b>duplicado</b>.</p>
  <p><a href="/">&larr; grupos</a> &middot; <a href="/frames?solo=dobles">solo dobles</a> &middot; <a href="/mosaico">mosaico</a>
    &middot; <a href="/chequeos">chequeos</a> &middot; <a href="/repaso">repaso ciego</a>
    &middot; <a href="/video">video</a> &middot; <a href="/sospechas">sospechas</a></p>
</div>
<script>
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
const lupa = document.getElementById('lupa'), lx = lupa.getContext('2d');
const COLOR = {persona: '#22c55e', no: '#ef4444', duplicado: '#9ca3af', ignorar: '#60a5fa'};
const SIGUIENTE = {persona: 'no', no: 'persona', duplicado: 'persona', ignorar: 'persona'};
const TECLA = {p: 'persona', n: 'no', d: 'duplicado', i: 'ignorar'};
let lista = [], pos = 0, datos = null, img = new Image(), abajo = null, raton = null, nombre = '', deshacer = [];

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
    if (b.doble) rect(b.caja, '#fb923c', 3, null, '¿doble?');
    else rect(b.caja, e ? COLOR[e] : '#facc15', e === 'persona' || !e ? 2 : 1,
              e && e !== 'persona' ? [4, 3] : null,
              (b.ajustada ? 'ajustada ' : '') + ({duplicado: 'dup', ignorar: 'ign'}[e] || '') || null);
    if (e === 'persona' || !e) {                    // handles: a wrong box is dragged, not redrawn
      const [x1, y1, x2, y2] = b.caja;
      [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([x, y]) => {
        g.fillStyle = e ? COLOR[e] : '#facc15';
        g.fillRect((x - ox) * s - 2, (y - oy) * s - 2, 4, 4);
      });
    }
  });
  datos.nuevas.forEach(b => {
    const ign = b.etiqueta === 'ignorar';
    rect(b.caja, b.doble ? '#fb923c' : (ign ? '#60a5fa' : '#22c55e'), b.doble ? 3 : 2, ign ? [4, 3] : null, ign ? 'ign' : '+');
    const [x1, y1, x2, y2] = b.caja;   // handles to grab: a drawn box can be resized by a corner
    [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([x, y]) => {
      g.fillStyle = '#22c55e';
      g.fillRect((x - ox) * s - 3, (y - oy) * s - 3, 6, 6);
    });
  });
  if (abajo && abajo.arrastrando && raton) {
    const o = abajo.redim ? abajo.redim.fija : {x: abajo.x, y: abajo.y};
    rect([Math.min(o.x, raton.x), Math.min(o.y, raton.y), Math.max(o.x, raton.x), Math.max(o.y, raton.y)],
         '#22c55e', 1, [3, 2]);
  }
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
async function poner(i, v) {
  const antes = datos.cajas.find(b => b.i === i);
  deshacer.push(() => pedir('/frames/caja', {i, v: antes && antes.corregida ? antes.etiqueta : null}));
  try { await pedir('/frames/caja', {i, v}); } catch (e) {}
  recargar();
}
async function deshacerUltimo() {
  const u = deshacer.pop();
  if (!u) return aviso('no hay nada que deshacer');
  try { await u(); } catch (e) {}
  recargar();
}
async function terminar() {
  try { await pedir('/frames/revisado', {f: datos.f, v: true}); } catch (e) { return; }   // the refusal is already in the notice
  lista[pos].revisado = true;
  if (pos + 1 < lista.length) ir(pos + 1); else { recargar(); aviso('ultimo frame de la lista'); }
}
function saltar(v) {
  const f = parseInt(v), k = lista.findIndex(x => x.f === f);
  if (k < 0) return aviso('el frame ' + v + ' no esta en la lista');
  document.getElementById('saltar').blur();
  ir(k);
}
function irSinRevisar() {
  for (let k = 1; k <= lista.length; k++) { const q = (pos + k) % lista.length; if (!lista[q].revisado) return ir(q); }
  aviso('todos los frames estan revisados');
}
// Playing the frames here, and not in another page, is what lets a box be fixed the moment it is seen wrong.
let cine = null;
function reproducir() {
  const b = document.getElementById('bplay');
  if (cine) { clearInterval(cine); cine = null; b.innerHTML = 'reproducir <span class="k">V</span>'; return; }
  b.innerHTML = 'pausa <span class="k">V</span>';
  cine = setInterval(() => { if (pos + 1 >= lista.length) return reproducir(); ir(pos + 1); }, 400);
}

function esquinaDe(p) {
  const cerca = 12 * img.naturalWidth / cv.width;   // 12 screen px, in pixels of the frame
  // Any box can be resized by a corner, drawn or proposed: a detection that covers only the legs is
  // fixed by dragging it, not by deleting and drawing it again.
  const todas = datos.nuevas.map(b => ({tipo: 'nueva', k: b.k, caja: b.caja}))
    .concat(datos.cajas.filter(b => !(ocultarNo() && b.etiqueta === 'no')).map(b => ({tipo: 'caja', i: b.i, caja: b.caja})));
  for (const b of todas) {
    const [x1, y1, x2, y2] = b.caja;
    for (const [x, y, fx, fy] of [[x1, y1, x2, y2], [x2, y1, x1, y2], [x1, y2, x2, y1], [x2, y2, x1, y1]])
      if (Math.abs(p.x - x) < cerca && Math.abs(p.y - y) < cerca)
        return Object.assign({fija: {x: fx, y: fy}}, b);
  }
  return null;
}
cv.addEventListener('mousedown', ev => {
  if (ev.button !== 0 || !datos) return;
  const p = punto(ev), e = esquinaDe(p);
  abajo = {x: p.x, y: p.y, arrastrando: !!e, redim: e};
});
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
    const f = datos.f;
    if (a.redim) {
      const {tipo, k, i, caja, fija} = a.redim;
      const nueva = [Math.min(fija.x, p.x), Math.min(fija.y, p.y), Math.max(fija.x, p.x), Math.max(fija.y, p.y)];
      try {
        if (tipo === 'nueva') {
          await pedir('/frames/mover', {f, k, caja: nueva});
          deshacer.push(() => pedir('/frames/mover', {f, k, caja}));
        } else {
          const antes = datos.cajas.find(b => b.i === i);
          await pedir('/frames/ajustar', {i, caja: nueva});
          deshacer.push(() => pedir('/frames/ajustar', {i, caja: antes.ajustada ? caja : null}));
        }
      } catch (e) {}
      return recargar();
    }
    try {
      const {k} = await pedir('/frames/nueva', {f, caja: [Math.min(a.x, p.x), Math.min(a.y, p.y), Math.max(a.x, p.x), Math.max(a.y, p.y)]});
      deshacer.push(() => pedir('/frames/borrar', {f, k}));
    } catch (e) {}
    return recargar();
  }
  const b = bajo(a);
  if (b && b.tipo === 'caja') poner(b.i, SIGUIENTE[b.etiqueta] || 'persona');
});
cv.addEventListener('contextmenu', async ev => {
  ev.preventDefault();
  const b = bajo(punto(ev));
  if (b && b.tipo === 'nueva') {
    const f = datos.f, caja = b.caja;
    await pedir('/frames/borrar', {f, k: b.k});
    deshacer.push(() => pedir('/frames/nueva', {f, caja}));
    recargar();
  }
});
document.addEventListener('keydown', async ev => {
  if (ev.target.tagName === 'INPUT' || !datos) return;
  const k = ev.key.toLowerCase();
  if (k === 'enter' || k === ' ') { ev.preventDefault(); terminar(); }
  else if (k === 'arrowright') ir(pos + 1);
  else if (k === 'arrowleft') ir(pos - 1);
  else if (k === 'u') irSinRevisar();
  else if (k === 'z') deshacerUltimo();
  else if (k === 'v') reproducir();
  else if (k === 'g') { ev.preventDefault(); document.getElementById('saltar').focus(); }
  else if (k === 'r') { await pedir('/frames/revisado', {f: datos.f, v: false}); lista[pos].revisado = false; recargar(); }
  else if (raton && TECLA[k]) {
    const b = bajo(raton);
    if (b && b.tipo === 'caja') poner(b.i, TECLA[k]);
    // A drawn box can only be a person or something to ignore.
    else if (b && b.tipo === 'nueva' && (TECLA[k] === 'persona' || TECLA[k] === 'ignorar')) {
      const f = datos.f, antes = datos.nuevas.find(n => n.k === b.k).etiqueta;
      deshacer.push(() => pedir('/frames/nueva_etiqueta', {f, k: b.k, v: antes}));
      try { await pedir('/frames/nueva_etiqueta', {f, k: b.k, v: TECLA[k]}); } catch (e) {}
      recargar();
    }
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


VIDEO = r"""<!doctype html><meta charset="utf-8"><title>Video de las etiquetas</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:0; padding:12px 16px; }
  img { width:100%; max-width:1280px; border-radius:6px; display:block; background:#000; }
  button { font:600 13px system-ui; padding:6px 12px; margin:4px 6px 4px 0; border-radius:99px;
           border:1px solid #3a3f4b; background:#1c1f27; color:#e6e9ef; cursor:pointer; }
  input[type=range] { width:min(1280px, 100%); }
  a { color:#93c5fd; } .k { display:inline-block; min-width:1.3em; padding:0 4px; border:1px solid #3a3f4b;
      border-radius:4px; text-align:center; font:12px monospace; }
</style>
<h2 id="titulo">Video de las etiquetas</h2>
<p>Los frames revisados, uno tras otro, con las cajas puestas: <b style="color:#4ade80">verde</b> persona,
<span style="color:#f87171">rojo</span> no, <span style="color:#9ca3af">gris</span> duplicado,
<span style="color:#60a5fa">azul</span> ignorar. Si ves a alguien sin caja verde, <b>para</b> y apreta
"corregir este frame": se abre ahi mismo para dibujarla.</p>
<img id="foto" alt="">
<input type="range" id="barra" min="0" value="0" oninput="pos = +this.value; pintar()">
<p>
  <button onclick="alternar()" id="bplay">reproducir <span class="k">espacio</span></button>
  <button onclick="paso(-1)"><span class="k">&larr;</span></button>
  <button onclick="paso(1)"><span class="k">&rarr;</span></button>
  <button onclick="velocidad(-1)">mas lento</button>
  <button onclick="velocidad(1)">mas rapido</button>
  <button onclick="window.open('/frames#' + lista[pos].f)">corregir este frame</button>
  <span id="estado"></span>
</p>
<p><a href="/">&larr; grupos</a> &middot; <a href="/frames">frames</a> &middot; <a href="/mosaico">mosaico</a>
 &middot; <a href="/chequeos">chequeos</a> &middot; <a href="/sospechas">sospechas del modelo</a></p>
<script>
let lista = [], pos = 0, fps = 5, tarea = null, nombre = '';
async function cargar() {
  const e = await (await fetch('/frames/estado')).json();
  nombre = e.nombre; document.title = nombre + ' - video';
  document.getElementById('titulo').textContent = nombre + ' - video de las etiquetas';
  lista = e.frames.filter(x => x.revisado);
  if (!lista.length) lista = e.frames;          // nothing reviewed yet: show them all anyway
  document.getElementById('barra').max = lista.length - 1;
  for (let k = 1; k < Math.min(8, lista.length); k++) new Image().src = '/miniatura/' + lista[k].f;
  pintar();
}
function pintar() {
  const x = lista[pos];
  document.getElementById('foto').src = '/miniatura/' + x.f;
  document.getElementById('barra').value = pos;
  document.getElementById('estado').textContent =
    `frame ${x.f} · ${pos + 1} de ${lista.length} · ${x.personas} persona${x.personas === 1 ? '' : 's'} · ${fps} fps`;
  if (pos + 3 < lista.length) new Image().src = '/miniatura/' + lista[pos + 3].f;
}
function paso(d) { pos = Math.max(0, Math.min(lista.length - 1, pos + d)); pintar(); }
function alternar() {
  if (tarea) { clearInterval(tarea); tarea = null; document.getElementById('bplay').textContent = 'reproducir'; return; }
  document.getElementById('bplay').textContent = 'pausa';
  tarea = setInterval(() => { if (pos + 1 >= lista.length) return alternar(); paso(1); }, 1000 / fps);
}
function velocidad(d) {
  fps = Math.max(1, Math.min(20, fps + d * (fps < 5 ? 1 : 2)));
  if (tarea) { alternar(); alternar(); } else pintar();
}
document.addEventListener('keydown', ev => {
  if (ev.key === ' ') { ev.preventDefault(); alternar(); }
  else if (ev.key === 'ArrowRight') paso(1);
  else if (ev.key === 'ArrowLeft') paso(-1);
});
cargar();
</script>
"""


SOSPECHAS = r"""<!doctype html><meta charset="utf-8"><title>Sospechas del modelo</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:0; padding:12px 16px; }
  .rejilla { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
  .rejilla a img { width:200px; height:200px; border-radius:4px; border:2px solid #2a2e38; display:block; }
  .rejilla a:hover img { border-color:#fb923c; }
  .rejilla div { text-align:center; font-size:12px; color:#9ca3af; }
  a { color:#93c5fd; } #aviso { color:#fb923c; }
</style>
<h2 id="titulo">Sospechas del modelo</h2>
<p>Cajas que un detector mas fuerte (RF-DETR y COCO, con el umbral mas bajo que el del etiquetado) encontro
<b>donde no hay ninguna caja nuestra</b>. En <b style="color:#ffe600">amarillo</b> lo que vio el modelo; en gris
las cajas del etiquetado. La mayoria es ruido: lo que importa es si aparece una persona que quedo sin caja.
<b>Clic</b> en una: se abre ese frame para dibujarla. Las que ya tienen caja desaparecen solas de esta lista.</p>
<p id="cuenta"></p>
<div class="rejilla" id="rejilla"></div>
<p id="aviso"></p>
<p><a href="/">&larr; grupos</a> &middot; <a href="/video">video</a> &middot; <a href="/chequeos">chequeos</a></p>
<script>
async function cargar() {
  const e = await (await fetch('/sospechas/estado')).json();
  document.title = e.nombre + ' - sospechas';
  document.getElementById('titulo').textContent = e.nombre + ' - sospechas del modelo';
  if (!e.hay_archivo) {
    document.getElementById('aviso').textContent = 'todavia no hay auditoria para este vuelo (falta ' + e.archivo + ')';
    return;
  }
  const top = e.frames.slice(0, 60);
  document.getElementById('cuenta').textContent =
    `${e.frames.length} cajas del modelo sin etiqueta nuestra` + (e.frames.length > 60 ? ', se muestran las 60 de mayor confianza' : '');
  document.getElementById('rejilla').innerHTML = top.map(s =>
    `<a href="/frames#${s.f}" target="_blank"><img loading="lazy" src="/sospecha/${s.f}/${s.caja.map(v => Math.round(v)).join(',')}?v=${Date.now()}">
     <div>f${s.f} · ${s.conf.toFixed(2)}</div></a>`).join('');
}
window.addEventListener('focus', cargar);
cargar();
</script>
"""


REPASO_PAGINA = r"""<!doctype html><meta charset="utf-8"><title>Repaso ciego</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:0; padding:16px; }
  img { max-width:100%; border-radius:6px; display:block; margin:8px 0; }
  input { font:16px system-ui; padding:6px 10px; width:5em; border-radius:6px; border:1px solid #3a3f4b;
          background:#1c1f27; color:#e6e9ef; }
  button { font:600 14px system-ui; padding:7px 14px; border-radius:99px; border:1px solid #3a3f4b;
           background:#1c1f27; color:#e6e9ef; cursor:pointer; }
  a { color:#93c5fd; } #veredicto { font-weight:600; } .mal { color:#fb923c; } .bien { color:#22c55e; }
</style>
<h2 id="titulo">Repaso ciego</h2>
<p>La herramienta te muestra, <b>sin las cajas</b>, algunos frames que ya diste por revisados, y te pregunta cuantas
personas ves. Compara tu respuesta con lo que dejaste etiquetado. No cambia ninguna etiqueta: mide si etiquetas
igual la segunda vez, que es la forma honesta de decir que el etiquetado es consistente.</p>
<p id="cuenta"></p>
<div id="caja">
  <img id="foto" alt="">
  <p>personas que ves en este frame: <input id="n" type="number" min="0" step="1"
      onkeydown="if (event.key === 'Enter') responder()"> <button onclick="responder()">responder</button></p>
  <p id="veredicto"></p>
</div>
<p><a href="/">&larr; grupos</a> &middot; <a href="/chequeos">chequeos</a></p>
<script>
let actual = null;
async function cargar() {
  const e = await (await fetch('/repaso/estado')).json();
  document.title = e.nombre + ' - repaso ciego';
  document.getElementById('titulo').textContent = e.nombre + ' - repaso ciego';
  const dichas = e.frames.filter(x => x.dicho !== null);
  document.getElementById('cuenta').textContent = e.frames.length
    ? `${e.contestadas} de ${e.frames.length} frames repasados · coinciden ${e.acuerdo}` +
      (e.contestadas ? ` (${Math.round(100 * e.acuerdo / e.contestadas)} %)` : '') +
      (dichas.length ? ' · ' + dichas.map(x => `f${x.f}: dijiste ${x.dicho}, tenia ${x.tenia}`).join(' · ') : '')
    : 'todavia no hay frames revisados: el repaso aparece cuando empieces a dar frames por revisados';
  actual = e.pendiente;
  document.getElementById('caja').hidden = actual === null;
  if (actual !== null) {
    document.getElementById('foto').src = '/imagen/' + actual;
    document.getElementById('n').value = '';
    document.getElementById('veredicto').textContent = '';
    document.getElementById('n').focus();
  } else if (e.frames.length) {
    document.getElementById('cuenta').textContent += ' · no queda ninguno por repasar';
  }
}
async function responder() {
  const n = parseInt(document.getElementById('n').value);
  if (isNaN(n)) return;
  const r = await (await fetch('/repaso', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({f: actual, n})})).json();
  const v = document.getElementById('veredicto');
  v.textContent = r.dicho === r.tenia ? `coincide: ${r.tenia}` : `dijiste ${r.dicho} y habias etiquetado ${r.tenia}`;
  v.className = r.dicho === r.tenia ? 'bien' : 'mal';
  setTimeout(cargar, 1600);
}
cargar();
</script>
"""


CHEQUEOS = r"""<!doctype html><meta charset="utf-8"><title>Chequeos</title>
<style>
  body { background:#12141a; color:#e6e9ef; font:14px system-ui; margin:0; padding:16px; max-width:900px; }
  .c { border:1px solid #2a2e38; border-left:5px solid #22c55e; border-radius:8px; padding:10px 14px; margin-bottom:10px; }
  .c.hay { border-left-color:#fb923c; } h3 { margin:0 0 4px; } p { margin:4px 0; }
  a { color:#93c5fd; } .f { display:inline-block; margin:2px 6px 2px 0; }
</style>
<h2 id="titulo">Chequeos</h2>
<p id="resumen"></p>
<div id="lista"></div>
<p><a href="/">&larr; grupos</a> &middot; <a href="/frames">frames</a> &middot; <a href="/mosaico">mosaico</a>
 &middot; <a href="/repaso">repaso ciego</a></p>
<script>
const QUE = [
  ['medias', 'Media persona con caja propia',
   'Una caja de persona esta dentro de otra caja de persona: el torso o las piernas de alguien que ya tiene su caja entera. Poner la chica en duplicado (tecla D).'],
  ['dobles', 'La misma persona en dos cajas',
   'Dos cajas de persona se pisan mas de la mitad. Si son la misma persona, una es duplicado; si son dos personas pegadas, esta bien asi.'],
  ['huecos', 'Frame sin nadie entre dos frames con alguien',
   'La persona estaba antes y despues, asi que lo mas probable es que siga ahi sin caja. Mirar el borde del frame y los frames movidos, y dibujar la caja si esta.'],
  ['sin_etiquetar', 'Cajas sin etiquetar', 'Frames con alguna caja amarilla. No se pueden dar por revisados asi.'],
  ['sin_revisar', 'Frames sin revisar', 'Todavia no pasaron por el mosaico ni por la revision frame a frame.'],
];

async function cargar() {
  const e = await (await fetch('/chequeos/estado')).json();
  document.title = e.nombre + ' - chequeos';
  document.getElementById('titulo').textContent = e.nombre + ' - chequeos';
  document.getElementById('resumen').textContent =
    `${e.revisados} de ${e.total} frames revisados · ${e.personas} cajas de persona en total`;
  document.getElementById('lista').innerHTML = QUE.map(([k, titulo, ayuda]) => {
    const fs = e[k] || [];
    const enlaces = fs.slice(0, 60).map(f => `<a class="f" href="/frames#${f}" target="_blank">${f}</a>`).join('') +
      (fs.length > 60 ? ` y ${fs.length - 60} mas` : '');
    return `<div class="c ${fs.length ? 'hay' : ''}"><h3>${titulo}: ${fs.length}</h3><p>${ayuda}</p><p>${enlaces || 'ninguno'}</p></div>`;
  }).join('');
}
window.addEventListener('focus', cargar);
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
            elif self.path == "/chequeos":
                self._responder(CHEQUEOS.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/chequeos/estado":
                with sesion.lock:
                    self._responder(json.dumps(sesion.chequeos()).encode("utf-8"))
            elif self.path == "/video":
                self._responder(VIDEO.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/sospechas":
                self._responder(SOSPECHAS.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/sospechas/estado":
                with sesion.lock:
                    self._responder(json.dumps(sesion.sospechas()).encode("utf-8"))
            elif self.path.startswith("/sospecha/"):
                try:
                    partes = self.path.split("?")[0].split("/")
                    f = int(partes[2])
                    caja = [float(x) for x in partes[3].split(",")]
                    with sesion.lock:
                        cuerpo = sesion.recorte_caja(f, caja)
                    self._responder(cuerpo, "image/jpeg")
                except Exception:
                    self._responder(b'{"error": "recorte"}', codigo=404)
            elif self.path == "/repaso":
                self._responder(REPASO_PAGINA.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/repaso/estado":
                with sesion.lock:
                    self._responder(json.dumps(sesion.repaso_estado()).encode("utf-8"))
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
                    elif self.path == "/frames/ajustar":
                        sesion.ajustar(int(d["i"]), d.get("caja"))
                        r = {"status": "ok"}
                    elif self.path == "/frames/nueva_etiqueta":
                        sesion.nueva_etiqueta(int(d["f"]), int(d["k"]), d.get("v"))
                        r = {"status": "ok"}
                    elif self.path == "/frames/mover":
                        sesion.mover(int(d["f"]), int(d["k"]), d["caja"])
                        r = {"status": "ok"}
                    elif self.path == "/repaso":
                        r = sesion.repasar(int(d["f"]), d.get("n"))
                    elif self.path == "/letras/renombrar":
                        r = sesion.renombrar(d.get("de"), d.get("a"))
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
    ap.add_argument("--sospechas", help="CSV frame,conf,x1,y1,x2,y2 de la auditoria con un modelo mas fuerte "
                                       "(por defecto olvidadas_<nombre>.csv junto a --cajas)")
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
               nombre=args.nombre, sospechas=args.sospechas)
    print("%d cajas en %d grupos, %d ya etiquetadas -> http://127.0.0.1:%d/  (frame por frame: /frames, %d de %d revisados)"
          % (len(s.filas), len(s.grupos), len(s.etiquetas), args.puerto, len(s.revisados), len(s.lista)), flush=True)
    servir(s, args.puerto)


if __name__ == "__main__":
    main()
