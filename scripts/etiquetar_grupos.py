"""Labels person / not-person by groups of similar-looking boxes instead of one box at a time.

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

    python scripts/etiquetar_grupos.py \
        --cajas  ../drone-geolocation/entrenamiento/valida_ident_cajas_vuelo2a.csv \
        --embs   ../drone-geolocation/entrenamiento/valida_ident_embs_vuelo2a.npy \
        --frames ../drone-geolocation/data/flight_01ago/20260801_184259/frames \
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

    python scripts/etiquetar_grupos.py ... \
        --contexto ../drone-geolocation/entrenamiento/identidad_cajas_02ago.csv \
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

Which flights it opens, what the labeller sees and what the page keeps:
docs/ARQUITECTURA.md.
"""
import argparse
import csv
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

ETIQUETAS = ("persona", "no")

_ENT = os.path.join("..", "drone-geolocation", "entrenamiento")
_DATOS = os.path.join("..", "drone-geolocation", "data", "flight_01ago")
VUELOS = {
    "2a": (os.path.join(_ENT, "valida_ident_cajas_vuelo2a.csv"), os.path.join(_ENT, "valida_ident_embs_vuelo2a.npy"),
           os.path.join(_DATOS, "20260801_184259", "frames"), os.path.join(_ENT, "grupos_vuelo2a.json")),
    "2b": (os.path.join(_ENT, "valida_ident_cajas_vuelo2b.csv"), os.path.join(_ENT, "valida_ident_embs_vuelo2b.npy"),
           os.path.join(_DATOS, "20260801_185326", "frames"), os.path.join(_ENT, "grupos_vuelo2b.json")),
}
_RAIZ_DATOS = os.path.join("..", "drone-geolocation", "data")
for _nombre, _frames in (("26jul", os.path.join(_RAIZ_DATOS, "20260726_195524", "frames")),
                         ("01ago_2a", os.path.join(_DATOS, "20260801_184259", "frames")),
                         ("01ago_2b", os.path.join(_DATOS, "20260801_185326", "frames")),
                         ("02ago", os.path.join(_RAIZ_DATOS, "flight_02ago", "20260802_133309", "frames")),
                         ("14jun", os.path.join(_RAIZ_DATOS, "flight_14jun", "20260614_225918", "frames")),
                         ("02ago_alto", os.path.join(_RAIZ_DATOS, "flight_02ago", "20260802_133309", "frames")),
                         ("02ago_huecos", os.path.join(_RAIZ_DATOS, "flight_02ago", "20260802_133309", "frames"))):
    VUELOS[_nombre] = (os.path.join(_ENT, "candidatas_%s.csv" % _nombre), os.path.join(_ENT, "candidatas_%s_embs.npy" % _nombre),
                       _frames, os.path.join(_ENT, "etiquetas_detector_%s.json" % _nombre),
                       os.path.join(_ENT, "candidatas_%s_frames.txt" % _nombre))
_PERSONA_POR_ALTURA = [(5.5, 183), (10.0, 168), (15.0, 109), (24.0, 62), (40.0, 42)]


def persona_px(alt):
    """Roughly how many pixels tall a person is at this altitude, read off the flights already labelled."""
    if alt <= _PERSONA_POR_ALTURA[0][0]:
        return _PERSONA_POR_ALTURA[0][1]
    for (a0, p0), (a1, p1) in zip(_PERSONA_POR_ALTURA, _PERSONA_POR_ALTURA[1:]):
        if alt <= a1:
            return int(round(p0 + (p1 - p0) * (alt - a0) / (a1 - a0)))
    return _PERSONA_POR_ALTURA[-1][1]


def _crear_seguidor(cv2):
    """The best correlation tracker this OpenCV has, by name rather than by version.

    CSRT lived in the main module, then in contrib, and OpenCV 5 dropped it from both; MIL is the one
    that has been there throughout. Asking for whichever exists keeps the labelling tool working on
    whatever the machine has installed instead of pinning it to one build.
    """
    for nombre in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        if hasattr(cv2, nombre):
            return getattr(cv2, nombre)()
    raise RuntimeError("este OpenCV no trae ningun seguidor (%s)" % cv2.__version__)


def _solapa(a, b):
    """Intersection over union of two boxes, used to leave alone what the reviewer already decided."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


MUESTRA = 24
LADO = 128
GRUESA = (0, 230, 255)
CONTEXTO = (255, 0, 255)
VECINA = (255, 255, 0)
REVISION = ("persona", "no", "duplicado", "ignorar")
REPASO = 0.05
SEMILLA_REPASO = 1234
LADO_MIN_NUEVA = 4


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


def iou_caja(a, b):
    """Intersection over union of two boxes: how much two boxes are the same box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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
        self.etiquetas = {}
        if os.path.exists(salida):
            previas = json.load(open(salida, encoding="utf-8"))["etiquetas"]
            local = {o: j for j, o in enumerate(self.orig)}
            self.etiquetas = {local[int(i)]: v for i, v in previas.items() if int(i) in local}
        self.grupos = {}
        for i, g in enumerate(self._agrupamiento(cajas, k)):
            self.grupos.setdefault(int(g), []).append(i)
        self.ruta_sospechas = sospechas or os.path.join(os.path.dirname(os.path.abspath(cajas)),
                                                        "olvidadas_%s.csv" % self.nombre)
        def dentro(f):
            return (desde is None or f >= desde) and (hasta is None or f <= hasta)
        self.lista = sorted({int(f) for f in list(lista_frames or []) + list(self.por_frame) if dentro(int(f))})
        self.ruta_revision = os.path.splitext(salida)[0] + "_frames.json"
        self.revisados, self.correcciones, self.nuevas, self.repaso, self.ajustes = set(), {}, {}, {}, {}
        self.pares_ok = set()
        if os.path.exists(self.ruta_revision):
            r = json.load(open(self.ruta_revision, encoding="utf-8"))
            local = {o: j for j, o in enumerate(self.orig)}
            self.revisados = {int(f) for f in r.get("revisados", [])}
            self.correcciones = {local[int(i)]: v for i, v in r.get("correcciones", {}).items() if int(i) in local}
            self.nuevas = {int(f): [[float(x) for x in c[:4]] + list(c[4:5]) for c in v] for f, v in r.get("nuevas", {}).items()}
            self.repaso = {int(f): int(n) for f, n in r.get("repaso", {}).items()}
            self.ajustes = {local[int(i)]: [float(x) for x in c] for i, c in r.get("ajustes", {}).items()
                            if int(i) in local}
            self.pares_ok = {tuple(sorted((local[int(a)], local[int(b)])))
                             for a, b in r.get("pares_ok", []) if int(a) in local and int(b) in local}

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
                        "muestra": [{"i": i, "etiqueta": self.final(i), "corregida": i in self.correcciones}
                                    for i in miembros[::paso][:MUESTRA]]})
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
                           "personas": sum(b["etiqueta"] == "persona" for b in d["cajas"])
                                       + sum(etiqueta_nueva(c) == "persona" for c in self.nuevas.get(f, [])),
                           "doble": any(b["doble"] for b in d["cajas"] + d["nuevas"]),
                           "sin": sum(b["etiqueta"] is None for b in d["cajas"]), "hueco": False})
        for k in range(1, len(frames) - 1):
            frames[k]["hueco"] = (frames[k]["personas"] == 0 and frames[k - 1]["personas"] > 0
                                  and frames[k + 1]["personas"] > 0)
        return {"frames": frames,
                "revisados": len(self.revisados & set(self.lista)), "nombre": self.nombre}

    def frame_cajas(self, f):
        """Everything the review page draws for frame f.

        A person box, proposed or drawn, is flagged "doble" when another person box of the same frame
        covers half of the smaller of the two: the same person boxed twice, or two people so close
        that the labeller should look again.
        """
        idx = self.por_frame.get(f, [])
        filas_persona = [i for i in idx if self.final(i) == "persona"]
        personas = [self._caja(i) for i in filas_persona]
        personas += [tuple(c[:4]) for c in self.nuevas.get(f, []) if etiqueta_nueva(c) == "persona"]

        def doble_de(i):
            """Whether box i shares its place with another person box that was not confirmed as a second person."""
            b = self._caja(i)
            return any(j != i and solape_menor(b, self._caja(j)) >= 0.5
                       and tuple(sorted((i, j))) not in self.pares_ok for j in filas_persona)

        def doble(b):
            return sum(solape_menor(b, p) >= 0.5 for p in personas) > 1 

        cajas = [{"i": i, "caja": self._caja(i), "etiqueta": self.final(i), "corregida": i in self.correcciones,
                  "ajustada": i in self.ajustes,
                  "conf": float(self.filas[i].get("conf") or 0), "fuentes": self.filas[i].get("fuentes") or "",
                  "doble": self.final(i) == "persona" and doble_de(i)} for i in idx]
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

    def seguir(self, f, caja, adelante=True, cuantos=40):
        """Carries a drawn box across the following frames with a visual tracker, so a person the
        detectors never proposed does not have to be drawn frame by frame.

        Somebody standing on a distant balcony is the case this exists for: the person barely moves,
        the drone does, and the box therefore has to move with the camera rather than stay put, which
        is why copying the same pixels to the neighbours does not work. A correlation tracker follows
        the patch instead, and stops as soon as it loses it, so what it leaves behind is a proposal
        the reviewer corrects rather than a claim.

        Boxes are only added where the frame has nothing of its own already overlapping: a frame that
        already carries a labelled person for that spot is left exactly as the reviewer left it.
        """
        import cv2
        if f not in self.lista:
            raise ValueError("frame fuera de la lista")
        caja = self._caja_valida(caja)
        pos = self.lista.index(f)
        orden = self.lista[pos + 1:pos + 1 + cuantos] if adelante else self.lista[max(0, pos - cuantos):pos][::-1]
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % f))
        if img is None:
            raise ValueError("no se pudo leer el frame %d" % f)
        seguidor = _crear_seguidor(cv2)
        x1, y1, x2, y2 = caja
        # OpenCV 5 wants whole pixels here, not the floats the review stores
        seguidor.init(img, (int(x1), int(y1), int(round(x2 - x1)), int(round(y2 - y1))))
        puestas, ultimo = 0, f
        for g in orden:
            im = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % g))
            if im is None:
                break
            ok, r = seguidor.update(im)
            if not ok:
                break
            nueva = [float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])]
            alto, ancho = im.shape[:2]
            if nueva[0] < 0 or nueva[1] < 0 or nueva[2] > ancho or nueva[3] > alto:
                break
            if any(_solapa(nueva, c) > 0.4 for c in self._cajas_de_persona(g)):
                ultimo = g
                continue
            try:
                self.nuevas.setdefault(g, []).append(self._caja_valida(nueva))
            except ValueError:
                break
            puestas += 1
            ultimo = g
        self._guardar_revision()
        return {"puestas": puestas, "hasta": ultimo, "frames": len(orden)}

    def _cajas_de_persona(self, f):
        """Every box the review currently calls a person in a frame, drawn ones included."""
        salida = []
        for i in self.por_frame.get(f, []):
            if self.final(i) == "persona":
                salida.append(list(self._caja(i)))
        for c in self.nuevas.get(f, []):
            if not (len(c) > 4 and c[4] == "ignorar"):
                salida.append(list(c[:4]))
        return salida

    def mover(self, f, k, caja):
        """Replaces a drawn box, which is what dragging one of its corners does: a box that came out
        too big or too small is fixed without drawing it again."""
        if not 0 <= k < len(self.nuevas.get(f, [])):
            raise ValueError("no hay caja dibujada %d en el frame %d" % (k, f))
        self.nuevas[f][k] = self._caja_valida(caja) + list(self.nuevas[f][k][4:5])
        self._guardar_revision()

    def propagar(self, i, ventana=12, umbral=0.7):
        """Gives the label of box i to the boxes that are the same box in the neighbouring frames.

        The camera moves little between two frames, so a wrong box repeats almost identical for a dozen
        frames: deciding it once and applying it to the ones that overlap by `umbral` is the difference
        between one click and twenty. Returns what changed, so the page can undo all of it at once.
        """
        v = self.final(i)
        if v is None:
            raise ValueError("esa caja todavia no tiene etiqueta")
        b = self._caja(i)
        f0 = int(self.filas[i]["frame"])
        if f0 not in self.lista:
            raise ValueError("frame fuera de la lista")
        k = self.lista.index(f0)
        cambiadas = []
        for f in self.lista[max(0, k - ventana):k + ventana + 1]:
            if f == f0:
                continue
            for j in self.por_frame.get(f, []):
                if self.final(j) == v or iou_caja(b, self._caja(j)) < umbral:
                    continue
                cambiadas.append([j, self.final(j)])
                self.correcciones[j] = v
        self._guardar_revision()
        return {"etiqueta": v, "cambiadas": len(cambiadas),
                "antes": [[self.orig[j], a] for j, a in cambiadas], "locales": [j for j, _ in cambiadas]}

    def resolver_doble(self, i):
        """Settles a pair of person boxes on the same person: the bigger one stays, the smaller becomes a
        duplicate. It is the decision that repeats most while reviewing, and it has one obvious answer."""
        b = self._caja(i)
        f = int(self.filas[i]["frame"])
        def area(c):
            return (c[2] - c[0]) * (c[3] - c[1])
        pareja = [j for j in self.por_frame.get(f, [])
                  if j != i and self.final(j) == "persona" and solape_menor(b, self._caja(j)) >= 0.5]
        if self.final(i) != "persona" or not pareja:
            raise ValueError("esa caja no es una persona encimada con otra")
        otra = max(pareja, key=lambda j: area(self._caja(j)))
        chica, grande = (i, otra) if area(b) <= area(self._caja(otra)) else (otra, i)
        antes = [[self.orig[chica], self.final(chica)], [self.orig[grande], self.final(grande)]]
        self.correcciones[chica] = "duplicado"
        self.correcciones[grande] = "persona"
        self._guardar_revision()
        return {"duplicado": self.orig[chica], "persona": self.orig[grande], "antes": antes,
                "locales": [chica, grande]}

    def resolver_frame(self, f, propagar=True):
        """Settles every pair of person boxes on the same person in frame f, and by default carries each
        decision to the neighbouring frames, where the same pair repeats almost identical.

        Reviewing flight 3 means 139 frames with a doubled box; one at a time that is 139 decisions, and
        the decision is always the same one.
        """
        antes, pares, propagadas = [], 0, 0
        while True:
            personas = [i for i in self.por_frame.get(f, []) if self.final(i) == "persona"]
            par = next(((i, j) for i in personas for j in personas
                        if i != j and solape_menor(self._caja(i), self._caja(j)) >= 0.5
                        and tuple(sorted((i, j))) not in self.pares_ok), None)
            if par is None:
                break
            r = self.resolver_grupo(par[0])
            antes += r["antes"]
            pares += 1
            if propagar:
                for chica in r["locales"]:
                    try:
                        p = self.propagar(chica)
                        antes += p["antes"]
                        propagadas += p["cambiadas"]
                    except ValueError:
                        pass
        return {"pares": pares, "propagadas": propagadas, "antes": antes}

    def etiquetar_encimadas(self, f=None, umbral=0.5):
        """Labels the boxes nobody labelled that lie on top of a box already decided.

        A box that covers half of a person already boxed is the same detection proposed twice, so it is a
        duplicate; one lying on something already ruled out is not a person either. Measured on flight 3,
        15 of the 30 boxes left without a label were of this kind: deciding them one by one is work that
        the overlap already answers. The ones that sit alone are left untouched: those need eyes.
        """
        frames = [f] if f is not None else list(self.lista)
        antes, cuenta = [], {"duplicado": 0, "no": 0}
        for fr in frames:
            decididas = [(self._caja(j), self.final(j)) for j in self.por_frame.get(fr, []) if self.final(j)]
            for i in self.por_frame.get(fr, []):
                if self.final(i) is not None:
                    continue
                b = self._caja(i)
                encima = [(solape_menor(b, c), e) for c, e in decididas if solape_menor(b, c) >= umbral]
                if not encima:
                    continue
                etiqueta = "duplicado" if max(encima)[1] == "persona" else "no"
                antes.append([self.orig[i], None])
                self.correcciones[i] = etiqueta
                cuenta[etiqueta] += 1
        self._guardar_revision()
        return {"duplicado": cuenta["duplicado"], "no": cuenta["no"], "antes": antes}

    def resolver_grupo(self, i):
        """Leaves one person standing in a pile of overlapping person boxes, however many there are.

        With three boxes on one person, settling them in pairs needs two decisions and can leave two
        standing; the group is what has one answer.
        """
        f = int(self.filas[i]["frame"])
        personas = [j for j in self.por_frame.get(f, []) if self.final(j) == "persona"]
        def area(j):
            return (self._caja(j)[2] - self._caja(j)[0]) * (self._caja(j)[3] - self._caja(j)[1])
        grupo, pendientes = [i], [i]
        while pendientes:
            a = pendientes.pop()
            for j in personas:
                if j not in grupo and solape_menor(self._caja(a), self._caja(j)) >= 0.5:
                    grupo.append(j)
                    pendientes.append(j)
        if len(grupo) < 2:
            raise ValueError("esa caja no esta encimada con otra persona")
        queda = max(grupo, key=area)
        antes = [[self.orig[j], self.final(j)] for j in grupo]
        for j in grupo:
            if j != queda:
                self.correcciones[j] = "duplicado"
        self.correcciones[queda] = "persona"
        self._guardar_revision()
        return {"grupo": len(grupo), "persona": self.orig[queda], "antes": antes,
                "locales": [j for j in grupo if j != queda]}

    def dos_personas(self, f):
        """Says that every pair of overlapping person boxes of frame f is two people standing together.

        It is the other answer to the same question the orange flag asks, and without it a real pair of
        people keeps the frame marked as a problem for ever.
        """
        personas = [i for i in self.por_frame.get(f, []) if self.final(i) == "persona"]
        nuevos = [(i, j) for k, i in enumerate(personas) for j in personas[k + 1:]
                  if solape_menor(self._caja(i), self._caja(j)) >= 0.5 and tuple(sorted((i, j))) not in self.pares_ok]
        if not nuevos:
            raise ValueError("en este frame no hay dos cajas de persona encimadas")
        for i, j in nuevos:
            self.pares_ok.add(tuple(sorted((i, j))))
        self._guardar_revision()
        return {"pares": len(nuevos), "filas": [[self.orig[i], self.orig[j]] for i, j in nuevos]}

    def deshacer_dos_personas(self, f):
        """Takes back the pairs of frame f confirmed as two people, so undo can put the flag back."""
        personas = [i for i in self.por_frame.get(f, []) if self.final(i) == "persona"]
        quitados = [p for p in list(self.pares_ok) if p[0] in personas and p[1] in personas]
        for p in quitados:
            self.pares_ok.discard(p)
        self._guardar_revision()
        return {"pares": len(quitados)}

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

    def alturas(self, cada=5):
        """The flight's altitude against its frames, with what is already reviewed marked on it.

        Choosing where to label has been a number typed into a command line by whoever happened to be
        looking, and that is how a stretch at three metres got proposed and a stretch at twenty-five
        did not. The flight's own shape answers it: altitude over time says where the drone was high,
        and the marks say what has already been taken, so the choice is made on what is there instead
        of on a memory of it.
        """
        ruta = os.path.join(os.path.dirname(os.path.abspath(self.frames)), "frames.csv")
        if not os.path.exists(ruta):
            return {"hay": False, "motivo": "este vuelo no tiene frames.csv al lado de la carpeta de frames"}
        alt = {}
        with open(ruta, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    alt[int(r["frame"])] = float(r["alt_agl"])
                except (KeyError, ValueError):
                    pass
        if not alt:
            return {"hay": False, "motivo": "frames.csv no trae alt_agl"}
        fs = sorted(alt)[::max(1, cada)]
        enlista = set(self.lista)
        return {"hay": True, "vuelo": self.nombre, "cada": cada,
                "frames": fs, "alt": [round(alt[f], 1) for f in fs],
                "en_lista": [f in enlista for f in fs],
                "revisado": [f in self.revisados for f in fs],
                "total": len(alt), "max": round(max(alt.values()), 1)}

    def tramo(self, desde, hasta):
        """What labelling a stretch would buy: how much of it is new, and how big a person looks there."""
        ruta = os.path.join(os.path.dirname(os.path.abspath(self.frames)), "frames.csv")
        alt = {}
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        alt[int(r["frame"])] = float(r["alt_agl"])
                    except (KeyError, ValueError):
                        pass
        fs = [f for f in sorted(alt) if desde <= f <= hasta]
        if not fs:
            return {"frames": 0}
        alturas = sorted(alt[f] for f in fs)
        mediana = alturas[len(alturas) // 2]
        aire = [f for f in fs if alt[f] > 3.0]
        return {"frames": len(fs), "en_aire": len(aire), "alt_mediana": round(mediana, 1),
                "alt_max": round(max(alturas), 1),
                "ya_en_lista": len([f for f in fs if f in set(self.lista)]),
                "ya_revisados": len([f for f in fs if f in self.revisados]),
                "persona_px": persona_px(mediana),
                "segundos": None}

    def tira_datos(self, desde, hasta, columnas=8, filas=4):
        """Where to cut, and what is in each cut, for a run of frames seen all at once.

        Reviewing frame by frame hides the kind of mistake that only shows up as a pattern: a box that
        drifts off a person over twenty frames looks fine in each one. The patch is centred on the
        median of the people the review knows about in the run, and made wide enough that they can move
        inside it, so the same ground is shown in every cell and the eye compares like with like.
        """
        frames = [f for f in self.lista if desde <= f <= hasta]
        if not frames:
            return {"frames": [], "centro": [0, 0], "lado": 400, "personas": {}}
        cajas = [c for f in frames for c in self._personas_del_frame(f)]
        if cajas:
            centro = [float(np.median([(c[0] + c[2]) / 2 for c in cajas])),
                      float(np.median([(c[1] + c[3]) / 2 for c in cajas]))]
            lado = int(max(300, 9 * np.median([c[3] - c[1] for c in cajas])))
        else:
            centro, lado = [960.0, 540.0], 900
        n = min(columnas * filas, len(frames))
        elegidos = ([frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)]
                    if n > 1 else list(frames[:1]))
        return {"frames": elegidos, "centro": centro, "lado": lado, "total": len(frames),
                "personas": {str(f): len(self._personas_del_frame(f)) for f in elegidos},
                "revisados": {str(f): (f in self.revisados) for f in elegidos}}

    def _personas_del_frame(self, f):
        """Every box the review calls a person in a frame, the drawn ones included."""
        salida = []
        for i in self.por_frame.get(f, []):
            if self.final(i) == "persona":
                salida.append(list(self._caja(i)))
        for c in self.nuevas.get(f, []):
            if not (len(c) > 4 and c[4] == "ignorar"):
                salida.append([float(x) for x in c[:4]])
        return salida

    def celda(self, f, cx, cy, lado, tam=190):
        """One cell of the strip: frame f with every person the review knows drawn on it.

        A side of zero means the whole frame instead of a patch of it. The patch answers whether a box
        follows its person, because every cell then shows the same ground; it cannot answer whether
        somebody is missing, because anyone standing outside the patch is invisible and the cell looks
        just as calm as one where nobody was missed. Those are two different questions and the view has
        to be told which one is being asked.
        """
        import cv2
        img = cv2.imread(os.path.join(self.frames, "frame_%04d.jpg" % f))
        if img is None:
            raise ValueError("no hay frame %d" % f)
        alto, ancho = img.shape[:2]
        if lado <= 0:
            entero = img.copy()
            for c in self._personas_del_frame(f):
                cv2.rectangle(entero, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])),
                              (0, 235, 0), max(2, ancho // 320))
            ok, buf = cv2.imencode(".jpg", cv2.resize(entero, (tam, max(1, int(tam * alto / ancho)))),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("no se pudo codificar el frame %d" % f)
            return buf.tobytes()
        lado = int(max(60, min(lado, min(alto, ancho))))
        x0 = int(np.clip(cx - lado / 2, 0, ancho - lado))
        y0 = int(np.clip(cy - lado / 2, 0, alto - lado))
        rec = img[y0:y0 + lado, x0:x0 + lado].copy()
        for c in self._personas_del_frame(f):
            x1, y1, x2, y2 = c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0
            if x2 < 0 or y2 < 0 or x1 > lado or y1 > lado:
                continue
            cv2.rectangle(rec, (int(x1), int(y1)), (int(x2), int(y2)), (0, 235, 0), max(1, lado // 220))
        ok, buf = cv2.imencode(".jpg", cv2.resize(rec, (tam, tam)), [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError("no se pudo codificar el frame %d" % f)
        return buf.tobytes()

    def _agrupamiento(self, cajas, k):
        """The grouping of the boxes, computed once per flight and remembered on disk.

        Hierarchical clustering over four thousand 512-dimensional embeddings takes sixteen seconds,
        and it ran on every start and on every flight change even though nothing it depends on ever
        moves: the embeddings are written once by proponer_cajas.py and never touched again. The
        answer is kept next to them and reused while the shape of the embeddings and the number of
        groups still match what produced it, which is what makes switching flights feel instant.
        """
        cache = os.path.splitext(cajas)[0] + "_grupos.npz"
        firma = np.array([self.emb.shape[0], self.emb.shape[1], k], dtype=np.int64)
        if os.path.exists(cache):
            try:
                d = np.load(cache)
                if np.array_equal(d["firma"], firma):
                    return d["grupos"]
            except Exception:
                pass
        grupos = np.asarray(agrupar(self.emb, k))
        try:
            np.savez(cache, grupos=grupos, firma=firma)
        except OSError:
            pass
        return grupos

    def _guardar_revision(self):
        datos = {"cajas": os.path.abspath(self.cajas), "clave": "fila del CSV de cajas",
                 "revisados": sorted(self.revisados),
                 "correcciones": {str(self.orig[i]): v for i, v in sorted(self.correcciones.items())},
                 "nuevas": {str(f): v for f, v in sorted(self.nuevas.items())},
                 "repaso": {str(f): n for f, n in sorted(self.repaso.items())},
                 "ajustes": {str(self.orig[i]): c for i, c in sorted(self.ajustes.items())},
                 "pares_ok": sorted([self.orig[a], self.orig[b]] for a, b in self.pares_ok)}
        tmp = self.ruta_revision + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(datos, fh, indent=1)
        os.replace(tmp, self.ruta_revision)
        self._anotar_historia(datos)

    def _anotar_historia(self, datos):
        """Appends one line per save with what the review contains, and keeps a copy of the file.

        Labelling mistakes are found long after they are made, and until now the only thing on disk was
        the current answer: there was no way to ask when a box became a duplicate or how many people the
        truth held yesterday. A line per save answers the first question by bracketing the change
        between two counts, and the kept copies answer the second by being openable.

        Copies are thinned as they age -- every save of the last hour, then one per hour, then one per
        day -- so a long session does not leave thousands of files and a week-old state is still there.
        """
        import time
        resumen = {"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "revisados": len(datos["revisados"]),
                   "correcciones": len(datos["correcciones"]),
                   "ajustes": len(datos["ajustes"]),
                   "dibujadas": sum(len(v) for v in datos["nuevas"].values()),
                   "personas": sum(1 for i in self.por_frame for j in self.por_frame[i]
                                   if self.final(j) == "persona")
                              + sum(1 for v in datos["nuevas"].values() for c in v
                                    if not (len(c) > 4 and c[4] == "ignorar"))}
        with open(self.ruta_revision.replace(".json", "_historia.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(resumen) + chr(10))
        copias = self.ruta_revision.replace(".json", "_copias")
        os.makedirs(copias, exist_ok=True)
        with open(os.path.join(copias, time.strftime("%Y%m%d_%H%M%S") + ".json"), "w", encoding="utf-8") as fh:
            json.dump(datos, fh, indent=1)
        self._ralear_copias(copias)

    @staticmethod
    def _ralear_copias(carpeta):
        """Keeps every copy of the last hour, one per hour of the last day, and one per day before that."""
        import time
        ahora = time.time()
        vistos, borrar = set(), []
        for nombre in sorted(os.listdir(carpeta), reverse=True):
            ruta = os.path.join(carpeta, nombre)
            edad = ahora - os.path.getmtime(ruta)
            if edad < 3600:
                continue
            cubo = nombre[:11] if edad < 86400 else nombre[:8]      # AAAAMMDD_HH or AAAAMMDD
            if cubo in vistos:
                borrar.append(ruta)
            else:
                vistos.add(cubo)
        for ruta in borrar:
            try:
                os.remove(ruta)
            except OSError:
                pass

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
                continue
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
        mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
        h, w = img.shape[:2]
        ox, oy = max(0, int(x1 - mx)), max(0, int(y1 - my))
        c = img[oy:min(h, int(y2 + my)), ox:min(w, int(x2 + mx))]
        esc = LADO / float(max(c.shape[:2]))
        c = cv2.resize(c, (max(1, int(c.shape[1] * esc)), max(1, int(c.shape[0] * esc))))
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
<a href="/frames" style="color:#93c5fd">todos, uno por uno</a> &middot; <a href="/chequeos" style="color:#93c5fd">chequeos</a> &middot; <a href="/tira">tira</a> &middot; <a href="/plan">donde etiquetar</a> &middot; <a href="/vuelos">vuelos</a> &middot;
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
  <button onclick="encimadasTodo()">resolver en TODO el vuelo las sin etiquetar encimadas</button>
  <label><input type="checkbox" id="ocultarNo" onchange="pintar()"> ocultar las "no"</label>
  <label><input type="checkbox" id="soloPersonas" onchange="pintar()"> solo personas</label>
  <label><input type="checkbox" id="propagarAuto" checked> al resolver, copiar a los vecinos</label>
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
    <b>Clic derecho</b>: en una caja dibujada la borra; en una del detector la pasa a <b>duplicado</b>
    (las del detector no se borran, se marcan). Con el raton encima de una dibujada,
    <span class="k">I</span> la pasa a ignorar y <span class="k">P</span> la devuelve a persona.
    <span class="k">R</span> desmarca revisado. <span class="k">Z</span> deshace lo ultimo.</p>
  <p><b>Cuando dos cajas verdes se pisan</b> (naranja "&iquest;doble?") hay solo dos respuestas:<br>
    <span class="k">1</span> <b>es la misma persona</b>: la caja chica pasa a duplicado, aqui y en los frames vecinos.<br>
    <span class="k">2</span> <b>son dos personas distintas</b>: las dos quedan y el frame deja de marcarse en naranja.<br>
    Funcionan sin poner el raton encima. <span class="k">A</span> hace lo mismo que <span class="k">1</span>.</p>
  <p><b>Cuando hay varias cajas encimadas y querés elegir una</b>: <span class="k">Tab</span> pasa de una a la
    siguiente del monton (queda con borde blanco "elegida") y <span class="k">P</span> <span class="k">N</span>
    <span class="k">D</span> <span class="k">I</span> actuan sobre esa, sin depender del raton. Al abrir un frame
    queda elegida sola la primera caja sin etiquetar.<br>
    <span class="k">3</span> etiqueta las que estan <b>encimadas con una ya decidida</b>: si pisa a una persona es
    duplicado, si pisa a una "no" es no. El boton de la barra lo hace en todo el vuelo.</p>
  <p><b>Para ir mas rapido</b>, con el raton sobre una caja:<br>
    <span class="k">X</span> resuelve el par de ESA caja (sin raton encima, resuelve todo el frame).<br>
    <span class="k">C</span> copia la etiqueta de esa caja a las cajas iguales de los frames vecinos.<br>
    <span class="k">S</span> sobre una caja que <b>dibujaste vos</b>: un seguidor la arrastra por los 40 frames
    siguientes y la dibuja en cada uno, para no dibujar a mano a alguien que ningun detector propuso
    (con <span class="k">Shift</span> va hacia atras). Deja propuestas: revisalas y corregi las que se desviaron.<br>
    <span class="k">F</span> salta al proximo frame con cajas encimadas o sin etiquetar. Todo se deshace con <span class="k">Z</span>.</p>
  <p>Un frame esta <b>revisado</b> cuando cada persona tiene UNA caja verde y nada mas es verde.
    Una caja corrida (piernas, sombra, media persona) de alguien que ya tiene la suya es <b>duplicado</b>.</p>
  <p><b><a href="/frames?solo=pendientes">cola de pendientes &rarr;</a></b> (huecos, sin revisar, sin etiquetar y
    encimadas, uno tras otro)</p>
  <p><a href="/">&larr; grupos</a> &middot; <a href="/frames?solo=dobles">solo dobles</a> &middot; <a href="/mosaico">mosaico</a>
    &middot; <a href="/chequeos">chequeos</a> &middot; <a href="/tira">tira</a> &middot; <a href="/plan">donde etiquetar</a> &middot; <a href="/vuelos">vuelos</a> &middot; <a href="/repaso">repaso ciego</a>
    &middot; <a href="/video">video</a> &middot; <a href="/sospechas">sospechas</a></p>
</div>
<script>
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
const lupa = document.getElementById('lupa'), lx = lupa.getContext('2d');
const COLOR = {persona: '#22c55e', no: '#ef4444', duplicado: '#9ca3af', ignorar: '#60a5fa'};
const SIGUIENTE = {persona: 'no', no: 'persona', duplicado: 'persona', ignorar: 'persona'};
const TECLA = {p: 'persona', n: 'no', d: 'duplicado', i: 'ignorar'};
let lista = [], pos = 0, datos = null, img = new Image(), abajo = null, raton = null, nombre = '', deshacer = [];
let elegida = null;   // the box the keys act on when Tab was used to pick it out of a pile

async function pedir(ruta, cuerpo) {
  const r = await fetch(ruta, cuerpo === undefined ? {} :
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cuerpo)});
  const d = await r.json();
  if (!r.ok) { aviso(d.error); throw new Error(d.error); }
  return d;
}
function aviso(t) { document.getElementById('aviso').textContent = t || ''; }
const ocultarNo = () => document.getElementById('ocultarNo').checked;
const soloPersonas = () => document.getElementById('soloPersonas').checked;
// What is drawn and what can be clicked: with "solo personas" everything that is not a person gets out of the way.
const visible = b => !(soloPersonas() && b.etiqueta && b.etiqueta !== 'persona') && !(ocultarNo() && b.etiqueta === 'no');

async function iniciar() {
  const e = await pedir('/frames/estado');
  // ?solo=dobles: only the frames where a person has two boxes, the ones that need a decision.
  const solo = new URLSearchParams(location.search).get('solo');
  // "pendientes" is the queue: everything still to decide, in order, so Enter walks through all of it.
  const pendiente = x => x.doble || x.sin || x.hueco || !x.revisado;
  lista = solo === 'dobles' ? e.frames.filter(x => x.doble)
        : solo === 'pendientes' ? e.frames.filter(pendiente) : e.frames;
  if (!lista.length) { lista = e.frames; aviso('no queda nada pendiente: se muestran todos los frames'); }
  nombre = e.nombre; document.title = nombre + (solo ? ' - ' + solo : ' - frames');
  if (solo === 'dobles') document.getElementById('modo').textContent = 'solo frames con una persona en dos cajas: ' + lista.length;
  if (solo === 'pendientes') document.getElementById('modo').textContent =
    'cola de pendientes: ' + lista.length + ' frames (Enter los va cerrando uno tras otro)';
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
  datos = d; aviso('');
  const sin = d.cajas.find(b => b.etiqueta === null);      // the first unlabelled box comes picked already
  elegida = sin ? sin.i : null;
  pintar();
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
    if (!visible(b)) return;
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
  if (elegida !== null) {                        // the picked box, so the keys are not a guess
    const b = datos.cajas.find(x => x.i === elegida);
    if (b) rect(b.caja, '#ffffff', 4, [6, 4], 'elegida');
  }
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
  const x = lista[pos], porque = [x.hueco ? 'HUECO: nadie aqui y si en los vecinos' : '',
                                  x.doble ? 'cajas encimadas' : '', x.sin ? x.sin + ' sin etiquetar' : '',
                                  x.revisado ? '' : 'sin revisar'].filter(Boolean).join(' · ');
  document.getElementById('donde').textContent = (pos + 1) + ' / ' + lista.length +
    (datos.revisado ? ' · revisado' : '') + (porque ? ' · ' + porque : '');
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
    if (!visible(b)) return;
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
async function encimadasTodo() {
  const r = await pedir('/frames/encimadas', {f: null});
  deshacer.push(async () => { for (const [j, v] of r.antes) await pedir('/frames/caja', {i: j, v}); });
  aviso('en todo el vuelo: ' + r.duplicado + ' pasaron a duplicado y ' + r.no + ' a no; las que estaban solas siguen sin etiquetar');
  recargar();
}
// Two overlapping person boxes have two possible answers, and only two: one key for each.
async function esLaMisma() {
  try {
    const r = await pedir('/frames/resolver_todo', {f: datos.f, propagar: document.getElementById('propagarAuto').checked});
    if (!r.pares) aviso('en este frame no hay cajas de persona encimadas');
    else {
      deshacer.push(async () => { for (const [j, v] of r.antes) await pedir('/frames/caja', {i: j, v}); });
      aviso('la misma persona: ' + r.pares + ' pares resueltos aqui y ' + r.propagadas + ' en los frames vecinos');
    }
  } catch (e) {}
  recargar();
}
async function sonDos() {
  const f = datos.f;
  try {
    const r = await pedir('/frames/dos_personas', {f});
    deshacer.push(() => pedir('/frames/deshacer_dos_personas', {f}));
    aviso('dos personas distintas: ' + r.pares + ' pares dejan de marcarse en naranja');
  } catch (e) {}
  recargar();
}
async function irAlProblema() {
  const e = await pedir('/frames/estado');     // asked again: what is a problem changes as you fix them
  lista = lista.map((x, k) => e.frames[k] || x);
  const quedan = lista.filter(x => x.doble || x.sin).length;
  for (let k = 1; k <= lista.length; k++) {
    const q = (pos + k) % lista.length;
    if (lista[q].doble || lista[q].sin) { await ir(q); return aviso('quedan ' + quedan + ' frames con algo que resolver'); }
  }
  aviso('no quedan frames con cajas encimadas ni sin etiquetar');
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
  // Only the boxes that count as a person can be grabbed: a duplicate lying on top of one used to steal
  // the corner and get moved instead of the box being fixed.
  const todas = datos.nuevas.filter(b => b.etiqueta !== 'ignorar').map(b => ({tipo: 'nueva', k: b.k, caja: b.caja}))
    .concat(datos.cajas.filter(b => visible(b) && (b.etiqueta === 'persona' || !b.etiqueta))
                       .map(b => ({tipo: 'caja', i: b.i, caja: b.caja})));
  for (const b of todas) {
    const [x1, y1, x2, y2] = b.caja;
    for (const [x, y, fx, fy] of [[x1, y1, x2, y2], [x2, y1, x1, y2], [x1, y2, x2, y1], [x2, y2, x1, y1]])
      if (Math.abs(p.x - x) < cerca && Math.abs(p.y - y) < cerca)
        return Object.assign({fija: {x: fx, y: fy}}, b);
  }
  return null;
}
const MOVER_MIN = 10;   // screen px before a press counts as a drag: below that it is a click, not a move
cv.addEventListener('mousedown', ev => {
  if (ev.button !== 0 || !datos) return;
  const p = punto(ev);
  // The corner is only remembered; the drag starts when the mouse actually moves, so a click near an
  // edge changes the label instead of resizing the box by accident.
  abajo = {x: p.x, y: p.y, arrastrando: false, redim: esquinaDe(p)};
});
cv.addEventListener('mousemove', ev => {
  raton = punto(ev);
  if (abajo && Math.hypot(raton.x - abajo.x, raton.y - abajo.y) * cv.width / img.naturalWidth > MOVER_MIN) abajo.arrastrando = true;
  if (!abajo) cv.style.cursor = esquinaDe(raton) ? 'nwse-resize' : 'crosshair';   // the cursor says what a drag would do
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
  if (!b) return aviso('clic derecho sobre una caja: la dibujada se borra, la del detector pasa a duplicado');
  if (b.tipo === 'nueva') {
    const f = datos.f, caja = b.caja;
    await pedir('/frames/borrar', {f, k: b.k});
    deshacer.push(() => pedir('/frames/nueva', {f, caja}));
    return recargar();
  }
  // A box of the CSV cannot be deleted: its row stays. Right-click marks it "duplicado", which is what
  // takes it out of the truth without pretending the detector never proposed it.
  poner(b.i, 'duplicado');
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
  else if (k === 'f') irAlProblema();
  else if (k === 'a' || k === '1') await esLaMisma();
  else if (k === '2') await sonDos();
  else if (k === '3') {                   // the boxes the overlap already answers, in one go
    try {
      const r = await pedir('/frames/encimadas', {f: datos.f});
      if (!r.duplicado && !r.no) aviso('en este frame no hay cajas sin etiquetar encimadas con otra');
      else {
        deshacer.push(async () => { for (const [j, v] of r.antes) await pedir('/frames/caja', {i: j, v}); });
        aviso('resueltas por encima: ' + r.duplicado + ' a duplicado y ' + r.no + ' a no');
      }
    } catch (e) {}
    recargar();
  }
  else if (k === 'c' && raton) {          // the same decision, on the same box, in the neighbouring frames
    const b = bajo(raton);
    if (!b || b.tipo !== 'caja') return aviso('C: poné el raton sobre la caja que querés propagar');
    try {
      const r = await pedir('/frames/propagar', {i: b.i});
      deshacer.push(async () => { for (const [j, v] of r.antes) await pedir('/frames/caja', {i: j, v}); });
      aviso('"' + r.etiqueta + '" aplicado a ' + r.cambiadas + ' cajas iguales de los frames vecinos');
    } catch (e) {}
    recargar();
  }
  else if (k === 's') {
    // S over a drawn box: a tracker carries it forward so a person nobody proposed is drawn once,
    // not once per frame. With shift it walks backwards instead.
    const b = raton && bajo(raton);
    if (!b || b.tipo !== 'nueva') { aviso('pone el raton sobre una caja que dibujaste y apreta S'); return; }
    aviso('siguiendo la caja...');
    try {
      const r = await pedir('/frames/seguir', {f: datos.f, caja: b.caja, adelante: !ev.shiftKey, cuantos: 40});
      aviso('puestas ' + r.puestas + ' cajas, hasta el frame ' + r.hasta + ' (revisalas y corregi las que se desviaron)');
    } catch (e) { aviso('el seguidor no pudo: ' + e); }
    recargar();
  }
  else if (k === 'g') { ev.preventDefault(); document.getElementById('saltar').focus(); }
  else if (k === 'r') { await pedir('/frames/revisado', {f: datos.f, v: false}); lista[pos].revisado = false; recargar(); }
  else if (k === 'x') {
    const b = raton && bajo(raton);
    if (b && b.tipo === 'caja') {          // a precise choice: settle the pair of THIS box
      try {
        const r = await pedir('/frames/resolver', {i: b.i});
        deshacer.push(async () => { for (const [j, v] of r.antes) await pedir('/frames/caja', {i: j, v}); });
        aviso('la chica quedo duplicado y la grande persona');
      } catch (e) {}
      recargar();
    } else await esLaMisma();              // no box under the cursor: do the whole frame instead of nothing
  }
  else if (ev.key === 'Tab') {            // cycle through the boxes piled under the cursor, or all of them
    ev.preventDefault();
    const monton = (raton ? datos.cajas.filter(b => visible(b) && raton.x >= b.caja[0] && raton.x <= b.caja[2]
                                                    && raton.y >= b.caja[1] && raton.y <= b.caja[3])
                          : datos.cajas.filter(visible));
    if (!monton.length) return aviso('no hay cajas donde elegir');
    const k0 = monton.findIndex(b => b.i === elegida);
    elegida = monton[(k0 + 1) % monton.length].i;
    const b = monton[(k0 + 1) % monton.length];
    aviso('elegida la caja ' + b.i + ' (' + (b.etiqueta || 'sin etiquetar') + '), ' + monton.length + ' encimadas: Tab pasa a la siguiente');
    pintar();
  }
  else if (TECLA[k] && elegida !== null && (!raton || !bajo(raton))) poner(elegida, TECLA[k]);
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
window.addEventListener('focus', () => { if (datos) recargar(); });   // back from another tab: redraw what changed
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
<p><a href="/vuelos">&larr; elegir otro vuelo</a> &middot; <a href="/tira">tira</a> &middot;
   <a href="/frames">frame por frame</a></p>
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
 &middot; <a href="/chequeos">chequeos</a> &middot; <a href="/tira">tira</a> &middot; <a href="/plan">donde etiquetar</a> &middot; <a href="/vuelos">vuelos</a> &middot; <a href="/sospechas">sospechas del modelo</a></p>
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
<p><a href="/">&larr; grupos</a> &middot; <a href="/video">video</a> &middot; <a href="/chequeos">chequeos</a> &middot; <a href="/tira">tira</a> &middot; <a href="/plan">donde etiquetar</a> &middot; <a href="/vuelos">vuelos</a></p>
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
<p><a href="/">&larr; grupos</a> &middot; <a href="/chequeos">chequeos</a> &middot; <a href="/tira">tira</a> &middot; <a href="/plan">donde etiquetar</a> &middot; <a href="/vuelos">vuelos</a></p>
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


OPCIONES = {}


def abrir_vuelo(nombre):
    """Builds the session of another flight with the options this run was started with.

    Restarting the tool to look at a second flight is what made mistakes easy to miss: comparing how a
    person was boxed on two days meant killing the server, and nobody does that in the middle of a
    review. The picker swaps the session instead; each flight keeps writing to its own files, so there
    is nothing to lose in the swap.
    """
    if nombre not in VUELOS or len(VUELOS[nombre]) < 5:
        raise ValueError("no conozco el vuelo %r" % nombre)
    cajas, embs, frames, salida, lista = VUELOS[nombre][:5]
    if not os.path.exists(cajas):
        raise ValueError("el vuelo %s no tiene candidatas todavia" % nombre)
    return Sesion(cajas, np.load(embs), frames, salida, OPCIONES.get("grupos", 30),
                  None, None, identidad=OPCIONES.get("identidad", False),
                  contexto=OPCIONES.get("contexto"), contexto_etiquetas=OPCIONES.get("contexto_etiquetas"),
                  lista_frames=[int(l) for l in open(lista)] if os.path.exists(lista) else None,
                  nombre=nombre, sospechas=OPCIONES.get("sospechas"))


def resumen_vuelo(nombre):
    """What a flight's labelling looks like, read from disk without loading it.

    The flight picker has to say what is done and what is not, and loading a whole session to answer
    that would cost the embeddings of every flight on every page view. Only the files are read.
    """
    cajas, _embs, frames, salida = VUELOS[nombre][:4]
    lista_txt = VUELOS[nombre][4] if len(VUELOS[nombre]) > 4 else None
    d = {"vuelo": nombre, "cajas": 0, "etiquetadas": 0, "personas": 0, "dibujadas": 0,
         "frames": 0, "revisados": 0, "existe": os.path.exists(cajas)}
    if not d["existe"]:
        return d
    with open(cajas, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    d["cajas"] = len(filas)
    etiquetas, revision = {}, {}
    if os.path.exists(salida):
        etiquetas = json.load(open(salida, encoding="utf-8")).get("etiquetas", {})
    ruta_rev = salida.replace(".json", "_frames.json")
    if os.path.exists(ruta_rev):
        revision = json.load(open(ruta_rev, encoding="utf-8"))
    def final(i):
        return revision.get("correcciones", {}).get(str(i), etiquetas.get(str(i)))
    d["etiquetadas"] = sum(1 for i in range(len(filas)) if final(i) is not None)
    d["personas"] = sum(1 for i in range(len(filas)) if final(i) == "persona")
    d["dibujadas"] = sum(len(v) for v in revision.get("nuevas", {}).values())
    d["personas"] += sum(1 for v in revision.get("nuevas", {}).values() for c in v
                         if not (len(c) > 4 and c[4] == "ignorar"))
    lista = {int(l) for l in open(lista_txt)} if lista_txt and os.path.exists(lista_txt) else set()
    lista |= {int(r["frame"]) for r in filas}
    d["frames"] = len(lista)
    d["revisados"] = len({int(x) for x in revision.get("revisados", [])} & lista)
    d["historia"] = salida.replace(".json", "_frames_historia.jsonl")
    return d


PAGINA_PLAN = """<!doctype html><meta charset="utf-8"><title>donde etiquetar</title>
<style>body{background:#0b1020;color:#e5e7eb;font:14px system-ui;margin:0;padding:18px}
h1{font-size:18px;margin:0 0 4px} p.s{color:#9ca3af;margin:0 0 12px;max-width:1000px}
svg{background:#0f172a;border-radius:8px;cursor:crosshair;touch-action:none}
a{color:#93c5fd;text-decoration:none} code{background:#111827;padding:3px 7px;border-radius:5px;display:inline-block}
table{border-collapse:collapse;margin-top:10px} td{padding:3px 14px 3px 0} td:first-child{color:#9ca3af}
.g{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px} .g img{border:2px solid #1f2937;border-radius:4px}</style>
<h1>Donde etiquetar: el vuelo visto por su altura</h1>
<p class="s">Cada punto es un frame. <b style="color:#4ade80">Verde</b>: ya revisado en ESTE vuelo.
<b style="color:#60a5fa">Azul</b>: esta en su lista pero sin revisar. <b style="color:#6b7280">Gris</b>: nunca se
propuso. La linea punteada son 12 m, por debajo de la cual la persona se ve grande y ya tenemos de sobra.
<b>Arrastra sobre el grafico</b> para elegir un tramo: abajo te digo cuanto material nuevo trae, de que tamano
se veria la persona ahi, y el comando exacto para proponer sus cajas.</p>
<div id="w"></div>
<div id="r"></div>
<p style="margin-top:14px"><a href="/vuelos">vuelos</a> &middot; <a href="/tira">tira</a> &middot;
   <a href="/video">video</a> &middot; <a href="/frames">frame por frame</a></p>
<script>
const W = 1180, H = 260, M = 34;
let D = null, a0 = null, a1 = null;
async function cargar() {
  D = await (await fetch('/plan.json')).json();
  if (!D.hay) { document.getElementById('w').textContent = D.motivo; return; }
  const n = D.frames.length, fmin = D.frames[0], fmax = D.frames[n - 1], amax = Math.max(12, D.max) * 1.1;
  const X = f => M + (f - fmin) / (fmax - fmin) * (W - M - 10);
  const Y = a => H - M - a / amax * (H - M - 12);
  let pts = '';
  for (let i = 0; i < n; i++) {
    const c = D.revisado[i] ? '#4ade80' : (D.en_lista[i] ? '#60a5fa' : '#6b7280');
    pts += `<rect x="${X(D.frames[i]).toFixed(1)}" y="${Y(D.alt[i]).toFixed(1)}" width="2" height="2" fill="${c}"/>`;
  }
  let ejes = `<line x1="${M}" y1="${Y(12)}" x2="${W - 10}" y2="${Y(12)}" stroke="#facc15" stroke-dasharray="4 4" opacity=".6"/>
    <text x="${W - 60}" y="${Y(12) - 4}" fill="#facc15" font-size="11">12 m</text>`;
  for (const a of [0, 10, 20, 30, 40]) if (a <= amax) ejes += `<text x="4" y="${Y(a) + 4}" fill="#6b7280" font-size="11">${a}</text>`;
  ejes += `<text x="${M}" y="${H - 8}" fill="#6b7280" font-size="11">frame ${fmin}</text>
           <text x="${W - 90}" y="${H - 8}" fill="#6b7280" font-size="11">${fmax}</text>`;
  document.getElementById('w').innerHTML =
    `<svg id="g" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${ejes}${pts}<rect id="sel" fill="#3b82f6" opacity=".25" y="0" height="${H - M}" width="0" x="0"/></svg>
     <p style="color:#9ca3af">vuelo <b>${D.vuelo}</b>, ${D.total} frames, altura maxima ${D.max} m
     &nbsp;(un punto cada ${D.cada} frames)</p>`;
  const g = document.getElementById('g');
  const fde = ev => { const b = g.getBoundingClientRect();
    const x = (ev.clientX - b.left) * W / b.width;
    return Math.round(fmin + Math.max(0, Math.min(1, (x - M) / (W - M - 10))) * (fmax - fmin)); };
  g.onpointerdown = ev => { a0 = fde(ev); a1 = null; g.setPointerCapture(ev.pointerId); };
  g.onpointermove = ev => { if (a0 === null) return; a1 = fde(ev);
    const x0 = X(Math.min(a0, a1)), x1 = X(Math.max(a0, a1));
    const s = document.getElementById('sel'); s.setAttribute('x', x0); s.setAttribute('width', Math.max(1, x1 - x0)); };
  g.onpointerup = () => { if (a0 !== null && a1 !== null) ver(Math.min(a0, a1), Math.max(a0, a1)); a0 = null; };
}
async function ver(d, h) {
  const r = await (await fetch(`/plan/tramo.json?desde=${d}&hasta=${h}`)).json();
  const nuevo = r.frames - r.ya_revisados;
  document.getElementById('r').innerHTML = `<table>
    <tr><td>tramo</td><td><b>${d} a ${h}</b></td></tr>
    <tr><td>frames</td><td>${r.frames} (${r.en_aire} con el dron en el aire)</td></tr>
    <tr><td>altura</td><td>mediana <b>${r.alt_mediana} m</b>, maxima ${r.alt_max} m</td></tr>
    <tr><td>la persona se veria</td><td><b>${r.persona_px} px</b> de alto
      ${r.persona_px < 90 ? '<span style="color:#4ade80">(regimen que falla: eso es lo que falta)</span>'
                          : '<span style="color:#fca5a5">(ya tenemos de sobra a este tamano)</span>'}</td></tr>
    <tr><td>ya revisados</td><td>${r.ya_revisados} &nbsp; <b>nuevo: ${nuevo}</b></td></tr></table>
    <p>Para proponer sus cajas:<br><code>python scripts/proponer_cajas.py --frames &lt;carpeta&gt;
    --salida &lt;prefijo&gt; --paso 3 --desde ${d} --hasta ${h}</code></p>
    <div class="g" id="p"></div>`;
  const paso = Math.max(1, Math.floor((h - d) / 8));
  let html = '';
  for (let f = d; f <= h && html.split('<img').length <= 8; f += paso)
    html += `<a href="/frames#${f}"><img src="/tira/celda/${f}?cx=0&cy=0&lado=0&tam=200"></a>`;
  document.getElementById('p').innerHTML = html;
}
cargar();
</script>"""


PAGINA_TIRA = """<!doctype html><meta charset="utf-8"><title>tira</title>
<style>body{background:#0b1020;color:#e5e7eb;font:14px system-ui;margin:0;padding:18px}
h1{font-size:18px;margin:0 0 4px} p.s{color:#9ca3af;margin:0 0 14px;max-width:900px}
input{background:#111827;color:#e5e7eb;border:1px solid #374151;border-radius:6px;padding:5px 8px;width:80px}
button{background:#1d4ed8;color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer}
a{color:#93c5fd;text-decoration:none}
.g{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.c{position:relative} .c img{display:block;border:2px solid #1f2937;border-radius:4px}
.c.sin img{border-color:#b91c1c} .c.rev img{border-color:#166534}
.c span{position:absolute;left:4px;top:3px;font:11px monospace;color:#d1fae5;text-shadow:0 0 3px #000}</style>
<h1>Tira: muchos frames de un tiron, con las cajas puestas</h1>
<p class="s">Sirve para ver de un vistazo lo que frame por frame no se nota: una caja que se despega de la
persona a lo largo de veinte frames se ve bien en cada uno por separado. Todas las celdas muestran el MISMO
pedazo de suelo, centrado donde esta la gente del tramo. El numero es el frame y cuanta gente tiene;
<b>borde rojo</b> = ese frame no tiene a nadie. Clic en una celda para abrirla frame por frame.<br>
<b>Cual de los dos modos</b>: "frame entero" es el unico que sirve para ver si FALTA alguien, porque muestra
todo lo que la camara vio; "acercar a la gente" muestra el mismo pedazo de suelo en todas las celdas y sirve
para ver si una caja se despego de su persona, pero esconde a quien este fuera de ese pedazo.</p>
<p>desde <input id="a" value="3377"> hasta <input id="b" value="3452">
   &nbsp;<label><input type="radio" name="m" value="entero" checked onchange="pintar()"> frame entero
   (para ver si <b>falta</b> alguien)</label>
   &nbsp;<label><input type="radio" name="m" value="zoom" onchange="pintar()"> acercar a la gente
   (para ver si una caja <b>se despego</b>)</label>
   &nbsp;<button onclick="pintar()">ver</button>
   &nbsp;<span id="q" style="color:#9ca3af"></span>
   &nbsp;&middot;&nbsp; <a href="/frames">frame por frame</a> &middot; <a href="/tira">tira</a> &middot; <a href="/plan">donde etiquetar</a> &middot; <a href="/vuelos">vuelos</a></p>
<div class="g" id="g"></div>
<script>
async function pintar() {
  const a = +document.getElementById('a').value, b = +document.getElementById('b').value;
  const d = await (await fetch(`/tira.json?desde=${a}&hasta=${b}`)).json();
  document.getElementById('q').textContent =
    `${d.frames.length} celdas de ${d.total} frames del tramo, vuelo ${d.vuelo}`;
  const entero = document.querySelector('input[name=m]:checked').value === 'entero';
  const lado = entero ? 0 : d.lado, tam = entero ? 320 : 190;
  document.getElementById('g').innerHTML = d.frames.map(f => {
    const n = d.personas[f], rev = d.revisados[f];
    return `<a class="c ${n ? (rev ? 'rev' : '') : 'sin'}" href="/frames#${f}">
      <img src="/tira/celda/${f}?cx=${d.centro[0]}&cy=${d.centro[1]}&lado=${lado}&tam=${tam}">
      <span>${f} (${n})</span></a>`;
  }).join('');
}
const h = location.hash.slice(1).split('-');
if (h.length === 2) { document.getElementById('a').value = h[0]; document.getElementById('b').value = h[1]; }
pintar();
</script>"""


PAGINA_VUELOS = """<!doctype html><meta charset="utf-8"><title>vuelos</title>
<style>body{background:#0b1020;color:#e5e7eb;font:15px system-ui;margin:0;padding:22px}
h1{font-size:20px;margin:0 0 6px} p.s{color:#9ca3af;margin:0 0 18px}
table{border-collapse:collapse;width:100%;max-width:1100px} th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #1f2937}
th:first-child,td:first-child{text-align:left} tr.act td{background:#132033}
a{color:#93c5fd;text-decoration:none} .b{background:#1d4ed8;color:#fff;padding:5px 12px;border-radius:6px;border:0;cursor:pointer}
.ok{color:#4ade80} .no{color:#fca5a5}</style>
<h1>Vuelos etiquetados</h1>
<p class="s">El vuelo en negrita es el que esta abierto. Cualquier boton cambia la sesion a ese vuelo sin
reiniciar nada: lo que estabas etiquetando ya quedo guardado en su propio archivo.<br>
<b>ver video</b> reproduce sus frames revisados con las cajas puestas, que es la forma de juzgar el etiquetado
entero de un vuelo de corrido; <b>tira</b> muestra muchos frames a la vez; <b>etiquetar</b> abre el frame por frame.</p>
<div id="t">cargando...</div>
<script>
async function pintar() {
  const d = await (await fetch('/vuelos.json')).json();
  document.getElementById('t').innerHTML = '<table><tr><th>vuelo</th><th>frames</th><th>revisados</th>' +
    '<th>cajas</th><th>etiquetadas</th><th>personas</th><th>dibujadas</th><th></th></tr>' +
    d.vuelos.map(v => {
      if (!v.existe) return `<tr><td>${v.vuelo}</td><td colspan="7" class="no">sin candidatas: correr proponer_cajas.py</td></tr>`;
      const act = v.vuelo === d.activo;
      const pend = v.frames - v.revisados;
      return `<tr class="${act ? 'act' : ''}"><td>${act ? '<b>' + v.vuelo + '</b>' : v.vuelo}</td>
        <td>${v.frames}</td><td class="${pend ? 'no' : 'ok'}">${v.revisados}${pend ? ' (faltan ' + pend + ')' : ''}</td>
        <td>${v.cajas}</td><td>${v.etiquetadas}</td><td>${v.personas}</td><td>${v.dibujadas}</td>
        <td><button class="b" onclick="abrir('${v.vuelo}', '/video')">ver video</button>
            <button class="b" style="background:#334155" onclick="abrir('${v.vuelo}', '/tira')">tira</button>
            <button class="b" style="background:#334155" onclick="abrir('${v.vuelo}', '/frames')">etiquetar</button></td></tr>`;
    }).join('') + '</table>';
}
async function abrir(v, destino) {
  document.getElementById('t').innerHTML = 'abriendo ' + v + '...';
  const r = await fetch('/vuelos/abrir', {method: 'POST', body: JSON.stringify({vuelo: v})});
  if (r.ok) location.href = destino || '/frames'; else { alert(await r.text()); pintar(); }
}
pintar();
</script>"""


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
            elif self.path == "/plan":
                self._responder(PAGINA_PLAN.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/plan.json":
                with sesion.lock:
                    self._responder(json.dumps(sesion.alturas()).encode("utf-8"))
            elif self.path.startswith("/plan/tramo.json"):
                q = parse_qs(urlparse(self.path).query)
                with sesion.lock:
                    self._responder(json.dumps(sesion.tramo(int(q["desde"][0]), int(q["hasta"][0]))).encode("utf-8"))
            elif self.path == "/tira" or self.path.startswith("/tira#"):
                self._responder(PAGINA_TIRA.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/tira.json"):
                q = parse_qs(urlparse(self.path).query)
                with sesion.lock:
                    d = sesion.tira_datos(int(q.get("desde", [0])[0]), int(q.get("hasta", [10 ** 9])[0]))
                d["vuelo"] = sesion.nombre
                self._responder(json.dumps(d).encode("utf-8"))
            elif self.path.startswith("/tira/celda/"):
                q = parse_qs(urlparse(self.path).query)
                f = int(urlparse(self.path).path.rsplit("/", 1)[1])
                try:
                    with sesion.lock:
                        cuerpo = sesion.celda(f, float(q["cx"][0]), float(q["cy"][0]),
                                              float(q["lado"][0]), int(q.get("tam", [190])[0]))
                    self._responder(cuerpo, "image/jpeg")
                except Exception as e:
                    self._responder(str(e).encode("utf-8"), "text/plain", 404)
            elif self.path == "/vuelos":
                self._responder(PAGINA_VUELOS.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/vuelos.json":
                self._responder(json.dumps({"activo": sesion.nombre,
                                            "vuelos": [resumen_vuelo(v) for v in sorted(VUELOS)
                                                       if len(VUELOS[v]) > 4]}).encode("utf-8"))
            elif self.path == "/historia.json":
                ruta = sesion.ruta_revision.replace(".json", "_historia.jsonl")
                lineas = open(ruta, encoding="utf-8").read().splitlines()[-200:] if os.path.exists(ruta) else []
                self._responder(json.dumps({"vuelo": sesion.nombre,
                                            "historia": [json.loads(l) for l in lineas if l.strip()]}).encode("utf-8"))
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
            nonlocal sesion
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
                    elif self.path == "/frames/propagar":
                        r = sesion.propagar(int(d["i"]))
                    elif self.path == "/frames/resolver":
                        r = sesion.resolver_doble(int(d["i"]))
                    elif self.path == "/frames/resolver_todo":
                        r = sesion.resolver_frame(int(d["f"]), bool(d.get("propagar", True)))
                    elif self.path == "/frames/encimadas":
                        r = sesion.etiquetar_encimadas(int(d["f"]) if d.get("f") is not None else None)
                    elif self.path == "/frames/dos_personas":
                        r = sesion.dos_personas(int(d["f"]))
                    elif self.path == "/frames/deshacer_dos_personas":
                        r = sesion.deshacer_dos_personas(int(d["f"]))
                    elif self.path == "/frames/ajustar":
                        sesion.ajustar(int(d["i"]), d.get("caja"))
                        r = {"status": "ok"}
                    elif self.path == "/frames/nueva_etiqueta":
                        sesion.nueva_etiqueta(int(d["f"]), int(d["k"]), d.get("v"))
                        r = {"status": "ok"}
                    elif self.path == "/vuelos/abrir":
                        sesion = abrir_vuelo(str(d["vuelo"]))
                        r = {"vuelo": sesion.nombre, "frames": len(sesion.lista)}
                    elif self.path == "/frames/seguir":
                        r = sesion.seguir(int(d["f"]), d["caja"], bool(d.get("adelante", True)),
                                          int(d.get("cuantos", 40)))
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
    OPCIONES.update({"grupos": args.grupos, "identidad": args.identidad, "contexto": args.contexto,
                     "contexto_etiquetas": args.contexto_etiquetas, "sospechas": args.sospechas})
    s = Sesion(args.cajas, np.load(args.embs), args.frames, args.salida, args.grupos, args.desde, args.hasta,
               identidad=args.identidad, contexto=args.contexto, contexto_etiquetas=args.contexto_etiquetas,
               lista_frames=[int(l) for l in open(args.lista_frames) if l.strip()] if args.lista_frames else None,
               nombre=args.nombre, sospechas=args.sospechas)
    print("%d cajas en %d grupos, %d ya etiquetadas -> http://127.0.0.1:%d/  (frame por frame: /frames, %d de %d revisados)"
          % (len(s.filas), len(s.grupos), len(s.etiquetas), args.puerto, len(s.revisados), len(s.lista)), flush=True)
    servir(s, args.puerto)


if __name__ == "__main__":
    main()
