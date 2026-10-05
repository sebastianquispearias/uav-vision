"""A second opinion on each candidate's crop: how much it looks like a person, according to CLIP.

The drone cannot tell a person from the objects its detector keeps confusing with one; none of its
own signals separate them. A general image-text model, which was never trained on these scenes and
so cannot have memorised them, can. Measured with scripts/medir_filtro_clip.py and nothing tuned
here:

    threshold 1.496   fixed on the 01ago flights, on crops degraded to what the radio carries
                      (128 px, JPEG quality 70), where it keeps 95 % of the people. On flight 3,
                      another day, it keeps 90.0 % of them and rejects 84.3 % of the non-people.
    flight 3 queue    applied unchanged to the operator's verification queue of that flight:
                      non-people 3 -> 0 and 7 -> 0, people 6 -> 5 and 11 -> 9 (flight tracker,
                      stand-in tracker). Not every person survives, which is why nothing is hidden.

That is a good ordering signal and a bad reason to delete anything, so the station uses it to
order and to mark, and the operator still decides.

The score is exactly the zero-shot score of that script -- same model (ViT-B-32-quickgelu, "openai"
weights), the same prompt lists imported from it, and

    score = logsumexp(100 * sim to POSITIVOS) - logsumexp(100 * sim to NEGATIVOS)

on the L2-normalised image and text embeddings. Below the threshold means "probably not a person".

torch and open_clip are imported only when a PuntuadorClip is built, so the ground station runs
without them; cargar() turns their absence into a warning instead of a crash.

CLASES_PUNTUADAS holds the classes a "person or not" score means anything for. A car scored
against "a person seen from a drone" would always be flagged, and demoting every car in a
search for cars is wrong. A POI with NO class is scored anyway: older drones do not name what
they report, and what they report is people.
"""
import base64
import hashlib
import os
import sys
import threading
from collections import OrderedDict

UMBRAL = 1.496

CLASES_PERSONA = ('person', 'pedestrian', 'people', None)


class PuntuadorClip:
    """
    The CLIP model with the prompt embeddings already computed: bytes or pixels in, score out.

    The text side is encoded once, when built. Every call after that is one image through the
    vision tower, which is the only part that depends on the crop.
    """

    def __init__(self, modelo='ViT-B-32-quickgelu', pesos='openai', dev=None):
        import open_clip
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from medir_filtro_clip import NEGATIVOS, POSITIVOS

        self.torch = torch
        self.dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo, _, self.prep = open_clip.create_model_and_transforms(
            modelo, pretrained=pesos, device=self.dev)
        self.modelo.eval()
        self.n_pos = len(POSITIVOS)
        with torch.no_grad():
            t = self.modelo.encode_text(
                open_clip.get_tokenizer(modelo)(POSITIVOS + NEGATIVOS).to(self.dev)).float()
            self.texto = t / t.norm(dim=-1, keepdim=True)

    def puntuar_rgb(self, rgb):
        """The score of an RGB uint8 array, as medir_filtro_clip scores one."""
        from PIL import Image
        torch = self.torch
        with torch.no_grad():
            f = self.modelo.encode_image(
                self.prep(Image.fromarray(rgb)).unsqueeze(0).to(self.dev)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            sim = 100.0 * f[0] @ self.texto.T
            return float(torch.logsumexp(sim[:self.n_pos], 0)
                         - torch.logsumexp(sim[self.n_pos:], 0))

    def puntuar(self, jpeg):
        """The score of a JPEG crop as it travels from the drone: decoded to RGB, nothing else."""
        import io

        import numpy as np
        from PIL import Image
        return self.puntuar_rgb(np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')))

    def __call__(self, jpeg):
        return self.puntuar(jpeg)


class Anotador:
    """
    What the station does with a scorer: writes the score onto a POI, once per distinct crop.

    A drone repeats the same candidate, with the same crop, on every report. The score is kept
    under a hash of the crop so each picture goes through the model once, and calls are
    serialised because one model instance is shared by the station's request threads. The cache
    is bounded, oldest out, so a long mission does not grow it without limit.

    Two fields are added, and only when a score exists: 'clip', the score, and
    'clip_no_persona', whether it is under the threshold. A POI with no crop, of a non-person
    class, or whose crop cannot be decoded is left exactly as it arrived.
    """

    def __init__(self, puntuar, umbral=UMBRAL, maximo=4096):
        self.puntuar = puntuar
        self.umbral = umbral
        self.maximo = maximo
        self._cache = OrderedDict()
        self._candado = threading.Lock()

    def puntaje(self, crop_b64):
        clave = hashlib.sha1(crop_b64.encode('ascii')).hexdigest()
        with self._candado:
            if clave in self._cache:
                self._cache.move_to_end(clave)
                return self._cache[clave]
            try:
                s = round(float(self.puntuar(base64.b64decode(crop_b64))), 3)
            except Exception:
                s = None
            self._cache[clave] = s
            while len(self._cache) > self.maximo:
                self._cache.popitem(last=False)
            return s

    def anotar(self, poi):
        if not poi.get('crop') or poi.get('cls') not in CLASES_PERSONA:
            return poi
        s = self.puntaje(poi['crop'])
        if s is not None:
            poi['clip'] = s
            poi['clip_no_persona'] = s < self.umbral
        return poi


def cargar(umbral=UMBRAL):
    """
    An Anotador with the real model, or None with a warning when the model cannot be loaded.

    The station must keep working without the score: it only ever orders and marks.
    """
    try:
        return Anotador(PuntuadorClip(), umbral=umbral)
    except Exception as e:
        print('AVISO: --clip pedido pero CLIP no se pudo cargar (%s: %s); la estacion sigue sin '
              'puntaje. open_clip esta en ../drone-geolocation/entrenamiento/venv.'
              % (type(e).__name__, e), flush=True)
        return None
