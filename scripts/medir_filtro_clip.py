"""
Does a general image-text model tell a person from what the detector confuses with one, across days?

Every labelled box of the three labelled flights is cropped and scored two ways:

    zero-shot   CLIP compares the crop with "a person seen from a drone" and with the objects the
                detector actually confused (cones, a hydrant, bins, poles...). Nothing is trained,
                so nothing can memorise a scene -- the failure every filter so far fell into.
    linear      a logistic probe on CLIP image features, trained on one DAY and tested on the
                other (02ago vs 01ago). Flights 2a and 2b were recorded eleven minutes apart in the
                same place; testing one against the other measured scene memory, not generalisation.

The reference it has to beat: OSNet appearance features, trained on one day and tested on the other,
gave AUC 0.55-0.80.

Runs in the training venv, which has open_clip:

    ../drone-geolocation/entrenamiento/venv/Scripts/python.exe scripts/medir_filtro_clip.py

THE WEIGHTS NEED THE MATCHING CONFIG. The OpenAI weights were trained with QuickGELU, and
loaded under the plain "ViT-B-32" config they run with the wrong activation, which open_clip
warns about, and separate worse. On 128 px crops: AUC 0.928 / 0.891 / 0.896 on
02ago / 01ago-2a / 01ago-2b, against 0.947 / 0.923 / 0.944 with the matching config.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
HNO = os.path.join(RAIZ, "..", "drone-geolocation")
ENT = os.path.join(HNO, "entrenamiento")

MODELO = "ViT-B-32-quickgelu"
PESOS = "openai"

POSITIVOS = ["an aerial photo of a person", "a person seen from a drone", "a pedestrian walking",
             "a person standing", "the legs of a person"]
NEGATIVOS = ["an aerial photo of a traffic cone", "a fire hydrant", "a trash bin", "a pole",
             "a bag on the ground", "a shadow on the ground", "a car", "a chair", "a plant",
             "an empty patch of ground"]


def conjuntos():
    """(name, day, frames dir, boxes [(frame, x1, y1, x2, y2)], is_person) for every labelled set."""
    out = []
    d = np.load(os.path.join(RAIZ, "demo", "data", "examen_v3_datos.npz"))["dets"]
    d = d[d[:, 1] >= 0.25]
    et = json.load(open(os.path.join(ENT, "identidad_gt_02ago.json"), encoding="utf-8"))["etiquetas"]
    ix = [i for i in range(len(d)) if 3000 <= d[i, 0] <= 3700 and et.get(str(i)) not in (None, "?")]
    out.append(("02ago", "02ago", os.path.join(HNO, "data", "flight_02ago", "20260802_133309", "frames"),
                [(int(d[i, 0]), *d[i, 2:6]) for i in ix],
                np.array([et[str(i)] not in ("X", "x") for i in ix])))
    for v, run in (("2a", "20260801_184259"), ("2b", "20260801_185326")):
        filas = list(csv.DictReader(open(os.path.join(ENT, "valida_ident_cajas_vuelo%s.csv" % v))))
        lab = json.load(open(os.path.join(ENT, "grupos_vuelo%s.json" % v), encoding="utf-8"))["etiquetas"]
        k = sorted(int(i) for i in lab)
        out.append(("01ago-" + v, "01ago", os.path.join(HNO, "data", "flight_01ago", run, "frames"),
                    [(int(filas[i]["frame"]), *(float(filas[i][c]) for c in ("x1", "y1", "x2", "y2"))) for i in k],
                    np.array([lab[str(i)] == "persona" for i in k])))
    return out


def recortar(img, x1, y1, x2, y2, margen=0.25, degradar=None):
    """The box with a margin, squared with padding: CLIP sees context, not a stretched sliver.

    degradar=N reproduces what actually reaches the ground station: the crop shrunk to N px on its
    longest side and round-tripped through JPEG at quality 70, the few kilobytes the radio budget
    allows. A filter that only works on full-resolution crops cannot run on the ground.
    """
    h, w = img.shape[:2]
    mx, my = margen * (x2 - x1), margen * (y2 - y1)
    c = img[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))]
    lado = max(c.shape[:2])
    lienzo = np.full((lado, lado, 3), 0, np.uint8)
    y0, x0 = (lado - c.shape[0]) // 2, (lado - c.shape[1]) // 2
    lienzo[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    if degradar:
        lienzo = cv2.resize(lienzo, (degradar, degradar), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", lienzo, [cv2.IMWRITE_JPEG_QUALITY, 70])
        lienzo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB)


def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def al_95(s, y):
    """Operating point that keeps 95 % of the people: how many non-people it rejects, and precision."""
    corte = np.percentile(s[y], 5)
    keep = s >= corte
    return 100.0 * (s[~y] < corte).mean(), 100.0 * y.mean(), 100.0 * y[keep].mean()


def sonda(Xtr, ytr, Xte, pasos=400):
    """Balanced logistic regression on L2-normalised features, in torch, so the venv needs no sklearn."""
    Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    peso = torch.where(y > 0, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    opt = torch.optim.LBFGS([w, b], max_iter=pasos)

    def cierre():
        opt.zero_grad()
        z = Xtr @ w + b
        perdida = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, weight=peso)
                   + 1e-3 * (w ** 2).sum())
        perdida.backward()
        return perdida
    opt.step(cierre)
    return (Xte @ w + b).detach().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--pesos", default=PESOS)
    ap.add_argument("--degradar", type=int, default=None,
                    help="lado en px del recorte que viaja (p.ej. 128), con JPEG calidad 70")
    args = ap.parse_args()
    import open_clip
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, _, prep = open_clip.create_model_and_transforms(args.modelo, pretrained=args.pesos, device=dev)
    modelo.eval()
    tok = open_clip.get_tokenizer(args.modelo)
    with torch.no_grad():
        t = modelo.encode_text(tok(POSITIVOS + NEGATIVOS).to(dev)).float()
        t = t / t.norm(dim=-1, keepdim=True)

    datos = {}
    for nombre, dia, frames, cajas, es in conjuntos():
        cache = os.path.join(ENT, "clip_%s_%s_%s%s.npz" % (args.modelo.replace("/", "-"), args.pesos, nombre,
                                                        "_d%d" % args.degradar if args.degradar else ""))
        if os.path.exists(cache):
            feats = np.load(cache)["feats"]
        else:
            feats, lote, img_de = [], [], {}
            for f, x1, y1, x2, y2 in cajas:
                if f not in img_de:
                    img_de = {f: cv2.imread(os.path.join(frames, "frame_%04d.jpg" % f))}
                lote.append(prep(Image.fromarray(recortar(img_de[f], x1, y1, x2, y2, degradar=args.degradar))))
                if len(lote) == 64:
                    with torch.no_grad():
                        feats.append(modelo.encode_image(torch.stack(lote).to(dev)).float().cpu().numpy())
                    lote = []
            if lote:
                with torch.no_grad():
                    feats.append(modelo.encode_image(torch.stack(lote).to(dev)).float().cpu().numpy())
            feats = np.vstack(feats)
            np.savez(cache, feats=feats)
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        datos[nombre] = (dia, feats, es)

    print("modelo %s/%s | %s\n" % (args.modelo, args.pesos, "  ".join(
        "%s: %d cajas (%d personas)" % (n, len(e), e.sum()) for n, (_, _, e) in datos.items())))
    print("ZERO-SHOT (sin entrenar nada)")
    print("%-10s %6s | %-52s" % ("vuelo", "AUC", "al umbral que conserva 95% de personas"))
    tt = t.cpu().numpy()
    for nombre, (_dia, feats, es) in datos.items():
        sim = 100.0 * feats @ tt.T
        s = (np.log(np.exp(sim[:, :len(POSITIVOS)]).sum(1)) - np.log(np.exp(sim[:, len(POSITIVOS):]).sum(1)))
        rech, p0, p1 = al_95(s, es)
        print("%-10s %6.3f | rechaza %4.1f%% de no-personas; precision %2.0f%% -> %2.0f%%" % (nombre, auc(s, es), rech, p0, p1))

    print("\nSONDA LINEAL ENTRE DIAS (entrena un dia, prueba el otro)")
    for tr_dia, te_dia in (("02ago", "01ago"), ("01ago", "02ago")):
        Xtr = np.vstack([f for d, f, _ in datos.values() if d == tr_dia])
        ytr = np.concatenate([e for d, _, e in datos.values() if d == tr_dia])
        for nombre, (dia, feats, es) in datos.items():
            if dia != te_dia:
                continue
            s = sonda(Xtr, ytr.astype(float), feats)
            rech, p0, p1 = al_95(s, es)
            print("entrena %-6s prueba %-9s AUC %.3f | rechaza %4.1f%%; precision %2.0f%% -> %2.0f%%"
                  % (tr_dia, nombre, auc(s, es), rech, p0, p1))
    print("\nreferencia OSNet entre dias: AUC 0.550-0.803")


if __name__ == "__main__":
    main()
