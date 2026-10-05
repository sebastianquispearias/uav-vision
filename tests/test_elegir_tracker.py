"""Any of boxmot's trackers, built by name, with our calibration actually delivered.

The chain has run one tracker since it was written, and boxmot ships ten. What is missing is not
a tracker zoo -- the library already resolves them by name -- but an adapter that says what each
one DID with the four thresholds and the buffer the flight was calibrated on.

Every section is a contrast that fails if the behaviour it claims is not there. Two of them exist
because the first version of the adapter got it wrong, in both directions, and both mistakes were
silent:

  1. Every tracker in boxmot's catalogue can be built, and an unknown name is refused loudly.
  2. **kwargs IS NOT ACCEPTANCE. All ten classes declare one, so the first version read "the
     class takes **kwargs" as "it accepts everything" and cheerfully reported five settings
     applied for trackers whose parameters are called something else entirely. The names simply
     vanished and the tracker ran on its own defaults.
  3. THE WHOLE MRO IS WALKED. These trackers forward their **kwargs to a common base that owns
     det_thresh, max_age and iou_threshold, so reading the subclass alone says a tracker ignores
     settings it does in fact honour. The second version under-reported for that reason.
  4. The translation does something measurable: without it, trackers that do not use BoT-SORT's
     spelling receive NOTHING. That is the comparison nobody wants to publish by accident, and
     the contrast is what makes the table worth reading.
  5. The default is unchanged. BoT-SORT receives all five settings with no approximation, which
     is the condition for the flight's numbers to stay comparable.
  6. An approximate rename is DECLARED. Same role is not the same quantity, and a comparison
     that hides the difference is the quiet version of the mistake this module prevents.
  7. A RENAME THAT RUNS IN THE OPPOSITE SENSE IS CONVERTED, not merely declared. BoT-SORT's
     match_thresh caps a cost of 1 - IoU and boxmot's iou_threshold floors an IoU, so the same
     number is the most permissive setting on one side and the strictest on the other. Passing
     0.85 straight through made five trackers demand an 0.85 overlap, and a published table
     scored one of them 0 of 7 people for it.
  8. CAMERA-MOTION COMPENSATION REACHES FOUR DIFFERENT MECHANISMS, and what RAN is not what was
     asked. boxmot defaults it on, two trackers honour it under a name no rename reaches, and
     one compensates on every frame with no parameter to stop it.

Needs boxmot, so on a machine without it the gate skips itself rather than failing.

Run with: ../drone-geolocation/entrenamiento/venv/Scripts/python.exe tests/test_elegir_tracker.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import boxmot
except ModuleNotFoundError:
    print("boxmot no instalado en este python: test saltado")
    sys.exit(0)

from uav_vision.trackers import TRADUCCION, _acepta, catalogo, construir

CAL = {"track_high_thresh": 0.35, "track_low_thresh": 0.2, "new_track_thresh": 0.4,
       "track_buffer": 40, "match_thresh": 0.85}


def clase_de(nombre):
    from boxmot.trackers.registry import TRACKER_DEFINITIONS
    ruta = TRACKER_DEFINITIONS[nombre].class_path
    modulo, _, cn = ruta.rpartition(".")
    return getattr(__import__(modulo, fromlist=[cn]), cn)


print("=" * 78)
print("1. EL CATALOGO: cada tracker de boxmot se construye, y un nombre falso se niega")
print("=" * 78)
cat = catalogo()
assert len(cat) >= 9, "boxmot trae menos trackers de los esperados: %d" % len(cat)
assert "botsort" in cat and cat["botsort"], "botsort deberia usar apariencia"
for nombre in cat:
    construir(nombre, **CAL)
print("  %d trackers construidos, %d usan apariencia"
      % (len(cat), sum(1 for v in cat.values() if v)))
try:
    construir("un_tracker_que_no_existe")
except ValueError as e:
    assert "no conoce" in str(e)
    print("  un nombre falso se niega nombrando los que si existen")
else:
    raise AssertionError("un nombre inventado tendria que levantar ValueError")

print()
print("=" * 78)
print("2. **kwargs NO ES ACEPTACION: un nombre ajeno no se cuenta como honrado")
print("=" * 78)
import inspect

firma = inspect.signature(clase_de("ocsort").__init__)
tiene_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in firma.parameters.values())
assert tiene_kwargs, "ocsort ya no declara **kwargs; esta seccion perdio su sentido"
acepta = _acepta(clase_de("ocsort"))
assert "track_high_thresh" not in acepta, "se esta contando un nombre que ocsort no declara"
_, ap, ig, _ = construir("ocsort", traducir=False, **CAL)
assert not ap, "sin traducir, ocsort no deberia honrar nada nuestro: %s" % sorted(ap)
assert len(ig) == len(CAL)
print("  ocsort declara **kwargs y aun asi honra 0 de %d sin traducir" % len(CAL))

print()
print("=" * 78)
print("3. SE RECORRE EL MRO: los parametros de la clase base cuentan")
print("=" * 78)
propios = [n for n, p in inspect.signature(clase_de("ocsort").__init__).parameters.items()
           if n != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD]
for base in ("det_thresh", "max_age", "iou_threshold"):
    assert base in acepta, "%s se honra por la base y el adaptador no lo ve" % base
    assert base not in propios, "%s ya es propio de OcSort; la seccion perdio su contraste" % base
print("  ocsort: %d nombres propios, %d por el MRO (det_thresh, max_age, iou_threshold entre ellos)"
      % (len(propios), len(acepta)))

print()
print("=" * 78)
print("4. LA TRADUCCION HACE ALGO: sin ella, siete de diez no reciben nada")
print("=" * 78)
print("  %-12s %-5s %-9s %-7s %s" % ("tracker", "reid", "sin trad.", "con", "sin equivalente"))
mejoraron = 0
for nombre, usa_reid in cat.items():
    _, sin, _, _ = construir(nombre, traducir=False, **CAL)
    _, con, falta, aprox = construir(nombre, **CAL)
    assert len(con) >= len(sin), "traducir no puede quitar ajustes en %s" % nombre
    if len(con) > len(sin):
        mejoraron += 1
    print("  %-12s %-5s %d de %d     %d de %d   %s"
          % (nombre, "si" if usa_reid else "no", len(sin), len(CAL), len(con), len(CAL),
             ", ".join(sorted(falta)) or "-"))
assert mejoraron >= 7, "la traduccion solo mejora %d trackers: la tabla esta vacia?" % mejoraron
print("  la traduccion mejora %d de los %d trackers" % (mejoraron, len(cat)))

print()
print("=" * 78)
print("5. EL DEFECTO NO CAMBIA: BoT-SORT recibe los cinco, sin aproximar ninguno")
print("=" * 78)
_, ap, ig, aprox = construir("botsort", **CAL)
assert ap == CAL, "la calibracion del vuelo ya no llega entera a BoT-SORT: %s" % sorted(ap)
assert not ig and not aprox, "BoT-SORT no deberia necesitar traduccion: %s %s" % (ig, aprox)
print("  los %d ajustes llegan con su propio nombre: %s" % (len(ap), ", ".join(sorted(ap))))

print()
print("=" * 78)
print("6. UN RENOMBRE QUE NO ES EXACTO SE DECLARA, no se esconde")
print("=" * 78)
con_aprox = {}
for nombre in cat:
    _, _, _, aprox = construir(nombre, **CAL)
    if aprox:
        con_aprox[nombre] = aprox
assert con_aprox, "nadie declara una traduccion aproximada: se perdio el aviso"
assert "botsort" not in con_aprox, "BoT-SORT no traduce nada, no puede aproximar"
seguridades = {s for pares in TRADUCCION.values() for _, s in pares}
assert seguridades == {"exacta", "aproximada"}, "etiquetas de seguridad inesperadas: %s" % seguridades
for nombre, aprox in sorted(con_aprox.items()):
    print("  %-12s %s" % (nombre, ", ".join(aprox)))
print("  %d de %d trackers reciben al menos un renombre que hay que declarar"
      % (len(con_aprox), len(cat)))

print()
print("=" * 78)
print("7. EL YAML DE BOXMOT NO SE CARGA: solo son no-defecto los valores que pasamos")
print("=" * 78)
AJ = {"use_cmc": True, "cmc_method": "sof", **CAL}
try:
    from boxmot.trackers.bbox.botsort import BotSort
except ImportError:
    from boxmot import BotSort
como_estaba = BotSort(reid_model=None, with_reid=True, **AJ)
con_adaptador, _, ig, apx = construir("botsort", embs_propias=True, **AJ)
assert not ig and not apx, "BoT-SORT no deberia necesitar traduccion: %s %s" % (ig, apx)
campos = sorted(set(AJ) | {"with_reid", "max_age", "det_thresh", "proximity_thresh",
                           "appearance_thresh", "frame_rate", "min_hits", "iou_threshold"})
difs = {k: (getattr(como_estaba, k, "<aus>"), getattr(con_adaptador, k, "<aus>"))
        for k in campos
        if getattr(como_estaba, k, "<aus>") != getattr(con_adaptador, k, "<aus>")}
assert not difs, (
    "el adaptador ya no construye el tracker que vuela. Diferencias: %s. "
    "Si esto falla por appearance_thresh o proximity_thresh, el YAML de boxmot volvio a "
    "entrar: sus archivos traen una busqueda de hiperparametros sobre MOT y mueven compuertas "
    "de apariencia que este vuelo nunca calibro." % difs)
print("  %d campos identicos al BotSort construido a mano, incluidos appearance_thresh=%s"
      % (len(campos), con_adaptador.appearance_thresh))
print("  y cada tracker del zoo recibe NUESTRO umbral, no el del YAML:")
for nombre in ("ocsort", "sfsort", "bytetrack"):
    t, _, _, _ = construir(nombre, **CAL)
    assert abs(getattr(t, "det_thresh", -1) - CAL["track_high_thresh"]) < 1e-9, (
        "%s no recibio nuestro umbral alto: det_thresh=%s" % (nombre, getattr(t, "det_thresh", None)))
    print("     %-10s det_thresh=%s" % (nombre, t.det_thresh))

print()
print("=" * 78)
print("8. SIN EL REGISTRO DE BOXMOT, el que vuela se construye igual")
print("=" * 78)
import uav_vision.trackers as T

assert T.hay_registro(), "este boxmot no trae registro; la seccion 8 perdio su contraste"
original = T.hay_registro
try:
    T.hay_registro = lambda: False
    de_respaldo, _, _, _ = construir("botsort", embs_propias=True, **AJ)
    assert type(de_respaldo).__name__ == type(con_adaptador).__name__
    difs = {k for k in campos
            if getattr(de_respaldo, k, "<aus>") != getattr(con_adaptador, k, "<aus>")}
    assert not difs, "el respaldo construye otro tracker: %s" % sorted(difs)
    assert list(T.catalogo()) == ["botsort"], "sin registro solo botsort puede ofrecerse"
    try:
        construir("ocsort")
    except ValueError as e:
        assert "no trae registro" in str(e)
    else:
        raise AssertionError("sin registro, ocsort tendria que negarse")
finally:
    T.hay_registro = original
print("  el camino de respaldo da el MISMO tracker, y el resto se niega nombrando el motivo")
print("  (boxmot movio sus modulos entre la version de una placa y la de la otra: por eso existe)")

print()
print("=" * 78)
print("9. EL RENOMBRE INVERTIDO SE CONVIERTE: match_thresh y iou_threshold van al reves")
print("=" * 78)
# El contraste no necesita el vuelo: la regla de cada uno se evalua sobre un IoU concreto.
# boxmot asocia con IoU > iou_threshold (stages.py:152, boost.py:196, hybrid.py:343);
# BoT-SORT asocia con 1 - IoU <= match_thresh (matching.py:79 y :35).
NUESTRO = 0.85



def acepta_botsort(iou):
    """BoT-SORT's rule: lap.lapjv caps the cost 1 - IoU at match_thresh (matching.py:35, :79)."""
    return (1.0 - iou) <= NUESTRO


def acepta_boxmot(iou, umbral):
    """boxmot's rule everywhere else: the IoU has to EXCEED the threshold (stages.py:152)."""
    return iou > umbral


print("  nuestro match_thresh=%.2f, o sea que BoT-SORT asocia desde IoU >= %.2f"
      % (NUESTRO, 1.0 - NUESTRO))
print("  %6s | %-18s | %-22s | %-22s" % ("IoU", "BoT-SORT 0.85", "boxmot SIN convertir", "boxmot convertido"))
discrepan = 0
for iou in (0.9, 0.5, 0.3, 0.2, 0.1):
    a, b, c = acepta_botsort(iou), acepta_boxmot(iou, NUESTRO), acepta_boxmot(iou, 1.0 - NUESTRO)
    discrepan += a != b
    print("  %6.2f | %-18s | %-22s | %-22s"
          % (iou, "asocia" if a else "rechaza", "asocia" if b else "RECHAZA",
             "asocia" if c else "rechaza"))
    assert a == c, "la conversion no reproduce la regla de BoT-SORT en IoU=%.2f" % iou
assert discrepan >= 3, (
    "sin convertir las dos reglas coinciden, asi que este contraste no prueba nada y la "
    "conversion no estaria justificada")
print("  pasar 0.85 crudo cambia la decision en %d de los 5 IoU probados; convertido, en 0"
      % discrepan)

for nombre in ("ocsort", "deepocsort", "hybridsort", "boosttrack", "occluboost"):
    t, _, _, apx = construir(nombre, embs_propias=cat[nombre], **CAL)
    assert abs(t.iou_threshold - (1.0 - CAL["match_thresh"])) < 1e-9, (
        "%s recibio iou_threshold=%s en vez de %s: el umbral mas permisivo de la calibracion "
        "se convirtio en el mas estricto posible" % (nombre, t.iou_threshold, 1.0 - CAL["match_thresh"]))
    assert any("opuesto" in a for a in apx), (
        "%s recibe el valor convertido pero no lo declara, y una tabla no puede perderlo" % nombre)
print("  los 5 trackers que USAN iou_threshold reciben %.2f y lo declaran"
      % (1.0 - CAL["match_thresh"]))

print()
print("=" * 78)
print("10. LA CMC QUE CORRE NO ES LA QUE SE PIDE, y la diferencia se lee de la instancia")
print("=" * 78)
from uav_vision.trackers import cmc_real

estados = {}
for nombre in cat:
    apagado, _, _, _ = construir(nombre, embs_propias=cat[nombre], use_cmc=False,
                                 cmc_method="sof", **CAL)
    encendido, _, _, _ = construir(nombre, embs_propias=cat[nombre], use_cmc=True,
                                   cmc_method="sof", **CAL)
    estados[nombre] = (cmc_real(apagado), cmc_real(encendido))
    print("  %-12s pedida off -> %-7s | pedida on -> %-7s" % (nombre, *estados[nombre]))

conmutables = {n for n, (a, b) in estados.items() if (a, b) == ("off", "on")}
sin_mecanismo = {n for n, (a, b) in estados.items() if (a, b) == ("ninguno", "ninguno")}
siempre = {n for n, (a, b) in estados.items() if (a, b) == ("on", "on")}
assert conmutables and sin_mecanismo and siempre, (
    "el desbalance de CMC desaparecio: conmutables=%s sin_mecanismo=%s siempre=%s. Si de verdad "
    "todos se comportan igual, esta seccion ya no prueba nada y hay que rehacerla."
    % (sorted(conmutables), sorted(sin_mecanismo), sorted(siempre)))
assert conmutables | sin_mecanismo | siempre == set(cat), (
    "algun tracker no cae en ninguno de los tres grupos: %s"
    % sorted(set(cat) - (conmutables | sin_mecanismo | siempre)))
print("  %d conmutables (%s)" % (len(conmutables), ", ".join(sorted(conmutables))))
print("  %d sin mecanismo (%s)" % (len(sin_mecanismo), ", ".join(sorted(sin_mecanismo))))
print("  %d compensan SIEMPRE, no hay parametro que lo apague (%s)"
      % (len(siempre), ", ".join(sorted(siempre))))
print("  y 'ninguno' NO es 'off': juntarlos cuenta a un tracker sin mecanismo como prueba de")
print("  que apagarla no hizo dano")

print()
print("TODO OK")
