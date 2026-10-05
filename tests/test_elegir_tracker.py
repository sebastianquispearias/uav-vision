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
print("6. UNA TRADUCCION APROXIMADA SE DECLARA, no se esconde")
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
print("  %d de %d trackers reciben al menos un ajuste renombrado SOLO por rol"
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
print("TODO OK")
