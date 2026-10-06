# How the system is put together

The source docstrings say **what** each thing is and how it is run.
[NOTES.md](../NOTES.md) says **where every number comes from**. This document says **how the
pieces fit**: what does not fit in a file header without the header ceasing to be readable.

Three components carry enough design to deserve a section: the ground station, the flight replay,
and the labelling tool.

The whole chain, one line per stage:

```
camera.py        capture -> YOLO -> BoT-SORT -> OSNet -> crop
pinhole_local.py pixel -> bearing ray
                 intersection with the ground
identity.py      accumulates per track, classifies static/mobile, fuses tracks into candidates
vision_protocol.py  broadcasts the POIs (it is the GrADyS protocol)
gs_mapa.py       the ground station, with a map
```

To read the whole chain running in a single file, without opening five:
`scripts/sistema_esencial.py`.

---

## The ground station (`scripts/banco_embedded/gs_mapa.py`)

### What is imported from the package, and why it is not copied

`fundir` and `pedidos_de_verificacion` do the fusion between drones, which needs **the same
numbers the drone uses** to fuse its own tracks, and `EMB_DIST_OBJETIVO` is the distance the
operator's click travels with. Both are **imported** instead of copied, because a threshold
written by hand in two places is how the station and the drones end up disagreeing about what
counts as the same object, or about what the click meant: a bug nobody would think to look for.

Without the package, because this file is meant to be droppable anywhere, the station **keeps
running**: it shows the two drones' targets unfused, and the click still means only a position,
which is what it meant before.

`R_TIERRA` is there because POIs arrive in **local metres**, x east and y north from the mission
origin, and with the origin's coordinates those same points become lat/lng. That conversion was
supposedly blocked waiting to agree a format with the group. It is not: both ends of this link
are ours.

### The station's state (the `ESTADO` dict)

The keys whose name does not explain itself:

| key | what it is |
|---|---|
| `pois` | the last list received, annotated. What the operator sees |
| `historia` | every report, for the trail |
| `drones` | drone id → the last thing heard from it |
| `veredictos_dir` | where the operator's verdicts and their crops are stored |
| `origen`, `origen_cli`, `desacuerdo` | the origin **in use**, which is the drone's if it declares one; what the operator **typed**, kept so the two can be contrasted; and the metres between them when they disagree |
| `nodos` | the drones' data-plane addresses, from `--nodos`, to push every order at them. Empty means the drones learn it by polling `/buscar` |
| `objetivo` | the target the operator fixed with "this is what I am looking for", in metres, or `None` |

**`buscar`, `buscar_v`, `buscar_epoca`.** What the operator asks **the drone** to look for. This is
**not** the screen's filter: hiding a class merely stops drawing it, whereas this changes what the
detector reports at all. `None` means "whatever the drone started with", and the counter lets a
drone notice a change without comparing lists. The **epoch** is which run of this station issued
the order: a restarted station counts from zero again, and without it a drone that took version 7
would ignore every new order below that.

**`pois_por_dron`.** One list per drone. A single shared list was replaced on every report, so a
second drone erased the first one's targets and the map flickered between the two views.

**`rastros`, `frame_actual`, `frames_dir`.** Where the bench replay is in the frame sequence, and
where the frames live. Only the bench sends this: a real drone sends coordinates, not photos, and
the link could not carry them. It exists so a demo can put what the camera saw beside what the map
made of it.

**`segunda`.** The ground second opinion, one entry per drone: what the operator asked for and what
came back. **Never the image**, which is served from disk via `/segunda.jpg`. A 1920x1080 frame is
300 KB, and carrying it inside a state poll that runs once a second would be four megabits of the
same image for as long as the operator looks at it.

`SEGUNDA` runs the second opinion in a **separate process** on purpose: RF-DETR lives in the
training venv and this station has to be droppable on a laptop with nothing installed. `CLIP` is
CLIP's optional second opinion on each crop (`filtro_clip.Anotador`), switched on by `--clip`;
`None` means the station never started one, and then no POI carries a score field.

### The constants that are the operator's decisions

All of them can be changed from the command line.

**`DRON_CALLADO_S` (`--callado-s`).** A drone that has gone this long without reporting is not
seeing anything now, and its targets stop counting towards the map: a pin labelled "seen by 1+2"
cannot survive one of the two going quiet. It is the same window the drones apply to what they
hear from each other.

**`CLIP_DESCARTA` (`--clip-descarta`).** Remove what CLIP calls a non-person instead of demoting
it.

**`OBJETIVO_RADIO_M`.** How far from the click a candidate may be and still be taken as the one the
operator meant. Same reasoning as the verdict, which only applies to the nearest POI: a click is a
gesture with the precision of a finger on a map, and reaching further would hand the drone the
appearance of somebody **standing next to** the person pointed at.

**`RODEO_RADIO_M`, `RODEO_ALTURA_M`.** Where the operator wants an aircraft to stop when sent to
look at a target from another side. Mission decisions, and nothing derives them: close enough for a
person to be more than a shape, far enough not to be over anybody's head, within whatever the
airspace allows. They live in the operator's instrument because **the layer that flies refuses to
invent them**.

**`RODEO_PUNTOS`.** How many stops. **ONE** by default, and the reasoning matters more than the
number: the question the operator is asking is "is that a person?", and one photograph from an
angle nobody has answers it. Twelve stops are twelve photographs of the same point, eleven of them
answering a question nobody asked, and at this radius each leg takes tens of seconds, so the whole
circuit is minutes during which that aircraft patrols nothing. The orbit is still there and the
protocol accepts any number, because "go and stay overhead" is in the mission and will be wanted;
what it is not is **the default**, because the default has to be the cheap answer to the frequent
question.

**`EN_BANCO` (`--banco`).** This station is driving boards on a desk, not aircraft. It only affects
what the page says about position, and it says exactly that instead of printing a number nobody
should believe.

### The startup, which has no `main()` to document

`--radio-rodeo`, `--altura-rodeo` and `--puntos-rodeo` are arguments of the operator's instrument
instead of numbers in the source, because all three are mission decisions and the layer that flies
refuses to invent them.

`--origen` is kept **apart** from the origin actually used: the typed value is what the operator
**believes**, and the whole point is being able to tell the two apart as soon as a drone declares
its own. Until one speaks, the typed value is all there is, so it seeds the one in use.

CLIP is imported only when asked for, so the station stays droppable anywhere without torch. The
second opinion is started **before the first report** and never on the first click, because loading
RF-DETR takes about 17 s and that function's criterion is that the operator waits less than five.

The port is probed before being bound. **Windows lets a second station bind a port that already has
one**, and then the reports go to whichever socket accepts first; the symptom is a map that stays
empty while the flight is clearly running, and it has cost two sessions. Refuse rather than guess.

---

## The flight replay (`scripts/replay_vuelo3.py`)

### Where the inputs come from

All three live in the flight archive by default. `demo/demo.py` points `UAV_VISION_DATOS` at a
self-contained copy, so the replay runs from a clone with no archive and no drone.

`CONF_MIN` is the same cut the identity analysis uses, and the rows of `embs_osnet.npy` correspond,
in order, to the detections above it. `LAT0`, `LNG0` are the ENU origin, which is the surveyed post
every analysis of flight 3 uses. `PIES` is the operator's surveyed position and `OBJ` the equipment
case, the flight-3 thief that stole the single consensus.

The cached person detections predate the class reaching the report, so they carry no class of their
own. Naming them costs nothing while they are alone, because a class never contradicts itself, and
it is what lets the station tell them from vehicles once both are on the same map.

`fusion_radius_m` is the expected projection noise of **this** scene (GPS sigma plus slant range
times heading error at its altitude) and it is the value validated offline.

### The switches, and why the default run is untouchable

The people-only run is **THIS REPOSITORY'S EQUIVALENCE GATE**: it has to keep printing 2.39 m, so
nothing about it changes unless asked. Every switch is opt-in for that reason.

| switch | what it does |
|---|---|
| `--vehiculos` | adds the vehicle path. Implied by `--vivo`, which only makes sense if there is something to switch to |
| `--dron`, `--pasada` | which drone this replay claims to be and which half of the flight it flies. The flight made two passes over the same ground several minutes apart, and the paper measures that the GPS bias between them is **independent**, so pass 1 and pass 2 stand in for two aircraft. It is the pseudo-swarm protocol, used here to exercise two drones with one camera |
| `--refuerzo` | lets tracks too short to open a candidate reinforce one that was already opened by a track that lasted |
| `--span`, `--miradas-min=N` | maturity by independent looks is the rule that governs. `--span` reproduces the rule every earlier number was measured with, which is the rule the 2.39 m gate is pinned on |
| `--preliminares` | shows the candidates that formed and did not mature. For a drone in orbit they are noise; a vehicle the drone crosses once on a sweep is exactly the case they exist for |

**`UAV_VISION_GS`, `--vivo`, `--velocidad`.** With `UAV_VISION_GS` set, the reports the protocol
produced are pushed to a running `gs_mapa`, at the pace they would have had during the flight
instead of appearing all at once. The flight lasted about 11 min, so the default 20x leaves it under
35 s, which is enough to click during.

**`--pistas=file.npz`.** Replaces the stand-in tracker with ids computed elsewhere;
`scripts/botsort_pistas.py` writes the ones the flight's BoT-SORT gives. A detection that tracker
left without an id reaches the protocol without an id, exactly as on the drone, and the identity
layer never sees it.

**`--evidencia-min=X`.** Separates **ASSOCIATING** from **EVIDENCE**. The tracker has already seen
every box and gave ids with all of them; a box below X keeps the id it helped build but never
reaches the protocol, so it adds neither an observation to the identity layer nor an impact to the
geolocation. The frames flown and the cadence are left alone, so the evidence floor is the only
thing that changes.

**`--foco=x,y`.** Fixes a target the way the operator's "this is what I am looking for" does, so
fixed-target mode can be judged by people and phantoms and not only by boxes. Without it, nothing
below the reporting threshold is ever loaded and the run is byte for byte the one that pins the
gate. The floor is **the camera's**, imported and not rewritten: this replay used to load from 0.10,
below both the camera's band and the tracker's own `track_low_thresh`, so it served boxes the drone
would never have given a tracker in the first place.

**`--plantilla=f.npy`.** The other half of the operator's click: what the target **looks like**.
With it, a doubted box is kept wherever it falls and not only inside the projected window. The
vector is read from a file so this script still knows nothing about who the flight's letters belong
to; building the template from the hand labels is the measuring script's job, not the replay's.

**`--sintetico=V`.** Adds a target with **known truth** that patrols east-west at V m/s across the
scene, for the whole flight. It is projected into every airborne frame using the flight's own poses,
and a frame that has it in view detects it with the probability measured for a real target in view
on this flight, with some pixel noise, under a single track id standing in for a tracker that holds
it. It is called 'boat' so its reports are distinguishable from the flight's. Nothing downstream
knows it is synthetic, so what comes out is what the chain does with a **moving** target: whether it
reports it, when, where and in how many pieces.

**`--actitud`, `--actitud-roll=+1/-1`.** Puts the airframe pitch recorded in `frames.csv` into every
ray, and adds roll with that sign. The sign is a switch because the August analysis could not
resolve it with this flight.

**`--candidatos=file.json`.** Writes each candidate together with the ids of the tracks fused into
it. That link is what allows a candidate to be traced back to the detections that fed it, and from
there to the label a human gave each box, which is how `scripts/personas_encontradas.py` scores the
chain **by person** instead of by box. It is written from the live run instead of saved as a file on
disk, so the scoreboard always describes the identity layer as it stands, not as it stood.

### How the two modes differ

**WITHOUT `--vivo`** nothing about the loop changes: it runs as fast as it can and the reports are
published at the end, which is what the equivalence gate measures. The station is spoken to with
the transport envelope and not with the raw report: `{"message": <json string>, "source": <node id>}`.

**WITH `--vivo`** the flight goes at wall-clock pace so there is time to click partway through, the
drone asks the station what it should be looking for, and each report goes out **as it is produced**.
That last part is what makes a class change visible: a batch sent at the end would show the final
answer and hide the moment it changed. The order is applied by the protocol's **own handler**, the
one a drone runs, so the replay exercises the code path that flies instead of a copy of it, and a
stale or repeated order is ignored just the same. What is printed is the second **of the flight** and
not the wall clock, because whether an order arrived in time is a question about the flight, not
about the operator.

With `--foco`, the operator points at what **the map** showed, not at the surveyed truth: fixing the
true position would measure a mode nobody can use. With `--pasada`, the flight is cut at the midpoint
of its **time** and not at a frame count, because the cadence varies and half the frames are not half
the flight.

What the final report **leaves out** is as informative as what it carries: a candidate that formed
and never matured is a target the drone crossed once and did not linger on, which is a property of
the flight plan and not of the detector.

---

## The labelling tool (`scripts/etiquetar_grupos.py`)

### The flights it opens

`VUELOS_LISTOS` are the flights whose boxes and embeddings are already on disk, so `--vuelo` fills
in the four paths. `VUELOS` are the flights with detector candidates from `proponer_cajas.py`, where
`--vuelo` also fills in `--lista-frames` and `--nombre`, and the labels go to
`etiquetas_detector_<vuelo>.json`, apart from any other labelling of that flight.

- **02ago** is the test flight, converted by `scripts/convertir_02ago.py` so it can be reviewed too.
- **02ago_alto** are its frames 9315-9865: the only unlabelled high-altitude material there is, and
  from the same day as the test, so training on it **flatters** the test score.
- **02ago_huecos** are the frames **between** the test windows, right up against them, so they serve
  to **measure** and never to train: without them, a candidate living there cannot be judged either
  way.
- **14jun** is the only material with a second person in it.

`_PERSONA_POR_ALTURA` is how many pixels tall a person came out, by altitude, **measured** over the
labelled boxes of these flights instead of derived from the optics: the camera looks forward and
down, so at low altitude the person is far away along the ground and does not grow the way a nadir
view would predict. The table is in [NOTES.md](../NOTES.md).

### What the labeller sees

`MUESTRA` is how many crops a group shows: enough to see what it is, few enough to load fast. `LADO`
is the crop side in pixels, and at 96 the text on a context box is a blur. The three colours are
BGR: yellow the box being labelled, magenta the flight's own detections (those of `--contexto`), and
cyan the other boxes of the CSV being labelled in the same frame.

`REVISION` holds the labels of the per-frame review. **"duplicado"** is a second box over a person
who already has one, and it is thrown away. **"ignorar"** is something that cannot be called one
thing or the other, like a lone foot or a person clipped to a sliver by the frame edge: the export
deletes it, so the detector is neither rewarded nor punished for finding it.

`REPASO` is the fraction of reviewed frames the blind re-check asks about again, and
`SEMILLA_REPASO` is fixed so reopening the tool asks about the same ones. `LADO_MIN_NUEVA` is the
side in pixels below which a drawn box is a slip of the mouse.

### What the page saves

A drawn box is `[x1, y1, x2, y2]` for a person, or `[x1, y1, x2, y2, "ignorar"]`. A CSV box whose
corners were dragged **is also** saved: the label belongs to the box, so the box has to be fixable,
or a detection covering only the legs stays wrong forever. Confirmed pairs are the ones the labeller
said are **two people standing together** and not one boxed twice; without recording them they stay
flagged forever and the frame never stops counting as a problem.

The crops are ordered **unlabelled first and then largest**, so the next click is always the one that
labels the most. Each crop carries **its own** final label, so a box corrected apart from its group
shows. A **gap** is a frame with nobody in it between two that do have somebody, and it is queued
along with every other kind of pending frame.

The crops are cut with a margin, because a box drawn tight at altitude cuts away the context that
tells a person from a post, and the borders are drawn **after** resizing, so their thickness is
thumbnail pixels whatever the size of the box.
